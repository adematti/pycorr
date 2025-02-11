"""Implements base two-point counter, to be extended when implementing a new engine."""

import os
import time
from collections import namedtuple

import numpy as np

from .utils import BaseClass, get_mpi, _make_array, _make_array_like, _nan_to_zero
from . import utils


TwoPointWeight = namedtuple('TwoPointWeight', ['sep', 'weight'])


class TwoPointCounterError(ValueError):

    """Exception raised when issue with two-point counting."""


def get_twopoint_counter(engine='corrfunc'):
    """
    Return :class:`BaseTwoPointCounter`-subclass corresponding to input engine name.

    Parameters
    ----------
    engine : string, default='corrfunc'
        Name of two-point counter engine, one of ["corrfunc", "analytic", "jackknife"].

    Returns
    -------
    counter : type
        Two-point counter class.
    """
    if isinstance(engine, str):

        if engine.lower() == 'corrfunc':
            from . import corrfunc  # adds counter to BaseTwoPointCounter._registry

        try:
            engine = BaseTwoPointCounter._registry[engine.lower()]
        except KeyError:
            raise TwoPointCounterError('Unknown two-point counter {}.'.format(engine))

    return engine


class RegisteredTwoPointCounter(type(BaseClass)):

    """Metaclass registering :class:`BaseTwoPointCounter`-derived classes."""

    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry[cls.name] = cls
        return cls


class MetaTwoPointCounter(type(BaseClass)):

    """Metaclass to return correct two-point counter engine."""

    def __call__(cls, *args, engine='corrfunc', **kwargs):
        return get_twopoint_counter(engine)(*args, **kwargs)


class TwoPointCounter(BaseClass, metaclass=MetaTwoPointCounter):
    """
    Entry point to two-point counter engines.

    Parameters
    ----------
    engine : string, default='corrfunc'
        Name of two-point counter engine, one of ["corrfunc", "analytical"].

    args : list
        Arguments for two-point counter engine, see :class:`BaseTwoPointCounter`.

    kwargs : dict
        Arguments for two-point counter engine, see :class:`BaseTwoPointCounter`.

    Returns
    -------
    engine : BaseTwoPointCounter
    """
    @classmethod
    def from_state(cls, state, load=False):
        """Return new two-point counter based on state dictionary."""
        state = state.copy()
        cls = get_twopoint_counter(state.pop('name'))
        new = cls.__new__(cls)
        new.__setstate__(state, load=load)
        return new


def _vlogical_and(*arrays):
    # & between any number of arrays
    toret = arrays[0].copy()
    for array in arrays[1:]: toret &= array
    return toret


def get_default_nrealizations(weights):
    """Return default number of realizations given input bitwise weights = the number of bits in input weights plus one."""
    return 1 + 8 * sum(weight.dtype.itemsize for weight in weights)


def get_inverse_probability_weight(*weights, noffset=1, nrealizations=None, default_value=0., correction=None, dtype='f8'):
    r"""
    Return inverse probability weight given input bitwise weights.
    Inverse probability weight is computed as::math:`\mathrm{nrealizations}/(\mathrm{noffset} + \mathrm{popcount}(w_{1} \& w_{2} \& ...))`.
    If denominator is 0, weight is set to default_value.

    Parameters
    ----------
    weights : int arrays
        Bitwise weights.

    noffset : int, default=1
        The offset to be added to the bitwise counts in the denominator (defaults to 1).

    nrealizations : int, default=None
        Number of realizations (defaults to the number of bits in input weights plus one).

    default_value : float, default=0.
        Default weight value, if the denominator is zero (defaults to 0).

    correction : 2D array, default=None
        Optionally, divide weight by ``correction[nbits1, nbits2]`` with ``nbits1``, ``nbits2`` the number of non-zero bits in weights.

    dtype : string, np.dtype
        Type for output weight.

    Returns
    -------
    weight : array
        IIP weight.
    """
    if nrealizations is None:
        nrealizations = get_default_nrealizations(weights[0])
    # denom = noffset + sum(utils.popcount(w1 & w2) for w1, w2 in zip(*weights))
    denom = noffset + sum(utils.popcount(_vlogical_and(*weight)) for weight in zip(*weights))
    mask = denom == 0
    denom[mask] = 1
    toret = np.empty_like(denom, dtype=dtype)
    toret[...] = nrealizations / denom
    if correction is not None:
        c = tuple(sum(utils.popcount(w) for w in weight) for weight in weights)
        toret /= correction[c]
    toret[mask] = default_value
    return toret


def _format_positions(positions, mode='auto', position_type='xyz', dtype=None, copy=True, mpicomm=None, mpiroot=None):
    # Format input array of positions
    # position_type in ["xyz", "rdd", "pos"]
    mode = mode.lower()
    position_type = position_type.lower()
    if position_type == 'auto':
        if mode in ['theta', 'angular']: position_type = 'rd'
        else: position_type = 'xyz'

    def __format_positions(positions):
        pt = position_type
        if position_type == 'pos':  # array of shape (N, 3)
            positions = np.array(positions, dtype=dtype, copy=copy)
            if positions.shape[-1] != 3:
                return None, 'For position type = {}, please provide a (N, 3) array for positions'.format(position_type)
            positions = positions.T
            pt = 'xyz'
        # Array of shape (3, N)
        positions = list(positions)
        for ip, p in enumerate(positions):
            # Cast to the input dtype if exists (may be set by previous positions)
            positions[ip] = np.array(p, dtype=dtype, copy=copy)

        size = len(positions[0])
        dt = positions[0].dtype
        if not np.issubdtype(dt, np.floating):
            return None, 'Input position arrays should be of floating type, not {}'.format(dt)
        for p in positions[1:]:
            if len(p) != size:
                return None, 'All position arrays should be of the same size'
            if p.dtype != dt:
                return None, 'All position arrays should be of the same type, you can e.g. provide dtype'
        if pt != 'auto' and len(positions) != len(pt):
            return None, 'For position type = {}, please provide a list of {:d} arrays for positions (found {:d})'.format(pt, len(pt), len(positions))

        if mode in ['theta', 'angular']:
            if pt == 'xyz':
                positions = utils.cartesian_to_sky(positions, degree=True)[:2]
            elif pt in ['rdd', 'rdz']:
                positions = positions[:2]
            elif pt != 'rd':
                return None, 'For mode = {}, position type should be one of ["xyz", "rdz", "rd"]'.format(mode)
        else:
            if pt == 'rdd':
                positions = utils.sky_to_cartesian(positions, degree=True)
            elif pt != 'xyz':
                return None, 'For mode = {}, position type should be one of ["pos", "xyz", "rdd"]'.format(mode)
        return list(positions), None

    error = None
    if mpiroot is None or (mpicomm.rank == mpiroot):
        if positions is not None and (position_type == 'pos' or not all(position is None for position in positions)):
            positions, error = __format_positions(positions)  # return error separately to raise on all processes
    if mpicomm is not None:
        error = mpicomm.allgather(error)
    else:
        error = [error]
    errors = [err for err in error if err is not None]
    if errors:
        raise TwoPointCounterError(errors[0])
    if mpiroot is not None and mpicomm.bcast(positions is not None if mpicomm.rank == mpiroot else None, root=mpiroot):
        n = mpicomm.bcast(len(positions) if mpicomm.rank == mpiroot else None, root=mpiroot)
        if mpicomm.rank != mpiroot: positions = [None] * n
        positions = [get_mpi().scatter(position, mpicomm=mpicomm, mpiroot=mpiroot) for position in positions]
    return positions


def _format_weights(weights, weight_type='auto', size=None, dtype=None, copy=True, mpicomm=None, mpiroot=None):
    # Format input weights, as a list of n_bitwise_weights uint8 arrays, and optionally a float array for individual weights.
    # Return formated list of weights, and n_bitwise_weights.

    def __format_weights(weights):
        islist = isinstance(weights, (tuple, list)) or getattr(weights, 'ndim', 1) == 2
        if not islist:
            weights = [weights]
        if all(weight is None for weight in weights):
            return [], 0
        individual_weights, bitwise_weights = [], []
        for w in weights:
            if np.issubdtype(w.dtype, np.integer):
                if weight_type == 'product_individual':  # enforce float individual weight
                    individual_weights.append(w)
                else:  # certainly bitwise weight
                    bitwise_weights.append(w)
            else:
                individual_weights.append(w)
        # any integer array bit size will be a multiple of 8
        bitwise_weights = utils.reformat_bitarrays(*bitwise_weights, dtype=np.uint8, copy=copy)
        n_bitwise_weights = len(bitwise_weights)
        weights = bitwise_weights
        if individual_weights:
            if len(individual_weights) > 1 or copy:
                weight = np.prod(individual_weights, axis=0, dtype=dtype)
            else:
                weight = individual_weights[0].astype(dtype, copy=False)
            weights += [weight]
        return weights, n_bitwise_weights

    weights, n_bitwise_weights = __format_weights(weights)
    if mpiroot is None:
        if mpicomm is not None:
            size_weights = mpicomm.allgather(len(weights))
            if len(set(size_weights)) != 1:
                raise ValueError('mpiroot = None but weights are None/empty on some ranks')
    else:
        n = mpicomm.bcast(len(weights) if mpicomm.rank == mpiroot else None, root=mpiroot)
        if mpicomm.rank != mpiroot: weights = [None] * n
        weights = [get_mpi().scatter(weight, mpicomm=mpicomm, mpiroot=mpiroot) for weight in weights]
        n_bitwise_weights = mpicomm.bcast(n_bitwise_weights, root=mpiroot)

    if size is not None:
        if not all(len(weight) == size for weight in weights):
            raise ValueError('All weight arrays should be of the same size as position arrays')
    return weights, n_bitwise_weights


class BaseTwoPointCounter(BaseClass, metaclass=RegisteredTwoPointCounter):
    """
    Base class for two-point counters.
    Extend this class to implement a new two-point counter engine.

    Attributes
    ----------
    wcounts : array
        (Optionally weighted) two-point counts.

    wnorm : float
        Two-point count normalization.
    """
    name = 'base'

    def __init__(self, mode, edges, positions1, positions2=None, weights1=None, weights2=None,
                 bin_type='auto', position_type='auto', weight_type='auto', weight_attrs=None,
                 twopoint_weights=None, selection_attrs=None, los='midpoint', boxsize=None, compute_sepsavg=True, dtype='f8',
                 nthreads=None, mpicomm=None, mpiroot=None, **kwargs):
        r"""
        Initialize :class:`BaseTwoPointCounter`, and run actual two-point counts
        (calling :meth:`run`), setting :attr:`wcounts` and :attr:`sep`.

        Parameters
        ----------
        mode : string
            Type of two-point counts, one of:

                - "theta": as a function of angle (in degree) between two particles
                - "s": as a function of distance between two particles
                - "smu": as a function of distance between two particles and cosine angle :math:`\mu` w.r.t. the line-of-sight
                - "rppi": as a function of distance transverse (:math:`r_{p}`) and parallel (:math:`\pi`) to the line-of-sight
                - "rp": same as "rppi", without binning in :math:`\pi`

        edges : tuple, array
            Tuple of bin edges (arrays), for the first (e.g. :math:`r_{p}`)
            and optionally second (e.g. :math:`\pi \in [-\infty, \infty]`, :math:`\mu \in [-1, 1]`) dimensions.
            In case of single-dimension binning (e.g. ``mode`` is "theta", "s" or "rp"),
            the single array of bin edges can be provided directly.
            Edges are inclusive on the low end, exclusive on the high end,
            i.e. a pair separated by :math:`s` falls in bin `i` if ``edges[i] <= s < edges[i+1]``.
            In case ``mode`` is "smu" however, the first :math:`\mu`-bin is also exclusive on the low end
            (increase the :math:`\mu`-range by a tiny value to include :math:`\mu = \pm 1`).
            Pairs at separation :math:`s = 0` are included in the :math:`\mu = 0` bin.
            Similarly, in case ``mode`` is "rppi", the first :math:`\pi`-bin is also exclusive on the low end
            and pairs at separation :math:`s = 0` are included in the :math:`\pi = 0` bin.
            In case of auto-correlation (no ``positions2`` provided), auto-pairs (pairs of same objects) are not counted.
            In case of cross-correlation, all pairs are counted.
            In any case, duplicate objects (with separation zero) will be counted.

        positions1 : list, array
            Positions in the first catalog. Typically of shape (3, N), but can be (2, N) when ``mode`` is "theta".
            See ``position_type``.

        positions2 : list, array, default=None
            Optionally, for cross-two-point counts, positions in the second catalog. See ``positions1``.

        weights1 : array, list, default=None
            Weights of the first catalog. Not required if ``weight_type`` is either ``None`` or "auto".
            See ``weight_type``.

        weights2 : array, list, default=None
            Optionally, for cross-two-point counts, weights in the second catalog. See ``weights1``.

        bin_type : string, default='auto'
            Binning type for first dimension, e.g. :math:`r_{p}` when ``mode`` is "rppi".
            Set to ``lin`` for speed-up in case of linearly-spaced bins.
            In this case, the bin number for a pair separated by a (3D, projected, angular...) separation
            ``sep`` is given by ``(sep - edges[0])/(edges[-1] - edges[0])*(len(edges) - 1)``,
            i.e. only the first and last bins of input edges are considered.
            Then setting ``compute_sepsavg`` is virtually costless.
            For non-linear binning, set to "custom".
            "auto" allows for auto-detection of the binning type:
            linear binning will be chosen if input edges are
            within ``rtol = 1e-05`` (relative tolerance) *or* ``atol = 1e-08``
            (absolute tolerance) of the array
            ``np.linspace(edges[0], edges[-1], len(edges))``.

        position_type : string, default='auto'
            Type of input positions, one of:

                - "rd": RA/Dec in degree, only if ``mode`` is "theta"
                - "rdd": RA/Dec in degree, distance, for any ``mode``
                - "xyz": Cartesian positions, shape (3, N)
                - "pos": Cartesian positions, shape (N, 3).

        weight_type : string, default='auto'
            The type of weighting to apply to provided weights. One of:

                - ``None``: no weights are applied.
                - "product_individual": each pair is weighted by the product of weights :math:`w_{1} w_{2}`.
                - "inverse_bitwise": each pair is weighted by :math:`\mathrm{nrealizations}/(\mathrm{noffset} + \mathrm{popcount}(w_{1} \& w_{2}))`.
                   Multiple bitwise weights can be provided as a list.
                   Individual weights can additionally be provided as float arrays.
                   In case of cross-correlations with floating weights, bitwise weights are automatically turned to IIP weights,
                   i.e. :math:`\mathrm{nrealizations}/(\mathrm{noffset} + \mathrm{popcount}(w_{1}))`.
                - "auto": automatically choose weighting based on input ``weights1`` and ``weights2``,
                   i.e. ``None`` when ``weights1`` and ``weights2`` are ``None``,
                   "inverse_bitwise" if one of input weights is integer, else "product_individual".

            In addition, angular upweights can be provided with ``twopoint_weights``.

        weight_attrs : dict, default=None
            Dictionary of weighting scheme attributes. In case ``weight_type`` is "inverse_bitwise",
            one can provide "nrealizations", the total number of realizations (*including* current one;
            defaulting to the number of bits in input weights plus one);
            "noffset", the offset to be added to the bitwise counts in the denominator (defaulting to 1)
            and "default_value", the default value of pairwise weights if the denominator is zero (defaulting to 0).
            The method used to compute the normalization with PIP weights can be specified with the keyword "normalization":
            if ``None`` or "total", normalization is given by eq. 22 of arXiv:1912.08803; "brute_force" (using OpenMP'ed C code)
            or "brute_force_npy" (slower, using numpy only methods; both methods match within machine precision) loop over all pairs;
            "counter" to normalize each pair by eq. 19 of arXiv:1912.08803.
            For normalizations "total" or "counter", "nalways" specifies the number of bits systematically set to 1 minus the number of bits systematically set to 0 (defaulting to 0).
            For example, for the "zero-truncated" estimator (arXiv:1912.08803), one would use noffset = 0, nalways = 1.

        twopoint_weights : WeightTwoPointEstimator, default=None
            Weights to be applied to each pair of particles.
            A :class:`WeightTwoPointEstimator` instance or any object with arrays ``sep``
            (separations) and ``weight`` (weight at given separation) as attributes
            (i.e. to be accessed through ``twopoint_weights.sep``, ``twopoint_weights.weight``)
            or as keys (i.e. ``twopoint_weights['sep']``, ``twopoint_weights['weight']``)
            or as element (i.e. ``sep, weight = twopoint_weights``).

        selection_attrs : dict, default=None
            To select pairs to be counted, provide mapping between the quantity (string)
            and the interval (tuple of floats),
            e.g. ``{'rp': (0., 20.)}`` to select pairs with transverse separation 'rp' between 0 and 20,
            `{'theta': (0., 20.)}`` to select pairs with separation angle 'theta' between 0 and 20 degrees.

        los : string, default='midpoint'
            Line-of-sight to be used when ``mode`` is "smu", "rppi" or "rp"; one of:

                - "x", "y" or "z": Cartesian axis
                - "midpoint": the mean position of the pair: :math:`\mathbf{\eta} = (\mathbf{r}_{1} + \mathbf{r}_{2})/2`
                - "firstpoint": the first position of the pair: :math:`\mathbf{\eta} = \mathbf{r}_{1}`
                - "endpoint": the second position of the pair: :math:`\mathbf{\eta} = \mathbf{r}_{2}`

            WARNING: "endpoint" is obtained by reversing "firstpoint" (which is the only line-of-sight implemented in the counter).
            This means, if "s" or "rp" edges starts at 0, and the number of "mu" or "pi" bins is even,
            zero separation pairs (due to duplicate objects) will be counted in ``counts[0, (counts.shape[1] - 1) // 2]`` instead of ``counts[0, counts.shape[1] // 2]``.

        boxsize : array, float, default=None
            For periodic wrapping, the side-length(s) of the periodic cube.

        compute_sepsavg : bool, default=True
            Set to ``False`` to *not* calculate the average separation for each bin.
            This can make the two-point counts faster if ``bin_type`` is "custom".
            In this case, :attr:`sep` will be set the midpoint of input edges.

        dtype : string, np.dtype, default='f8'
            Array type for positions and weights.
            If ``None``, defaults to type of first ``positions1`` array.
            Double precision is highly recommended in case ``mode`` is "theta",
            ``twopoint_weights`` is provided (due to cosine), or ``compute_sepsavg`` is ``True``.

        nthreads : int, default=None
            Number of OpenMP threads to use.

        mpicomm : MPI communicator, default=None
            The MPI communicator, to MPI-distribute calculation.

        mpiroot : int, default=None
            In case ``mpicomm`` is provided, if ``None``, input positions and weights are assumed to be scattered across all ranks.
            Else the MPI rank where input positions and weights are gathered.

        kwargs : dict
            Two-point counter engine-specific options.
        """
        self.attrs = kwargs
        self.mpicomm = mpicomm
        if self.mpicomm is None and mpiroot is not None:
            raise TwoPointCounterError('mpiroot is not None, but no mpicomm provided')
        self._set_nthreads(nthreads)
        self._set_mode(mode)
        self._set_boxsize(boxsize)
        self._set_edges(edges, bin_type=bin_type)
        self._set_los(los)
        self._set_compute_sepsavg(compute_sepsavg)
        self._set_positions(positions1, positions2, position_type=position_type, dtype=dtype, copy=False, mpiroot=mpiroot)
        self._set_weights(weights1, weights2, weight_type=weight_type, twopoint_weights=twopoint_weights, weight_attrs=weight_attrs, copy=False, mpiroot=mpiroot)
        self._set_selection(selection_attrs)
        self._set_zeros()
        self._set_reversible()
        self.wnorm = self.normalization()
        t0 = time.time()
        self.run()
        t1 = time.time()
        if not self.with_mpi or self.mpicomm.rank == 0:
            self.log_debug('Two-point counts computed in elapsed time {:.2f} s.'.format(t1 - t0))
        del self.positions1, self.positions2, self.weights1, self.weights2

    def run(self):
        """
        Method that computes the actual two-point counts and set :attr:`wcounts` and :attr:`sep`,
        to be implemented in your new engine.
        """
        raise NotImplementedError('Implement method "run" in your {}'.format(self.__class__.__name__))

    def _set_nthreads(self, nthreads):
        if nthreads is None:
            self.nthreads = int(os.getenv('OMP_NUM_THREADS', '1'))
        else:
            self.nthreads = int(nthreads)

    def _set_mode(self, mode):
        self.mode = mode.lower()

    def _set_reversible(self):
        self.is_reversible = self.mode in ['theta', 's', 'rp']
        if self.mode in ['smu', 'rppi']:
            self.is_reversible = self.autocorr or (self.los_type not in ['firstpoint', 'endpoint'])  # even smu is reversible for midpoint los, i.e. positions1 <-> positions2

    def _set_zeros(self):
        self._set_default_seps()
        self.wcounts = np.zeros_like(self.sep, dtype='f8')
        self.ncounts = np.zeros_like(self.sep, dtype='i8')

    def _set_compute_sepsavg(self, compute_sepsavg):
        if 'compute_sepavg' in self.attrs:
            self.log_warning('Use compute_sepsavg instead of compute_sepavg.')
            compute_sepsavg = self.attrs.pop('compute_sepavg')
        if np.ndim(compute_sepsavg) == 0:
            compute_sepsavg = (compute_sepsavg,) * self.ndim
        self.compute_sepsavg = [bool(c) for c in compute_sepsavg]
        if len(self.compute_sepsavg) != self.ndim:
            raise TwoPointCounterError('compute_sepsavg must be either a boolean or its length must match number of dimensions = {:d} for mode = {:d}'.format(self.ndim, self.mode))

    def _set_edges(self, edges, bin_type='auto'):
        if np.ndim(edges[0]) == 0:
            edges = (edges,)
        self.edges = [np.array(edge, dtype='f8') for edge in edges]
        if self.mode in ['smu', 'rppi']:
            if not self.ndim == 2:
                raise TwoPointCounterError('A tuple of edges should be provided to two-point counter in mode {}'.format(self.mode))
        else:
            if not self.ndim == 1:
                raise TwoPointCounterError('Only one edge array should be provided to two-point counter in mode {}'.format(self.mode))
        if self.mode in ['smu', 'rppi']:
            edges = self.edges[1]
            if np.allclose(edges[0], 0.):
                axis = {'smu': 'mu', 'rppi': 'pi'}[self.mode]
                import warnings
                nedges = 2 * len(edges) - 1
                warnings.warn('{} edges starting at 0 is deprecated, please use symmetric binning; I am assuming np.linspace({:.4f}, {:.4f}, {:d})!'.format(axis, -edges[-1], edges[-1], nedges))
                self.edges[1] = np.linspace(-edges[-1], edges[-1], nedges)
        if np.any(self.edges[0] < 0.):
            raise TwoPointCounterError('First edges must be >= 0')
        if not all(np.all(np.diff(edges) > 0.) for edges in self.edges):
            raise TwoPointCounterError('Edges must be strictly increasing')
        if self.mode == 'smu' and not np.all((self.edges[1] >= -1.01) & (self.edges[1] <= 1.01)):
            raise TwoPointCounterError('In mode smu, mu-edges must be in [-1, 1]')
        self._set_bin_type(bin_type)

    def _set_bin_type(self, bin_type):
        self.bin_type = bin_type.lower()
        allowed_bin_types = ['lin', 'custom', 'auto']
        if self.bin_type not in allowed_bin_types:
            raise TwoPointCounterError('bin type should be one of {}'.format(allowed_bin_types))
        if self.bin_type == 'auto':
            edges = self.edges[0]
            if np.allclose(edges, np.linspace(edges[0], edges[-1], len(edges))):
                self.bin_type = 'lin'

    @property
    def shape(self):
        """Return shape of obtained counts :attr:`wcounts`."""
        return tuple(len(edges) - 1 for edges in self.edges)

    @property
    def ndim(self):
        """Return binning dimensionality."""
        return len(self.edges)

    @property
    def periodic(self):
        """Whether periodic wrapping is used (i.e. :attr:`boxsize` is not ``None``)."""
        return self.boxsize is not None

    @property
    def with_mpi(self):
        """Whether to use MPI."""
        if not hasattr(self, 'mpicomm'): self.mpicomm = None
        return self.mpicomm is not None and self.mpicomm.size > 1

    def _set_positions(self, positions1, positions2=None, position_type='auto', dtype=None, copy=False, mpiroot=None):
        self.positions1 = _format_positions(positions1, mode=self.mode, position_type=position_type, dtype=dtype, copy=copy, mpicomm=self.mpicomm, mpiroot=mpiroot)
        self.dtype = self.positions1[0].dtype
        self.positions2 = _format_positions(positions2, mode=self.mode, position_type=position_type, dtype=self.dtype, copy=copy, mpicomm=self.mpicomm, mpiroot=mpiroot)
        self.autocorr = self.positions2 is None
        if self.periodic:
            self.positions1 = [p % b.astype(p.dtype) for p, b in zip(self.positions1, self.boxsize)]
            if not self.autocorr:
                self.positions2 = [p % b.astype(p.dtype) for p, b in zip(self.positions2, self.boxsize)]

        self._size1 = self._size2 = len(self.positions1[0])
        if not self.autocorr: self._size2 = len(self.positions2[0])
        self.size1, self.size2 = self._size1, self._size2
        if self.with_mpi:
            self.size1, self.size2 = self.mpicomm.allreduce(self._size1), self.mpicomm.allreduce(self._size2)

    def _set_weights(self, weights1, weights2=None, weight_type='auto', twopoint_weights=None, weight_attrs=None, copy=False, mpiroot=None):

        if weight_type is not None: weight_type = weight_type.lower()
        allowed_weight_types = [None, 'auto', 'product_individual', 'inverse_bitwise']
        if weight_type not in allowed_weight_types:
            raise TwoPointCounterError('weight_type should be one of {}'.format(allowed_weight_types))
        self.weight_type = weight_type

        weight_attrs = weight_attrs or {}
        self.weight_attrs = {}

        self.n_bitwise_weights = 0
        if weight_type is None:
            self.weights1 = self.weights2 = []

        else:
            self.weight_attrs.update(nalways=weight_attrs.get('nalways', 0), normalization=weight_attrs.get('normalization', None), correction=weight_attrs.get('correction', None))
            noffset = weight_attrs.get('noffset', 1)
            if int(noffset) != noffset:
                raise TwoPointCounterError('Only integer offset accepted')
            noffset = int(noffset)
            default_value = weight_attrs.get('default_value', 0.)
            self.weight_attrs.update(noffset=noffset, default_value=default_value)

            self.weights1, n_bitwise_weights1 = _format_weights(weights1, weight_type=self.weight_type, size=self._size1, dtype=self.dtype, copy=copy, mpicomm=self.mpicomm, mpiroot=mpiroot)

            def get_nrealizations(n_bitwise_weights):
                nrealizations = weight_attrs.get('nrealizations', None)
                if nrealizations is None:  # 8 because we ask for 8-bit integers, whatever dtype is
                    nrealizations = n_bitwise_weights * 8 + 1
                return nrealizations

            self.weights2, n_bitwise_weights2 = _format_weights(weights2, weight_type=self.weight_type, size=self._size2, dtype=self.dtype, copy=copy, mpicomm=self.mpicomm, mpiroot=mpiroot)
            self.same_shotnoise = self.autocorr and bool(self.weights2)

            if self.same_shotnoise:
                self.positions2 = self.positions1
                self.autocorr = False

            if self.autocorr:

                nrealizations = get_nrealizations(n_bitwise_weights1)
                self.weight_attrs.update(nrealizations=nrealizations)
                self.weights2 = self.weights1
                self.n_bitwise_weights = n_bitwise_weights1

            else:
                if n_bitwise_weights2 == n_bitwise_weights1:

                    nrealizations = get_nrealizations(n_bitwise_weights1)
                    self.n_bitwise_weights = n_bitwise_weights1

                else:

                    if n_bitwise_weights2 == 0:
                        indweights = self.weights1[n_bitwise_weights1] if len(self.weights1) > n_bitwise_weights1 else 1.
                        nrealizations = get_nrealizations(n_bitwise_weights1)
                        self.weights1 = [get_inverse_probability_weight(self.weights1[:n_bitwise_weights1], nrealizations=nrealizations,
                                                                        noffset=noffset, default_value=default_value, dtype=self.dtype) * indweights]
                        self.n_bitwise_weights = 0
                        if not self.with_mpi or self.mpicomm.rank == 0:
                            self.log_info('Setting IIP weights for first catalog.')
                    elif n_bitwise_weights1 == 0:
                        indweights = self.weights2[n_bitwise_weights2] if len(self.weights2) > n_bitwise_weights2 else 1.
                        nrealizations = get_nrealizations(n_bitwise_weights2)
                        self.weights2 = [get_inverse_probability_weight(self.weights2[:n_bitwise_weights2], nrealizations=nrealizations,
                                                                        noffset=noffset, default_value=default_value, dtype=self.dtype) * indweights]
                        self.n_bitwise_weights = 0
                        if not self.with_mpi or self.mpicomm.rank == 0:
                            self.log_info('Setting IIP weights for second catalog.')
                    else:
                        raise TwoPointCounterError('Incompatible length of bitwise weights: {:d} and {:d} bytes'.format(n_bitwise_weights1, n_bitwise_weights2))

                self.weight_attrs.update(nrealizations=nrealizations)

        if len(self.weights1) == len(self.weights2) + 1:
            self.weights2.append(np.ones(self._size2, dtype=self.dtype))
        elif len(self.weights1) == len(self.weights2) - 1:
            self.weights1.append(np.ones(self._size1, dtype=self.dtype))
        elif len(self.weights1) != len(self.weights2):
            raise TwoPointCounterError('Something fishy happened with weights; number of weights1/weights2 is {:d}/{:d}'.format(len(self.weights1), len(self.weights2)))

        normalization = self.weight_attrs['normalization'] = self.weight_attrs.get('normalization', 'total') or 'total'
        allowed_normalizations = ['total', 'brute_force', 'brute_force_npy', 'counter']
        if normalization not in allowed_normalizations:
            raise TwoPointCounterError('normalization should be one of {}'.format(allowed_normalizations))

        if self.n_bitwise_weights and self.weight_attrs['normalization'] == 'counter' and self.weight_attrs['correction'] is None:
            nrealizations, nalways = self.weight_attrs['nrealizations'], self.weight_attrs['nalways']
            noffset = self.weight_attrs['noffset']
            joint = utils.joint_occurences(nrealizations, noffset=noffset + nalways, default_value=self.weight_attrs['default_value'])
            correction = np.ones((1 + self.n_bitwise_weights * 8,) * 2, dtype=self.dtype)
            cmin, cmax = nalways, min(nrealizations - noffset, self.n_bitwise_weights * 8)
            for c1 in range(cmin, 1 + cmax):
                for c2 in range(cmin, 1 + cmax):
                    correction[c1][c2] = joint[c1 - nalways][c2 - nalways] if c2 <= c1 else joint[c2 - nalways][c1 - nalways]
                    correction[c1][c2] /= (nrealizations / (noffset + c1) * nrealizations / (noffset + c2))
            self.weight_attrs['correction'] = correction

        self.twopoint_weights = twopoint_weights
        self.cos_twopoint_weights = None
        if twopoint_weights is not None:
            if self.periodic:
                raise TwoPointCounterError('Cannot use angular weights in case of periodic boundary conditions (boxsize)')
            try:
                sep = twopoint_weights.sep
                weight = twopoint_weights.weight
            except AttributeError:
                try:
                    sep = twopoint_weights['sep']
                    weight = twopoint_weights['weight']
                except (IndexError, TypeError):
                    sep, weight = twopoint_weights
            # just to make sure we use the correct dtype
            sep = np.cos(np.radians(np.array(sep, dtype=self.dtype)))
            argsort = np.argsort(sep)
            self.cos_twopoint_weights = TwoPointWeight(sep=np.array(sep[argsort], dtype=self.dtype), weight=np.array(weight[argsort], dtype=self.dtype))

    def _set_selection(self, selection_attrs=None):
        self.selection_attrs = {str(name): tuple(float(v) for v in value) for name, value in (selection_attrs or {}).items()}

    def _mpi_decompose(self):
        if self.with_mpi:
            smoothing = np.max(self.edges[0])
            if self.mode == 'theta':
                smoothing = 2 * np.sin(0.5 * np.deg2rad(smoothing))
            elif self.mode == 'rppi':
                smoothing = np.sqrt(smoothing**2 + np.max(self.edges[1])**2)
            elif self.mode == 'rp':
                smoothing = np.inf
            return get_mpi().domain_decompose(self.mpicomm, smoothing, self.positions1, weights1=self.weights1,
                                              positions2=self.positions2, weights2=self.weights2, boxsize=self.boxsize)
        return (self.positions1, self.weights1), (self.positions2, self.weights2)

    def _get_default_seps(self):
        return [(edges[1:] + edges[:-1]) / 2. for edges in self.edges]

    def _set_default_seps(self):
        self.seps = list(np.meshgrid(*self._get_default_seps(), indexing='ij'))

    def _set_los(self, los):
        self.los_type = los.lower()
        allowed_los = ['midpoint', 'endpoint', 'firstpoint', 'x', 'y', 'z']
        if self.los_type not in allowed_los:
            raise TwoPointCounterError('los should be one of {}'.format(allowed_los))
        if self.periodic and self.mode != 's':
            allowed_los = ['x', 'y', 'z']
            if self.los_type not in allowed_los:
                raise TwoPointCounterError('los should be one of {} in case of periodic boundary conditions (boxsize)'.format(allowed_los))

    def _set_boxsize(self, boxsize):
        self.boxsize = boxsize
        if self.periodic:
            if self.mode == 'theta':
                raise TwoPointCounterError('boxsize must not be provided with mode = theta (no periodic conditions available)')
            self.boxsize = _make_array(boxsize, 3, dtype='f8')

    def _sum_auto_weights(self):
        """Return auto-counts, that are pairs of same objects."""
        if not self.autocorr and not self.same_shotnoise:
            return 0.
        weights = 1.
        if self.cos_twopoint_weights is not None:
            weights *= np.interp(1., self.cos_twopoint_weights.sep, self.cos_twopoint_weights.weight, left=1., right=1.)
        if not self.weights1:
            return self.size1 * weights
        # up to now weights is scalar
        if self.n_bitwise_weights:
            weights *= get_inverse_probability_weight(self.weights1[:self.n_bitwise_weights], self.weights2[:self.n_bitwise_weights], nrealizations=self.weight_attrs['nrealizations'],
                                                      noffset=self.weight_attrs['noffset'], default_value=self.weight_attrs['default_value'], correction=self.weight_attrs['correction'],
                                                      dtype=self.dtype)
        for ii in range(self.n_bitwise_weights, len(self.weights1)):
            weights *= self.weights1[ii] * self.weights2[ii]
        # assert weights.size == len(self.positions1[0])
        weights = np.sum(weights)
        if self.with_mpi:
            weights = self.mpicomm.allreduce(weights)
        return weights

    def normalization(self):
        r"""
        Return two-point count normalization, i.e., in case of cross-correlation:

        .. math::

            \left(\sum_{i=1}^{N_{1}} w_{1,i}\right) \left(\sum_{j=1}^{N_{2}} w_{2,j}\right)

        with the sums running over the weights of the first and second catalogs, and in case of auto-correlation:

        .. math::

            \left(\sum_{i=1}^{N_{1}} w_{1,i}\right)^{2} - \sum_{i=1}^{N_{1}} w_{1,i}^{2}

        """
        method = self.weight_attrs.get('normalization', 'total')
        noffset = self.weight_attrs['noffset']
        nrealizations = self.weight_attrs['nrealizations']
        nalways = self.weight_attrs['nalways']
        default_value = self.weight_attrs['default_value']

        def get_individual_weights(weights):
            indweights = weights[self.n_bitwise_weights:]
            if indweights: return np.prod(indweights, axis=0)

        if self.n_bitwise_weights and method != 'counter':

            if 'brute_force' in method:

                if self.with_mpi:
                    from . import mpi

                weights1, weights2 = self.weights1, self.weights2
                size1, size2 = len(weights1[0]), len(weights2[0])
                indweights1, indweights2 = (get_individual_weights(w) for w in [weights1, weights2])
                bitweights1, bitweights2 = (utils.reformat_bitarrays(*w[:self.n_bitwise_weights], dtype='i8') for w in [weights1, weights2])
                sumw_auto = 0.
                if self.autocorr or self.same_shotnoise:
                    tmp = get_inverse_probability_weight(bitweights1, bitweights2, nrealizations=nrealizations, noffset=noffset, default_value=default_value, correction=self.weight_attrs['correction'], dtype=self.dtype)
                    if indweights1 is not None: tmp *= indweights1 * indweights2
                    sumw_auto = tmp.sum()

                def slice_array(array, start, stop):
                    # Simple enough, but maybe just import mpytools?
                    if not self.with_mpi:
                        return array[start:stop]
                    cumsize = np.cumsum([0] + self.mpicomm.allgather(len(array)))[self.mpicomm.rank]
                    start, stop = max(start - cumsize, 0), max(stop - cumsize, 0)
                    return mpi.gather(array[start:stop], mpiroot=None, mpicomm=self.mpicomm)

                _slab_npairs_max = 1000 * 1000
                if method.endswith('npy'):
                    csize2 = size2
                    if self.with_mpi:
                        csize2 = self.mpicomm.allreduce(size2)
                    nslabs = min(size1 * csize2 // _slab_npairs_max + 1, csize2)
                    sumw_cross = 0
                    for islab in range(nslabs):
                        start, stop = islab * csize2 // nslabs, (islab + 1) * csize2 // nslabs
                        bw1, bw2 = bitweights1, [slice_array(w, start, stop) for w in bitweights2]
                        bw1, bw2 = zip(*[np.meshgrid(ww1, ww2, indexing='ij') for ww1, ww2 in zip(bw1, bw2)])
                        tmp = get_inverse_probability_weight(bw1, bw2, nrealizations=nrealizations, noffset=noffset, default_value=default_value, correction=self.weight_attrs['correction'], dtype=self.dtype)
                        if indweights1 is not None: tmp *= (indweights1[:, None] * slice_array(indweights2, start, stop))
                        sumw_cross += tmp.sum()
                else:
                    from ._utils import sum_weights
                    idtype = 'i{:d}'.format(self.dtype.itemsize)
                    bitweights1, bitweights2 = (np.column_stack(utils.reformat_bitarrays(*w[:self.n_bitwise_weights], dtype=idtype)) for w in [weights1, weights2])
                    if self.with_mpi:
                        sumw_cross = 0.
                        for irank in range(self.mpicomm.size):
                            bw2 = mpi.bcast(bitweights2, mpiroot=irank, mpicomm=self.mpicomm)
                            iw2 = mpi.bcast(indweights2, mpiroot=irank, mpicomm=self.mpicomm) if indweights1 is not None else None
                            sumw_cross += sum_weights(indweights1, iw2, bitweights1, bw2, noffset, default_value / nrealizations, nthreads=self.nthreads)
                    else:
                        sumw_cross = sum_weights(indweights1, indweights2, bitweights1, bitweights2, noffset, default_value / nrealizations, nthreads=self.nthreads)
                    sumw_cross *= nrealizations
                sumw = sumw_cross - sumw_auto
                if self.with_mpi:
                    sumw = self.mpicomm.allreduce(sumw)
                return sumw

            def binned_weights(weights, weights2=None):
                indweights = get_individual_weights(weights)
                bitweights = weights[:self.n_bitwise_weights]
                if weights2 is not None:
                    if indweights is not None:
                        indweights *= get_individual_weights(weights2)
                    bitweights = [w1 & w2 for w1, w2 in zip(bitweights, weights2[:self.n_bitwise_weights])]
                w = np.bincount(utils.popcount(*bitweights),
                                weights=indweights, minlength=self.n_bitwise_weights * 8 + 1)
                if self.with_mpi:
                    w = self.mpicomm.allreduce(w)
                c = np.flatnonzero(w)
                return w[c], c

            w1, c1 = binned_weights(self.weights1)
            if self.autocorr:
                w2, c2 = w1, c1
            else:
                w2, c2 = binned_weights(self.weights2)
            joint = utils.joint_occurences(nrealizations, noffset=noffset + nalways, default_value=default_value)
            sumw_auto = 0
            if self.autocorr or self.same_shotnoise:
                wsq, csq = binned_weights(self.weights1, self.weights2)
                for csq_, wsq_ in zip(csq, wsq):
                    sumw_auto += joint[csq_ - nalways][csq_ - nalways] * wsq_
            sumw_cross = 0
            for c1_, w1_ in zip(c1, w1):
                for c2_, w2_ in zip(c2, w2):
                    sumw_cross += w1_ * w2_ * (joint[c1_ - nalways][c2_ - nalways] if c2_ <= c1_ else joint[c2_ - nalways][c1_ - nalways])
            return sumw_cross - sumw_auto

        indweights1, indweights2 = get_individual_weights(self.weights1), get_individual_weights(self.weights2)
        if self.n_bitwise_weights and method == 'counter':
            wiip = get_inverse_probability_weight(self.weights1[:self.n_bitwise_weights], nrealizations=nrealizations,
                                                  noffset=noffset, default_value=default_value, dtype=self.dtype)
            indweights1 = wiip * (1 if indweights1 is None else indweights1)
            wiip = get_inverse_probability_weight(self.weights2[:self.n_bitwise_weights], nrealizations=nrealizations,
                                                  noffset=noffset, default_value=default_value, dtype=self.dtype)
            indweights2 = wiip * (1 if indweights2 is None else indweights2)

        if indweights1 is None:

            if self.autocorr or self.same_shotnoise:
                return 1. * self.size1 * (self.size1 - 1.)
            return 1. * self.size1 * self.size2

        # individual_weights
        sumw_auto = 0.
        if self.autocorr or self.same_shotnoise:
            sumw_auto = np.sum(indweights1 * indweights2)
            if self.with_mpi:
                sumw_auto = self.mpicomm.allreduce(sumw_auto)
        sumw1, sumw2 = indweights1.sum(), indweights2.sum()
        if self.with_mpi:
            sumw1, sumw2 = self.mpicomm.allreduce(sumw1), self.mpicomm.allreduce(sumw2)
        sumw_cross = sumw1 * sumw2
        return sumw_cross - sumw_auto

    @property
    def sep(self):
        """Array of separation values of first dimension (e.g. :math:`s` if :attr:`mode` is "smu")."""
        return self.seps[0]

    @sep.setter
    def sep(self, sep):
        self.seps[0] = sep

    @property
    def compute_sepavg(self):
        """Whether to compute average of separation values for first dimension (e.g. :math:`s` if :attr:`mode` is "smu")."""
        return self.compute_sepsavg[0]

    @compute_sepavg.setter
    def compute_sepavg(self, compute_sepavg):
        self.compute_sepsavg[0] = compute_sepavg

    def normalized_wcounts(self):
        """Return normalized two-point counts, i.e. :attr:`wcounts` divided by :meth:`normalization`."""
        #with np.errstate(divide='ignore', invalid='ignore'):
        #    return _nan_to_zero(self.wcounts / self.wnorm)
        toret = np.zeros_like(self.wcounts)
        nonzero = self.wcounts != 0.
        toret[nonzero] = self.wcounts[nonzero] / (self.wnorm[nonzero] if np.ndim(self.wnorm) else self.wnorm)
        return toret

    def sepavg(self, axis=0, method=None):
        r"""
        Return average of separation for input axis.

        Parameters
        ----------
        axis : int, default=0
            Axis; if :attr:`mode` is "smu", 0 is :math:`s`, 1 is :math:`mu`;
            if :attr:`mode` is "rppi", 0 is :math:`r_{p}`, 1 is :math:`\pi`.

        method : str, default=None
            If ``None``, return average separation from :attr:`seps`.
            If 'mid', return bin mid-points.

        Returns
        -------
        sepavg : array
            1D array of size :attr:`shape[axis]`.
        """
        axis = axis % self.ndim
        if method is None:
            if self.compute_sepsavg[axis]:
                axes_to_sum_over = tuple(ii for ii in range(self.ndim) if ii != axis)
                with np.errstate(divide='ignore', invalid='ignore'):
                    toret = np.sum(_nan_to_zero(self.seps[axis]) * self.wcounts, axis=axes_to_sum_over) / np.sum(self.wcounts, axis=axes_to_sum_over)
            else:
                toret = self.seps[axis][tuple(Ellipsis if ii == axis else 0 for ii in range(self.ndim))]
        elif isinstance(method, str):
            allowed_methods = ['mid']
            method = method.lower()
            if method not in allowed_methods:
                raise TwoPointCounterError('method should be one of {}'.format(allowed_methods))
            elif method == 'mid':
                toret = self._get_default_seps()[axis]
        return toret

    def __getitem__(self, slices):
        """Call :meth:`slice`."""
        new = self.copy()
        if isinstance(slices, tuple):
            new.slice(*slices)
        else:
            new.slice(slices)
        return new

    def select(self, *xlims):
        """
        Restrict counts to provided coordinate limits in place.

        For example:

        .. code-block:: python

            counts.select((0, 0.3))  # restrict first axis to (0, 0.3)
            counts.select(None, (0, 0.2))  # restrict second axis to (0, 0.2)
            statistic.select((0, 30, 4))   # rebin to match step size of 4 and restrict to (0, 30)

        """
        if len(xlims) > self.ndim:
            raise IndexError('Too many limits: statistics is {:d}-dimensional, but {:d} were indexed'.format(self.ndim, len(xlims)))
        slices = []
        for iaxis, xlim in enumerate(xlims):
            if xlim is None:
                slices.append(slice(None))
            elif len(xlim) == 3:
                factor = int(xlim[2] / np.diff(self.edges[iaxis]).mean() + 0.5)
                if not np.allclose(np.diff(self.edges[iaxis][::factor]), xlim[2]):
                    import warnings
                    with np.printoptions(threshold=40):
                        warnings.warn('Unable to match step {} with edges {}'.format(xlim[2], self.edges[iaxis]))
                slices.append(slice(0, (self.shape[iaxis] // factor) * factor, factor))
            elif len(xlim) != 2:
                raise ValueError('Input limits must be a tuple (min, max) or (min, max, step)')
        self.slice(*slices)
        slices = []
        for iaxis, xlim in enumerate(xlims):
            if xlim is None:
                slices.append(slice(None))
            else:
                x = self.sepavg(axis=iaxis, method='mid')
                indices = np.flatnonzero((x >= xlim[0]) & (x <= xlim[1]))
                if indices.size:
                    slices.append(slice(indices[0], indices[-1] + 1, 1))
                else:
                    slices.append(slice(0))
        self.slice(*slices)
        return self

    def slice(self, *slices):
        """
        Slice counts in place. If slice step is not 1, use :meth:`rebin`.
        For example:

        .. code-block:: python

            counts.slice(slice(0, 10, 2), slice(0, 6, 3)) # rebin by factor 2 (resp. 3) along axis 0 (resp. 1), up to index 10 (resp. 6)
            counts[:10:2,:6:3] # same as above, but return new instance.

        """
        inslices = list(slices) + [slice(None)] * (self.ndim - len(slices))
        if len(inslices) > self.ndim:
            raise IndexError('Too many indices: statistics is {:d}-dimensional, but {:d} were indexed'.format(self.ndim, len(slices)))
        slices, eslices, factor = [], [], []
        for iaxis, sl in enumerate(inslices):
            start, stop, step = sl.indices(self.wcounts.shape[iaxis])
            if step < 0:
                raise IndexError('Positive slicing step only supported')
            slices.append(slice(start, stop, 1))
            eslices.append(slice(start, stop + 1, 1))
            factor.append(step)
        slices = tuple(slices)
        names = ['seps', 'wcounts', 'ncounts']
        if np.ndim(self.wnorm) > 0: names.append('wnorm')
        for name in names:
            if hasattr(self, name):
                tmp = getattr(self, name)
                if isinstance(tmp, list):
                    setattr(self, name, [tt[slices] for tt in tmp])
                else:
                    setattr(self, name, tmp[slices])
        self.edges = [edges[eslice] for edges, eslice in zip(self.edges, eslices)]
        if not all(f == 1 for f in factor):
            self.rebin(factor=factor)
        return self

    def rebin(self, factor=1):
        """
        Rebin two-point counts, by factor(s) ``factor``.
        Input factors must divide :attr:`shape`.

        Warning
        -------
        If current instance is the result of :meth:`concatenate_x`,
        rebinning is exact only if ``factor`` divides each of the constant-:attr:`wnorm` chunks.
        """
        if np.ndim(factor) == 0:
            factor = (factor,)
        factor = list(factor) + [1] * (self.ndim - len(factor))
        if len(factor) > self.ndim:
            raise ValueError('Too many rebinning factors: statistics is {:d}-dimensional, but got {:d} factors'.format(self.ndim, len(factor)))
        if any(s % f for s, f in zip(self.shape, factor)):
            raise ValueError('Rebinning factor must divide shape {}'.format(self.shape))
        new_shape = tuple(s // f for s, f in zip(self.shape, factor))
        normalized_wcounts = self.normalized_wcounts()
        rebinned_normalized_wcounts = utils.rebin(normalized_wcounts, new_shape, statistic=np.sum)
        if np.ndim(self.wnorm) > 0: self.wnorm = utils.rebin(self.wnorm, new_shape, statistic=np.mean)  # somewhat conventional...
        self.wcounts = rebinned_normalized_wcounts * self.wnorm
        if hasattr(self, 'ncounts'):
            self.ncounts = utils.rebin(self.ncounts, new_shape, statistic=np.sum)  # somewhat conventional...
        self.edges = [edges[::f] for edges, f in zip(self.edges, factor)]
        seps = self.seps
        self._set_default_seps()  # reset self.seps to default
        for idim, (sep, compute_sepavg) in enumerate(zip(seps, self.compute_sepsavg)):
            if compute_sepavg:
                with np.errstate(divide='ignore', invalid='ignore'):
                    self.seps[idim] = utils.rebin(_nan_to_zero(sep) * normalized_wcounts, new_shape, statistic=np.sum) / rebinned_normalized_wcounts
        return self

    def reverse(self):
        """Return counts for reversed positions1/weights1 and positions2/weights2."""
        if not self.is_reversible:
            raise TwoPointCounterError('These counts are not reversible')
        new = self.deepcopy()
        new.size1, new.size2 = new.size2, new.size1
        if new.mode in ['smu', 'rppi'] and not self.autocorr:
            for name in ['wcounts', 'ncounts', 'wnorm', 'sep']:
                if hasattr(new, name):
                    tmp = getattr(self, name)
                    if np.ndim(tmp):
                        setattr(new, name, tmp[:, ::-1])
        return new

    def wrap(self):
        r"""Return new 'smu' or 'rppi' two-point counts with 2nd coordinate wrapped to positive values, :math:`\mu > 0` or :math:`\pi > 0`."""
        if self.mode in ['smu', 'rppi']:
            new = self.deepcopy()
            if self.shape[1] % 2:
                raise TwoPointCounterError('These counts cannot be wrapped as 2nd dimension is {} % 2 = 1'.format(self.shape[1]))
            mid = self.shape[1] // 2
            sl_neg, sl_pos = slice(mid - 1, None, -1), slice(mid, None, 1)
            if not np.allclose(new.edges[1][mid:], - new.edges[1][mid::-1]):
                raise TwoPointCounterError('These counts cannot be wrapped as 2nd dimension edges are not symmetric; {} != {}'.format(new.edges[1][mid:], - new.edges[1][mid::-1]))
            new.edges[1] = new.edges[1][mid:]
            for name in ['ncounts', 'wcounts']:
                if hasattr(new, name):
                    tmp = getattr(new, name)
                    tmp = tmp[..., sl_neg] + tmp[..., sl_pos]
                    setattr(new, name, tmp)
            if np.ndim(self.wnorm):
                new.wnorm = self.wnorm[..., sl_pos]
            new._set_default_seps()
            for idim, compute_sepavg in enumerate(new.compute_sepsavg):
                if compute_sepavg:
                    sep = _nan_to_zero(self.seps[idim])
                    with np.errstate(divide='ignore', invalid='ignore'):
                        new.seps[idim] = (sep[..., sl_neg] * self.wcounts[..., sl_neg] + sep[..., sl_pos] * self.wcounts[..., sl_pos]) / new.wcounts
            return new
        raise TwoPointCounterError('These counts be wrapped')

    @classmethod
    def concatenate_x(cls, *others):
        """
        Concatenate input two-point counts along :attr:`sep`;
        useful when running two-point counts at different particle densities,
        e.g. high density on small scales, and lower density on larger scales,
        to keep computing time tractable.

        Warning
        -------
        :attr:`wnorm` is cast to a :attr:`ndim` array.
        """
        if len(others) == 1 and utils.is_sequence(others[0]):
            others = others[0]
        others = sorted(others, key=lambda other: np.mean(other.edges[0]))  # rank input counts by mean scale
        new = others[0].deepcopy()
        if len(others) > 1:
            new.wnorm = _make_array_like(new.wnorm, new.wcounts)
        names = ['wcounts']
        if hasattr(new, 'ncounts'): names = ['ncounts'] + names
        for iother, other in enumerate(others[1:]):
            mid = (other.edges[0][:-1] + other.edges[0][1:]) / 2.
            mask_low, mask_high = np.flatnonzero(mid < new.edges[0][0]), np.flatnonzero(mid > new.edges[0][-1])
            new.edges[0] = np.concatenate([other.edges[0][mask_low], new.edges[0], other.edges[0][mask_high + 1]], axis=0)
            for name in names:
                tmp = getattr(other, name)
                setattr(new, name, np.concatenate([tmp[mask_low], getattr(new, name), tmp[mask_high]], axis=0))
            wnorm = _make_array_like(other.wnorm, other.wcounts)
            new.wnorm = np.concatenate([wnorm[mask_low], new.wnorm, wnorm[mask_high]], axis=0)
            for idim in range(new.ndim):
                new.seps[idim] = np.concatenate([other.seps[idim][mask_low], new.seps[idim], other.seps[idim][mask_high]], axis=0)
        return new

    def normalize(self, wnorm):
        """
        Rescale both :attr:`wcounts` and :attr:`wnorm` such that new :attr:`wnorm` matches ``wnorm``.
        This is useful when combining counts in various regions.

        Parameters
        ----------
        wnorm : float
            New normalization :attr:`wnorm`.

        Returns
        -------
        new : BaseTwoPointCounter
            Normalized counts.
        """
        factor = wnorm / self.wnorm
        return self * factor

    @classmethod
    def sum(cls, *others):
        """
        Sum input two-point counts; useful when splitting input sample of particles;
        e.g. https://arxiv.org/pdf/1905.01133.pdf.

        Warning
        -------
        If > 1 input two-point counts, :attr:`size1`, :attr:`size2` attributes will be lost.
        Input two-point counts must have same edges for this operation to make sense
        (no checks performed).
        """
        if len(others) == 1 and utils.is_sequence(others[0]):
            others = others[0]
        new = others[0].deepcopy()
        if len(others) > 1:
            new.size1 = new.size2 = 0  # we do not know the total size
        for name in ['ncounts', 'wcounts', 'wnorm']:
            if hasattr(new, name):
                setattr(new, name, sum(getattr(other, name) for other in others))
        new._set_default_seps()  # reset self.seps to default
        for idim, compute_sepavg in enumerate(new.compute_sepsavg):
            if compute_sepavg:
                with np.errstate(divide='ignore', invalid='ignore'):
                    new.seps[idim] = sum(_nan_to_zero(other.seps[idim]) * other.wcounts for other in others) / new.wcounts
        return new

    def __add__(self, other):
        return self.sum(self, other)

    def __radd__(self, other):
        if other == 0: return self.deepcopy()
        return self.__add__(other)

    def __mul__(self, factor):
        new = self.deepcopy()
        for name in ['wcounts', 'wnorm']:
            setattr(new, name, getattr(new, name) * factor)
        return new

    def __rmul__(self, factor):
        return self.__mul__(factor)

    def deepcopy(self):
        import copy
        return copy.deepcopy(self)

    def __getstate__(self):
        state = {}
        for name in ['name', 'autocorr', 'is_reversible', 'seps', 'ncounts', 'wcounts', 'wnorm', 'size1', 'size2', 'mode', 'edges', 'bin_type',
                     'boxsize', 'los_type', 'compute_sepsavg', 'weight_attrs', 'cos_twopoint_weights', 'selection_attrs', 'dtype', 'attrs']:
            if hasattr(self, name):
                state[name] = getattr(self, name)
                if name == 'cos_twopoint_weights' and state[name] is not None:
                    state[name] = tuple(state[name])
        return state

    def __setstate__(self, state, load=False):
        super(BaseTwoPointCounter, self).__setstate__(state=state, load=load)
        if getattr(self, 'cos_twopoint_weights', None) is not None:
            self.cos_twopoint_weights = TwoPointWeight(*self.cos_twopoint_weights)
        if load:
            if not hasattr(self, 'selection_attrs'):
                self.selection_attrs = {}
            if hasattr(self, 'is_reversable'):
                self.is_reversible = self.is_reversable
            # wnorm and wcounts were allowed to be int at some point...
            self.wnorm = np.asarray(self.wnorm, dtype='f8')
            self.wcounts = np.asarray(self.wcounts, dtype='f8')
            if self.mode == 'rppi' and self.is_reversible and np.all(self.edges[1] >= 0.):
                import warnings
                warnings.warn('Loaded pair count is assumed to have been produced with < 20220909 version; if so please save it again to disk to remove this warning;'
                              'else these must be sliced pair counts: in this case, sorry I removed the slicing, please do counts[:, counts.shape[1]:]!')
                self.edges[1] = np.concatenate([-self.edges[1][:0:-1], self.edges[1]], axis=0)
                self.seps[0] = np.concatenate([self.seps[0][...,::-1], self.seps[0]], axis=-1)
                self.seps[1] = np.concatenate([-self.seps[1][...,::-1], self.seps[1]], axis=-1)
                for name in ['ncounts', 'wcounts', 'wnorm']:
                    if hasattr(self, name):
                        tmp = getattr(self, name)
                        if np.ndim(tmp):
                            tmp = np.concatenate([tmp[..., ::-1], tmp], axis=-1)
                            setattr(self, name, tmp)
        return self

    def save(self, filename):
        """Save two-point counts to ``filename``."""
        if not self.with_mpi or self.mpicomm.rank == 0:
            super(BaseTwoPointCounter, self).save(filename)

    def save_txt(self, filename, fmt='%.12e', delimiter=' ', header=None, comments='# '):
        """
        Save two-point counts as txt file.

        Warning
        -------
        Attributes are not all saved, hence there is :meth:`load_txt` method.

        Parameters
        ----------
        filename : str
            File name.

        fmt : str, default='%.12e'
            Format for floating types.

        delimiter : str, default=' '
            String or character separating columns.

        header : str, list, default=None
            String that will be written at the beginning of the file.
            If multiple lines, provide a list of one-line strings.

        comments : str, default=' #'
            String that will be prepended to the header string.
        """
        if not self.with_mpi or self.mpicomm.rank == 0:
            self.log_info('Saving {}.'.format(filename))
            utils.mkdir(os.path.dirname(filename))
            formatter = {'int_kind': lambda x: '%d' % x, 'float_kind': lambda x: fmt % x}
            if header is None: header = []
            elif isinstance(header, str): header = [header]
            else: header = list(header)
            for name in ['mode', 'autocorr', 'size1', 'size2', 'los_type', 'bin_type']:
                value = getattr(self, name, None)
                if value is None:
                    value = 'None'
                elif any(name.startswith(key) for key in ['mode', 'los_type', 'bin_type']):
                    value = str(value)
                else:
                    value = np.array2string(np.array(value), separator=delimiter, formatter=formatter).replace('\n', '')
                header.append('{} = {}'.format(name, value))
            coords_names = {'smu': ('s', 'mu'), 'rppi': ('rp', 'pi')}.get(self.mode, (self.mode,))
            assert len(coords_names) == self.ndim
            labels = []
            for name in coords_names:
                labels += ['{}mid'.format(name), '{}avg'.format(name)]
            labels += ['wcounts', 'wnorm']
            mids = np.meshgrid(*[(edges[:-1] + edges[1:]) / 2. for edges in self.edges], indexing='ij')
            columns = []
            for idim in range(self.ndim):
                columns += [mids[idim].flat, self.seps[idim].flat]
            columns += [self.wcounts.flat, _make_array_like(self.wnorm, self.wcounts).flat]
            columns = [[np.array2string(value, formatter=formatter) for value in column] for column in columns]
            widths = [max(max(map(len, column)) - len(comments) * (icol == 0), len(label)) for icol, (column, label) in enumerate(zip(columns, labels))]
            widths[-1] = 0  # no need to leave a space
            header.append((' ' * len(delimiter)).join(['{:<{width}}'.format(label, width=width) for label, width in zip(labels, widths)]))
            widths[0] += len(comments)
            with open(filename, 'w') as file:
                for line in header:
                    file.write(comments + line + '\n')
                for irow in range(len(columns[0])):
                    file.write(delimiter.join(['{:<{width}}'.format(column[irow], width=width) for column, width in zip(columns, widths)]) + '\n')


def normalization(weights1, weights2=None, weight_type='auto', weight_attrs=None, dtype='f8',
                  nthreads=None, mpicomm=None, mpiroot=None):
    r"""
    Initialize :class:`BaseTwoPointCounter`, and run actual two-point counts
    (calling :meth:`run`), setting :attr:`wcounts` and :attr:`sep`.

    Parameters
    ----------
    weights1 : int, array, list
        Weights (or local size, if no weights) of the first catalog.
        See ``weight_type``.

    weights2 : array, list, default=None
        Optionally, for cross-two-point counts, weights (or local size, if no weights) in the second catalog. See ``weights1``.

    weight_type : string, default='auto'
        The type of weighting to apply to provided weights. One of:

            - ``None``: no weights are applied.
            - "product_individual": each pair is weighted by the product of weights :math:`w_{1} w_{2}`.
            - "inverse_bitwise": each pair is weighted by :math:`\mathrm{nrealizations}/(\mathrm{noffset} + \mathrm{popcount}(w_{1} \& w_{2}))`.
               Multiple bitwise weights can be provided as a list.
               Individual weights can additionally be provided as float arrays.
               In case of cross-correlations with floating weights, bitwise weights are automatically turned to IIP weights,
               i.e. :math:`\mathrm{nrealizations}/(\mathrm{noffset} + \mathrm{popcount}(w_{1}))`.
            - "auto": automatically choose weighting based on input ``weights1`` and ``weights2``,
               i.e. ``None`` when ``weights1`` and ``weights2`` are ``None``,
               "inverse_bitwise" if one of input weights is integer, else "product_individual".

    weight_attrs : dict, default=None
        Dictionary of weighting scheme attributes. In case ``weight_type`` is "inverse_bitwise",
        one can provide "nrealizations", the total number of realizations (*including* current one;
        defaulting to the number of bits in input weights plus one);
        "noffset", the offset to be added to the bitwise counts in the denominator (defaulting to 1)
        and "default_value", the default value of pairwise weights if the denominator is zero (defaulting to 0).
        One can also provide "nalways", stating the number of bits systematically set to 1 minus the number of bits systematically set to 0 (defaulting to 0).
        These will only impact the normalization factors, if not computed with the brute-force approach.
        For example, for the "zero-truncated" estimator (arXiv:1912.08803), one would use noffset = 0, nalways = 1.
        The method used to compute the normalization with PIP weights can be specified with the keyword "normalization":
        if ``None``, normalization is given by eq. 22 of arXiv:1912.08803; "brute_force" (using OpenMP'ed C code)
        or "brute_force_npy" (slower, using numpy only methods; both methods match within machine precision) loop over all pairs;
        "counter" to normalize each pair by eq. 19 of arXiv:1912.08803.

    dtype : string, np.dtype, default='f8'
        Array type for weights.
        If ``None``, defaults to type of first ``weights1`` array if a floating-point array is provided, else 'f8'.
        Double precision is highly recommended.

    nthreads : int, default=None
        Number of OpenMP threads to use.

    mpicomm : MPI communicator, default=None
        The MPI communicator, to MPI-distribute calculation.

    mpiroot : int, default=None
        In case ``mpicomm`` is provided, if ``None``, input weights are assumed to be scattered across all ranks.
        Else the MPI rank where input weights are gathered.
    """
    # Somewhat hacky, but works!
    self = BaseTwoPointCounter.__new__(BaseTwoPointCounter)
    self.mpicomm = mpicomm
    if self.mpicomm is None and mpiroot is not None:
        raise TwoPointCounterError('mpiroot is not None, but no mpicomm provided')
    self._set_nthreads(nthreads)
    self.dtype = None if dtype is None else np.dtype(dtype)
    self._size1 = self._size2 = None
    self.autocorr = weights2 is None or isinstance(weights2, (tuple, list)) and not weights2
    if self.with_mpi:
        self.autocorr = all(self.mpicomm.allgather(self.autocorr))
    if self.autocorr:
        weights2 = None

    def size_weights(weights):
        import numbers
        local_is_size = is_size = isinstance(weights, numbers.Number)
        if self.with_mpi: is_size = any(self.mpicomm.allgather(is_size))
        if is_size:
            size = csize = weights
            if self.with_mpi:
                csize = self.mpicomm.allreduce(size if local_is_size else 0)
            return size, csize
        return None, None

    self._size1, self.size1 = size_weights(weights1)
    if self.size1 is not None:
        weights1 = None
    self._size2, self.size2 = size_weights(weights2)
    if self.size2 is not None:
        weights2 = None

    self._set_weights(weights1, weights2, weight_type=weight_type, weight_attrs=weight_attrs, copy=False, mpiroot=mpiroot)
    if self.dtype is None:
        if self.weights1[self.n_bitwise_weights:]:
            self.dtype = self.weights1[-1].dtype
        else:
            self.dtype = 'f8'
        self.dtype = np.dtype(self.dtype)
    return self.normalization()


class AnalyticTwoPointCounter(BaseTwoPointCounter):
    """
    Analytic two-point counter. Assume periodic wrapping and no data weights.

    Attributes
    ----------
    wcounts : array
        Analytical two-point counts.
    """
    name = 'analytic'

    def __init__(self, mode, edges, boxsize, size1=10, size2=None, los='z', selection_attrs=None):
        r"""
        Initialize :class:`AnalyticTwoPointCounter`, and set :attr:`wcounts` and :attr:`sep`.

        Parameters
        ----------
        mode : string
            Two-point counting mode, one of:

                - "s": two-point counts as a function of distance between two particles
                - "smu": two-point counts as a function of distance between two particles and cosine angle :math:`\mu` w.r.t. the line-of-sight
                - "rppi": two-point counts as a function of distance transverse (:math:`r_{p}`) and parallel (:math:`\pi`) to the line-of-sight
                - "rp": same as "rppi", without binning in :math:`\pi`

        edges : tuple, array
            Tuple of bin edges (arrays), for the first (e.g. :math:`r_{p}`)
            and optionally second (e.g. :math:`\pi`) dimensions.
            In case of single-dimension binning (e.g. ``mode`` is "theta", "s" or "rp"),
            the single array of bin edges can be provided directly.

        boxsize : array, float
            The side-length(s) of the periodic cube.

        size1 : int, default=2
            Length of the first catalog.

        size2 : int, default=None
            Optionally, for cross-two-point counts, length of second catalog.

        los : string, default='z'
            Line-of-sight to be used when ``mode`` is "rp", in case of non-cubic box;
            one of cartesian axes "x", "y" or "z".
        """
        self.attrs = {}
        self._set_mode(mode)
        self._set_boxsize(boxsize)
        self._set_edges(edges)
        self._set_los(los)
        self.size1 = size1
        self.size2 = size2
        self.autocorr = size2 is None
        self._set_compute_sepsavg(False)
        self._set_default_seps()
        self._set_reversible()
        self.wnorm = self.normalization()
        self._set_selection(selection_attrs)
        self.run()

    def run(self):
        """Set analytical two-point counts."""
        if any(name != "rp" for name in self.selection_attrs.keys()):
            raise TwoPointCounterError("Analytic random counts not implemented for selections other than rp")
        rp_selection = self.selection_attrs.get("rp", None)
        if rp_selection is not None and self.mode not in ['s', 'smu']:
            raise TwoPointCounterError("Analytic random counts not implemented for rp selection with mode {}".format(self.mode))
        if self.mode == 's':
            v = 4. / 3. * np.pi * self.edges[0]**3
            dv = np.diff(v, axis=0)
            if rp_selection:
                v_rpcut = [4. / 3. * np.pi * (self.edges[0]**3 - np.fmax(self.edges[0]**2 - rp_cut**2, 0)**1.5) for rp_cut in rp_selection] # volume of the intersection of a cylinder with radius rp_cut and a sphere with radius s
                v_rpcut = v_rpcut[1] - v_rpcut[0] # the volume that is removed between two rp cut limits
                dv_rpcut = np.diff(v_rpcut, axis=0) # volume in each bin removed by rp selection
                dv = np.where(dv_rpcut >= 1e-8 * dv, dv_rpcut, 0) # ensure that the volume is not negative and further remove small positive values that may arise due to rounding errors; assumes that dv is accurate
        elif self.mode == 'smu':
            # we bin in mu
            v = 2. / 3. * np.pi * self.edges[0][:, None]**3 * self.edges[1]
            dv = np.diff(np.diff(v, axis=0), axis=-1)
            if rp_selection:
                s, mu = self.edges[0][:, None], self.edges[1][None, :]
                sin_theta = np.sqrt(1 - mu**2) * np.ones_like(s)
                v_rpcut = []
                for rp_cut in rp_selection:
                    ss = s * np.ones_like(mu); c = ss * sin_theta > rp_cut; ss[c] = rp_cut / sin_theta[c] # this prevents division by zero, should work when rp_cut = 0, infinity or s = 0
                    r = ss * sin_theta # = min(rp_cut, s * sin(theta))
                    h = ss * mu # = cot(theta) * r, but avoids ambiguity/division by zero
                    v_rpcut.append(2. / 3. * np.pi * (s**3 - (s**2 - r**2)**1.5 + r**2 * h + 2 * (mu > 0) * ((s**2 - r**2)**1.5 - np.fmax(s**2 - rp_cut**2, 0)**1.5))) # volume of the intersection of a cylinder with radius rp_cut and a spherical sector/cone between -1 and mu with radius s.
                    # it can be decomposed into (1) intersection of a cylinder with the sphere, (2) a usual cylinder, (3) a usual cone and (4) only for mu>0 - intersection of the sphere with the space between two cylinders.
                    # r is the radius of (1-3) from the line of sight; h is the height of the cylinder (2) and the cone (3).
                    # the radii of cylinders for (4) are r and R = min(rp_cut, s), and r <= R always.
                    # it may seem that (4) becomes a more complicated shape if r < s * sin(theta), but this can only happen if rp_cut < s * sin(theta) <= s, then r = R = rp_cut, and we obtain zero volume as it should be.
                v_rpcut = v_rpcut[1] - v_rpcut[0] # the volume that is removed between two rp cut limits
                dv_rpcut = np.diff(np.diff(v_rpcut, axis=0), axis=-1) # volume in each bin removed by rp selection
                dv = np.where(dv_rpcut >= 1e-8 * dv, dv_rpcut, 0) # ensure that the volume is not negative and further remove small positive values that may arise due to rounding errors; assumes that dv is accurate
        elif self.mode == 'rppi':
            v = np.pi * self.edges[0][:, None]**2 * self.edges[1]
            dv = np.diff(np.diff(v, axis=0), axis=1)
        elif self.mode == 'rp':
            v = np.pi * self.edges[0]**2 * self.boxsize['xyz'.index(self.los_type)]
            dv = np.diff(v, axis=0)
        else:
            raise TwoPointCounterError('No analytic randoms provided for mode {}'.format(self.mode))
        self.wcounts = self.normalization() * dv / self.boxsize.prod()

    def normalization(self):
        """
        Return two-point count normalization, i.e., in case of cross-correlation ``size1 * size2``,
        and in case of auto-correlation ``size1 * (size1 - 1)``.
        """
        if self.autocorr:
            return self.size1 * (self.size1 - 1)
        return self.size1 * self.size2
