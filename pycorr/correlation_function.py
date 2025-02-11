"""Implements high-level interface to estimate 2-point correlation function."""

import time
import logging

from .twopoint_estimator import get_twopoint_estimator, TwoPointEstimator
from .twopoint_counter import TwoPointCounter, AnalyticTwoPointCounter
from .twopoint_jackknife import JackknifeTwoPointCounter


def TwoPointCorrelationFunction(mode, edges, data_positions1, data_positions2=None, randoms_positions1=None, randoms_positions2=None, shifted_positions1=None, shifted_positions2=None,
                                data_weights1=None, data_weights2=None, randoms_weights1=None, randoms_weights2=None, shifted_weights1=None, shifted_weights2=None,
                                data_samples1=None, data_samples2=None, randoms_samples1=None, randoms_samples2=None, shifted_samples1=None, shifted_samples2=None,
                                D1D2_weight_type='auto', D1R2_weight_type='auto', R1D2_weight_type='auto', R1R2_weight_type='auto', S1S2_weight_type='auto', D1S2_weight_type='auto', S1D2_weight_type='auto', S1R2_weight_type='auto',
                                D1D2_twopoint_weights=None, D1R2_twopoint_weights=None, R1D2_twopoint_weights=None, R1R2_twopoint_weights=None, S1S2_twopoint_weights=None, D1S2_twopoint_weights=None, S1D2_twopoint_weights=None, S1R2_twopoint_weights=None,
                                estimator='auto', boxsize=None, selection_attrs=None, mpicomm=None, mpiroot=None, **kwargs):
    r"""
    Compute two-point counts and correlation function estimation, optionally with jackknife realizations.

    Note
    ----
    To compute the cross-correlation of samples 1 and 2, provide ``data_positions2``
    (and optionally ``randoms_positions2``, ``shifted_positions2`` for the selection function / shifted random catalogs of population 2).
    To compute (with the correct normalization estimate) the auto-correlation of sample 1, but with 2 weights, provide ``data_positions1``
    (but no ``data_positions2``, nor ``randoms_positions2`` and ``shifted_positions2``), ``data_weights1`` and ``data_weights2``;
    ``randoms_weights2`` and ``shited_weights2`` default to ``randoms_weights1`` and ``shited_weights1``, resp.

    Parameters
    ----------
    mode : string
        Type of correlation function, one of:

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

    data_positions1 : array
        Positions in the first data catalog. Typically of shape (3, N), but can be (2, N) when ``mode`` is "theta".
        See ``position_type``.

    data_positions2 : array, default=None
        Optionally, for cross-correlations, data positions in the second catalog. See ``data_positions1``.

    randoms_positions1 : array, default=None
        Optionally, positions of the random catalog representing the first selection function.
        If no randoms are provided, and estimator is "auto", or "natural",
        :class:`NaturalTwoPointEstimator` will be used to estimate the correlation function,
        with analytical two-point counts for R1R2.

    randoms_positions2 : array, default=None
        Optionally, for cross-correlations, positions of the random catalog representing the second selection function.
        See ``randoms_positions1``.

    shifted_positions1 : array, default=None
        Optionally, in case of BAO reconstruction, positions of the first shifted catalog.

    shifted_positions2 : array, default=None
        Optionally, in case of BAO reconstruction, positions of the second shifted catalog.

    data_weights1 : array, default=None
        Weights in the first data catalog. Not required if ``weight_type`` is either ``None`` or "auto".
        See ``weight_type``.

    data_weights2 : array, default=None
        Optionally, for cross-two-point counts, weights in the second data catalog. See ``data_weights1``.

    randoms_weights1 : array, default=None
        Optionally, weights of the first random catalog. See ``data_weights1``.

    randoms_weights2 : array, default=None
        Optionally, for cross-correlations, weights of the second random catalog.
        See ``randoms_weights1``.

    shifted_weights1 : array, default=None
        Optionally, weights of the first shifted catalog. See ``data_weights1``.

    shifted_weights2 : array, default=None
        Optionally, weights of the second shifted catalog. See ``shifted_weights1``.

    data_samples1 : array, default=None
        Optionally, (integer) labels of subsamples for the first data catalog.
        This is used to obtain an estimate of the covariance matrix from jackknife realizations.

    data_samples2 : array, default=None
        Same as ``data_samples1``, for the second data catalog.

    randoms_samples1 : array, default=None
        Same as ``data_samples1``, for the first randoms catalog.

    randoms_samples2 : array, default=None
        Same as ``data_samples1``, for the second randoms catalog.

    shifted_samples1 : array, default=None
        Same as ``data_samples1``, for the first shifted catalog.

    shifted_samples2 : array, default=None
        Same as ``data_samples1``, for the second shifted catalog.

    D1D2_weight_type : string, default='auto'
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

        In addition, angular upweights can be provided with ``D1D2_twopoint_weights``, ``D1R2_twopoint_weights``, etc.

    D1R2_weight_type : string, default='auto'
        Same as ``D1D2_weight_type``, for D1R2 two-point counts.

    R1D2_weight_type : string, default='auto'
        Same as ``D1D2_weight_type``, for R1D2 two-point counts.

    R1R2_weight_type : string, default='auto'
        Same as ``D1D2_weight_type``, for R1R2 two-point counts.

    S1S2_weight_type : string, default='auto'
        Same as ``D1D2_weight_type``, for S1S2 two-point counts.

    D1S2_weight_type : string, default='auto'
        Same as ``D1D2_weight_type``, for D1S2 two-point counts.

    S1D2_weight_type : string, default='auto'
        Same as ``D1D2_weight_type``, for S1D2 two-point counts.

    D1D2_twopoint_weights : WeightTwoPointEstimator, default=None
        Weights to be applied to each pair of particles between first and second data catalogs.
        A :class:`WeightTwoPointEstimator` instance or any object with arrays ``sep``
        (separations) and ``weight`` (weight at given separation) as attributes
        (i.e. to be accessed through ``twopoint_weights.sep``, ``twopoint_weights.weight``)
        or as keys (i.e. ``twopoint_weights['sep']``, ``twopoint_weights['weight']``)
        or as element (i.e. ``sep, weight = twopoint_weights``).

    D1R2_twopoint_weights : WeightTwoPointEstimator, default=None
        Weights to be applied to each pair of particles between first data catalog and second randoms catalog.
        See ``D1D2_twopoint_weights``.

    R1D2_twopoint_weights : WeightTwoPointEstimator, default=None
        Weights to be applied to each pair of particles between second data catalog and first randoms catalog.
        See ``D1D2_twopoint_weights``.

    R1R2_twopoint_weights : WeightTwoPointEstimator, default=None
        Weights to be applied to each pair of particles between first and second randoms catalogs.
        See ``D1D2_twopoint_weights``.

    S1S2_twopoint_weights : WeightTwoPointEstimator, default=None
        Weights to be applied to each pair of particles between first and second shifted catalogs.
        See ``D1D2_twopoint_weights``.

    D1S2_twopoint_weights : WeightTwoPointEstimator, default=None
        Weights to be applied to each pair of particles between first data catalog and second shifted catalog.
        See ``D1D2_twopoint_weights``.

    S1D2_twopoint_weights : WeightTwoPointEstimator, default=None
        Weights to be applied to each pair of particles between second data catalog and first shifted catalog.
        See ``D1D2_twopoint_weights``.

    estimator : string, default='auto'
        Estimator name, one of ["auto", "natural", "landyszalay", "davispeebles", "weight", "residual"].
        If "auto", "landyszalay" will be chosen if random or shifted catalog(s) is/are provided, else "natural".

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

    selection_attrs : dict, default=None
        To select pairs to be counted, provide mapping between the quantity (string)
        and the interval (tuple of floats),
        e.g. ``{'rp': (0., 20.)}`` to select pairs with transverse separation 'rp' between 0 and 20,
        ``{'theta': (0., 20.)}`` to select pairs with separation angle 'theta' between 0 and 20 degrees.
        One can additionally provide e.g. ``'counts': ['D1D2', 'D1R2']`` to specify counts for which the selection is to be applied,
        except analytic counts; defaults to all counts.

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
        If ``None``, defaults to type of first array of positions.
        Double precision is highly recommended in case ``mode`` is "theta",
        ``twopoint_weights`` is provided (due to cosine), or ``compute_sepsavg`` is ``True``.
        dtype='f8' highly recommended for ``mode = 'theta'`` or for :math:`\theta`-cut.

    nthreads : int, default=None
        Number of OpenMP threads to use.

    mpicomm : MPI communicator, default=None
        The MPI communicator, to MPI-distribute calculation.

    mpiroot : int, default=None
        In case ``mpicomm`` is provided, if ``None``, input positions and weights are assumed to be scattered across all ranks.
        Else the MPI rank where input positions and weights are gathered.

    kwargs : dict
        Counter engine-specific options, e.g. for corrfunc:
        - 'isa': one of 'fallback', 'sse42', 'avx', 'fastest'
        - 'mesh_refine_factors': an integer for ech dimension (2 for ``mode = 'theta'``),
          which increases the resolution of the grid used to speed-up pair counting.

        One can also provide precomputed two-point counts, e.g. R1R2.

    Returns
    -------
    estimator : BaseTwoPointEstimator
        Estimator with correlation function estimation :attr:`BaseTwoPointEstimator.corr`
        at separations :attr:`BaseTwoPointEstimator.sep`.
    """
    logger = logging.getLogger('TwoPointCorrelationFunction')
    log = mpicomm is None or mpicomm.rank == 0
    t0 = time.time()

    def is_none(array):
        if mpicomm is None or mpiroot is None:
            return array is None
        return mpicomm.bcast(array is None, root=mpiroot)

    def is_same(array1, array2):
        if mpicomm is None or mpiroot is None:
            return array1 is array2
        return mpicomm.bcast(array1 is array2, root=mpiroot)

    with_randoms = not is_none(randoms_positions1)
    with_shifted = not is_none(shifted_positions1)
    #with_shifted = not (is_none(shifted_positions1) and is_none(shifted_positions2))
    with_jackknife = not is_none(data_samples1)
    Estimator = get_twopoint_estimator(estimator, with_DR=with_randoms or with_shifted, with_jackknife=with_jackknife)
    if log: logger.info('Using estimator {}.'.format(Estimator))

    # This could work, but prefer to remain explicit / simple for now: rely on catalogs 1 to decide on the default estimator
    # and transferring shifted_weights1 = randoms_weights1 breaks the convention that None weights are unit weights
    #if with_shifted:
    #    # allow for pre- x post-recon, or just propagate missing properties from non-shifted randoms
    #    if is_none(shifted_positions1): shifted_positions1 = randoms_positions1
    #    if is_none(shifted_weights1): shifted_weights1 = randoms_weights1
    #    if is_none(shifted_samples1): shifted_samples1 = randoms_samples1
    #    # allow for post- x pre-recon, or just propagate missing properties from non-shifted randoms
    #    if is_none(shifted_positions2): shifted_positions2 = randoms_positions2
    #    if is_none(shifted_weights2): shifted_weights2 = randoms_weights2
    #    if is_none(shifted_samples2): shifted_samples2 = randoms_samples2

    positions = {'D1': data_positions1, 'D2': data_positions2, 'R1': randoms_positions1, 'R2': randoms_positions2, 'S1': shifted_positions1, 'S2': shifted_positions2}
    weights = {'D1': data_weights1, 'D2': data_weights2, 'R1': randoms_weights1, 'R2': randoms_weights2, 'S1': shifted_weights1, 'S2': shifted_weights2}
    samples = {'D1': data_samples1, 'D2': data_samples2, 'R1': randoms_samples1, 'R2': randoms_samples2, 'S1': shifted_samples1, 'S2': shifted_samples2}
    twopoint_weights = {'D1D2': D1D2_twopoint_weights, 'D1R2': D1R2_twopoint_weights, 'R1D2': R1D2_twopoint_weights, 'R1R2': R1R2_twopoint_weights,
                        'S1S2': S1S2_twopoint_weights, 'D1S2': D1S2_twopoint_weights, 'S1D2': S1D2_twopoint_weights, 'S1R2': S1R2_twopoint_weights, 'R1S2': S1R2_twopoint_weights}
    weight_type = {'D1D2': D1D2_weight_type, 'D1R2': D1R2_weight_type, 'R1D2': R1D2_weight_type, 'R1R2': R1R2_weight_type,
                   'S1S2': S1S2_weight_type, 'D1S2': D1S2_weight_type, 'S1D2': S1D2_weight_type, 'S1R2': S1R2_weight_type, 'R1S2': S1R2_weight_type}  # RS and SR only used by 'residual' estimator
    if selection_attrs is None:
        selection_attrs = {name: None for name in twopoint_weights}
    else:
        selection_attrs = dict(selection_attrs)
        counts = selection_attrs.pop('counts', None)
        if counts is None:
            counts = twopoint_weights.keys()
        selection_attrs = {name: selection_attrs if name in counts else None for name in twopoint_weights}

    autocorr, same_shotnoise = False, False
    precomputed = {}
    for ilabel, (label1, label2) in enumerate(Estimator.requires(with_reversed=True, with_shifted=with_shifted)):
        label12 = label1 + label2
        if ilabel == 0 and is_none(positions[label2]):
            same_shotnoise = not is_none(weights[label2])
            autocorr = not same_shotnoise
        precomputed[label12] = kwargs.pop(label12, None)

    if log: logger.info('Running {}-correlation.'.format('auto' if autocorr else 'cross'))

    if same_shotnoise:
        for name in ['D', 'R', 'S']:
            if is_none(positions[name + '2']): positions[name + '2'] = positions[name + '1']
            if is_none(weights[name + '2']): weights[name + '2'] = weights[name + '1']
            if is_none(samples[name + '2']): samples[name + '2'] = samples[name + '1']
        if log: logger.info('Assuming same shot noise.')

    counts = {}
    for label1, label2 in Estimator.requires(with_reversed=True, with_shifted=with_shifted):
        label12 = label1 + label2
        pre = precomputed.get(label12, None)
        if pre is not None:
            if log: logger.info('Using precomputed two-point counts {}.'.format(label12))
            counts[label12] = pre
            continue
        label21 = label2.replace('2', '1') + label1.replace('1', '2')
        if autocorr and label21 in counts and counts[label21].is_reversible:
            continue
        if label12 == 'R1R2' and not with_randoms:
            if log: logger.info('Analytically computing two-point counts {}.'.format(label12))
            size1 = counts['D1D2'].size1
            size2 = None
            if not autocorr and not same_shotnoise:
                size2 = counts['D1D2'].size2
            if boxsize is None:
                raise ValueError('boxsize must be provided for analytic two-point counts {}.'.format(label12))
            counts[label12] = AnalyticTwoPointCounter(mode, edges, boxsize, size1=size1, size2=size2, selection_attrs=selection_attrs[label12])
            continue
        if log: logger.info('Computing two-point counts {}.'.format(label12))
        twopoint_kwargs = {'twopoint_weights': twopoint_weights[label12], 'weight_type': weight_type[label12], 'selection_attrs': selection_attrs[label12]}
        if autocorr:
            if label2[:-1] == label1[:-1]:
                label2 = None
            else:
                # label12 is D1R2, but we only have R1, so switch label2 to R1; same for D1S2
                label2 = label2.replace('2', '1')
        # In case of autocorrelation, los = firstpoint or endpoint, R1D2 = R1D1 should get the same angular weight as D1R2 = D2R1
        if (autocorr and label2 is not None) or (same_shotnoise and label2[:-1] != label1[:-1]):
            for name in twopoint_kwargs:
                if twopoint_kwargs[name] is None: twopoint_kwargs[name] = locals()[name][label21]

        jackknife_kwargs = {}
        with_jackknife = not is_none(samples[label1])
        if with_jackknife:
            Counter = JackknifeTwoPointCounter
            jackknife_kwargs['samples1'] = samples[label1]
            jackknife_kwargs['samples2'] = samples[label2] if label2 is not None else None
        else:
            Counter = TwoPointCounter

        for label in [label1] + ([label2] if label2 is not None else []):
            if is_none(positions[label]): raise ValueError('{} must be provided'.format(label))

        positions2 = positions[label2] if label2 is not None else None
        if same_shotnoise and label2[:-1] == label1[:-1] and is_same(positions2, positions[label1]):  # D2 = D1, R2 = R1 positions, but different weights; this is to remove the correct amount of auto-pairs at s = 0
            positions2 = None

        counts[label12] = Counter(mode, edges, positions[label1], positions2=positions2,
                                  weights1=weights[label1], weights2=weights[label2] if label2 is not None else None,
                                  boxsize=boxsize, mpicomm=mpicomm, mpiroot=mpiroot,
                                  **jackknife_kwargs, **twopoint_kwargs, **kwargs)

    toret = Estimator(**counts)
    if log: logger.info('Correlation function computed in elapsed time {:.2f} s.'.format(time.time() - t0))
    return toret


TwoPointCorrelationFunction.from_state = TwoPointEstimator.from_state
TwoPointCorrelationFunction.load = TwoPointEstimator.load
