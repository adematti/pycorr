import os
import tempfile

import numpy as np

from pycorr import BoxSubsampler, KMeansSubsampler, TwoPointCounter, JackknifeTwoPointCounter, utils, setup_logging


def generate_catalogs(size=200, boxsize=(1000,)*3, offset=(1000,0,0), n_individual_weights=1, n_bitwise_weights=0, seed=42):
    rng = np.random.RandomState(seed=seed)
    toret = []
    for i in range(2):
        positions = [o + rng.uniform(0., 1., size)*b for o, b in zip(offset, boxsize)]
        weights = utils.pack_bitarrays(*[rng.randint(0, 2, size) for i in range(64*n_bitwise_weights)], dtype=np.uint64)
        #weights = utils.pack_bitarrays(*[rng.randint(0, 2, size) for i in range(33)], dtype=np.uint64)
        #weights = [rng.randint(0, 0xffffffff, size, dtype=np.uint64) for i in range(n_bitwise_weights)]
        weights += [rng.uniform(0.5, 1., size) for i in range(n_individual_weights)]
        toret.append(positions + weights)
    return toret


def test_subsampler():

    mpi = False
    try:
        from pycorr import mpi
    except ImportError:
        pass

    boxsize = np.array([1000.]*3)
    boxcenter = np.array([100., 0., 0.])
    catalog = generate_catalogs(size=1000, boxsize=boxsize, offset=boxcenter-boxsize/2.)[0]
    nsamples = 27
    subsampler = BoxSubsampler(boxsize=boxsize, boxcenter=boxcenter, nsamples=nsamples)
    assert np.allclose(subsampler.boxsize, boxsize, rtol=1e-2)
    labels = subsampler.label(catalog[:3])
    assert np.max(labels) < nsamples

    subsampler = BoxSubsampler(positions=catalog[:3], nsamples=nsamples)
    assert np.allclose(subsampler.boxsize, boxsize, rtol=1e-2)
    labels = subsampler.label(catalog[:3])
    assert np.max(labels) < nsamples

    for nside in [None, 512]:

        nsamples = 100
        subsampler = KMeansSubsampler(mode='angular', positions=catalog[:3], nsamples=nsamples, nside=nside, random_state=42, position_type='xyz')
        labels = subsampler.label(catalog[:3])
        assert np.max(labels) < nsamples

        if mpi:
            mpicomm = mpi.COMM_WORLD
            subsampler_mpi = KMeansSubsampler(mode='angular', positions=catalog[:3], nsamples=nsamples, nside=nside, random_state=42, position_type='xyz', mpicomm=mpicomm, mpiroot=0)
            labels_mpi = subsampler.label(catalog[:3])
            assert np.allclose(labels_mpi, labels)
            if mpicomm.rank == 0:
                labels_mpi = subsampler.label(catalog[:3])
            else:
                labels_mpi = None
            labels_mpi = mpi.broadcast_array(labels_mpi, mpicomm=mpicomm, root=0)
            assert np.allclose(labels_mpi, labels)


def test_twopoint_counter(mode='s'):

    list_engine = ['corrfunc']
    edges = np.linspace(1,100,11)
    size = 1000
    boxsize = (1000,)*3
    list_options = []
    list_options.append({})
    if mode not in ['theta', 'rp']:
        list_options.append({'boxsize':boxsize})
        list_options.append({'autocorr':True, 'boxsize':boxsize})

    list_options.append({'autocorr':True})
    list_options.append({'n_individual_weights':1, 'bin_type':'custom'})
    list_options.append({'n_individual_weights':1, 'n_bitwise_weights':1})
    list_options.append({'n_individual_weights':1, 'n_bitwise_weights':1, 'iip':1, 'dtype':'f4'})
    list_options.append({'n_individual_weights':1, 'n_bitwise_weights':1, 'bitwise_type': 'i4', 'iip':1})
    list_options.append({'n_individual_weights':2, 'n_bitwise_weights':2, 'iip':2, 'position_type':'rdd', 'weight_attrs':{'nrealizations':42,'noffset':3}})

    list_options.append({'n_individual_weights':1, 'n_bitwise_weights':2, 'iip':2, 'weight_attrs':{'noffset':0,'default_value':0.8}})
    if mode == 'theta':
        list_options.append({'n_individual_weights':2, 'n_bitwise_weights':2, 'iip':2, 'position_type':'rd'})

    from collections import namedtuple
    TwoPointWeight = namedtuple('TwoPointWeight', ['sep', 'weight'])
    twopoint_weights = TwoPointWeight(np.logspace(-4, 0, 40), np.linspace(4., 1., 40))
    #list_options.append({'autocorr':True, 'twopoint_weights':twopoint_weights})
    list_options.append({'autocorr':True, 'n_individual_weights':2, 'n_bitwise_weights':2, 'twopoint_weights':twopoint_weights, 'dtype':'f8'})
    mpi = False
    try:
        from pycorr import mpi
    except ImportError:
        pass
    if mpi:
        print('Has MPI')
        list_options.append({'mpicomm':mpi.COMM_WORLD})
        list_options.append({'n_individual_weights':1, 'mpicomm':mpi.COMM_WORLD})
        list_options.append({'n_individual_weights':2, 'n_bitwise_weights':2, 'twopoint_weights':twopoint_weights, 'mpicomm':mpi.COMM_WORLD})

    if mode == 'smu':
        edges = (edges, np.linspace(0,1,21))
    elif mode == 'rppi':
        edges = (edges, np.linspace(0,40,41))
    elif mode == 'theta':
        edges = np.linspace(1e-1,10,21) # below 1e-5 for float64 (1e-1 for float32), self pairs are counted by Corrfunc
    for engine in list_engine:
        for options in list_options:
            options = options.copy()
            n_individual_weights = options.pop('n_individual_weights',0)
            n_bitwise_weights = options.pop('n_bitwise_weights',0)
            npos = 3
            data1, data2 = generate_catalogs(size, boxsize=boxsize, n_individual_weights=n_individual_weights, n_bitwise_weights=n_bitwise_weights)

            subsampler = KMeansSubsampler(mode='angular', positions=data1[:npos], nsamples=5, nside=512, random_state=42, position_type='xyz')
            data1.append(subsampler.label(data1[:npos]))
            data2.append(subsampler.label(data2[:npos]))

            autocorr = options.pop('autocorr', False)
            options.setdefault('boxsize', None)
            options['los'] = 'x' if options['boxsize'] is not None else 'midpoint'
            bin_type = options.pop('bin_type', 'auto')
            mpicomm = options.pop('mpicomm', None)
            bitwise_type = options.pop('bitwise_type', None)
            iip = options.pop('iip', False)
            position_type = options.pop('position_type', 'xyz')
            dtype = options.pop('dtype', None)
            weight_attrs = options.get('weight_attrs', {}).copy()

            def setdefaultnone(di, key, value):
                if di.get(key, None) is None:
                    di[key] = value

            setdefaultnone(weight_attrs, 'nrealizations', n_bitwise_weights * 64 + 1)
            setdefaultnone(weight_attrs, 'noffset', 1)
            set_default_value = 'default_value' in weight_attrs
            setdefaultnone(weight_attrs, 'default_value', 0)
            if set_default_value:
                for w in data1[npos:npos+n_bitwise_weights] + data2[npos:npos+n_bitwise_weights]: w[:] = 0 # set to zero to make sure default_value is used

            def wiip(weights):
                denom = weight_attrs['noffset'] + utils.popcount(*weights)
                mask = denom == 0
                denom[mask] = 1.
                toret = weight_attrs['nrealizations']/denom
                toret[mask] = weight_attrs['default_value']
                return toret

            def dataiip(data):
                return data[:npos] + [wiip(data[npos:npos+n_bitwise_weights])] + data[npos+n_bitwise_weights:]

            if iip == 1:
                data1 = dataiip(data1)
            elif iip == 2:
                data2 = dataiip(data2)
            if iip:
                n_bitwise_weights = 0
                weight_attrs['nrealizations'] = None

            itemsize = np.dtype('f8' if dtype is None else dtype).itemsize
            tol = {'atol':1e-8, 'rtol':1e-3} if itemsize <= 4 else {'atol':1e-8, 'rtol':1e-6}

            if bitwise_type is not None and n_bitwise_weights > 0:

                def update_bit_type(data):
                    return data[:npos] + utils.reformat_bitarrays(*data[npos:npos+n_bitwise_weights], dtype=bitwise_type) + data[npos+n_bitwise_weights:]

                data1 = update_bit_type(data1)
                data2 = update_bit_type(data2)

            if position_type != 'xyz':

                if position_type == 'rd': npos = 2

                def update_pos_type(data):
                    rdd = list(utils.cartesian_to_sky(data[:3]))
                    if position_type == 'rdd':
                        return rdd + data[3:]
                    if position_type == 'rd':
                        return rdd[:2] + data[3:]
                    raise ValueError('Unknown position type {}'.format(position_type))

                data1 = update_pos_type(data1)
                data2 = update_pos_type(data2)

            def run_ref(ii=None, **kwargs):
                positions1, weights1, samples1 = data1[:npos], data1[npos:-1], data1[-1]
                positions2, weights2, samples2 = data2[:npos], data2[npos:-1], data2[-1]
                if ii is not None:
                    mask = samples1 == ii
                    positions1, weights1 = [position[~mask] for position in positions1], [weight[~mask] for weight in weights1]
                    mask = samples2 == ii
                    positions2, weights2 = [position[~mask] for position in positions2], [weight[~mask] for weight in weights2]
                return TwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=positions1, positions2=None if autocorr else positions2,
                                       weights1=weights1, weights2=None if autocorr else weights2, position_type=position_type, bin_type=bin_type,
                                       dtype=dtype, **kwargs, **options)

            def run(pass_none=False, **kwargs):
                return JackknifeTwoPointCounter(mode=mode, edges=edges, engine=engine, positions1=None if pass_none else data1[:npos], weights1=None if pass_none else data1[npos:-1],
                                               positions2=None if pass_none or autocorr else data2[:npos], weights2=None if pass_none or autocorr else data2[npos:-1],
                                               samples1=None if pass_none else data1[-1], samples2=None if pass_none or autocorr else data2[-1],
                                               position_type=position_type, bin_type=bin_type, dtype=dtype, **kwargs, **options)

            def assert_allclose(res1, res2):
                assert np.allclose(res2.wcounts, res1.wcounts, **tol)
                assert np.allclose(res2.wnorm, res1.wnorm, **tol)
                if n_individual_weights == n_bitwise_weights == 0:
                    mask = ~np.isnan(res2.sep)
                    assert np.allclose(res2.sep[mask], res1.sep[mask], **tol)
                assert res1.size1 == res2.size1
                assert res1.size2 == res2.size2

            ref = run_ref()
            test = run()
            assert_allclose(test, ref)

            nsplits = 10
            test = JackknifeTwoPointCounter.concatenate(*[run(samples=samples) for samples in np.array_split(np.unique(data1[-1]), nsplits)])
            assert_allclose(test, ref)

            ii = data1[-1][0]
            ref_ii = run_ref(ii=ii)
            test_ii = test.realization(ii, correction=None)
            assert_allclose(test_ii, ref_ii)

            with tempfile.TemporaryDirectory() as tmp_dir:
                fn = os.path.join(tmp_dir, 'tmp.npy')
                test.save(fn)
                test2 = JackknifeTwoPointCounter.load(fn)
                assert_allclose(test2, ref)
                test2.rebin((2,2) if len(edges) == 2 else (2,))
                assert np.allclose(np.sum(test2.wcounts), np.sum(ref.wcounts))

            if mpicomm is not None:
                test_mpi = run(mpicomm=mpicomm, pass_none=mpicomm.rank != 0, mpiroot=0, nprocs_per_real=2)
                assert_allclose(test_mpi, test)
                data1 = [mpi.scatter_array(d, root=0, mpicomm=mpicomm) for d in data1]
                data2 = [mpi.scatter_array(d, root=0, mpicomm=mpicomm) for d in data2]
                test_mpi = run(mpicomm=mpicomm)
                assert_allclose(test_mpi, test)


if __name__ == '__main__':

    setup_logging()
    test_subsampler()
    for mode in ['theta','s','smu','rppi','rp']:
        test_twopoint_counter(mode=mode)