"""main.py
test to use NumPy
"""

import time
import numpy as np

SHAPE = (1024, 1024)
N_MEASURES = 100
SAVE_PATH = "./data/process_time_python.csv"
RANDOM_SEED = 1234


def measure(func: callable, ntimes: int, *args, **kwargs):
    """measure process time of a function
    """
    elapsed = []
    for _ in range(ntimes):
        st = time.time()
        _ = func(*args, **kwargs)
        elapsed.append(time.time() - st)
    print("averaged process time over {0} times: {1:.6f} +/- {2:.6f} msec.".format(
        ntimes, np.mean(elapsed) * 1e3, np.std(elapsed) * 1e3
    ))
    return [np.mean(elapsed) * 1e3, np.std(elapsed) * 1e3]


def measure_add(a, b, ntimes: int):
    """measure process time of a function
    """
    elapsed = []
    for _ in range(ntimes):
        st = time.time()
        _ = a + b
        elapsed.append(time.time() - st)
    print("averaged process time over {0} times: {1:.6f} +/- {2:.6f} msec.".format(
        ntimes, np.mean(elapsed) * 1e3, np.std(elapsed) * 1e3
    ))
    return [np.mean(elapsed) * 1e3, np.std(elapsed) * 1e3]


def measure_pow(a, p, ntimes: int):
    """measure process time of a function
    """
    elapsed = []
    for _ in range(ntimes):
        st = time.time()
        _ = a**p
        elapsed.append(time.time() - st)
    print("averaged process time over {0} times: {1:.6f} +/- {2:.6f} msec.".format(
        ntimes, np.mean(elapsed) * 1e3, np.std(elapsed) * 1e3
    ))
    return [np.mean(elapsed) * 1e3, np.std(elapsed) * 1e3]


def measure_index_more_than(a, v, ntimes: int):
    """measure process time of a function
    """
    elapsed = []
    for _ in range(ntimes):
        st = time.time()
        _ = a > v
        elapsed.append(time.time() - st)
    print("averaged process time over {0} times: {1:.6f} +/- {2:.6f} msec.".format(
        ntimes, np.mean(elapsed) * 1e3, np.std(elapsed) * 1e3
    ))
    return [np.mean(elapsed) * 1e3, np.std(elapsed) * 1e3]


def measure_fill(a, v, ntimes: int):
    """measure process time of a function
    """
    elapsed = []
    for _ in range(ntimes):
        st = time.time()
        a[:] = v
        elapsed.append(time.time() - st)
    print("averaged process time over {0} times: {1:.6f} +/- {2:.6f} msec.".format(
        ntimes, np.mean(elapsed) * 1e3, np.std(elapsed) * 1e3
    ))
    return [np.mean(elapsed) * 1e3, np.std(elapsed) * 1e3]


# def return_index_more_than(a: np.ndarray, v: float):
#     """wrapper of `a > v`"""
#     return a > v


# def return_fill(a: np.ndarray, v: float):
#     """wrapper of `a[:] = v`"""
#     a[:] = v


def main(n_meas: int):
    """main function"""
    np.random.seed(RANDOM_SEED)
    elapsed = []
    print("Array creation:")
    print("np.arange")
    elapsed.append(
        measure(np.arange, n_meas, *(0., 10., 10. / (SHAPE[0] * SHAPE[1])))
    )

    print("np.linspace")
    elapsed.append(
        measure(np.linspace, n_meas, *(0., 10., SHAPE[0] * SHAPE[1]))
    )

    print("np.ones")
    elapsed.append(measure(np.ones, n_meas, *(SHAPE,)))

    print("np.zeros")
    elapsed.append(measure(np.zeros, n_meas, *(SHAPE,)))

    print("np.full")
    elapsed.append(measure(np.full, n_meas, *(SHAPE, 7.0)))

    print("np.eye")
    elapsed.append(measure(np.eye, n_meas, *(SHAPE[0],)))

    print("Randomize:")
    print("Normal distribution")
    elapsed.append(measure(np.random.normal, n_meas, *(0.0, 1.0, SHAPE)))

    print("Poisson distribution")
    elapsed.append(measure(np.random.poisson, n_meas, *(10., SHAPE)))

    print("Uniform distribution")
    elapsed.append(measure(np.random.uniform, n_meas, *(0.0, 1.0, SHAPE)))

    print("Mathematics:")
    vec = np.random.normal(0.0, 1.0, SHAPE[0] * SHAPE[1])
    vec2 = np.random.normal(0.0, 1.0, SHAPE[0] * SHAPE[1])
    mat = np.random.normal(0.0, 1.0, SHAPE)
    mat2 = np.random.normal(0.0, 1.0, SHAPE)
    mat3 = mat.copy()

    print("mat.dot(mat2)")
    elapsed.append(measure(mat.dot, n_meas, *(mat2,)))

    print("mat.dot(vec)")
    elapsed.append(measure(mat.dot, n_meas, *(vec[:SHAPE[0]],)))

    print("vec.dot(mat)")
    elapsed.append(measure(vec[:SHAPE[0]].dot, n_meas, *(mat,)))

    print("mat.dot(vec2)")
    elapsed.append(measure(vec.dot, n_meas, *(vec2,)))

    print("mat + mat2")
    elapsed.append(measure_add(mat, mat2, n_meas))

    print("mat**3")
    elapsed.append(measure_pow(mat, 3, n_meas))

    print("np.sqrt(mat)")
    elapsed.append(measure(np.sqrt, n_meas, *(mat,)))

    print("mat > 0.5")
    elapsed.append(measure_index_more_than(mat, 0.5, n_meas))

    print("np.sum(mat)")
    elapsed.append(measure(mat.sum, n_meas))
    
    print("np.sum(mat, axis=1)")
    elapsed.append(measure(mat.sum, n_meas, axis=1))

    print("mat.mean()")
    elapsed.append(measure(mat.mean, n_meas))
    
    print("mat.mean(axis=1)")
    elapsed.append(measure(mat.mean, n_meas, axis=1))

    print("np.allclose(mat, mat2, atol=1e-8)")
    elapsed.append(measure(np.allclose, n_meas, *(mat, mat2), atol=1e-8))

    print("np.diag(mat)")
    elapsed.append(measure(np.diag, n_meas, *(mat,)))

    print("Array manipulation:")
    print("mat.fill(3.0)")
    elapsed.append(measure(mat3.fill, n_meas, *(3.0,)))

    print("mat[:] = mat2")
    elapsed.append(measure_fill(mat3, mat2, n_meas))

    print("np.concatenate((mat, mat2), axis=1)")
    elapsed.append(measure(np.concatenate, n_meas, *((mat3, mat2),), axis=1))
    
    print("np.stack((mat, mat2), axis=1)")
    elapsed.append(measure(np.stack, n_meas, *((mat3, mat2),), axis=1))

    print("np.expand_dims(mat, axis=1)")
    elapsed.append(measure(np.expand_dims, n_meas, *(mat3,), axis=1))

    print("mat.transpose()")
    elapsed.append(measure(mat.transpose, n_meas))

    print("mat.flatten()")
    elapsed.append(measure(mat.flatten, n_meas))

    print("TYpe conversion:")
    print("convert u8 to f32")
    mat = np.random.uniform(0., 255., SHAPE).astype(np.uint8)
    elapsed.append(measure(mat.astype, n_meas, *(np.float32,)))

    print("convert u8 to f32")
    elapsed.append(measure(mat.astype, n_meas, *(np.int32,)))

    print("convert i8 to u8")
    mat = np.random.uniform(0., 127., SHAPE).astype(np.int8)
    elapsed.append(measure(mat.astype, n_meas, *(np.uint8,)))

    print("convert f32 to i32")
    mat = np.random.normal(0., 1000., SHAPE).astype(np.float32)
    elapsed.append(measure(mat.astype, n_meas, *(np.int32,)))

    np.savetxt(SAVE_PATH, elapsed, delimiter=",")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        N_MEASURES = int(sys.argv[1])
    main(N_MEASURES)
