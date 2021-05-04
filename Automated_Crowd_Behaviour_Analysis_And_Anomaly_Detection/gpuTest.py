import tensorflow as tf


def usingGPU():
    print("Is built with CUDA: ", tf.test.is_built_with_cuda())
    # print("Is GPU available: ", tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
    print(tf.config.list_physical_devices('GPU'))
