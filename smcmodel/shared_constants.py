import tensorflow as tf
import numpy as np

_dtypes = {
    'float32': {
        'tensorflow': tf.float32,
        'numpy': np.float32
    },
    'float64': {
        'tensorflow': tf.float64,
        'numpy': np.float64
    },
    'int32': {
        'tensorflow': tf.int32,
        'numpy': np.int32
    },
    'int64': {
        'tensorflow': tf.int64,
        'numpy': np.int64
    },
    'bool': {
        'tensorflow': tf.bool,
        'numpy': np.bool_
    },
}
