import tensorflow as tf


def sparse_dense_matmult_batch(sp_a, b):
    """
    Multiplies a tf.SparseTensor sp_a with an additional batch dimension with a 2 dimensional
    dense matrix. Returns a 3 dimensional dense matrix.
    """

    def map_function(x):
        i, dense_slice = x[0], x[1]
        sparse_slice = tf.sparse.reshape(
            tf.sparse.slice(sp_a, [i, 0, 0], [1, sp_a.dense_shape[1], sp_a.dense_shape[2]]),
            [sp_a.dense_shape[1], sp_a.dense_shape[2]],
        )
        mult_slice = tf.sparse.sparse_dense_matmul(sparse_slice, dense_slice)
        return mult_slice

    elems = (tf.range(0, sp_a.dense_shape[0], delta=1, dtype=tf.int64), b)
    return tf.map_fn(map_function, elems, fn_output_signature=tf.float32)
