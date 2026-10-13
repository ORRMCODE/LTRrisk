
import tensorflow as tf

def count_queries(ds):
    # one element = one query in your pipeline
    return ds.reduce(tf.constant(0, dtype=tf.int64), lambda acc, _: acc + 1)
