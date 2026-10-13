import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def prepare_dataset(
    ds: tf.data.Dataset,
    rng_seed: int = 42,
    max_list_size = 200,
    batch_size = 128
):
    """Prepares a training dataset by applying padding/truncating/etc."""
    tf.random.set_seed(rng_seed)
    features_to_keep = set(["label", "float_features"])
    ds = ds.map(lambda e: {k: v for k, v in e.items() if k in features_to_keep})
    ds = ds.map(lambda e: (e.pop("float_features"), e.pop("label")))
    ds = ds.map(lambda x, y: (tf.sign(x)*tf.math.log1p(tf.abs(x)), 2.**y-1.))
    ds = ds.map(lambda x, y: (x, y, tf.ones_like(y, dtype=bool)))


  # not every query has the same number of pre-selected documents in the dataset
  # therefore, we add padding so that the matrices can be batched together
    def pad(t):
        shape = tf.concat([tf.maximum(tf.constant([max_list_size])-tf.shape(t)[0], 0), tf.shape(t)[1:]], 0)
        return tf.concat([t, tf.zeros(shape, dtype=t.dtype)], 0)

    ds = ds.map(lambda x, y, z: (pad(x), pad(y), pad(z)))
    ds = ds.map(lambda x, y, z: (x[:max_list_size, :], y[:max_list_size], z[:max_list_size]))

#   ds = ds.cache()
    ds = ds.shuffle(1000, seed=rng_seed)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = tfds.as_numpy(ds)
    return ds



def prepare_dataset_ids(
    ds: tf.data.Dataset,
    rng_seed: int = 42,
    max_list_size: int = 200,
    batch_size: int = 128,
    qid_table: tf.lookup.StaticHashTable | None = None,
):
    """
    If qid_table is not None:
        - map query_id -> contiguous int ids via the lookup table
    If qid_table is None:
        - keep the original query_id from the dataset (string/bytes/etc.).
    """

    tf.random.set_seed(rng_seed)
    features_to_keep = {"label", "float_features", "query_id"}

    # Keeps only needed features
    ds = ds.map(lambda e: {k: v for k, v in e.items() if k in features_to_keep})

    # Maps to (float_features, label, query_id or continuous_id)
    def map_to_tuple(e):
        x = e["float_features"]
        y = e["label"]

        if qid_table is not None:
            # Use contiguous IDs
            qid = tf.cast(qid_table.lookup(e["query_id"]), tf.int32)
        else:
            # Use raw dataset IDs (e.g., string/bytes)
            qid = e["query_id"]

        return x, y, qid

    ds = ds.map(map_to_tuple)

    # Feature and label transforms
    ds = ds.map(
        lambda x, y, qid: (
            tf.sign(x) * tf.math.log1p(tf.abs(x)),  # feature transform
            2.0**y - 1.0,                           # label transform
            qid
        )
    )

    # Adds mask
    ds = ds.map(lambda x, y, qid: (x, y, tf.ones_like(y, dtype=bool), qid))

    def pad(t):
        shape = tf.concat(
            [
                tf.maximum(tf.constant([max_list_size]) - tf.shape(t)[0], 0),
                tf.shape(t)[1:],
            ],
            axis=0,
        )
        return tf.concat([t, tf.zeros(shape, dtype=t.dtype)], axis=0)

    # Pads lists to max_list_size
    ds = ds.map(
        lambda x, y, z, qid: (
            pad(x),
            pad(y),
            pad(z),
            qid   # usually scalar, no padding needed
        )
    )

    # Clips to max_list_size (safety)
    ds = ds.map(
        lambda x, y, z, qid: (
            x[:max_list_size, :],
            y[:max_list_size],
            z[:max_list_size],
            qid
        )
    )

    ds = ds.shuffle(1000, seed=rng_seed)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return tfds.as_numpy(ds)

def build_query_id_table(raw_ds: tf.data.Dataset, key_name: str = "query_id"):
    """
    Scan the raw dataset once to collect all query_ids and build a mapping:
        original_id -> continuous_id in [0, num_unique_ids).

    Returns:
        table: tf.lookup.StaticHashTable
        num_ids: int, number of unique query_ids
    """
    # Keeps only the query_id feature
    qid_ds = raw_ds.map(lambda e: e[key_name])

    all_qids = []
    for qid in tfds.as_numpy(qid_ds):
        # qid can be bytes, string, or int; just collect them
        all_qids.append(qid)

    all_qids = np.array(all_qids)
    unique_qids = np.unique(all_qids)

    # Converts keys to tf.constant (strings or ints as they are)
    keys = tf.constant(unique_qids)

    # Values are 0..num_unique_qids-1 (continuous)
    values = tf.range(len(unique_qids), dtype=tf.int64)

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=-1  # in case of unseen ID
    )

    return table, len(unique_qids)
