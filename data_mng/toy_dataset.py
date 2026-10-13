import numpy as np


def _make_split(rng, n_queries, n_features, max_list_size, batch_size, variable_list_size):

    """Generates a single dataset split as a list of (x, y, z) numpy batches."""
    xs, ys, zs = [], [], []

    for _ in range(n_queries):
        list_size = (
            rng.integers(10, max_list_size + 1) if variable_list_size else max_list_size
        )

        x_raw = rng.standard_normal((list_size, n_features)).astype(np.float32)
        y_raw = rng.integers(0, 5, size=(list_size,)).astype(np.float32)

        # Mirrors prepare_dataset transforms
        x = np.sign(x_raw) * np.log1p(np.abs(x_raw))
        y = 2.0 ** y_raw - 1.0
        z = np.ones(list_size, dtype=bool)

        # Pad to max_list_size
        pad_len = max_list_size - list_size
        x = np.concatenate([x, np.zeros((pad_len, n_features), dtype=np.float32)], axis=0)
        y = np.concatenate([y, np.zeros(pad_len, dtype=np.float32)], axis=0)
        z = np.concatenate([z, np.zeros(pad_len, dtype=bool)], axis=0)

        xs.append(x)
        ys.append(y)
        zs.append(z)

    xs = np.stack(xs)
    ys = np.stack(ys)
    zs = np.stack(zs)

    batches = []
    for start in range(0, n_queries, batch_size):
        end = start + batch_size
        batches.append((xs[start:end], ys[start:end], zs[start:end]))

    return batches


def get_single_obs_dataset(
    n_features: int = 136,
    max_list_size: int = 200,
    rng_seed: int = 42,
):
    """
    Returns (trn_ds, val_ds, tst_ds) each containing a single query (batch size = 1),
    mirroring the structure of prepare_dataset_ids output.

    Each split is a one-element list of (x, y, z, qid) with shapes:
      x:   (1, max_list_size, n_features)
      y:   (1, max_list_size)
      z:   (1, max_list_size)
      qid: (1,)
    """
    rng = np.random.default_rng(rng_seed)
    splits = []
    for split_idx in range(3):  # trn, val, tst
        x_raw = rng.standard_normal((max_list_size, n_features)).astype(np.float32)
        y_raw = rng.integers(0, 5, size=(max_list_size,)).astype(np.float32)

        # Same transforms as in prepare_dataset_ids
        x = np.sign(x_raw) * np.log1p(np.abs(x_raw))
        y = 2.0 ** y_raw - 1.0
        z = np.ones(max_list_size, dtype=bool)

        # Dummy query id for this split; matches tf.int32 from prepare_dataset_ids
        qid = np.array([split_idx], dtype=np.int32)  # shape (1,)

        # Add batch dimension
        splits.append([(x[None], y[None], z[None], qid)])

    return tuple(splits)



def get_random_dataset(
    n_queries: int = 256,
    n_features: int = 136,
    max_list_size: int = 200,
    batch_size: int = 128,
    variable_list_size: bool = True,
    trn_val_tst_split: tuple = (0.7, 0.15, 0.15),
    rng_seed: int = 42,
):
    """
    Returns (trn_ds, val_ds, tst_ds) mirroring the mslr_web fold structure.

    Args:
        n_queries:          Total number of queries across all splits.
        n_features:         Number of float features per document.
        max_list_size:      Maximum documents per query (padded/truncated).
        batch_size:         Queries per batch.
        variable_list_size: If True, each query gets a random doc count <= max_list_size.
        trn_val_tst_split:  Fraction of queries assigned to train/val/test.
        rng_seed:           Random seed for reproducibility.

    Returns:
        Tuple (trn_ds, val_ds, tst_ds), each a list of (x, y, z) numpy batches:
          x: (batch, max_list_size, n_features)
          y: (batch, max_list_size)
          z: (batch, max_list_size)  — False where padded
    """
    assert abs(sum(trn_val_tst_split) - 1.0) < 1e-6, "Split fractions must sum to 1."

    rng = np.random.default_rng(rng_seed)

    fracs = trn_val_tst_split
    n_trn = int(n_queries * fracs[0])
    n_val = int(n_queries * fracs[1])
    n_tst = n_queries - n_trn - n_val

    splits = []
    for n in (n_trn, n_val, n_tst):
        splits.append(_make_split(rng, n, n_features, max_list_size, batch_size, variable_list_size))

    return tuple(splits)  # (trn_ds, val_ds, tst_ds)