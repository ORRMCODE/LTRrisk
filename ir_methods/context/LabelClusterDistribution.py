"""
Streaming clustering on (feats, labels, mask) batches using JAX.

Each batch is:
  feats:  [Q, D, F]
  labels: [Q, D] with values of the form 2^y - 1 (e.g., 0,1,3,7,15,...)
  mask:   [Q, D]  (True/1 = valid doc, False/0 = ignored)

Features are flattened across Q, D into a 2D matrix [N, F'] where F' = F
(we do NOT append labels as features anymore).

Masked (False) positions are excluded from clustering.

Supports:
  - Streaming / mini-batch K-means
  - Configurable number of clusters
  - Per-cluster label distributions over discrete classes y
    where original_label = 2^y - 1
  - Prediction of cluster IDs and cluster label distributions for new batches
  - Saving and loading of trained centroids and label distributions
"""

import os
from typing import Iterable, Tuple, Optional, Sequence

import functools

import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.nn as jnn
import numpy as np

Array = jnp.ndarray

# Types for a single batch:
#   feats:  [Q, D, F]
#   labels: [Q, D]
#   mask:   [Q, D]

FeatBatch = Sequence[Array]
LabelBatch = Sequence[Array]
MaskBatch = Sequence[Array]


@jax.jit
def _assign_clusters(x: Array, centroids: Array) -> Array:
    """
    Assign each point in x to the nearest centroid (squared Euclidean).

    x:         [N, F']
    centroids: [K, F']
    returns:   cluster indices [N] (int)
    """
    distances = jnp.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
    return jnp.argmin(distances, axis=1)


@jax.jit
def _streaming_update(
    centroids: Array,
    counts: Array,
    batch_sums: Array,
    batch_counts: Array,
) -> Tuple[Array, Array]:
    """
    Update global centroids with mini-batch statistics.

    centroids:    [K, F']
    counts:       [K]
    batch_sums:   [K, F'] (sum of x for each cluster in this batch)
    batch_counts: [K]     (number of points in each cluster in this batch)

    returns (new_centroids, new_counts)
    """
    new_counts = counts + batch_counts           # [K]
    counts_nonzero = jnp.clip(new_counts, 1.0)   # [K]

    old_contrib = centroids * counts[:, None]    # [K, F']
    new_contrib = old_contrib + batch_sums       # [K, F']
    updated_centroids = new_contrib / counts_nonzero[:, None]

    # If a cluster is still empty, keep old centroid
    updated_centroids = jnp.where(
        (new_counts > 0)[:, None],
        updated_centroids,
        centroids,
    )
    return updated_centroids, new_counts


class StreamingFeatureClusterer:
    """
    Streaming / mini-batch K-means on batches of (feats, labels, mask).

    Each batch is:
      feats:  [Q, D, F]
      labels: [Q, D] with integer values of the form 2^y - 1
              (e.g., 0,1,3,7,15,...)
      mask:   [Q, D] (bool or 0/1)

    Internally we convert labels to class indices y via:
        y = round(log2(label + 1))
    and work with y ∈ {0, 1, ..., num_label_classes-1}.

    Parameters:
      - num_clusters: number of clusters K
      - num_label_classes: number of possible y-classes.
          If None, inferred from the first (masked) batch as max(y) + 1.
      - num_epochs: how many passes over trn_dataset (maximum)
      - model_path: optional path for saving/loading (.npz)
      - tol: early-stopping threshold on max centroid movement (L2).
             If <= 0, disabled.
      - patience_batches: how many consecutive "small-movement" batches
                          before stopping
      - min_batches: minimum number of batches to process before early
                     stopping is allowed
    """

    def __init__(
        self,
        num_clusters: int,
        num_label_classes: Optional[int] = None,
        include_label_as_feature: bool = False,  # deprecated, must be False
        num_epochs: int = 25,
        model_path: str = None,
        tol: float = 1e-3,
        patience_batches: int = 10,
        min_batches: int = 100,
    ):
        if include_label_as_feature:
            raise ValueError(
                "include_label_as_feature is no longer supported. "
                "Labels are now used only to compute per-cluster label distributions."
            )

        self.num_clusters = num_clusters
        self.num_label_classes = num_label_classes
        self.include_label_as_feature = False  # keep attribute for compatibility
        self.num_epochs = num_epochs
        self.model_path = model_path

        # Early-stopping settings
        self.tol = tol
        self.patience_batches = patience_batches
        self.min_batches = min_batches

        self.centroids: Optional[Array] = None           # [K, F']
        self.counts: Optional[Array] = None              # [K]
        self.label_counts: Optional[Array] = None        # [K, L]
        self.label_distributions: Optional[Array] = None # [K, L]

    def _check_fitted(self):
        if self.centroids is None or self.label_distributions is None:
            raise RuntimeError("Clusterer is not fitted yet. Call .train(...) first.")

    # ---------------------- Training ---------------------- #
    def train(
        self,
        rng: Array,
        trn_dataset: Iterable[Tuple[Array, Array, Array]],
        force_train: bool = False,
    ):
        """
        Train (fit) streaming K-means over trn_dataset.

        trn_dataset yields: (feats, labels, mask, ...)
          feats:  [Q, D, F]
          labels: [Q, D] with values of the form 2^y - 1
          mask:   [Q, D]
        """
        # 1. If we already have a saved model and user does not force training, load and exit
        if (not force_train) and self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)
            print("Loaded existing trained model, skipping training.")
            return

        # 2. Otherwise, proceed with training
        trn_iter = iter(trn_dataset)
        try:
            first_feats, first_labels, first_mask, _ = next(trn_iter)
        except StopIteration:
            raise ValueError("Training dataset is empty; cannot fit clusterer.")

        # ---- flatten first batch ----
        if first_feats.ndim != 3:
            raise ValueError(f"Expected feats [Q,D,F], got {first_feats.shape}")
        if first_labels.ndim != 2 or first_mask.ndim != 2:
            raise ValueError(
                f"Expected labels/mask [Q,D], got labels {first_labels.shape}, mask {first_mask.shape}"
            )

        Q, D, F = first_feats.shape
        feats_flat = first_feats.reshape(-1, F)   # [Q*D, F]
        labels_flat = first_labels.reshape(-1)    # [Q*D]

        # --- transform labels: label = 2^y - 1 -> y = round(log2(label + 1)) ---
        if jnp.any(labels_flat < 0):
            raise ValueError("Labels must be non-negative integers.")

        labels_y = jnp.log2(labels_flat + 1.0)       # float
        labels_y_rounded = jnp.round(labels_y)       # nearest integer

        # tolerate tiny numerical noise, e.g. 2.9999998
        if not jnp.allclose(labels_y, labels_y_rounded, atol=1e-6):
            raise ValueError(
                "Labels must be of the form 2^y - 1 (e.g., 0,1,3,7,15,...). "
                f"Got non-integer exponents: {labels_y}"
            )

        labels_flat = labels_y_rounded.astype(jnp.int32)  # now class indices y
        # -----------------------------------------------------------------------

        mask_flat = first_mask.reshape(-1).astype(bool)   # [Q*D]

        x0 = feats_flat[mask_flat]                        # [N0, F]
        labels0 = labels_flat[mask_flat]                  # [N0], y-classes
        if x0.shape[0] == 0:
            raise ValueError(
                "First batch has no valid (masked) points; cannot initialize K-means."
            )

        # Infer num_label_classes from y-indices if not provided
        if self.num_label_classes is None:
            max_y = int(jnp.max(labels0))
            if max_y < 0:
                raise ValueError("Transformed labels (y) must be non-negative.")
            self.num_label_classes = max_y + 1

        L = self.num_label_classes  # number of label classes y

        N0, Fp = x0.shape  # F' (here it's just F; we don't append labels anymore)
        if N0 < self.num_clusters:
            raise ValueError(
                f"First batch has {N0} valid points but num_clusters={self.num_clusters}. "
                "Increase batch size or reduce num_clusters."
            )

        rng_init, _ = jrng.split(rng)
        init_indices = jrng.choice(
            rng_init, N0, shape=(self.num_clusters,), replace=False
        )
        centroids = x0[init_indices]  # [K, F']
        counts = jnp.zeros((self.num_clusters,), jnp.float32)
        label_counts = jnp.zeros((self.num_clusters, L), jnp.float32)

        def process_batch(centroids, counts, label_counts, feats, labels, mask):
            if feats.ndim != 3:
                raise ValueError(f"Expected feats [Q,D,F], got {feats.shape}")
            if labels.ndim != 2 or mask.ndim != 2:
                raise ValueError(
                    f"Expected labels/mask [Q,D], got labels {labels.shape}, mask {mask.shape}"
                )

            Qb, Db, Fb = feats.shape
            feats_flat_b = feats.reshape(-1, Fb)      # [Q*D, F]
            labels_flat_b = labels.reshape(-1)        # [Q*D]

            # --- transform labels: 2^y - 1 -> y ---
            if jnp.any(labels_flat_b < 0):
                raise ValueError("Labels must be non-negative integers.")
            labels_y_b = jnp.log2(labels_flat_b + 1.0)       # float
            labels_y_b_round = jnp.round(labels_y_b)         # nearest integer

            if not jnp.allclose(labels_y_b, labels_y_b_round, atol=1e-6):
                raise ValueError(
                    "Labels must be of the form 2^y - 1 (e.g., 0,1,3,7,15,...). "
                    f"Got non-integer exponents in batch: {labels_y_b}"
                )

            labels_flat_b = labels_y_b_round.astype(jnp.int32)  # y-classes
            # --------------------------------------------------

            mask_flat_b = mask.reshape(-1).astype(bool)   # [Q*D]

            xb = feats_flat_b[mask_flat_b]                # [Nb, F]
            labels_sel_b = labels_flat_b[mask_flat_b]     # [Nb], already y-indices
            if xb.shape[0] == 0:
                return centroids, counts, label_counts, 0.0  # nothing to update

            # Assign clusters
            assignments = _assign_clusters(xb, centroids)              # [Nb]
            one_hot = jnn.one_hot(assignments, self.num_clusters)      # [Nb, K]

            batch_sums = one_hot.T @ xb              # [K, F']
            batch_counts = jnp.sum(one_hot, axis=0)  # [K]

            new_centroids, new_counts = _streaming_update(
                centroids, counts, batch_sums, batch_counts
            )

            # Update label counts per cluster
            # labels_sel_b are y in [0, L-1]
            idx = assignments * L + labels_sel_b      # [Nb]
            idx = idx.astype(jnp.int32)
            batch_label_counts_flat = jnp.bincount(
                idx,
                length=self.num_clusters * L,
                weights=jnp.ones_like(idx, dtype=jnp.float32),
            )
            batch_label_counts = batch_label_counts_flat.reshape(self.num_clusters, L)

            new_label_counts = label_counts + batch_label_counts

            # Movement of centroids in this batch (max L2 over clusters)
            diff = new_centroids - centroids                     # [K, F']
            movement = jnp.max(jnp.linalg.norm(diff, axis=-1))   # scalar

            return new_centroids, new_counts, new_label_counts, movement

        # Optionally also use first batch to update centroids and label counts
        centroids, counts, label_counts, _ = process_batch(
            centroids, counts, label_counts, first_feats, first_labels, first_mask
        )

        # Early-stopping bookkeeping
        batches_seen = 1
        no_change_in_a_row = 0
        use_early_stop = (self.tol > 0.0 and self.patience_batches > 0)

        stop = False

        # Remaining epochs
        for epoch in range(self.num_epochs):
            if stop:
                break

            if epoch == 0:
                epoch_iter = trn_iter  # we've already consumed first batch
            else:
                epoch_iter = iter(trn_dataset)

            for feats, labels, mask, _ in epoch_iter:
                centroids, counts, label_counts, movement = process_batch(
                    centroids, counts, label_counts, feats, labels, mask
                )
                batches_seen += 1

                if use_early_stop and batches_seen >= self.min_batches:
                    movement_val = float(movement)
                    if movement_val < self.tol:
                        no_change_in_a_row += 1
                    else:
                        no_change_in_a_row = 0

                    if no_change_in_a_row >= self.patience_batches:
                        print(
                            f"Early stopping at epoch {epoch}, batch {batches_seen} "
                            f"(movement {movement_val:.6f} < tol {self.tol})"
                        )
                        stop = True
                        break  # break inner loop

        # Store final parameters
        self.centroids = centroids
        self.counts = counts
        self.label_counts = label_counts

        # Convert counts to distributions [K, L]
        totals = jnp.sum(label_counts, axis=1, keepdims=True)  # [K,1]
        safe_totals = jnp.where(totals > 0.0, totals, 1.0)
        self.label_distributions = label_counts / safe_totals

        # Save model if a path is provided
        if self.model_path:
            self.save_model(self.model_path)

    # ---------------------- Inference ---------------------- #
    def predict_clusters(
        self,
        feats: Array,
        labels: Array,
        mask: Array,
        Qids: Array,
    ) -> Tuple[Array, Array]:
        """
        Predict cluster index and cluster label distribution for each document in a batch.

        Inputs:
          feats:  [Q, D, F]
          labels: [Q, D] (not used for prediction; only feats are used)
          mask:   [Q, D]

        Returns:
          cluster_ids:         [Q, D] (int, -1 where mask==False)
          cluster_label_dists: [Q, D, L] (float, 0.0 where mask==False)
        """
        self._check_fitted()

        if feats.ndim != 3:
            raise ValueError(f"Expected feats [Q,D,F], got {feats.shape}")
        if labels.ndim != 2 or mask.ndim != 2:
            raise ValueError(
                f"Expected labels/mask [Q,D], got labels {labels.shape}, mask {mask.shape}"
            )

        Q, D, F = feats.shape
        L = self.num_label_classes

        feats_flat = feats.reshape(-1, F)          # [Q*D, F]
        mask_flat = mask.reshape(-1).astype(bool)  # [Q*D]

        x = feats_flat[mask_flat]                  # [N, F]
        if x.shape[0] == 0:
            # No valid docs: everything is masked
            cluster_ids = -jnp.ones((Q, D), dtype=jnp.int32)
            cluster_dists = jnp.zeros((Q, D, L), dtype=jnp.float32)
            return cluster_ids, cluster_dists

        # Assign clusters
        assignments = _assign_clusters(x, self.centroids)           # [N]

        # Map assignments to label distributions
        dists_flat = self.label_distributions[assignments]          # [N, L]

        # Reconstruct full [Q*D] arrays including masked-out docs
        ids_full = -jnp.ones((Q * D,), dtype=jnp.int32)
        dists_full = jnp.zeros((Q * D, L), dtype=dists_flat.dtype)

        ids_full = ids_full.at[mask_flat].set(assignments.astype(jnp.int32))
        dists_full = dists_full.at[mask_flat].set(dists_flat)

        cluster_ids = ids_full.reshape(Q, D)
        cluster_label_dists = dists_full.reshape(Q, D, L)

        return cluster_ids, cluster_label_dists

    # ---------------------- Persistence ---------------------- #
    def save_model(self, path: str):
        """
        Save centroids, counts, label distributions, and config to disk.

        Uses NumPy .npz format (compressed) for simplicity.
        """
        self._check_fitted()

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        np.savez_compressed(
            path,
            centroids=np.array(self.centroids),
            counts=np.array(self.counts),
            label_counts=np.array(self.label_counts),
            label_distributions=np.array(self.label_distributions),
            num_clusters=np.array(self.num_clusters),
            num_label_classes=np.array(self.num_label_classes),
            num_epochs=np.array(self.num_epochs),
        )
        print(f"Clusterer saved to {path}")

    def load_model(self, path: str):
        """
        Load centroids, counts, label distributions, and config from disk.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        data = np.load(path)
        self.centroids = jnp.array(data["centroids"])
        self.counts = jnp.array(data["counts"])
        self.label_counts = jnp.array(data["label_counts"])
        self.label_distributions = jnp.array(data["label_distributions"])

        # Restore config
        self.num_clusters = int(data["num_clusters"])
        self.num_label_classes = int(data["num_label_classes"])
        self.include_label_as_feature = False
        self.num_epochs = int(data["num_epochs"])

        print(f"Clusterer loaded from {path}")


@functools.partial(jax.jit, static_argnames=('include_label_as_feature',))
def assign_clusters_with_label_distributions(
    feats: Array,                 # [Q, D, F]
    labels: Array,                # [Q, D] (unused, kept for API symmetry)
    mask: Array,                  # [Q, D] (bool or 0/1)
    centroids: Array,             # [K, F']
    label_distributions: Array,   # [K, L]
    include_label_as_feature: bool = False,
) -> tuple[Array, Array]:
    """
    JIT-compilable function that assigns each [Q,D] position to a centroid
    and returns the per-cluster label distribution per position.

    - feats: [Q, D, F]
    - labels: [Q, D] (ignored)
    - mask: [Q, D] (False -> output is "invalid")
    - centroids: [K, F']
    - label_distributions: [K, L]

    Returns:
      cluster_ids: [Q, D], int32, -1 where mask == False
      label_dists: [Q, D, L], same dtype as label_distributions,
                   0.0 where mask == False
    """
    if include_label_as_feature:
        raise ValueError("include_label_as_feature is no longer supported.")

    Q, D, F = feats.shape
    K, L = label_distributions.shape

    # Flatten features
    feats_flat = feats.reshape(-1, F)          # [Q*D, F]

    # Compute distances to centroids and assign clusters for ALL positions
    diffs = feats_flat[:, None, :] - centroids[None, :, :]   # [Q*D, K, F']
    dists = jnp.sum(diffs * diffs, axis=-1)                  # [Q*D, K]
    assignments = jnp.argmin(dists, axis=1)                  # [Q*D]

    # Map assignments to label distributions
    dists_flat = label_distributions[assignments]            # [Q*D, L]

    # Reshape back to [Q, D] / [Q, D, L]
    ids_2d = assignments.reshape(Q, D)                       # [Q, D]
    dists_2d = dists_flat.reshape(Q, D, L)                   # [Q, D, L]

    mask_bool = mask.astype(bool)                            # [Q, D]

    # Apply mask: invalid positions get id = -1, distribution = 0
    invalid_ids = -jnp.ones((Q, D), dtype=jnp.int32)
    cluster_ids = jnp.where(mask_bool, ids_2d.astype(jnp.int32), invalid_ids)

    invalid_dists = jnp.zeros((Q, D, L), dtype=dists_2d.dtype)
    label_dists = jnp.where(mask_bool[..., None], dists_2d, invalid_dists)

    return cluster_ids, label_dists