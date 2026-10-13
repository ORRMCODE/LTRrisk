import jax
import jax.numpy as jnp
import jax.random as jrng

from collections import defaultdict
import copy

import auxiliary as aux


# -------------------- Metric wrappers -------------------- #
def metric_of_label(metric_fn, *, metric_args):
    """
    Deterministic metric on observed labels (no sampling).

    preds:   (Q,D)
    targets: (Q,D)
    mask:    (Q,D)

    Returns:
        scalar (mean over queries)
    """

    def fn(preds, targets, mask):
        # q_metric_fn: (preds, targets, mask, **metric_args) -> (Q,)
        q_metric_fn = aux.per_query_metric_rng(metric_fn)
        q_values = q_metric_fn(preds, targets, mask, **metric_args)  # (Q,)
        return jnp.mean(q_values)  # scalar

    return jax.jit(fn)


def make_local_reduction_fn(metric_fn, risk_fn, *, metric_args, risk_args):
    """
    Local risk: mean over queries of risk over samples.

    preds:           (Q,D)
    sampled_targets: (Q,D,S)
    mask:            (Q,D)

    Steps:
        1) For each sample s, compute metric per query -> (Q,S)
        2) For each query, apply risk over S -> (Q,)
        3) Average over queries -> scalar
    """

    def fn(preds, sampled_targets, mask):
        # metric_samples: (Q,S)
        qs_metric_fn = aux.per_query_metric_rng(
                        aux.per_sample_metric(
                            metric_fn
                        )
                    )
        metric_samples = qs_metric_fn(preds, sampled_targets, mask, **metric_args)
        per_query_risk = jax.vmap(
            lambda qs: risk_fn(qs, **risk_args),
            in_axes=0
        )(metric_samples)

        return jnp.mean(per_query_risk)

    return jax.jit(fn)


def make_global_reduction_fn(metric_fn, risk_fn, *, metric_args, risk_args):
    """
    Global risk: risk of the query-averaged metric.

    preds:           (Q,D)
    sampled_targets: (Q,D,S)
    mask:            (Q,D)

    Steps:
        1) For each sample s, compute metric per query -> (Q,S)
        2) Average over queries -> (S,)
        3) Apply risk over S -> scalar
    """

    def fn(preds, sampled_targets, mask):
        q_metric_fn = aux.per_query_metric_rng(metric_fn)
        metric_samples = jax.vmap(
            lambda t_ds: q_metric_fn(preds, t_ds, mask, **metric_args),
            in_axes=2,  # iterate over S
            out_axes=1  # result (Q,S)
        )(sampled_targets)

        # mean over queries -> (S,)
        per_sample_mean = jnp.mean(metric_samples, axis=0)
        return risk_fn(per_sample_mean, **risk_args)

    return jax.jit(fn)


# -------------------- Main Evaluation Manager -------------------- #
class EvaluationManager:
    """
    metrics_dict structure:

        metrics_dict = {
            "_epoch": {
                dataset_name: [epoch_0, epoch_1, ...],
                ...
            },
            dataset_name: {
                "label": {
                    metric_name: [label_epoch_0, label_epoch_1, ...],
                },
                "dists": {
                    dist_name: {
                        metric_name: {
                            "local":  { risk_name: [local_epoch_0, ...],  ... },
                            "global": { risk_name: [global_epoch_0, ...], ... },
                        },
                        ...
                    },
                    ...
                },
            },
            ...
        }

    - Label metrics are aggregated per (dataset, metric), independent of distribution.
    - Local/global risk metrics are per (dataset, dist, metric, risk).

    PERFORMANCE NOTE
    -----------------
    For risk-based metrics, the heavy work is done in a **per-distribution jitted
    batch function** built by `build_dist_batch_fns()`. This function:

        - runs the sampler (to get sampled_targets),
        - computes all (metric, risk) pairs (local and global) for that dist
          in a single JAX call.

    `update_dist_for_batch` then only does light Python aggregation.
    """

    def __init__(self, rng):
        self.rng = rng
        self.model_state = None
        self.model_variables = None
        self.last_eval_epoch = {}

        # Registered functions and their args
        self.metrics = {}
        self.metric_args = {}
        self.metric_eval_risk = {}  # metric_name -> bool

        self.samplers = {}
        self.sampler_args = {}
        self.sampler_envs = {}  # stored but not passed (to avoid JAX static issues)

        self.risks = {}
        self.risk_args = {}

        # Compiled JAX functions (per metric / risk)
        self._label_fns = {}   # metric_name -> jitted label metric
        self._local_fns = {}   # (metric_name, risk_name) -> jitted local reduction
        self._global_fns = {}  # (metric_name, risk_name) -> jitted global reduction

        # Per-distribution jitted batch functions:
        # dist_name -> (preds, feats, targets, mask, rng_key) -> (local_vals, global_vals)
        self._dist_batch_fns = {}

        # Fixed ordering for metric/risk indices when stacking results
        self._metric_names_eval_risk = []
        self._risk_names = []

        # Storage
        self.metrics_dict = {"_epoch": {}}

        # Epoch buffers
        self.reset_epoch_buffers()

    # ---------------- Model ---------------- #
    def set_model_state(self, model_state):
        self.model_state = model_state

    def apply_control(self, batch_features, rng_key):
        """
        Apply model to features to obtain predictions.
        """
        return aux.prd_control_fn(
            self.model_state,
            self.model_state.params,
            batch_features,
            rng_key,
        )

    # ---------------- Registration ---------------- #
    def register_risk(self, name, fn, **risk_args):
        """
        Register a risk functional.

        fn: risk_fn(samples, **risk_args) -> scalar
        """
        self.risks[name] = fn
        self.risk_args[name] = risk_args

    def register_metric(self, name, fn, eval_risk, **metric_args):
        """
        Register a metric.

        name: str
        fn: metric function (used inside aux.per_query_metric_rng)
        eval_risk: bool
            If True, we also compute local/global risk using this metric.
        """
        self.metrics[name] = fn
        self.metric_args[name] = metric_args
        self.metric_eval_risk[name] = eval_risk

    def register_sampler(self, dist, fn, sampler_envs=None, **sampler_args):
        """
        Register a sampler for a given distribution name.

        Expected sampler signature (matches how we call it):

            fn(features, targets, mask, rng_key, **args) -> (Q,D,S)

        sampler_envs is stored but not passed into the sampler here, to avoid
        JAX issues with non-array / string static arguments. If you need env,
        you can either:
            - bake it into fn via closure, or
            - include it inside sampler_args in a JAX-compatible way.
        """
        self.samplers[dist] = fn
        self.sampler_args[dist] = sampler_args
        self.sampler_envs[dist] = sampler_envs  # kept for possible external use

        # If batch fns have already been built, the user should rebuild them
        # after adding new samplers.
        if self._dist_batch_fns:
            # Do not rebuild automatically (might be expensive); instead:
            # mark as stale.
            self._dist_batch_fns = {}

    # ---------------- Compilation ---------------- #
    def precompile_all(self):
        """
        Precompile label/local/global functions for all registered metrics and risks.

        NOTE: This only builds per-metric/per-risk functions. To get the
        high-performance per-distribution batch functions used in
        `update_dist_for_batch`, you must also call:

            self.build_dist_batch_fns()

        after all samplers are registered.
        """
        for metric_name, metric_fn in self.metrics.items():
            # deterministic label metric
            self._label_fns[metric_name] = metric_of_label(
                metric_fn, metric_args=self.metric_args[metric_name]
            )

            if self.metric_eval_risk[metric_name]:
                for risk_name, risk_fn in self.risks.items():
                    key = (metric_name, risk_name)

                    self._local_fns[key] = make_local_reduction_fn(
                        metric_fn,
                        risk_fn,
                        metric_args=self.metric_args[metric_name],
                        risk_args=self.risk_args[risk_name],
                    )

                    self._global_fns[key] = make_global_reduction_fn(
                        metric_fn,
                        risk_fn,
                        metric_args=self.metric_args[metric_name],
                        risk_args=self.risk_args[risk_name],
                    )

        # Per-distribution batch fns depend on samplers as well, so they are
        # built in a separate step by build_dist_batch_fns().

    def build_dist_batch_fns(self):
        """
        Build and jit per-distribution batch evaluation functions.

        Must be called *after*:
            - all metrics and risks are registered,
            - precompile_all() has been called,
            - all samplers are registered.

        Result:
            self._dist_batch_fns[dist_name](preds, feats, targets, mask, rng_key)
            returns (local_vals, global_vals) where:

                - local_vals:  shape (num_metric_risk_pairs,)
                - global_vals: shape (num_metric_risk_pairs,)

            The ordering of metric/risk pairs is:

                for metric_name in metric_names_eval_risk:
                    for risk_name in risk_names:
                        ...

            and the same ordering is used when aggregating results.
        """
        if not self._local_fns or not self._global_fns:
            raise RuntimeError(
                "precompile_all() must be called before build_dist_batch_fns()."
            )

        metric_names_eval_risk = [
            m for m, do_eval in self.metric_eval_risk.items() if do_eval
        ]
        risk_names = list(self.risks.keys())

        self._metric_names_eval_risk = metric_names_eval_risk
        self._risk_names = risk_names
        self._dist_batch_fns = {}

        for dist_name, sampler_fn in self.samplers.items():
            # Filter out any rng_key from sampler_args to avoid double-passing
            sampler_args = {
                k: v for k, v in self.sampler_args[dist_name].items()
                if k != "rng_key"
            }

            # Collect jitted metric/risk functions in a static structure
            local_fns = []
            global_fns = []

            for metric_name in metric_names_eval_risk:
                for risk_name in risk_names:
                    fn_key = (metric_name, risk_name)
                    local_fns.append(self._local_fns[fn_key])
                    global_fns.append(self._global_fns[fn_key])

            # Use tuples to emphasize immutability in the closure
            local_fns = tuple(local_fns)
            global_fns = tuple(global_fns)

            def make_dist_batch_fn(
                sampler_fn, sampler_args, local_fns, global_fns
            ):
                @jax.jit
                def dist_batch_fn(preds, feats, targets, mask, rng_key):
                    # sampling: expected shape (Q,D,S)
                    sampled_targets = sampler_fn(
                        feats,
                        targets,
                        mask,
                        rng_key=rng_key,
                        **sampler_args,
                    )

                    local_vals = []
                    global_vals = []

                    # The loops are over a small, static number of metric/risk pairs
                    for lf, gf in zip(local_fns, global_fns):
                        local_vals.append(lf(preds, sampled_targets, mask))
                        global_vals.append(gf(preds, sampled_targets, mask))

                    local_vals = jnp.stack(local_vals)   # (M*R,)
                    global_vals = jnp.stack(global_vals) # (M*R,)
                    return local_vals, global_vals

                return dist_batch_fn

            self._dist_batch_fns[dist_name] = make_dist_batch_fn(
                sampler_fn, sampler_args, local_fns, global_fns
            )

    # ---------------- Epoch buffers ---------------- #
    def reset_epoch_buffers(self):
        """
        Reset accumulators used within a single epoch.
        """

        # Label metrics: key = (dataset, metric)
        self._label_sum = defaultdict(float)

        # Local risk: key = (dataset, dist, metric, risk)
        self._local_sum = defaultdict(float)

        # Counts:
        #   - label: (dataset, metric)
        #   - local: (dataset, dist, metric, risk)
        self._count = defaultdict(int)

        # Global risk: key = (dataset, dist, metric, risk)
        self._global_vals = defaultdict(list)

    # ---------------- Aggregation helpers (kept for possible external use) ---------------- #
    def aggregate_global(self, fn_key, acc_key, preds, targets_s, masks):
        """
        Legacy helper: calls per-metric global fn directly.
        Not used in the new fast path, but kept for compatibility.
        """
        global_val = self._global_fns[fn_key](preds, targets_s, masks)
        self._global_vals[acc_key].append(global_val)
        return global_val

    def aggregate_local(self, fn_key, acc_key, preds, targets_s, masks, Q):
        """
        Legacy helper: calls per-metric local fn directly.
        Not used in the new fast path, but kept for compatibility.
        """
        local_val = self._local_fns[fn_key](preds, targets_s, masks)
        self._local_sum[acc_key] += local_val * Q
        self._count[acc_key] += Q
        return local_val

    def aggregate_label(self, dataset_name, metric_name, preds, targets, masks, Q):
        """
        Label aggregation is per (dataset, metric), independent of distribution.
        """
        key = (dataset_name, metric_name)
        label_val = self._label_fns[metric_name](preds, targets, masks)
        self._label_sum[key] += label_val * Q
        self._count[key] += Q
        return label_val

    # ---------------- Batch-level updates ---------------- #
    def _compute_preds(self, batch_features, batch_targets):
        if self.model_state is not None:
            self.rng, rng = jrng.split(self.rng)
            preds = self.apply_control(batch_features, rng)
        else:
            preds = jnp.zeros_like(batch_targets)
        return preds

    def update_labels_for_batch(self, dataset_name, preds, batch_targets, batch_mask):
        """
        Update label (deterministic) metrics for a single batch.
        """
        Q = batch_targets.shape[0]

        for metric_name in self.metrics.keys():
            self.aggregate_label(
                dataset_name,
                metric_name,
                preds,
                batch_targets,
                batch_mask,
                Q,
            )

    def update_dist_for_batch(
        self,
        dataset_name,
        dist_name,
        preds,
        batch_features,
        batch_targets,
        batch_mask,
    ):
        """
        Update local/global risk metrics for a single batch and one distribution.

        Heavy work is done inside a jitted per-distribution batch function
        built by build_dist_batch_fns().
        """
        if not self._dist_batch_fns:
            raise RuntimeError(
                "build_dist_batch_fns() must be called after precompile_all() "
                "and sampler registration before evaluating distributions."
            )
        if dist_name not in self._dist_batch_fns:
            raise KeyError(
                f"No jitted batch function for dist '{dist_name}'. "
                "Did you call build_dist_batch_fns() after registering this sampler?"
            )

        # --- rng for sampler / metrics ---
        self.rng, sampler_rng = jrng.split(self.rng)

        dist_fn = self._dist_batch_fns[dist_name]
        local_vals, global_vals = dist_fn(
            preds, batch_features, batch_targets, batch_mask, sampler_rng
        )

        Q = preds.shape[0]

        # Map back from flat index to (metric_name, risk_name)
        idx = 0
        for metric_name in self._metric_names_eval_risk:
            for risk_name in self._risk_names:
                acc_key = (dataset_name, dist_name, metric_name, risk_name)

                local_val = float(local_vals[idx])
                global_val = float(global_vals[idx])
                idx += 1

                # local aggregation (weighted by Q)
                self._local_sum[acc_key] += local_val * Q
                self._count[acc_key] += Q

                # global aggregation (simple list of batch-level values)
                self._global_vals[acc_key].append(global_val)

    # ---------------- Append aggregates to metrics_dict ---------------- #
    def append_agg_label(self, dataset_name, metric_name):
        """
        Store aggregated label metric for (dataset, metric) into metrics_dict.
        """
        key = (dataset_name, metric_name)
        if self._count[key] == 0:
            return

        mean_val = float(self._label_sum[key] / self._count[key])

        ds_entry = self.metrics_dict.setdefault(
            dataset_name, {"label": {}, "dists": {}}
        )
        label_dict = ds_entry.setdefault("label", {})
        metric_list = label_dict.setdefault(metric_name, [])
        metric_list.append(mean_val)

    def append_agg_local(self, dataset_name, dist_name, metric_name, risk_name):
        """
        Store aggregated local risk for (dataset, dist, metric, risk).
        """
        key = (dataset_name, dist_name, metric_name, risk_name)
        if self._count[key] == 0:
            return

        mean_val = float(self._local_sum[key] / self._count[key])

        ds_entry = self.metrics_dict.setdefault(
            dataset_name, {"label": {}, "dists": {}}
        )
        dists_dict = ds_entry.setdefault("dists", {})
        dist_entry = dists_dict.setdefault(dist_name, {})
        metric_entry = dist_entry.setdefault(
            metric_name, {"local": {}, "global": {}}
        )

        local_dict = metric_entry.setdefault("local", {})
        vals = local_dict.setdefault(risk_name, [])
        vals.append(mean_val)

    def append_agg_global(self, dataset_name, dist_name, metric_name, risk_name):
        """
        Store aggregated global risk for (dataset, dist, metric, risk).
        """
        key = (dataset_name, dist_name, metric_name, risk_name)
        if not self._global_vals[key]:
            return

        mean_val = float(sum(self._global_vals[key]) / len(self._global_vals[key]))

        ds_entry = self.metrics_dict.setdefault(
            dataset_name, {"label": {}, "dists": {}}
        )
        dists_dict = ds_entry.setdefault("dists", {})
        dist_entry = dists_dict.setdefault(dist_name, {})
        metric_entry = dist_entry.setdefault(
            metric_name, {"local": {}, "global": {}}
        )

        global_dict = metric_entry.setdefault("global", {})
        vals = global_dict.setdefault(risk_name, [])
        vals.append(mean_val)

    # ------------------ End epoch ------------------- #
    def end_epoch(self, dataset_name, epoch=None):
        """
        Finalize the current epoch: push buffered aggregates into metrics_dict
        and reset buffers.
        """
        # Labels: one per (dataset, metric), independent of dist
        for metric_name in self.metrics.keys():
            self.append_agg_label(dataset_name, metric_name)

        # Local / global risk: per (dataset, dist, metric, risk)
        for dist_name in self.samplers.keys():
            for metric_name in self.metrics.keys():
                if not self.metric_eval_risk[metric_name]:
                    continue

                for risk_name in self.risks.keys():
                    self.append_agg_local(
                        dataset_name, dist_name, metric_name, risk_name
                    )
                    self.append_agg_global(
                        dataset_name, dist_name, metric_name, risk_name
                    )

        # Epoch bookkeeping
        if epoch is not None:
            if (dataset_name not in self.last_eval_epoch) or (
                self.last_eval_epoch[dataset_name] is None
            ):
                self.last_eval_epoch[dataset_name] = []
            self.last_eval_epoch[dataset_name].append(epoch)
            self.metrics_dict["_epoch"] = self.last_eval_epoch

        # Reset buffers for the next epoch
        self.reset_epoch_buffers()

    # ---------------- Dataset evaluation ---------------- #
    def evaluate_dataset(self, *, dataset, dataset_name, dist_names=None, epoch=None):
        """
        Evaluate a full dataset by iterating over batches.

        Args:
            dataset: iterable of
                (batch_feats, batch_targets, batch_mask, batch_Qids)
            dataset_name: str
            dist_names: list of distribution names to evaluate (default: all)
            epoch: optional epoch index (stored in metrics_dict["_epoch"])

        REQUIREMENT:
            - precompile_all() must have been called,
            - build_dist_batch_fns() must have been called (if risk-based metrics
              are to be evaluated).
        """
        if dist_names is None:
            dist_names = list(self.samplers.keys())

        for batch_data in dataset:
            batch_feats, batch_targets, batch_mask, batch_Qids = batch_data

            # --- compute predictions once per batch ---
            preds = self._compute_preds(batch_feats, batch_targets)

            # --- update label metrics (independent of dist) ---
            self.update_labels_for_batch(
                dataset_name, preds, batch_targets, batch_mask
            )

            # --- update risk-based metrics per distribution ---
            for dist_name in dist_names:
                self.update_dist_for_batch(
                    dataset_name,
                    dist_name,
                    preds,
                    batch_feats,
                    batch_targets,
                    batch_mask,
                )

        self.end_epoch(dataset_name, epoch=epoch)

    # ---------------- Accessor ---------------- #
    def get_metrics(self, copy_dict=True):
        """
        Return the metrics dictionary.
        If copy_dict=True, return a deep copy.
        """
        if copy_dict:
            return copy.deepcopy(self.metrics_dict)
        return self.metrics_dict

    # ----------------- Log - Loss --------------------#
    def log_loss(self, dataset_name, loss_value, name="loss"):
        """
        Log a scalar loss into metrics_dict for a given dataset.

        This is typically called once per epoch (but you can also call
        it per batch if you prefer, as long as you interpret the list accordingly).

        Args:
            dataset_name: e.g. "TRN", "VAL", "TST"
            loss_value:   scalar (Python float or JAX array 0-d)
            name:         identifier under which the loss is stored
                          (default: "loss"; e.g. "train_loss", "val_loss" if needed)
        """
        # Ensure the dataset entry exists
        ds_entry = self.metrics_dict.setdefault(
            dataset_name,
            {"label": {}, "dists": {}}
        )

        # Create / use a "label" dict inside this dataset entry
        loss_dict = ds_entry.setdefault("label", {})

        # Create / use the list for this particular loss name
        loss_list = loss_dict.setdefault(name, [])

        # Convert to plain float for serialization
        loss_list.append(float(loss_value))

    def log_info(self, dataset_name, info_value, name):

        # Ensure the dataset entry exists
        ds_entry = self.metrics_dict.setdefault(
            dataset_name,
            {"label": {}, "dists": {}, "info":{}}
        )

        # Create / use a "label" dict inside this dataset entry
        loss_dict = ds_entry.setdefault("info", {})

        # Create / use the list for this particular loss name
        loss_list = loss_dict.setdefault(name, [])

        # Convert to plain float for serialization
        loss_list.append(info_value)