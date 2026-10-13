import matplotlib.pyplot as plt
import numpy as np
import os


def _as_float_array_old(seq):
    """Convert a sequence of scalars / 0-d arrays to a 1D float numpy array."""
    return np.array(
        [x.item() if hasattr(x, "item") else x for x in seq],
        dtype=float,
    )


def plot_metrics_old(
    evaluation_data,
    dataset,
    sampler,
    file_name,
    plot_folder="./plots",
    plot_label=True,
):
    """
    Plot label, local, and global metrics for a given dataset and sampler (distribution).

    Works with the new EvaluationManager structure:

        evaluation_data["_epoch"][dataset] -> [epochs]
        evaluation_data[dataset]["label"][metric] -> [label values]
        evaluation_data[dataset]["dists"][sampler][metric]["local/global"][risk] -> [...]

    Parameters
    ----------
    evaluation_data : dict
        Metrics dictionary from EvaluationManager.get_metrics().
    dataset : str
        Dataset name key.
    sampler : str
        Distribution / sampler name under evaluation_data[dataset]["dists"].
    file_name : str
        Suffix for saved plot filename.
    plot_folder : str
        Folder to save plots.
    plot_label : bool
        If True, plot the label curve for each metric. If False, only plot
        local/global risk curves.
    """

    # Epochs
    epochs = evaluation_data["_epoch"][dataset]
    epochs = _as_float_array(epochs)

    ds_entry = evaluation_data[dataset]
    label_entry = ds_entry["label"]
    dists_entry = ds_entry.get("dists", {})
    dist_entry = dists_entry.get(sampler, {})

    # All metrics that have label values
    metrics = list(label_entry.keys())
    scopes = ["global", "local"]

    # Figure setup
    fig_ratio = (1 + np.sqrt(5)) / 2
    h_size = 5
    w_size = h_size * fig_ratio
    n_rows = len(metrics)
    n_cols = len(scopes)

    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False)

    for i, metric in enumerate(metrics):
        # Label metric values
        label_values = label_entry[metric]
        label_values = _as_float_array(label_values)

        metric_dist_entry = dist_entry.get(metric, {})  # may be {} for eval_risk=False

        for j, scope in enumerate(scopes):
            ax = axes[i, j]

            # Plot label (optional)
            if plot_label:
                ax.plot(
                    epochs,
                    label_values,
                    label=f"{metric}-label",
                    linewidth=2,
                )

            # If this metric has risk-based values for this sampler & scope, plot them
            scope_entry = metric_dist_entry.get(scope, {})
            for functional, values in scope_entry.items():
                values = _as_float_array(values)
                ax.plot(
                    epochs,
                    values,
                    label=f"{metric}-{functional}",
                    linewidth=2,
                )

            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric)
            ax.set_title(f"{scope.capitalize()} metrics")
            ax.legend()
            ax.grid(True, color="gray", linestyle=":")

    # Adjust layout
    fig.set_figheight(h_size * n_rows)
    fig.set_figwidth(w_size * n_cols)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.tight_layout()

    # Save
    os.makedirs(plot_folder, exist_ok=True)
    plot_filename = os.path.join(plot_folder, f"{dataset}_{file_name}.png")
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")

    plt.show()


def _as_float_array(seq):
    """Convert a sequence of scalars / 0-d arrays to a 1D float numpy array."""
    return np.array(
        [x.item() if hasattr(x, "item") else x for x in seq],
        dtype=float,
    )


def plot_metrics(
    evaluation_data,
    dataset,
    sampler,
    file_name,
    plot_folder="./plots",
    plot_label=True,
):
    """
    Plot label, local, and global metrics for a given dataset and sampler (distribution).

    Structure:

        evaluation_data["_epoch"][dataset] -> [epochs]
        evaluation_data[dataset]["label"][metric] -> [label values]
        evaluation_data[dataset]["dists"][sampler][metric]["local/global"][risk] -> [...]

    Behaviour of plot_label:
        - True  -> always plot label curve for each metric.
        - False -> plot label only for metrics that have NO risk-based values
                  (i.e. metrics evaluated only on labels).
    """

    # Epochs
    epochs = evaluation_data["_epoch"][dataset]
    epochs = _as_float_array(epochs)

    ds_entry = evaluation_data[dataset]
    label_entry = ds_entry["label"]
    dists_entry = ds_entry.get("dists", {})
    dist_entry = dists_entry.get(sampler, {})

    # All metrics that have label values
    metrics = list(label_entry.keys())
    scopes = ["global", "local"]

    # Figure setup
    fig_ratio = (1 + np.sqrt(5)) / 2
    h_size = 5
    w_size = h_size * fig_ratio
    n_rows = len(metrics)
    n_cols = len(scopes)

    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False)

    for i, metric in enumerate(metrics):
        # Label metric values
        label_values = label_entry[metric]
        label_values = _as_float_array(label_values)

        # Risk-based data for this metric and sampler (may be empty)
        metric_dist_entry = dist_entry.get(metric, {})

        # Check if this metric has ANY risk-based values (local or global) for this sampler
        has_risk_values = bool(metric_dist_entry.get("local")) or bool(
            metric_dist_entry.get("global")
        )

        for j, scope in enumerate(scopes):
            ax = axes[i, j]

            # Decide whether to plot label for this metric
            # - If plot_label is True -> always plot label.
            # - If plot_label is False:
            #       - If has_risk_values -> do NOT plot label (used for comparison).
            #       - If not has_risk_values -> still plot label (label-only metric).
            if plot_label or not has_risk_values:
                ax.plot(
                    epochs,
                    label_values,
                    label=f"{metric}-label",
                    linewidth=2,
                )

            # Plot risk-based values for this sampler & scope, if any
            scope_entry = metric_dist_entry.get(scope, {})
            for functional, values in scope_entry.items():
                values = _as_float_array(values)
                ax.plot(
                    epochs,
                    values,
                    label=f"{metric}-{functional}",
                    linewidth=2,
                )

            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric)
            ax.set_title(f"{scope.capitalize()} metrics")
            ax.legend()
            ax.grid(True, color="gray", linestyle=":")

    # Adjust layout
    fig.set_figheight(h_size * n_rows)
    fig.set_figwidth(w_size * n_cols)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.tight_layout()

    # Save
    os.makedirs(plot_folder, exist_ok=True)
    plot_filename = os.path.join(plot_folder, f"{dataset}_{file_name}.png")
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")

    plt.show()