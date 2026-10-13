import os
import json
import csv
import numpy as np



class ArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        # Anything with .tolist (NumPy/JAX arrays) -> list
        if hasattr(obj, "tolist"):
            return obj.tolist()
        # 0‑d arrays with .item() -> scalar
        if hasattr(obj, "item"):
            return obj.item()
        return super().default(obj)

def save_dictionary(metrics_dic, out_folder, trn_sampler, trn_fnal, trn_fnal_desc):
    out_folder_dic = os.path.join(out_folder, "dic")
    os.makedirs(out_folder_dic, exist_ok=True)
    json_path = os.path.join(
        out_folder_dic,
        f"UTRAIN={trn_sampler}__FTRAIN={trn_fnal}__DESC={trn_fnal_desc}.json",
    )

    with open(json_path, "w") as f:
        json.dump(metrics_dic, f, indent=4, cls=ArrayEncoder)

def _as_float_array(seq):
    """Convert a sequence of scalars / 0-d arrays to a 1D float numpy array."""
    return np.array(
        [x.item() if hasattr(x, "item") else x for x in seq],
        dtype=float,
    )

def save_dictionary_old(metrics_dic, out_folder, trn_sampler, trn_fnal, trn_fnal_desc):
    out_folder_dic = os.path.join(out_folder, "dic")
    os.makedirs(out_folder_dic, exist_ok=True)
    json_path = os.path.join(out_folder_dic, f"UTRAIN={trn_sampler}__FTRAIN={trn_fnal}__DESC={trn_fnal_desc}.json")
    with open(json_path, "w") as f:
        json.dump(metrics_dic, f, indent=4)


# def save_dictionary(metrics_dic, out_folder, trn_sampler, trn_fnal, trn_fnal_desc):
#     out_folder_dic = os.path.join(out_folder, "dic")
#     os.makedirs(out_folder_dic, exist_ok=True)
#     json_path = os.path.join(
#         out_folder_dic,
#         f"UTRAIN={trn_sampler}__FTRAIN={trn_fnal}__DESC={trn_fnal_desc}.json",
#     )
#     json_safe_dic = _make_json_safe(metrics_dic)
#     with open(json_path, "w") as f:
#         json.dump(json_safe_dic, f, indent=4)

def export_metrics(
    evaluation_data,
    dataset,
    sampler,
    file_name,
    out_folder="./metrics",
    save_json=True,
    save_csv=True,
):
    """
    Flatten evaluation_data into a table-like structure and save as JSON/CSV.

    New structure:

        evaluation_data["_epoch"][dataset] -> [epochs]
        evaluation_data[dataset]["label"][metric] -> [label values]
        evaluation_data[dataset]["dists"][sampler][metric]["local/global"][risk] -> [...]

    Each row has:
        epoch, metric, scope, functional, value

    scope ∈ {"label", "local", "global"}.
    functional:
        - "label" for label scope
        - risk name for local/global
    """

    # Epochs
    epochs = evaluation_data["_epoch"][dataset]
    epochs = _as_float_array(epochs)

    ds_entry = evaluation_data[dataset]
    label_entry = ds_entry["label"]
    dists_entry = ds_entry.get("dists", {})
    dist_entry = dists_entry.get(sampler, {})

    # All metrics (from labels); some may not have risk-based entries for this sampler
    metrics = list(label_entry.keys())
    scopes = ["global", "local"]

    rows = []

    for metric in metrics:
        # --- Label values (always present for this metric) ---
        label_values = label_entry[metric]
        label_values = _as_float_array(label_values)

        for epoch, value in zip(epochs, label_values):
            rows.append(
                {
                    "epoch": float(epoch),
                    "metric": metric,
                    "scope": "label",
                    "functional": "label",
                    "value": float(value),
                }
            )

        # --- Scoped values (global / local) if this metric has risk entries for this sampler ---
        metric_dist_entry = dist_entry.get(metric, {})

        for scope in scopes:
            scope_entry = metric_dist_entry.get(scope, {})
            # keys under this scope are the risk functionals
            for functional, values in scope_entry.items():
                values = _as_float_array(values)

                for epoch, value in zip(epochs, values):
                    rows.append(
                        {
                            "epoch": float(epoch),
                            "metric": metric,
                            "scope": scope,
                            "functional": functional,
                            "value": float(value),
                        }
                    )

    # Ensure output folder exists
    os.makedirs(out_folder, exist_ok=True)
    base_path = os.path.join(out_folder, f"{dataset}_{file_name}")

    # Save JSON
    if save_json:
        json_path = base_path + ".json"
        with open(json_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Metrics saved as JSON: {json_path}")

    # Save CSV
    if save_csv:
        csv_path = base_path + ".csv"
        fieldnames = ["epoch", "metric", "scope", "functional", "value"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Metrics saved as CSV: {csv_path}")