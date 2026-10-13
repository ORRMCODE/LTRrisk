import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="Training script for IR / portfolio model."
    )

    # --- General configuration ---
    parser.add_argument("--rng_key", type=int, default=42,
                        help="Random seed for JAX.")
    parser.add_argument("--ltr_dataset", type=str, default="mslr_web/10k_fold1",
                        help="TFDS LTR dataset name.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for dataset")
    parser.add_argument("--variational", action="store_true",
                        help="Use variational risk functionals (VARIATIONAL=True).")

    # --- Training risk functional / sampler ---
    parser.add_argument("--trn_loss", type=str, default="v_exposure_loss",
                        help = "Training loss (TRN_LOSS)")
    parser.add_argument("--trn_sampler", type=str, default="cluster_stakes",
                        help="Training sampler name (TRN_SAMPLER).")
    parser.add_argument("--trn_fnal", type=str, default="Q10",
                        help="Training risk functional name (TRN_FNAL).")
    parser.add_argument("--trn_fnal_desc", type=str, default="POP",
                        help="Training functional description (TRN_FNAL_DESC).")

    parser.add_argument("--loss_q", type=float, default=0.9,
                        help="Quantile level for LOSS_Q.")
    parser.add_argument("--tradeoff", type=float, default=1e0,
                        help="Tradeoff coefficient for tailored trade-off functional.")

    parser.add_argument("--topn", type=int, default=10,
                        help="Top-N cutoff for DCG / NDCG.")
    parser.add_argument("--relsamples", type=int, default=1000,
                        help="Number of relevance samples (RELSAMPLES).")
    parser.add_argument("--ranksamples", type=int, default=60,
                        help="Number of ranking samples (RANKSAMPLES).")

    # --- Context models ---
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Number of label classes (NUM_CLASSES).")

    parser.add_argument("--l_model_epochs", type=int, default=35,
                        help="Epochs for label model (L_MODEL_EPOCHS).")
    parser.add_argument("--fit_l_model", action="store_true",
                        help="Force training of label model (FIT_L_MODEL=True).")

    parser.add_argument("--num_clusters", type=int, default=25,
                        help="Number of clusters for uncertainty clusterer.")
    parser.add_argument("--cluster_epochs", type=int, default=25,
                        help="Epochs for uncertainty clusterer.")
    parser.add_argument("--fit_u_cluster", action="store_true",
                        help="Force training of uncertainty clusterer (FIT_U_CLUSTER=True).")

    parser.add_argument("--labeldist_num_clusters", type=int, default=25,
                        help="Number of clusters for label-distribution clusterer.")
    parser.add_argument("--labeldist_cluster_epochs", type=int, default=25,
                        help="Epochs for label-distribution clusterer.")
    parser.add_argument("--labeldist_fit_cluster", action="store_true",
                        help="Force training of label-distribution clusterer (LABELDIST_FIT_CLUSTER=True).")

    parser.add_argument("--constant_concentration", type=float, default=5.0,
                        help="Constant concentration parameter for sampler context.")

    # --- Output paths / training hyperparams ---
    parser.add_argument("--plots_path", type=str, default="./plots_toy",
                        help="Path to save plots.")
    parser.add_argument("--files_path", type=str, default="./metrics_toy",
                        help="Path to save exported metrics.")
    parser.add_argument("--evaluate_every", type=int, default=15,
                        help="Evaluate the whole dataset every evaluate_every epochs")


    parser.add_argument("--max_epochs", type=int, default=601,
                        help="Max epochs for optimize_vrisk.")
    parser.add_argument("--learning_rate_loss", type=float, default=1e-2,
                        help="Learning rate for loss parameters.")
    parser.add_argument("--learning_rate_aux", type=float, default=1e-2,
                        help="Learning rate for auxiliary parameters.")

                        
    return parser.parse_args()