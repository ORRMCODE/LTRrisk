import os
import time

# ---------------------------------------------------------------------
# XLA / JAX configuration (must be set before importing jax)
# ---------------------------------------------------------------------
os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.98"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer="

import jax
import jax.random as jrng
import jax.numpy as jnp
import tensorflow_datasets as tfds
import json

from typing import Any, Dict

import data_mng
import training
import ir_methods
import portfolio
import portfolio.grd_fnctnl as grd_fnctnl
import portfolio.sto_fnctnl as sto_fnctnl
import auxiliary as aux
import auxiliary.wrappers as aux_wrap
import visualize
import tailored_functions as tailored

from samplers import SamplerContext, make_sampler

# NEW: import CLI argument parser
from cli_args import get_args


def main():
    # -----------------------------------------------------------------
    # Parse CLI arguments
    # -----------------------------------------------------------------
    args = get_args()

    # -----------------------------------------------------------------
    # General configuration (read from args instead of hard-coded)
    # -----------------------------------------------------------------
    RNG_KEY = args.rng_key
    LTR_DATASET = args.ltr_dataset

    VARIATIONAL = args.variational
    TRN_SAMPLER = args.trn_sampler
    TRN_FNAL = args.trn_fnal
    TRN_FNAL_DESC = args.trn_fnal_desc

    LOSS_Q = args.loss_q
    TRADEOFF = args.tradeoff

    TOPN = args.topn
    RELSAMPLES = args.relsamples
    RANKSAMPLES = args.ranksamples

    NUM_CLASSES = args.num_classes

    LABELDIST_NUM_CLUSTERS = args.labeldist_num_clusters
    LABELDIST_CLUSTER_EPOCHS = args.labeldist_cluster_epochs
    LABELDIST_CLUSTER_MODEL_PATH = f"./misc/ir_methods/context/{LTR_DATASET}/clusterer_model_label_distribution.npz"
    LABELDIST_FIT_CLUSTER = args.labeldist_fit_cluster

    # -----------------------------------------------------------------
    # RNG setup
    # -----------------------------------------------------------------
    rng_key = jrng.key(RNG_KEY)
    rng_key, rng_labcluster_trn = jrng.split(rng_key, 2)
    # -----------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------
    trn_ds_tf = tfds.load(LTR_DATASET, split="train")
    val_ds_tf = tfds.load(LTR_DATASET, split="vali")
    tst_ds_tf = tfds.load(LTR_DATASET, split="test")

    qid_table, num_qids = data_mng.build_query_id_table(trn_ds_tf)

    trn_ds = data_mng.prepare_dataset_ids(trn_ds_tf, batch_size=args.batch_size, qid_table=qid_table, rng_seed = RNG_KEY)
    val_ds = data_mng.prepare_dataset_ids(val_ds_tf, batch_size=args.batch_size, qid_table=None, rng_seed = RNG_KEY)
    tst_ds = data_mng.prepare_dataset_ids(tst_ds_tf, batch_size=args.batch_size, qid_table=None, rng_seed = RNG_KEY)

    # Paths from args instead of hard-coded
    PLOTS_PATH = args.plots_path
    FILES_PATH = args.files_path

    print(num_qids)

    # -----------------------------------------------------------------
    # Precompute quantile weights (uses RELSAMPLES)
    # -----------------------------------------------------------------
    Q_Wi_mat90_HD = portfolio.W_quantile_HD(N_samples=RELSAMPLES, q=0.90)
    Q_Wi_mat75_HD = portfolio.W_quantile_HD(N_samples=RELSAMPLES, q=0.75)
    Q_Wi_mat50_HD = portfolio.W_quantile_HD(N_samples=RELSAMPLES, q=0.50)
    Q_Wi_mat25_HD = portfolio.W_quantile_HD(N_samples=RELSAMPLES, q=0.25)
    Q_Wi_mat10_HD = portfolio.W_quantile_HD(N_samples=RELSAMPLES, q=0.10)

    Q_Wi_mat90 = portfolio.W_quantile_pop(N_samples=RELSAMPLES, q=0.90)
    Q_Wi_mat75 = portfolio.W_quantile_pop(N_samples=RELSAMPLES, q=0.75)
    Q_Wi_mat50 = portfolio.W_quantile_pop(N_samples=RELSAMPLES, q=0.50)
    Q_Wi_mat25 = portfolio.W_quantile_pop(N_samples=RELSAMPLES, q=0.25)
    Q_Wi_mat10 = portfolio.W_quantile_pop(N_samples=RELSAMPLES, q=0.10)

    # -----------------------------------------------------------------
    # Training risk functional configuration (uses LOSS_Q, TRADEOFF)
    # -----------------------------------------------------------------
    vari_train_functionals = {
        f"NCQ{100*LOSS_Q:.0f}": {
            "fn": portfolio.var_fnctnl.variational_CQ,
            "args": {"alpha": LOSS_Q},
        },
        f"NEQ{100*LOSS_Q:.0f}": {
            "fn": portfolio.var_fnctnl.variational_EQ,
            "args": {"alpha": LOSS_Q},
        },
        f"trade_off_mean_NCQ{100*LOSS_Q:.0f}_{TRADEOFF:.1e}": {
            "fn": tailored.vari_trade_off_cost,
            "args": {
                "cost": TRADEOFF,
                "gain_fnal": portfolio.var_fnctnl.expectation,
                "risk_fnal": portfolio.var_fnctnl.variational_CQ,
                "gain_fnal_args": {},
                "risk_fnal_args": {
                        "alpha": LOSS_Q
                    },
            },
        },
        f"trade_off_mean_NEQ{100*LOSS_Q:.0f}_{TRADEOFF:.1e}": {
            "fn": tailored.vari_trade_off_cost,
            "args": {
                "cost": TRADEOFF,
                "gain_fnal": portfolio.var_fnctnl.expectation,
                "risk_fnal": portfolio.var_fnctnl.variational_EQ,
                "gain_fnal_args": {},
                "risk_fnal_args": {                
                            "alpha": LOSS_Q,
                        },
            },
        },
    }

    esti_train_functionals = {
        "expectation": {
            "fn": portfolio.var_fnctnl.expectation,
            "args": {},
        },
        f"Q{100*(1-LOSS_Q):.0f}": {
            "fn": portfolio.var_fnctnl.quantile_pop,
            "args": {"q": LOSS_Q},
        },
        f"NAT{100*LOSS_Q:.0f}": {
            "fn": portfolio.var_fnctnl.avg_tail,
            "args": {"alpha": LOSS_Q},
        },
        f"NCQ{100*LOSS_Q:.0f}": {
            "fn": portfolio.var_fnctnl.cvar_pop,
            "args": {"alpha": LOSS_Q},
        },
        f"NEQ{100*LOSS_Q:.0f}": {
            "fn": portfolio.var_fnctnl.evar_z,
            "args": {
                "alpha": LOSS_Q,
                "maxiter": 45,
            },
        },
        f"trade_off_mean_semidev_{TRADEOFF:.1e}": {
            "fn": tailored.vari_trade_off_cost,
            "args": {
                "cost": TRADEOFF,
                "gain_fnal": portfolio.var_fnctnl.expectation,
                "risk_fnal": portfolio.var_fnctnl.u_semidev,
                "gain_fnal_args": {},
                "risk_fnal_args": {},
            },
        },
        f"trade_off_mean_stddev_{TRADEOFF:.1e}": {
            "fn": tailored.vari_trade_off_cost,
            "args": {
                "cost": TRADEOFF,
                "gain_fnal": portfolio.var_fnctnl.expectation,
                "risk_fnal": portfolio.var_fnctnl.stddev,
                "gain_fnal_args": {},
                "risk_fnal_args": {},
            },
        },
        f"trade_off_mean_semivar_{TRADEOFF:.1e}": {
            "fn": tailored.vari_trade_off_cost,
            "args": {
                "cost": TRADEOFF,
                "gain_fnal": portfolio.var_fnctnl.expectation,
                "risk_fnal": portfolio.var_fnctnl.u_semivar,
                "gain_fnal_args": {},
                "risk_fnal_args": {},
            },
        },
        f"trade_off_mean_var_{TRADEOFF:.1e}": {
            "fn": tailored.vari_trade_off_cost,
            "args": {
                "cost": TRADEOFF,
                "gain_fnal": portfolio.var_fnctnl.expectation,
                "risk_fnal": portfolio.var_fnctnl.variance,
                "gain_fnal_args": {},
                "risk_fnal_args": {},
            },
        },
        f"trade_off_mean_Q{100*(1-LOSS_Q):.0f}_{TRADEOFF:.1e}": {
            "fn": tailored.vari_trade_off_cost,
            "args": {
                "cost": TRADEOFF,
                "gain_fnal": portfolio.var_fnctnl.expectation,
                "risk_fnal": portfolio.var_fnctnl.quantile_pop,
                "gain_fnal_args": {},
                "risk_fnal_args": {
                        "q": LOSS_Q
                    },
            },
        },
        f"trade_off_mean_NCQ{100*LOSS_Q:.0f}_{TRADEOFF:.1e}": {
            "fn": tailored.vari_trade_off_cost,
            "args": {
                "cost": TRADEOFF,
                "gain_fnal": portfolio.var_fnctnl.expectation,
                "risk_fnal": portfolio.var_fnctnl.cvar_pop,
                "gain_fnal_args": {},
                "risk_fnal_args": {
                        "alpha": LOSS_Q
                    },
            },
        },
        f"trade_off_mean_NEQ{100*LOSS_Q:.0f}_{TRADEOFF:.1e}": {
            "fn": tailored.vari_trade_off_cost,
            "args": {
                "cost": TRADEOFF,
                "gain_fnal": portfolio.var_fnctnl.expectation,
                "risk_fnal": portfolio.var_fnctnl.evar_z,
                "gain_fnal_args": {},
                "risk_fnal_args": {                
                            "alpha": LOSS_Q,
                            "maxiter": 45,
                        },
            },
        },
    }

    if VARIATIONAL:
        trn_risk_fnal = vari_train_functionals.get(TRN_FNAL)
    else:
        trn_risk_fnal = esti_train_functionals.get(TRN_FNAL)

    # -----------------------------------------------------------------
    # Loss definition (uses TOPN, RANKSAMPLES)
    # -----------------------------------------------------------------

    @aux_wrap.per_query_loss_rng
    @aux_wrap.per_sample_loss
    def v_exposure_loss(scores, labels, mask, rng_key):
        return ir_methods.exposure_dcg_loss_fn(
            scores,
            labels,
            mask,
            rng_key,
            n_samples=RANKSAMPLES,
            top_K=TOPN,
        )

    @aux_wrap.per_query_loss_rng
    def v_exposure_loss_entropy_tradeoff(scores, labels, mask, rng_key):
        @aux_wrap.per_sample_loss
        def exposure_loss_sample(scores, labels, mask, rng_key):
            return ir_methods.exposure_dcg_loss_fn(
                        scores,
                        labels,
                        mask,
                        rng_key,
                        n_samples=RANKSAMPLES,
                        top_K=TOPN,
                    )
        policy_entropy = ir_methods.exposure_entropy(scores, labels, mask, rng_key, n_samples = RANKSAMPLES, top_K=TOPN)        
        policy_losses = exposure_loss_sample(scores, labels, mask, rng_key)
        return policy_losses-TRADEOFF*policy_entropy

    training_loss_dic = {
        "v_exposure_loss": v_exposure_loss,
        "v_exposure_loss_entropy_tradeoff": v_exposure_loss_entropy_tradeoff,
    }
    training_loss = training_loss_dic.get(args.trn_loss)

    # ----- K-means Label Distribution -----
    time_start = time.time()
    from ir_methods.context.LabelClusterDistribution import StreamingFeatureClusterer as LabelDistClusterer

    label_dist_clusterer = LabelDistClusterer(
        num_clusters=LABELDIST_NUM_CLUSTERS,
        num_label_classes=NUM_CLASSES,
        num_epochs=LABELDIST_CLUSTER_EPOCHS,
        model_path=LABELDIST_CLUSTER_MODEL_PATH,
    )

    print("Fitting label-distribution clusterer...")
    label_dist_clusterer.train(rng_labcluster_trn, trn_ds, force_train=LABELDIST_FIT_CLUSTER)
    label_dist_model = (label_dist_clusterer.centroids, label_dist_clusterer.label_distributions)

    time_end = time.time()
    elapsed_time = time_end - time_start
    elapsed_time_min, elapsed_time_sec = divmod(elapsed_time, 60)
    print(f"[Label-dist clusterer] Execution time: {int(elapsed_time_min)}m {elapsed_time_sec:.2f}s")

    # -----------------------------------------------------------------
    # Build samplers (train + eval) using the shared factory
    # -----------------------------------------------------------------
    sampler_ctx = SamplerContext(
        cluster_label_model=label_dist_model,
        n_samples=RELSAMPLES,
    )

    trn_sampler_cfg = make_sampler(TRN_SAMPLER, sampler_ctx)

    EVAL_SAMPLERS = [
        TRN_SAMPLER,
    ]

    samplers_dic: Dict[str, Dict[str, Any]] = {
        kind: {
            **make_sampler(kind, sampler_ctx),
            "envs": ["TRN", "VAL", "TST"],
        }
        for kind in EVAL_SAMPLERS
    }

    # -----------------------------------------------------------------
    # Metrics / risks configuration (uses TOPN, LOSS_Q)
    # -----------------------------------------------------------------
    metrics_dic = {
        "gumbel_dcg": {
            "fn": ir_methods.exposure_dcg_metric_fn,
            "args": {
                "top_K": TOPN,
                "n_samples": RANKSAMPLES,
            },
            "eval_risk": True,
        },
        "dcg": {
            "fn": ir_methods.dcg_metric,
            "args": {
                "topn": TOPN,
            },
            "eval_risk": False,
        },
        "ndcg": {
            "fn": ir_methods.ndcg_metric,
            "args": {
                "topn": TOPN,
            },
            "eval_risk": False,
        },
        "gini_exposure": {
            "fn": ir_methods.tsallis_exposure,
            "args": {
                "n_samples": RANKSAMPLES,
                "top_K": TOPN,
                "q_def": 2,
            },
            "eval_risk": False,
        },
        "entropy_exposure": {
            "fn": ir_methods.entropy_exposure,
            "args": {
                "n_samples": RANKSAMPLES,
                "top_K": TOPN,
            },
            "eval_risk": False,
        },
        "N_docs_exposure_95": {
            "fn": ir_methods.N_docs_exposure,
            "args": {
                "n_samples": RANKSAMPLES,
                "top_K": TOPN,
                "p_ineq": 0.95,
            },
            "eval_risk": False,
        },
    }

    rng_key = aux.assign_rng_key(rng_key, samplers_dic)
    rng_key = aux.assign_rng_key(rng_key, metrics_dic)

    rng_key, rng_boot_key = jrng.split(rng_key)

    risks_dic = {
        "expectation": {
            "fn": portfolio.expectation,
            "args": {},
        },
        f"Q10": {
            "fn": portfolio.Q_weight,
            "args": {
                "wi": Q_Wi_mat10_HD
            },
        },
        f"Q25": {
            "fn": portfolio.Q_weight,
            "args": {
                "wi": Q_Wi_mat25_HD
            },
        },
        f"Q50": {
            "fn": portfolio.Q_weight,
            "args": {
                "wi": Q_Wi_mat50_HD
            },
        },
        f"NCQ90": {
            "fn": aux.polarity_transform(portfolio.CVaR_Q),
            "args": {
                "alpha": 0.90,
                "EVaR_bound": True,
                "maxiter": 45,
            },
        },
        f"NCQ75": {
            "fn": aux.polarity_transform(portfolio.CVaR_Q),
            "args": {
                "alpha": 0.75,
                "EVaR_bound": True,
                "maxiter": 45,
            },
        },
        f"NCQ50": {
            "fn": aux.polarity_transform(portfolio.CVaR_Q),
            "args": {
                "alpha": 0.5,
                "EVaR_bound": True,
                "maxiter": 45,
            },
        },
        f"NEQ90": {
            "fn": aux.polarity_transform(portfolio.Entropic_Q),
            "args": {
                "alpha": 0.9,
                "maxiter": 45,
            },
        },
        f"NEQ75": {
            "fn": aux.polarity_transform(portfolio.Entropic_Q),
            "args": {
                "alpha": 0.75,
                "maxiter": 45,
            },
        },
        f"NEQ50": {
            "fn": aux.polarity_transform(portfolio.Entropic_Q),
            "args": {
                "alpha": 0.50,
                "maxiter": 45,
            },
        },
        "semidev": {
            "fn": portfolio.sto_fnctnl.lower_semideviation,
            "args": {},
        },
        "stddev": {
            "fn": portfolio.sto_fnctnl.standard_deviation,
            "args": {},
        },
        # f"trade_off_mean_semidev_{TRADEOFF:.1e}": {
        #     "fn": tailored.trade_off_cost,
        #     "args": {
        #         "cost": 0.0-TRADEOFF,
        #         "gain_fnal": portfolio.sto_fnctnl.expectation,
        #         "risk_fnal": portfolio.sto_fnctnl.standard_deviation,
        #         "gain_fnal_args": {},
        #         "risk_fnal_args": {},
        #     },
        # },
    }

    # -----------------------------------------------------------------
    # Training (uses args.max_epochs, args.learning_rate_*)
    # -----------------------------------------------------------------
    time_start = time.time()

    metrics_dic = training.optimize_vrisk(
        rng_key,
        (trn_ds, val_ds, tst_ds),
        num_qids,
        trn_risk_fnal,
        training_loss,
        trn_sampler_cfg,
        samplers_dic,
        metrics_dic,
        risks_dic,
        variational=VARIATIONAL,
        max_epochs=args.max_epochs,
        learning_rate_loss=args.learning_rate_loss,
        learning_rate_aux=args.learning_rate_aux,
        evaluate_every=args.evaluate_every,
        evaluate_test=True,
        verbose=0,
        cluster_centroids = label_dist_clusterer.centroids,
        n_rank_samples=RANKSAMPLES,
        top_K=TOPN,
    )

    time_end = time.time()
    elapsed_time = time_end - time_start
    elapsed_time_min, elapsed_time_sec = divmod(elapsed_time, 60)
    print(f"[Training] Execution time: {int(elapsed_time_min)}m {elapsed_time_sec:.2f}s")

    # -----------------------------------------------------------------
    # Visualization and export (uses PLOTS_PATH, FILES_PATH)
    # -----------------------------------------------------------------


    for eval_sampler in samplers_dic:
        title_words = [
            f"UTRAIN={TRN_SAMPLER}",
            f"FTRAIN={TRN_FNAL}",
            f"UEVAL={eval_sampler}",
            f"DESC={TRN_FNAL_DESC}",
        ]
        title = "__".join(title_words)

        visualize.plot_metrics(metrics_dic, "VAL", eval_sampler, title,
                               plot_folder=PLOTS_PATH, plot_label=False)
        visualize.plot_metrics(metrics_dic, "TRN", eval_sampler, title,
                               plot_folder=PLOTS_PATH, plot_label=False)

        visualize.export_metrics(metrics_dic, "VAL", eval_sampler, title, out_folder=FILES_PATH)
        visualize.export_metrics(metrics_dic, "TRN", eval_sampler, title, out_folder=FILES_PATH)

    visualize.save_dictionary(metrics_dic, FILES_PATH, TRN_SAMPLER, TRN_FNAL, TRN_FNAL_DESC)

if __name__ == "__main__":
    main()