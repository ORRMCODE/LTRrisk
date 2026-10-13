import jax
import jax.random as jrng
import jax.numpy as jnp
import jax.nn as jnn

import optax
from flax import linen as nn
from flax.training import train_state
from flax import struct as flax_struct

from evaluation import EvaluationManager

from training.NN_control import NNPolicy, NNAuxVariational

from training.trn_step import RiskTrainer, VariationalTrainer
from training.exposure_cluster import v_exposure_per_cluster
from training.exposure_cluster import dataset_exposure_per_cluster
import gc
import time
from jax import tree_util

###################### Debug ###################################
def pytree_has_nan(tree):
#     leaves = tree_util.tree_leaves(tree)
#     return any(bool(jnp.isnan(leaf).any()) for leaf in leaves)
    leaves = tree_util.tree_leaves(tree)
    return any(bool((~jnp.isfinite(leaf)).any()) for leaf in leaves)
###################### Debug ###################################

class TrainState(train_state.TrainState):
    batch_stats: dict = flax_struct.field(pytree_node=True)

def trainer_uryasev(
                rng,
                dataset,
                N_Q_ID,
                trainer,  # pre-constructed VariationalTrainer
                dic_samplers,
                dic_metrics,
                dic_risks,
                variational=True,
                max_epochs=15,
                learning_rate_loss=0.01,
                learning_rate_aux=0.1,
                evaluate_every=10,
                evaluate_test=True,
                verbose=0,
                cluster_centroids=None,
                n_rank_samples = None,
                top_K = None
                ):

    train_ds, vali_ds, test_ds = dataset
    rng, init_rng, init_aux_rng, trn_rng, eval_rng = jrng.split(rng, 5)

    init_feats, init_labels, init_mask, init_Qids = next(iter(train_ds))

    optimizerLoss = optax.adamaxw(learning_rate=learning_rate_loss)
    optimizerAux = optax.adamaxw(learning_rate=learning_rate_aux)
    modelLoss = NNPolicy()
    modelAux = NNAuxVariational(N_Q_ID+1)

    modelLoss_params = modelLoss.init(init_rng, init_feats)
    if "batch_stats" in modelLoss_params:
        modelLoss_state = TrainState.create(
            apply_fn=modelLoss.apply,
            params=modelLoss_params["params"],
            batch_stats=modelLoss_params["batch_stats"],
            tx=optimizerLoss
        )
    else:
        modelLoss_state = train_state.TrainState.create(
            apply_fn=modelLoss.apply,
            params=modelLoss_params["params"],
            tx=optimizerLoss
        )

    modelAux_params = modelAux.init(init_aux_rng, init_Qids)
    if "batch_stats" in modelAux_params:
        modelAux_state = TrainState.create(
            apply_fn=modelAux.apply,
            params=modelAux_params["params"],
            batch_stats=modelAux_params["batch_stats"],
            tx=optimizerAux
        )
    else:
        modelAux_state = train_state.TrainState.create(
            apply_fn=modelAux.apply,
            params=modelAux_params["params"],
            tx=optimizerAux
        ) 

    eval_mng = EvaluationManager(eval_rng)
    eval_mng.set_model_state(modelLoss_state)

    for name, value in dic_samplers.items():
        eval_mng.register_sampler(name, value["fn"], value["envs"], **value["args"])
    for name, value in dic_metrics.items():
        eval_mng.register_metric(name, value["fn"], value["eval_risk"], **value["args"])
    for name, value in dic_risks.items():
        eval_mng.register_risk(name, value["fn"], **value["args"])

    # 1) compile per-metric/per-risk functions
    eval_mng.precompile_all()

    # 2) build per-distribution jitted batch functions (NEW, required)
    eval_mng.build_dist_batch_fns()

    eval_mng.log_loss(dataset_name="TRN", loss_value=0)
    eval_mng.evaluate_dataset(dataset_name="TRN", dataset=train_ds, epoch=0)
    eval_mng.evaluate_dataset(dataset_name="VAL", dataset=vali_ds, epoch=0)
    eval_mng.evaluate_dataset(dataset_name="TST", dataset=test_ds, epoch=0)

    if cluster_centroids is not None:
        _expo_0, _n_docs_0 = dataset_exposure_per_cluster(vali_ds, trainer, modelLoss_state, rng, n_rank_samples, top_K, cluster_centroids)
        eval_mng.log_info(dataset_name="TRN", info_value=(_expo_0, _n_docs_0), name="cluster_exposure")
        val_c_exposure, val_c_docs = dataset_exposure_per_cluster(vali_ds, trainer, modelLoss_state, rng, n_rank_samples, top_K, cluster_centroids)
        eval_mng.log_info(dataset_name="VAL", info_value=(val_c_exposure, val_c_docs), name="cluster_exposure")
        if evaluate_test:
            tst_c_exposure, tst_c_docs = dataset_exposure_per_cluster(test_ds, trainer, modelLoss_state, rng, n_rank_samples, top_K, cluster_centroids)
            eval_mng.log_info(dataset_name="TST", info_value=(tst_c_exposure, tst_c_docs), name="cluster_exposure")

    for epoch in range(1, max_epochs):
        tot_loss = 0
        n_batch = 0
        
        first_epoch = True

        cluster_exposure = jnp.zeros(cluster_centroids.shape[0])
        cluster_n_docs = jnp.zeros(cluster_centroids.shape[0])        

        time_start = time.time()
        for batch_data in train_ds:
            batch_feats, batch_labels, batch_mask, batch_qids = batch_data
            trn_rng, rng_sampler, rng_loss, rng_aux = jrng.split(trn_rng, 4)

            if variational:        
                t_aux, _ = trainer.trn_auxiliary(
                                            modelAux_state, 
                                            modelAux_state.params, 
                                            batch_qids, 
                                            rng_aux)
            else:
                t_aux = jnp.zeros_like(batch_qids)

            batch_data_samples = trainer.sampling_step(batch_data, rng_sampler)

            if variational:
                modelLoss_state, _, loss_q_s  = trainer.trainLoss_step(modelLoss_state, t_aux, batch_data_samples, rng_loss)
                modelAux_state, batch_loss, t_aux = trainer.trainAux_step(modelAux_state, loss_q_s, batch_data_samples, rng_aux)
            else:
                modelLoss_state, batch_loss, loss_q_s  = trainer.trainLoss_step(modelLoss_state, t_aux, batch_data_samples, rng_loss)


            tot_loss += batch_loss*batch_labels.shape[0]/N_Q_ID

            if cluster_centroids is not None:
                if (epoch % evaluate_every) == 0:
                    scores_q_d, _ = trainer.trn_control_fn(
                                                modelLoss_state, 
                                                modelLoss_state.params, 
                                                batch_feats, 
                                                rng_loss
                                        )
                    cluster_exposure_q, cluster_n_docs_q = v_exposure_per_cluster(
                                                                scores_q_d,
                                                                batch_feats,
                                                                batch_labels, 
                                                                batch_mask,
                                                                rng_loss,
                                                                n_rank_samples = n_rank_samples,
                                                                top_K = int(top_K),
                                                                centroids = cluster_centroids
                                                        )
                    cluster_exposure += jnp.sum(cluster_exposure_q, axis=0)
                    cluster_n_docs += jnp.sum(cluster_n_docs_q, axis=0)


        time_end = time.time()
        elapsed_time = time_end - time_start
        elapsed_time_min, elapsed_time_sec = divmod(elapsed_time, 60)
        print(f"epoch {epoch}: ", tot_loss)
        print(f"Execution time: {int(elapsed_time_min)}m {elapsed_time_sec:.2f}s")
        print('Clusters exposure: ', cluster_exposure, cluster_n_docs, cluster_exposure/cluster_n_docs)
        if (epoch % evaluate_every) == 0:

            eval_mng.set_model_state(modelLoss_state)
            eval_mng.evaluate_dataset(dataset=train_ds, dataset_name="TRN", epoch=epoch)
            eval_mng.evaluate_dataset(dataset=vali_ds, dataset_name="VAL", epoch=epoch)

            if evaluate_test:
                eval_mng.evaluate_dataset(dataset=test_ds, dataset_name="TST", epoch=epoch)

            eval_mng.log_loss(dataset_name="TRN", loss_value=-tot_loss)
            if cluster_centroids is not None:
                eval_mng.log_info(dataset_name="TRN", info_value=(cluster_exposure, cluster_n_docs), name="cluster_exposure")
                val_c_exposure, val_c_docs = dataset_exposure_per_cluster(vali_ds, trainer, modelLoss_state, rng_loss, n_rank_samples, top_K, cluster_centroids)
                eval_mng.log_info(dataset_name="VAL", info_value=(val_c_exposure, val_c_docs), name="cluster_exposure")
                if evaluate_test:
                    tst_c_exposure, tst_c_docs = dataset_exposure_per_cluster(test_ds, trainer, modelLoss_state, rng_loss, n_rank_samples, top_K, cluster_centroids)
                    eval_mng.log_info(dataset_name="TST", info_value=(tst_c_exposure, tst_c_docs), name="cluster_exposure")
            # gc.collect()

    return eval_mng.get_metrics()


def optimize_vrisk(
                    rng, 
                    dataset, N_Q_ID,  
                    risk_fnal, 
                    loss_fn,
                    sampler_fn, 
                    dic_samplers, dic_metrics, dic_risks, **kwargs
                ):

    trainer = VariationalTrainer( 
                risk_fnal["fn"],
                risk_fnal["args"],
                loss_fn,
                sampler_fn["fn"],
                sampler_fn["args"],
                rng
    )
    return trainer_uryasev(rng, 
                           dataset, N_Q_ID,
                           trainer, 
                           dic_samplers, dic_metrics, dic_risks, **kwargs)