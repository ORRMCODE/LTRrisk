import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.lax as jlax

from auxiliary import trn_control_fn, prd_control_fn
from functools import partial


class LossTrainer:

    def __init__(self, loss_fn, verbose=0):

        self.trn_control_fn = trn_control_fn
        self.loss_fn = loss_fn
        self.verbose = verbose

        # Build and jit compile once
        self._grad_fn = jax.jit(
                            jax.value_and_grad(self._calculate_loss, 
                                               argnums=1, 
                                               has_aux=True
                            )
                        )
    
    @partial(jax.jit, static_argnums=0)
    def _calculate_loss(self, state, params, batch_data, rng_loss, rng_dropout):
        feat, labels, mask, _ = batch_data

        scores, new_model_state = self.trn_control_fn(
            state, params, feat, rng_dropout
        )

        loss_value = jnp.mean(
            self.loss_fn(
                scores, labels, mask, rng_loss
            )
        )

        return loss_value, new_model_state

    def train_step(self, state, batch_data, rng_key):

        rng_key, rng_loss, rng_dropout = jrng.split(rng_key, 3)

        (loss, new_model_state), grads = self._grad_fn(
                                                    state,
                                                    state.params,
                                                    batch_data,
                                                    rng_loss,
                                                    rng_dropout
                                            )

        if getattr(state, "batch_stats", None) is not None:

            state = state.apply_gradients(
                grads=grads,
                batch_stats=new_model_state["batch_stats"],
            )
        else:
            state = state.apply_gradients(grads=grads)

        return state, loss


#########################################################################################################################
class RiskTrainer:

    def __init__(
                self, 
                sampler_fn,
                sampler_params,
                n_samples,
                broker_fn,
                verbose=0
            ):

        self.trn_control_fn = trn_control_fn

        self.broker_fn = broker_fn

        self.sampler_fn = sampler_fn
        self.sampler_params = sampler_params

        self.n_samples = n_samples

        self.verbose = verbose


        # Build and jit compile once
        self._grad_fn = jax.value_and_grad(self._calculate_loss, 
                               argnums=1, 
                               has_aux=True
                            )

    def _calculate_loss(self, state, params, batch_data, rng_key, rng_dropout):
        feats, labels, mask, _ = batch_data

        scores, new_model_state = self.trn_control_fn(
            state, params, feats, rng_dropout
        )

        rng_broker, rng_sampler = jrng.split(rng_key)

        label_q_d_s = jlax.stop_gradient(
                            self.sampler_fn(feats, labels, mask, rng_sampler, self.n_samples, **self.sampler_params)
                        )

        loss_q = self.broker_fn(scores, label_q_d_s, mask, rng_broker)
        loss_value = jnp.mean(loss_q)

        return loss_value, new_model_state

    def train_step(self, state, batch_data, rng_key):

        rng_key, rng_loss, rng_dropout = jrng.split(rng_key, 3)

        (loss, new_model_state), grads = self._grad_fn(
                                                    state,
                                                    state.params,
                                                    batch_data,
                                                    rng_loss,
                                                    rng_dropout
                                            )

        if getattr(state, "batch_stats", None) is not None:
            
            state = state.apply_gradients(
                grads=grads,
                batch_stats=new_model_state["batch_stats"],
            )
        else:
            state = state.apply_gradients(grads=grads)

        return state, loss

############################################################################################################
############################################################################################################
class VariationalTrainer:

    def __init__(
                self, 
                risk_fnal,
                risk_fnal_params,
                loss_fn,
                sampler_fn,
                sampler_params,
                # n_samples,
                rng_key = jrng.key(42),
                verbose=0
                ):

        self.trn_control_fn = trn_control_fn
        self.trn_auxiliary = trn_control_fn

        self.loss_fn = loss_fn
        self.sampler_fn = sampler_fn
        self.sampler_params = sampler_params

        # self.n_samples = n_samples

        self.risk_fnal = risk_fnal
        self.risk_fnal_params = risk_fnal_params

        self.verbose = verbose
        self.first_epoch = True
        # Build and jit compile once
        self._grad_loss_fn = jax.value_and_grad(self._calculate_loss, 
                               argnums=1, 
                               has_aux=True
                            )
        self._grad_aux_fn = jax.value_and_grad(self._calculate_aux, 
                        argnums=1, 
                        has_aux=True
                    )

    def _calculate_loss(self, state, params, t_vnal, batch_data, rng_key):
        feats, labels, mask, q_ids, label_q_d_s = batch_data

        rng_dropout, rng_loss = jrng.split(rng_key)

        scores, new_model_state = self.trn_control_fn(
            state, params, feats, rng_dropout
        )

        loss_q_s = self.loss_fn(scores, label_q_d_s, mask, rng_loss)

        bad_rows = ~jnp.all(jnp.isfinite(loss_q_s), axis=1)

        loss_q   = self.risk_fnal(loss_q_s, t_vnal, **self.risk_fnal_params)
        loss_value = jnp.mean(loss_q)
        return loss_value, (new_model_state, loss_q_s)

    def _calculate_aux(self, state, params, loss_samples, batch_data, rng_key):
        feats, labels, mask, q_ids, label_q_d_s = batch_data
        t_aux, new_model_state = self.trn_auxiliary(
            state, params, q_ids, rng_key
        )
        loss_q = self.risk_fnal(loss_samples, t_aux,  **self.risk_fnal_params)
        loss_value = jnp.mean(loss_q)
        return loss_value, (new_model_state, t_aux)


    def sampling_step(self, batch_data, rng_key):
        feats, labels, mask, q_ids = batch_data
        label_qds = jlax.stop_gradient(
                        self.sampler_fn(feats, labels, mask, rng_key, **self.sampler_params)
                    )
        return feats, labels, mask, q_ids, label_qds

    def trainAux_step(self, state, loss_samples, batch_data, rng_key):

        (loss, model_output), grads = self._grad_aux_fn(
                                                    state,
                                                    state.params,
                                                    loss_samples,
                                                    batch_data,
                                                    rng_key
                                            )
        new_model_state, t_vnal = model_output

        if getattr(state, "batch_stats", None) is not None:

            state = state.apply_gradients(
                grads=grads,
                batch_stats=new_model_state["batch_stats"],
            )
        else:
            state = state.apply_gradients(grads=grads)

        return state, loss, t_vnal

    def trainLoss_step(self, state, t_vnal, batch_data, rng_key):

        (loss, model_output), grads = self._grad_loss_fn(
                                                    state,
                                                    state.params,
                                                    t_vnal,
                                                    batch_data,
                                                    rng_key
                                            )
        new_model_state, loss_q_s = model_output
        if getattr(state, "batch_stats", None) is not None:

            state = state.apply_gradients(
                grads=grads,
                batch_stats=new_model_state["batch_stats"],
            )
        else:
            state = state.apply_gradients(grads=grads)

        return state, loss, loss_q_s