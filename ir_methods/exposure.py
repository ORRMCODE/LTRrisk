import jax 
import jax.numpy as jnp
import jax.lax as jlax

#### Call it exposure primitive #####
@jax.jit
def exposure_prim(scores, rank_exposure, rankings, mask):
    
    rank_exposure=jlax.stop_gradient(rank_exposure)
    rankings = jlax.stop_gradient(rankings)

    # rankings [N, D] matrix of sampled rankings from PL ranking model
    K = rank_exposure.size
    n_samples = rankings.shape[0]
    
    scores = jnp.where(mask, scores, -jnp.inf)
#    scores = scores - jlax.stop_gradient(scores.max())

    n_docs = jnp.sum(mask)

    # [N, D] log
    ranked_scores = scores[rankings]
    ranked_mask = mask[rankings]

    # [N, K] log denominator of placement prob. per position per sample
    log_denom = jlax.cumlogsumexp(ranked_scores, axis=1, reverse=True)[:,:K]

    # rax implementation cannot handle masks properly
    # the following trick avoids division by zero
    log_denom = jnp.where(ranked_mask[:,:K], log_denom, 0)
    
    # [N, K-1] log prob. of sampled ranking up to pos. k (not including)
    log_prefix = ranked_scores[:,:K-1] - log_denom[:,:K-1]
    log_prefix = jnp.cumsum(jnp.pad(log_prefix, ((0,0),(1,0))), axis=1)
    log_prefix = jnp.where(ranked_mask[:,:K], log_prefix, 0)
    
    # [N, K, D] placement prob. of every document per position & sample
    logprob_all = scores[None,None,:] - log_denom[:,:,None]

    placement_mask = jnp.zeros((logprob_all.shape[0], logprob_all.shape[1]+1, logprob_all.shape[2]), dtype=bool)
    placement_mask = placement_mask.at[jnp.arange(n_samples)[:,None], 
                                    jnp.arange(1,K)[None,:], 
                                    rankings[:,:(K-1)]].set(True)

    placement_mask = placement_mask.at[:, jnp.minimum(n_docs, K),:].set(True)[:,:-1,:]
    placement_mask = jnp.cumsum(placement_mask, axis=1).astype(bool)
    masked_logprob = jnp.where(placement_mask, -jnp.inf, logprob_all)
    
    prob_all = jnp.exp(masked_logprob)

    # [N, K, D] loss value used for computing policy gradient
    loss_K_D = prob_all + jlax.stop_gradient(prob_all)*log_prefix[:,:,None]
    exposure_prob_prod = prob_all * rank_exposure[None,:,None]

    # [N, D] exposure of document per sample
    sample_exposure = jnp.sum(exposure_prob_prod, axis=1)

    # [D] expected exposure of document estimated from samples
    mean_exposure = jnp.mean(sample_exposure, axis=0)

    # [N, K, D] leave-one-out baseline corrections
    baseline = mean_exposure[None,:] - sample_exposure/n_samples
    baseline *= n_samples / (n_samples-1)
    baseline = jlax.stop_gradient(baseline)

    # [N, K, D] baseline-corrected rewards for every doc., pos., sample
    rewards = rank_exposure[None,:,None] - baseline[:,None,:]
    loss_inside_K = jnp.sum(rewards * loss_K_D, axis=1)

    no_prob_all = 1-prob_all[:,K-1,:]
    loss_outside_K = no_prob_all + jlax.stop_gradient(no_prob_all)*log_prefix[:,K-1,None]
    loss_outside_K = jnp.where(placement_mask[:,K-1], 0, loss_outside_K)
    
    # loss_outside_K = jnp.where(mask[None,:], 0, loss_outside_K)
    loss_outside_K = jnp.where(mask[None,:], loss_outside_K, 0)
    loss_outside_K *= -baseline

    loss = jnp.mean(loss_inside_K + loss_outside_K, axis=0)
    grad_loss = jlax.stop_gradient(mean_exposure) + loss - jlax.stop_gradient(loss)
    return grad_loss
# trick: value is expected exposure but gradient comes from loss

@jax.jit
def exposure(scores, rank_exposure, rankings, mask):

    # rankings [N, D] matrix of sampled rankings from PL ranking model
    K = rank_exposure.size
    n_samples = rankings.shape[0]
    scores = jnp.where(mask, scores, -jnp.inf)

    n_docs = jnp.sum(mask)

    # [N, D] log
    ranked_scores = scores[rankings]
    ranked_mask = mask[rankings]

    # [N, K] log denominator of placement prob. per position per sample
    log_denom = jlax.cumlogsumexp(ranked_scores, axis=1, reverse=True)[:,:K]

    # rax implementation cannot handle masks properly
    # the following trick avoids division by zero
    log_denom = jnp.where(ranked_mask[:,:K], log_denom, 0)
    
    # [N, K-1] log prob. of sampled ranking up to pos. k (not including)
    log_prefix = ranked_scores[:,:K-1] - log_denom[:,:K-1]
    log_prefix = jnp.cumsum(jnp.pad(log_prefix, ((0,0),(1,0))), axis=1)
    log_prefix = jnp.where(ranked_mask[:,:K], log_prefix, 0)
    
    # [N, K, D] placement prob. of every document per position & sample
    logprob_all = scores[None,None,:] - log_denom[:,:,None]

    placement_mask = jnp.zeros((logprob_all.shape[0], logprob_all.shape[1]+1, logprob_all.shape[2]), dtype=bool)
    placement_mask = placement_mask.at[
                                        jnp.arange(n_samples)[ :, None], 
                                        jnp.arange(1,K)[None, :], 
                                        rankings[:, :(K-1)]
                                    ].set(True)

    placement_mask = placement_mask.at[:, jnp.minimum(n_docs, K), :].set(True)[:,:-1,:]
    placement_mask = jnp.cumsum(placement_mask, axis=1).astype(bool)
    masked_logprob = jnp.where(placement_mask, -jnp.inf, logprob_all)

    prob_all = jnp.exp(masked_logprob)
    # [N, K, D] loss value used for computing policy gradient

    loss_K_D = prob_all + jlax.stop_gradient(prob_all)*log_prefix[:,:,None]
    exposure_prob_prod = prob_all * rank_exposure[None,:,None]

    # [N, D] exposure of document per sample
    sample_exposure = jnp.sum(exposure_prob_prod, axis=1)
    # [D] expected exposure of document estimated from samples
    mean_exposure = jnp.mean(sample_exposure, axis=0)

    return mean_exposure