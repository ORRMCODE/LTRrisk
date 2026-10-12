import jax
import jax.scipy as jsp
import jax.numpy as jnp
import jax.random as jrng
import jax.nn as jnn
import jax.lax as lax
import rax



import ir_methods.auxiliary as aux_ir
import ir_methods.exposure as aux_exp

from functools import partial

from auxiliary import jit_bincount
from ir_methods.control_sampling import gumbel_rankings
from ir_methods.exposure import exposure_prim


@partial(jax.jit, static_argnames=["topn"])
def dcg_metric(scores, labels, mask, *, rng_key, topn):
    id_fn = lambda x: x
    return rax.dcg_metric(scores, labels, where = mask, topn = topn, key = rng_key, gain_fn=id_fn)

@partial(jax.jit, static_argnames=["topn"])
def ndcg_metric(scores, labels, mask, *,  rng_key, topn):
    id_fn = lambda x: x
    return rax.ndcg_metric(scores, labels, where=mask, topn=topn, key=rng_key, gain_fn = id_fn)

@jax.jit
def KLfair_metric(scores, labels, mask,*, rng_key):
    
    norm_scores=jnn.softmax(scores, where = mask)
    norm_labels=labels/jnp.sum(labels, where = mask)
    non_zero_labels = jnp.where(norm_labels>1e-16, mask, False)
    return -jnp.sum(norm_labels*jnp.log(norm_scores/norm_labels), where=non_zero_labels)


@partial(jax.jit, static_argnames=["n_samples"])
def topK_alpha_diverse_doc(scores, labels, mask, *, rng_key, n_samples, alpha):

    rng_, rng_ranks = jrng.split(rng_key, 2)
    n_valid_docs = jnp.sum(mask)
    ranks_mat = gumbel_rankings(scores, mask, rng_ranks, n_samples)
    n_docs = scores.shape[0]

    def global_unique_prefix_count(A):
        
        num_ranks, num_docs = A.shape        

        # Map everything to unique integers
        flat_A = A.ravel()
        sort_idx = jnp.argsort(flat_A)
        sorted_A = flat_A[sort_idx]
        
        # Identify transitions between different values
        is_different = jnp.concatenate([jnp.array([True]), sorted_A[1:] != sorted_A[:-1]])
        unique_ids = jnp.cumsum(is_different) - 1
        
        # Map the IDs back to the original matrix shape
        inv_sort_idx = jnp.argsort(sort_idx)
        A_indexed = unique_ids[inv_sort_idx].reshape(num_ranks, num_docs)
        
        # number of docs as the static limit for our mask size
        
        initial_seen = jnp.zeros(num_docs, dtype=bool)

        def body_fun(seen_mask, col_ids):
            
            # already_seen[i] is True if col_ids[i] has appeared in a previous column
            already_seen = seen_mask[col_ids]
            
            # Now handle duplicates within this specific column
            # Sort the column IDs to find transitions

            col_sort = jnp.argsort(col_ids)
            sorted_col = col_ids[col_sort]
            is_first_in_col_sorted = jnp.concatenate([jnp.array([True]), sorted_col[1:] != sorted_col[:-1]])
            
            # Map that "first in col" info back to original order
            inv_col_sort = jnp.argsort(col_sort)
            is_first_in_col = is_first_in_col_sorted[inv_col_sort]
            
            # A value counts as "new" if it's the first time in this col 
            # AND it hasn't appeared in any column before this one.
            is_truly_new = is_first_in_col & (~already_seen)
            new_count = jnp.sum(is_truly_new)
            
            # Update mask: mark all values in this column as seen
            new_mask = seen_mask.at[col_ids].set(True)
            
            return new_mask, new_count

        # Execute the Scan across the columns
        _, new_arrivals_per_col = lax.scan(body_fun, initial_seen, A_indexed.T)
        
        return jnp.cumsum(new_arrivals_per_col)
    
    ############################################################################
    submatrix_sizes = jnp.arange(1, n_docs + 1) * n_samples
    # min(submatrix_size, n_valid_docs)
    denominator = jnp.minimum(submatrix_sizes, n_valid_docs)
    # Vectorized division with safety for zero-docs
    topk_j = jnp.where(denominator > 0, global_unique_prefix_count(ranks_mat) / denominator, n_valid_docs)

    return jnp.argmin(jnp.abs(topk_j-alpha))+1
    #############################################################################


@partial(jax.jit, static_argnames=["n_samples", "topn"])
def jaccard_index(scores, labels, mask, *, rng_key, n_samples, topn):
    
    rng_, rng_ranks = jrng.split(rng_key, 2)
    n_valid_docs = jnp.sum(mask)
    ranks_mat = gumbel_rankings(scores, mask, rng_ranks, n_samples)
    n_docs = scores.shape[0]
    
    ranks_mat = jnp.where(mask[ranks_mat], ranks_mat, -2)

    def count_unique_docs(A):
        flat_A = A.ravel()
        sort_idx = jnp.argsort(flat_A)
        sorted_A = flat_A[sort_idx]
        is_different = jnp.concatenate([jnp.array([True]), sorted_A[1:] != sorted_A[:-1]])
        unique_ids = jnp.sum(is_different)
        return unique_ids

    def count_common_docs(A, padding_id = -1):
        
        """
        Counts how many unique document IDs are present in every row of A.
        A: (N, D) - N rows, D depth
        padding id is to an id to that tells the counter don't count this slot
        """
        N, D = A.shape
    
        # 1. Ensure elements are unique within each row
        # We sort each row and nullify duplicates
        A_sorted_rows = jnp.sort(A, axis=1)
        is_duplicate_in_row = jnp.concatenate([
            jnp.zeros((N, 1), dtype=bool),
            A_sorted_rows[:, 1:] == A_sorted_rows[:, :-1]
        ], axis=1)
        
        # Replace duplicates and existing padding with the sentinel
        sentinel_flag= (A_sorted_rows == padding_id)
        row_uniques = jnp.where(is_duplicate_in_row | sentinel_flag, 
                                padding_id, A_sorted_rows)
        
        # 2. Flatten and sort globally
        flat = row_uniques.ravel()
        flat_sorted = jnp.sort(flat)
        
        # 3. Identify blocks of size N
        # Since we have at most 1 occurrence per row, any ID appearing N times
        # in the flat array MUST be in the intersection.
        
        # Check 1: Is this the end of a block of N identical elements?
        # (Checking if the element N-1 positions ago is the same)
        is_n_occurences = (flat_sorted == jnp.roll(flat_sorted, N - 1))
        
        # Check 2: Is this the start of a new ID? (or the very last element)
        flag_new_id=(flat_sorted[:-1] != flat_sorted[1:])
        is_block_end = jnp.concatenate([ flag_new_id, jnp.array([True])])
        
        # Valid intersection members:
        # Must be the end of a block, have N occurrences, and not be padding.
        is_intersect = is_block_end & is_n_occurences & (flat_sorted != padding_id)
        
        # Size is the sum of these occurrences
        return jnp.sum(is_intersect)


    size_of_union = count_unique_docs(ranks_mat[:,:topn])
    size_of_intersection = count_common_docs(ranks_mat[:, :topn])

    return jnp.maximum(1,size_of_intersection-1)/jnp.maximum(size_of_union-1, 1)


@partial(jax.jit, static_argnames=["n_samples", "top_K", "p_ineq"])
def N_docs_exposure(scores, labels, mask, *, rng_key, n_samples, top_K, p_ineq=0.95):
    weights = aux_ir.dcg_weights(top_K)
    _, rng_ranks = jrng.split(rng_key, 2)
    # Number of valid documents (as float for division)
    N_valid_docs = jnp.sum(mask).astype(jnp.float32)
    # Sample rankings
    ranks_mat = gumbel_rankings(scores, mask, rng_ranks, n_samples)
    # Exposure per document
    exposure_coeffs = aux_exp.exposure(scores, weights, ranks_mat, mask)
    total_exposure = jnp.sum(exposure_coeffs, where=mask, keepdims=True)
    exposure_coeffs = exposure_coeffs / total_exposure
    # Sort by decreasing exposure
    sorted_idx = jnp.argsort(exposure_coeffs, descending=True)
    sorted_coeffs = exposure_coeffs[sorted_idx]
    # Cumulative exposure
    cum_coeffs = jnp.cumsum(sorted_coeffs)
    # Index of smallest prefix that reaches ~p_ineq exposure
    doc_threshold = jnp.argmin(jnp.abs(cum_coeffs - p_ineq))
    # Convert index -> count (index + 1), then normalize by N_valid_docs
    return (doc_threshold + 1) / N_valid_docs


@partial(jax.jit, static_argnames=["n_samples", "top_K", "q_def"])
def tsallis_exposure(scores, labels, mask, *, rng_key, n_samples, top_K, q_def=2):
    # weights and exposure computation as before
    weights = aux_ir.dcg_weights(top_K)
    _, rng_ranks = jrng.split(rng_key, 2)
    ranks_mat = gumbel_rankings(scores, mask, rng_ranks, n_samples)
    exposure_coeffs = aux_exp.exposure(scores, weights, ranks_mat, mask)

    # Zero out invalid docs explicitly
    exposure_coeffs = jnp.where(mask, exposure_coeffs, 0.0)

    # Sort all docs by exposure
    sorted_idx = jnp.argsort(exposure_coeffs, descending=True)

    # Reorder exposures and mask
    exposure_sorted = exposure_coeffs[sorted_idx]
    mask_sorted = mask[sorted_idx]

    # Normalize only over valid docs
    Z = jnp.sum(exposure_sorted, where=mask_sorted)

    # To be safe with potential Z == 0 (e.g., all invalid), add a small epsilon or use lax.cond
    Z = jnp.where(jnp.isclose(Z, 0.0, atol=1e-7), 1.0, Z)

    norm_coeffs = exposure_sorted / Z

    # Tsallis entropy over valid docs only
    tsallis = (1.0 / (q_def - 1.0)) * (
        1.0 - jnp.sum(jnp.power(norm_coeffs, q_def), where=mask_sorted)
    )

    return tsallis


@partial(jax.jit, static_argnames=["n_samples", "top_K"])
def entropy_exposure(scores, labels, mask, *, rng_key, n_samples, top_K):
    weights = aux_ir.dcg_weights(top_K)
    _, rng_ranks = jrng.split(rng_key, 2)
    ranks_mat = gumbel_rankings(scores, mask, rng_ranks, n_samples)
    exposure_coeffs = aux_exp.exposure(scores, weights, ranks_mat, mask)

    # Optional: zero out invalid docs (not strictly necessary if you always use `where=mask`)
    exposure_coeffs = jnp.where(mask, exposure_coeffs, 0.0)

    # Normalize only over valid docs
    Z = jnp.sum(exposure_coeffs, where=mask)
    Z = jnp.where(jnp.isclose(Z, 0.0, atol=1e-7), 1.0, Z)

    norm_coeffs = exposure_coeffs / Z

    # Valid docs with nonzero probability
    mask_pos = mask & ~jnp.isclose(norm_coeffs, 0.0, atol=1e-16)

    # Safe log: log(1) = 0 for masked-out entries
    safe_norm = jnp.where(mask_pos, norm_coeffs, 1.0)

    entropy = -jnp.sum(
        safe_norm * jnp.log(safe_norm),
        where=mask_pos,
    )
    return entropy


@partial(jax.jit, static_argnames=["n_samples", "top_K"])
def exposure_dcg_metric_fn(scores, labels, mask, rng_key, *, n_samples, top_K):

    _, rng_ranks = jrng.split(rng_key, 2)    
    
    K = scores.shape[0] if top_K is None else top_K
    weights = aux_ir.dcg_weights(K)

    ranking_samples = gumbel_rankings(scores, mask, rng_ranks, n_samples)
    exposures = exposure_prim(scores, weights, ranking_samples, mask=mask)
    # relevances = aux_ir.rel_qd_fn(labels, 0.1)
    # relevances is a [D] vector of relevance labels per doc.
    relevance_loss = jnp.sum(exposures * labels, where=mask)
    return relevance_loss