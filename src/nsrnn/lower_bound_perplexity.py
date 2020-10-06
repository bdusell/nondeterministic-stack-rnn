import math

from .lang_algorithm.pcfg import string_log_probability

def compute_lower_bound_perplexity(sampler, parser, num_valid_lengths, samples):
    """
    Compute the lower bound perplexity of a specific set of strings generated
    by a PCFG subject to length constraints.
    """
    # Let p_G be the probability distribution over strings defined by the PCFG.
    # For a given string w with length l, we compute p(w) as
    # p(w) = p(l) p_G(w) / \sum_{w' where |w'| = l } p_G(w').
    length_log_prob = -math.log(num_valid_lengths)
    parts = compute_lower_bound_parts(sampler, parser, samples)
    return parts_to_perplexity(parts, num_valid_lengths)

def compute_lower_bound_parts(sampler, parser, samples):
    r"""
    Compute \sum_w log p(w) / \sum_{w' where |w'| = |w|} p_G(w') and
    \sum_w |w| for w in samples.
    """
    total_neg_log_prob = 0.0
    total_len = 0
    num_samples = 0
    for sample in samples:
        length = len(sample)
        # Re-parse the string wrt the grammar to compute p_G(w).
        gw_log_prob = string_log_probability(parser, sample)
        # Get \sum_{w' where |w'| = l} p_G(w') from the table of inside
        # weights.
        summed_gw_prob = sampler.get_inside_probability(parser.grammar.start, length)
        summed_gw_log_prob = math.log(summed_gw_prob)
        total_neg_log_prob -= gw_log_prob - summed_gw_log_prob
        total_len += length
        num_samples += 1
    return total_neg_log_prob, total_len, num_samples

def parts_to_perplexity(parts, num_valid_lengths):
    neg_log_prob, total_len, num_samples = parts
    length_neg_log_prob = math.log(num_valid_lengths)
    neg_log_prob += length_neg_log_prob * num_samples
    return math.exp(neg_log_prob / total_len)
