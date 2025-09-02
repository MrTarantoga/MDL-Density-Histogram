# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: binding=False
# cython: cdivision=True

import numpy as np
cimport cython
from libc.math cimport floor, log, INFINITY, exp

cdef const double PI = 3.1415926535897932384626433832795028841971693993751058209749445923078164062

cdef double log_fac_by_Ramanujan(const unsigned long long n):  
    assert n >= 0, "N is only allowed to be [0, inf)"
    cdef double term_1, term_2, term_3, result

    if n <= 1:
        return log(1.0)
    else:
        term_1 = <double>n * log(n) - <double>n
        term_2 = log(<double>n * <double>(1 + 4 * n * (1 + 2 * n))) / 6.0
        term_3 = log(PI) / 2.0
        result = term_1 + term_2 + term_3
        return result

def quantize_data(double [:] x, double epsilon):
    assert len(x) > 2, "The x must contain [2, inf)"
    assert epsilon > 0, "Epsilon must be positive"

    n_bins = int(np.ceil((np.max(x) - np.min(x)) / epsilon)) + 1
    bins = np.arange(
        np.min(x), np.max(x) + n_bins * epsilon + 1e-10, epsilon
    )
    data_disc = []
    for lower_bin, upper_bin in zip(bins, bins[1:]):
       number_bins = np.argwhere(np.logical_and(x < upper_bin, x >= lower_bin) == True).shape[0]
       data_disc += [lower_bin]*number_bins
    return np.asarray(data_disc)

def generate_candidate_cut_points(double [:] x, double epsilon):
    """Generate candidate cut points between data points (Equation 22)"""
    assert len(x) > 2, "The x must contain [2, inf)"
    assert epsilon > 0, "Epsilon must be positive"

    x = np.sort(x)
    candidates = []
    
    for _x in x:
        candidates.extend([_x - epsilon/2, _x + epsilon/2])
    
    # Remove duplicates and sort
    candidates = sorted(list(set(candidates)))
    
    # Remove implicit boundaries
    implicit_lower = min(x) - epsilon/2 
    candidates = np.delete(candidates, np.argwhere(candidates == implicit_lower).flatten())

    return candidates

cpdef unsigned long long[:] precompute_n_e(double [:] data, double [:] candidates):
    """Precompute n_e: number of data points in [x_min, c_e] (Section 4)"""
    n_e = np.zeros(len(candidates), dtype=np.uint64)  # Include E+1
    cdef Py_ssize_t i, d
    cdef double c

    for i, c in enumerate(candidates):
        # Count data 
        for d in range(len(data)):
            if data[d] < c:
                n_e[i] += 1
    return n_e

cdef double compute_parametric_complexity(unsigned long long n, unsigned long long K):
    cdef double R_prev2 = 1.0  # R_n_h(K-2)
    cdef double R_current = 0.0  # R_n_h(K)
    cdef double R_prev1 = 0.0  # R_n_h(K-1)
    cdef unsigned long long h2, h1, k
    cdef double term_1, term_2, term_3
    cdef double log_n = log(n)
    
    if K == 0:
        return 1.0

    if K >= 1:
        # Precompute log(n) once for all h2 iterations
        for h2 in range(n + 1):
            h1 = n - h2
            term_1 = log_fac_by_Ramanujan(n) - (log_fac_by_Ramanujan(h1) + log_fac_by_Ramanujan(h2))
            
            if h1 != 0:
                term_2 = (log(<double>h1) - log_n) * <double>h1
            else:
                term_2 = 0.0
                
            if h2 != 0:
                term_3 = (log(<double>h2) - log_n) * <double>h2
            else:
                term_3 = 0.0
                
            R_current += exp(term_1 + term_2 + term_3)
        
    if K >= 2:
        R_prev1 = R_current  # R_n_h(1)
        for k in range(2, K + 1):
            R_current = R_prev1 + (<double>n / <double>(k-1)) * R_prev2
            R_prev2 = R_prev1
            R_prev1 = R_current

    return R_current

cdef double dp_func_init(unsigned long long [:] n_e, 
                         double [:] x, 
                         unsigned long long n, 
                         double [:] candidates, 
                         unsigned long long e, 
                         double epsilon):
    cdef double term1, term2, best_score, candidates_val, x_min
    cdef unsigned long long n_e_val = <unsigned long long>n_e[e]

    if n_e_val != 0:
        candidates_val = <double>candidates[e]
        x_min = <double>np.min(x)
        
        term1 = log(epsilon * <double>n_e_val)
        term2 = log((candidates_val - (x_min - epsilon / 2.0)) * <double>n)
        best_score = -<double>n_e_val * (term1 - term2)
        return best_score
    else:
        return 0.0

cdef tuple dp_func(unsigned long long [:] n_e, 
                   double [:] x, 
                   const unsigned long long n, 
                   double [:] candidates, 
                   const unsigned long long K, 
                   const unsigned long long E, 
                   const unsigned long long e, 
                   const double epsilon, 
                   double[:, :] DP_table,
                   dict[tuple[unsigned long long, unsigned long long], double] cache):
    cdef double e_prime_best_score = INFINITY
    cdef double best_score = INFINITY
    cdef unsigned long long e_prime
    cdef unsigned long long n_k
    cdef double bin_width
    cdef double term1, term2, term3, score
    cdef double R, R_prime

    for e_prime in range(K-1, e):
        n_k = n_e[e] - n_e[e_prime]
        bin_width = candidates[e] - candidates[e_prime]

        if n_k != 0:
            term1 = <double>n_k * (log(epsilon * <double>n_k) - log(bin_width * <double>n))
        else:
            term1 = 0.0

        if (n_e[e], K) in cache:
            R = cache[n_e[e], K]
        else:
            R = compute_parametric_complexity(n_e[e], K)
            cache[n_e[e], K] = R

        if (n_e[e_prime], K-1) in cache:
            R_prime = cache[n_e[e_prime], K-1]
        else:
            R_prime = compute_parametric_complexity(n_e[e_prime], K - 1)
            cache[n_e[e_prime], K-1] = R_prime

        term2 = log(R / R_prime)
        term3 = log(<double>(E - K + 4) / <double>K)

        assert DP_table[K-1, e_prime] != INFINITY, "DP_table contain a infinity error, that is not allowed"

        score = DP_table[K-1, e_prime] - term1 + term2 + term3
        if score < best_score:
            e_prime_best_score = <double>e_prime
            best_score = score

    return (e_prime_best_score, best_score)



def mdl_optimal_histogram(double [:] data, 
                          double epsilon=0.1, 
                          unsigned long long K_max=10):
    cdef double[:] data_view = data
    cdef double[:] K_scores
    cdef unsigned long long n = data_view.shape[0]
    cdef unsigned long long i
    cdef unsigned long long K, e, K_best, e_pos,
    cdef double[:] candidates
    cdef unsigned long long[:] n_e
    cdef unsigned long long E
    cdef double[:, :] dp_table_score
    cdef int[:, :] dp_table_e_prime
    cdef double best_score
    cdef list optimal_cut_points

    # Check for valid input data
    assert len(data) > 2, "The data must contain more than 2 datapoints"
    assert epsilon > 0, "The epsilon must be positive"

    # Step 0: Create lookup dictionary for parametric compute_parametric_complexity
    cdef dict[tuple[unsigned long long, unsigned long long], double] cpc_cache = {}

    # Step 1: Quantize data
    data = quantize_data(data, epsilon)

    # Step 3: Generate candidate cut points
    candidates = generate_candidate_cut_points(data, epsilon)
    E = candidates.shape[0] - 1  # Exclude implicit outer boundary

    # Step 4: Precompute n_e
    n_e = precompute_n_e(data, candidates)

    assert K_max > 1, "K_max must be [2, inf)"

    # Step 5: Initialize DP tables
    dp_table_score = np.full((K_max + 1, E + 1), INFINITY, dtype=np.float64)
    dp_table_e_prime = np.full((K_max + 1, E + 1), -1, dtype=np.int32)

    # Initialize DP table for K=0
    for e in range(E + 1):
        dp_table_score[0, e] = dp_func_init(n_e, data, n, candidates, e, epsilon)

    # Fill DP table for K >= 1
    for K in range(1, K_max + 1):
        for e in range(K, E + 1):
            dp_table_e_prime[K, e], dp_table_score[K, e] = dp_func(
                n_e, data, n, candidates, K, E, e, epsilon, dp_table_score, cpc_cache
            )

    # Find best K
    K_best = 0
    K_scores = dp_table_score[:, E]

    best_score = dp_table_score[0, E]
    for K in range(1, K_max + 1):
        if dp_table_score[K, E] < best_score:
            best_score = dp_table_score[K, E]
            K_best = K

    # Backtrack to find optimal cut points
    optimal_cut_points = [candidates[E]]
    e_pos = E
    for K in range(K_best - 1, 0, -1):
        e_pos = dp_table_e_prime[K, e_pos]
        optimal_cut_points.append(candidates[e_pos])

    # Add min and sort
    optimal_cut_points.append(np.min(data))
    optimal_cut_points.sort()

    return np.array(optimal_cut_points, dtype=np.float64), np.array(K_scores, dtype=np.float64)
