#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: infer_types=True

from libc.math cimport log, exp


cdef inline double __pfunc__(double i, double j, double t):
    """
    Function to compute pairwise comparison probabilities of non-ties
    :param i: merit of the winning item
    :param j: merit of the loosing item
    :param t: difference threshold
    :return: probability of item i beating item j
    """
    return log(exp(i) / (exp(i) + exp(j) * exp(t)))


cdef inline double __tfunc__(double i, double j, double t):
    """
    Function to compute pairwise comparison probabilities of ties
    :param i: merit of the winning item
    :param j: merit of the loosing item
    :param t: difference threshold
    :return: probability of item i beating item j
    """
    f1 = exp(i) * exp(j) * (exp(t) * exp(t) - 1)
    f2 = (exp(i) + exp(j) * exp(t)) * (exp(i) * exp(t) + exp(j))
    return log(f1 / f2)


cdef inline double __rfunc__(double i, double l):
    """
    Function to compute regularized probability
    :param i: item merit
    :param l: regularization factor
    :return: value of __pfunc__ for matches with dummy item weighted by l
    """
    return l * (__pfunc__(i, 1, 0) + __pfunc__(1, i, 0))


def __log_likelihood__(double[::1] merits, int[:,:] comparisons, double regularization, double threshold):
    """
    Log-Likelihood Function
    :param merits: merit vector
    :param comparisons: comparison matrix, shape=(n_comparisons, 3); each row indicates (index_a, index_b, tie)
    :param regularization: regularization parameter
    :param threshold: difference threshold
    :return: log-likelihood value
    """
    cdef:
        double k = 0   # Maximization sum
        int i          # Loop variable
        Py_ssize_t n_comparisons = comparisons.shape[0]
        Py_ssize_t n_items = merits.shape[0]
        int arg1, arg2, tie
    # Summing Edge Probabilities
    for i in range(n_comparisons):
        arg1 = comparisons[i,0]
        arg2 = comparisons[i,1]
        tie = comparisons[i,2]
        if tie == 0:
            k += __pfunc__(merits[arg1], merits[arg2], threshold)
        else:
            k += __tfunc__(merits[arg1], merits[arg2], threshold)
    # Regularization
    for i in range(n_items):
        k += __rfunc__(merits[i], regularization)
    return -1 * k
