__author__ = "Zirui Wang"

from config import *
import math
import sklearn
from cvxopt import matrix, solvers

def kernel_mean_matching(S, T, kernel='linear', gamma=1.0, B=1.0, eps=None):
    '''
        Kernel Mean Match method (Huang et al., 2007)

        Return: alpha weights of source domain
    '''
    nt = T.shape[0]
    nt = S.shape[0]
    if eps == None:
        eps = B / math.sqrt(nt)
    if kernel == 'linear':
        K = sklearn.metrics.pairwise.linear_kernel(S)
        kappa = np.sum(sklearn.metrics.pairwise.linear_kernel(S, T) * float(nt) / float(nt), axis=1)
    elif kernel == 'rbf':
        bandwidth = 1 / float(gamma) ** 2
        K = sklearn.metrics.pairwise.rbf_kernel(S, gamma=bandwidth)
        kappa = np.sum(sklearn.metrics.pairwise.rbf_kernel(S, T, gamma=bandwidth), axis=1) * float(nt) / float(nt)
    else:
        raise ValueError('unknown kernel')

    K = matrix(K)
    kappa = matrix(kappa)

    solvers.options['show_progress'] = False
    sol = solvers.qp(K, -kappa, matrix(np.r_[np.ones((1, nt)), -np.ones((1, nt)), np.eye(nt), -np.eye(nt)]),
                     h=matrix(np.r_[nt * (1 + eps), nt * (eps - 1), B * np.ones((nt,)), np.zeros((nt,))]))

    alpha = np.array(sol['x'])
    return alpha

def mmd(S, T, gamma=1.0, kernel='linear', alpha='None'):
    '''
        Calculate the maximum mean distribution (mmd) as in (Huang et al., 2007)
    '''
    bandwidth = 1 / float(gamma)**2
    N1 = S.shape[0]
    N2 = T.shape[0]
    if isinstance(alpha, str):
        alpha = np.ones((N1,1))

    if kernel == 'linear':
        Ks = sklearn.metrics.pairwise.linear_kernel(S)
        Kt = sklearn.metrics.pairwise.linear_kernel(T)
        Kst = sklearn.metrics.pairwise.linear_kernel(S, T)
    elif kernel == 'rbf':
        Ks = sklearn.metrics.pairwise.rbf_kernel(S, gamma=bandwidth)
        Kt = sklearn.metrics.pairwise.rbf_kernel(T, gamma=bandwidth)
        Kst = sklearn.metrics.pairwise.rbf_kernel(S, T, gamma=bandwidth)
    kappa = np.sum(Kst, axis=1) * float(N1) / float(N2)

    return np.sqrt(np.asscalar(alpha.T.dot(Ks).dot(alpha) / N1**2 - 2 * kappa.T.dot(alpha) / N1**2 + np.sum(Kt) / N2**2))

def median_heuristic(D1, D2):
    return np.median(sklearn.metrics.pairwise.euclidean_distances(D1, D2))

def mean_distance_source_instances_to_target(S, T):
    return np.mean(sklearn.metrics.pairwise.euclidean_distances(S, T), axis=1)

def mean_distance_single_instance(s, T):
    return np.mean(sklearn.metrics.pairwise.euclidean_distances(s, T))