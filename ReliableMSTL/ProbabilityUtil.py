__author__ = "Zirui Wang"

from config import *

def pick_index_with_probability(weights):
    '''
        Randomly pick an index with weights
    '''
    total = np.sum(weights)
    r = random.uniform(0, total)
    upto = 0
    for index in xrange(len(weights)):
        if upto + weights[index] >= r:
            return index
        upto += weights[index]
    assert False, "Shouldn't get here"

def normalize_weights(weights):
    return weights/np.sum(weights)

def KL_Divergence(weights):
    kl_log = np.log(weights) - np.log(np.ones(len(weights)) / float(len(weights)))
    kl = 0.0
    for i in xrange(len(weights)):
        kl += weights[i] * kl_log[i]
    return kl