__author__ = "Zirui Wang"

"""
	Description:
		A simple simulator to demonstrate how to use AMSAT method
		experiment parameters can be tuned below
"""

from config import *

def MultiSouece_Active_Learning_Simulator(exp_params):
    # k sources, each with n instances with d dimensions
    k, n, d = exp_params["k"], exp_params["n"], exp_params["d"]
    X_source, Y_source, source_labeled_indices = [], [], []
    for i in xrange(k):
        X_source.append(np.random.rand(n,d))
        Y_source.append(np.random.choice([-1, 1], n).reshape(-1, 1))
        source_labeled_indices .append(np.random.choice(xrange(X_source[i].shape[0]), n/2, replace=False))
    X_target_train = np.random.rand(n, d)
    X_target_test, Y_target_test = np.random.rand(n,d), np.random.choice([-1, 1], n).reshape(-1, 1)

    # Initialize the model with model parameters
    params = exp_params2model_params(exp_params)
    model = ReliableMultiSourceModel(X_source, Y_source, X_target_train, source_labeled_indices, params)
    model.train_models(exp_params["base_model"])

    # Evaluate before active learning
    Y_predicted_before_active_learning = []
    for i in xrange(X_target_test.shape[0]):
        Y_predicted_before_active_learning.append(model.multi_source_classify_PWMSTL(X_target_test[i].reshape(1, -1)))
    print "Accuracy before active learning: ", accuracy_score(Y_predicted_before_active_learning, Y_target_test)

    # Perform active learning
    for i_round in xrange(exp_params["budget"]):
        model.perform_active_learning()

    # Evaluate after active learning
    Y_predicted_after_active_learning = []
    for i in xrange(X_target_test.shape[0]):
        Y_predicted_after_active_learning.append(model.multi_source_classify_PWMSTL(X_target_test[i].reshape(1, -1)))
    print "Accuracy after active learning: ", accuracy_score(Y_predicted_after_active_learning, Y_target_test)

# Experimental Parameters
if __name__ == '__main__':
    exp_params = {}

    # Core parameters required for PW-MSTL
    exp_params["start_mode"] = "warm"
    exp_params["b1"] = 1.0
    exp_params["tau_lambda"] = 1.0
    exp_params["rho"] = 1.0
    exp_params["beta_1"] = 10.0
    exp_params["beta_2"] = 10.0
    exp_params["mu"] = 0.1
    exp_params["max_alpha"] = 10.0
    exp_params["AL_method"] = "AMSAT"

    # Dummy parameters
    exp_params["base_model"] = "svm"

    # Experiment specific parameters
    exp_params["k"] = 10
    exp_params["n"] = 100
    exp_params["d"] = 10
    exp_params["budget"] = 100

    MultiSouece_Active_Learning_Simulator(exp_params)
