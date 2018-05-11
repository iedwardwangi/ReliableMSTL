__author__ = "Zirui Wang"

from config import *

class ReliableMultiSourceModel(object):

    def __init__(self, X_source, Y_source, X_target, source_labeled_indices, params):
        self.X_source = X_source
        self.Y_source = Y_source
        self.n_domain = len(X_source)
        self.source_labeled_indices = source_labeled_indices
        self.source_unlabeled_indices = [0] * self.n_domain
        self.X_target = X_target
        self.params = params
        self.n_T = X_target.shape[0]

        # Active Learning params
        self.Y_source_labeled = []
        self.AL_models = [0] * self.n_domain
        self.tau = np.zeros((self.n_domain, self.n_domain))
        self.AL_method = self.params["AL_method"]
        self.b1 = self.params["b1"]
        self.start_mode = self.params["start_mode"]
        self.max_alpha = self.params["max_alpha"]
        self.source_labeled_ratio = [0] * self.n_domain

        # Transfer Learning params
        self.mu = self.params["mu"]
        self.rho = self.params["rho"]
        self.beta_1 = self.params["beta_1"]
        self.beta_2 = self.params["beta_2"]
        self.hasTrained = False
        self.models = [0] * self.n_domain
        self.alpha_list = [0] * self.n_domain
        self.source_mmd = np.zeros(self.n_domain)
        self.source_constant_prox_weights = [0] * self.n_domain
        self.source_constant_prox_weights = np.zeros(self.n_domain)
        self.omega = np.zeros(self.n_domain)

        # Initialize weights
        self.initialize_active_learning()
        self.train_source_constant_mmd_weights()
        self.calculate_source_labeled_ratio()
        self.train_alpha_list()

    ######################################################################################################
    # Initialize Methods
    ######################################################################################################

    def initialize_active_learning(self):
        '''
            Initialize all active learning required params
        '''
        self.initialize_labeled_Y()
        self.initialize_unlabeled_indices()
        self.train_AL_models()
        self.initialize_tau()

    def train_AL_models(self):
        '''
            This method trains the perceptron used during the active learning procedure
        '''
        for i in xrange(self.n_domain):
            # For simplicity, the linear SVC is used here. Feel free to modify to any kernel svm/other base model.
            X_train, Y_train = self.X_source[i][self.source_labeled_indices[i]], self.Y_source_labeled[i]
            self.AL_models[i] = LinearSVC(loss="hinge").fit(X_train, Y_train)

    def initialize_labeled_Y(self):
        '''
            This method initializes the labeled Y data for both active learning and transfer learning
        '''
        for i in xrange(self.n_domain):
            self.Y_source_labeled.append(self.Y_source[i][self.source_labeled_indices[i]].reshape(-1))

    def initialize_tau(self):
        '''
            Initialize the value of tau matrix
        '''
        if self.start_mode == "cold":
            self.tau = np.ones((self.n_domain, self.n_domain))/float(self.n_domain-1)
            for source_i in xrange(self.n_domain):
                self.tau[source_i][source_i] = 0
        elif self.start_mode == "warm":
            self.tau = np.zeros((self.n_domain, self.n_domain))
            for k in xrange(self.n_domain):
                for m in xrange(self.n_domain):
                    if k != m:
                        self.tau[k][m] = np.exp(
                            self.beta_1 * self.AL_models[m].score(self.X_source[k][self.source_labeled_indices[k]],
                                                     self.Y_source_labeled[k]))
                self.tau[k] = self.tau[k] / np.sum(self.tau[k])
        else:
            raise Exception("Error: shouldn't get here")

    def initialize_unlabeled_indices(self):
        '''
            Initialize the list of unlabeled indices for each source to be selected
        '''
        for source_id in xrange(self.n_domain):
            self.reset_unlabeled_indices_for_source(source_id)

    def reset_unlabeled_indices_for_source(self, source_id):
        '''
            Reset the unlabeled list for a single source to [i for i not in labeled]
        '''
        self.source_unlabeled_indices[source_id] = [i for i in xrange(self.X_source[source_id].shape[0]) if
                                    i not in self.source_labeled_indices[source_id]]

    ######################################################################################################
    # Weight Util Methods
    ######################################################################################################

    def calculate_source_labeled_ratio(self):
        '''
            Calculate the ratio beta as: beta_i * m = m_i where m is total number of all labeled sources
        '''
        for source_id in xrange(self.n_domain):
            self.source_labeled_ratio[source_id] = float(len(self.source_labeled_indices[source_id]))
        self.source_labeled_ratio = self.source_labeled_ratio / np.sum(self.source_labeled_ratio)


    def train_alpha_list(self):
        '''
            Train KMM weights for each source
        '''
        for i in xrange(self.n_domain):
            X_source = self.X_source[i]
            alpha = kernel_mean_matching(X_source, self.X_target, kernel='linear', B=self.max_alpha)
            self.alpha_list[i] = alpha

    def train_source_constant_mmd_weights(self):
        '''
            Train fixed weights based on MMD(S_i, T)
        '''
        for i in xrange(self.n_domain):
            X_source = self.X_source[i]
            source_mmd = mmd(X_source, self.X_target)
            self.source_mmd[i] = source_mmd
            self.source_constant_prox_weights[i] = np.exp(-1 * self.beta_2 * (source_mmd**self.rho))
        self.source_constant_prox_weights = self.source_constant_prox_weights / np.sum(self.source_constant_prox_weights)

    def train_source_labeled_constant_prox_weights(self):
        '''
            Train fixed weights based on MMD(S_i[labeled], T)
        '''
        for i in xrange(self.n_domain):
            X_source = self.X_source[i][self.source_labeled_indices[i]]
            source_mmd = mmd(X_source, self.X_target)
            self.source_labeled_constant_prox_weights[i] = np.exp(-1 * self.beta_2 * (source_mmd ** self.rho))
        self.source_labeled_constant_prox_weights = self.source_labeled_constant_prox_weights / np.sum(
            self.source_labeled_constant_prox_weights)

    def train_omega(self):
        '''
            Calculate the omega which measures both proximity and reliability
        '''
        gamma = np.zeros((self.n_domain, self.n_domain))
        for k in xrange(self.n_domain):
            for m in xrange(self.n_domain):
                if k != m:
                    gamma[k][m] = np.exp(
                        self.beta_1 * self.models[m].score(self.X_source[k][self.source_labeled_indices[k]],
                                                      self.Y_source_labeled[k]))
            gamma[k] = (gamma[k] / np.sum(gamma[k]))
        gamma = self.mu * np.eye(self.n_domain) + (1 - self.mu) * gamma

        self.omega = self.source_constant_prox_weights.dot(gamma)
        self.omega = self.omega / np.sum(self.omega)

    def train_models(self, model):
        '''
            Train models for each source
        '''
        if model == "perceptron":
            for i in xrange(self.n_domain):
                X_train, Y_train = self.X_source[i][self.source_labeled_indices[i]], self.Y_source_labeled[i]
                self.models[i] = Perceptron(max_iter=5).fit(X_train, Y_train)
            self.hasTrained = True
        elif model == "svm":
            # For simplicity, the linear SVC is used here. Feel free to modify to any kernel svm/other base model.
            for i in xrange(self.n_domain):
                X_train, Y_train = self.X_source[i][self.source_labeled_indices[i]], self.Y_source_labeled[i]
                self.models[i] = LinearSVC(loss="hinge").fit(X_train, Y_train)
            self.hasTrained = True
        else:
            print "Error: shouldn't get here"

        if self.hasTrained:
            # or use the labeled instances: self.train_source_labeled_constant_prox_weights()
            self.train_source_constant_mmd_weights()
            self.train_omega()

    ######################################################################################################
    # Active Learning Method
    ######################################################################################################

    def perform_active_learning(self, source_i=-1):
        '''
            Perform a round of active learning to label one instance

            Return: 1 if the query is successful, 0 otherwise (all instances in that source have been labeled)
        '''
        if self.AL_method == "random":
            # Randomly pick an instance from a random source
            if source_i == -1:
                source_i = random.choice(xrange(self.n_domain))
            if len(self.source_labeled_indices[source_i]) < self.X_source[source_i].shape[0]:
                active_index = self.pick_active_index(source_i, "random")
                self.query_instance(source_i, active_index)
                return 1
            return 0
        if self.AL_method == "proximity":
            # Pick an instance from the most similar source
            source_i = np.random.choice(self.n_domain, p=self.source_constant_prox_weights)
            if len(self.source_labeled_indices[source_i]) < self.X_source[source_i].shape[0]:
                active_index = self.pick_active_index(source_i, "proximity")
                self.query_instance(source_i, active_index)
                return 1
            return 0
        elif self.AL_method == "uncertainty":
            # Pick the most uncertainty instance
            if source_i == -1:
                raise Exception("A source_id must be provided for uncertainty sampling.")
            if len(self.source_labeled_indices[source_i]) < self.X_source[source_i].shape[0]:
                active_index = self.pick_active_index(source_i, "uncertainty")
                self.query_instance(source_i, active_index, retrain=True)
                return 1
            return 0
        elif self.AL_method == "representative":
            # Pick the most representative instance
            if source_i == -1:
                raise Exception("A source_id must be provided for representative sampling.")
            if len(self.source_labeled_indices[source_i]) < self.X_source[source_i].shape[0]:
                active_index = self.pick_active_index(source_i, "KMM")
                self.query_instance(source_i, active_index)
                return 1
            return 0
        elif self.AL_method == "AMSAT":
            # Pick according to KL divergence and based on KMM weights
            p = KL_Divergence(self.source_labeled_ratio)
            if np.random.binomial(1, p) == 0:
                # Exploitation
                self.train_omega_ALModels()
                source_p = self.omega # or we can add additional control variable: source_p = np.exp(self.beta_4 * self.omega)
            else:
                # Exploration
                source_p = 1 / self.source_labeled_ratio
            source_p = normalize_weights(source_p)
            source_i = np.random.choice(self.n_domain, p=source_p)
            active_index = self.pick_active_index(source_i, "KMM_Weighted_Uncertainty")
            self.query_instance(source_i, active_index, retrain=True)
            self.calculate_source_labeled_ratio()
            return 1
        else:
            raise Exception("Error: shouldn't get here")

    def query_instance(self, source_i, active_index, retrain=False):
        '''
            Query an instance and retrain active learning model
        '''
        if active_index not in self.source_unlabeled_indices[source_i]:
            print "Warning: trying to label an already labeled instance"

        self.source_labeled_indices[source_i] = np.append(self.source_labeled_indices[source_i], active_index)
        self.Y_source_labeled[source_i] = np.append(self.Y_source_labeled[source_i], self.Y_source[source_i][active_index][0])
        self.source_unlabeled_indices[source_i].remove(active_index)
        if retrain:
            X_train, Y_train = self.X_source[source_i][self.source_labeled_indices[source_i]], self.Y_source_labeled[source_i]
            self.AL_models[source_i] = LinearSVC(loss="hinge").fit(X_train, Y_train)

    def pick_active_index(self, source_id, criteria):
        '''
            Select an instance in the source according to the criteria
        '''
        if criteria == "random":
            return random.choice(self.source_unlabeled_indices[source_id])
        elif criteria == "uncertainty":
            uncertainty_list = np.abs(self.AL_models[source_id].decision_function(self.X_source[source_id][self.source_unlabeled_indices[source_id]]))
            return self.source_unlabeled_indices[source_id][np.argmin(uncertainty_list.reshape(-1))]
        elif criteria == "proximity":
            distance_list = mean_distance_source_instances_to_target(self.source_unlabeled_indices[source_id], self.X_target)
            return self.source_unlabeled_indices[source_id][np.argmin(distance_list.reshape(-1))]
        elif criteria == "KMM":
            return self.source_unlabeled_indices[source_id][np.argmax(self.alpha_list[source_id][self.source_unlabeled_indices[source_id]])]
        elif criteria == "KMM_Weighted_Uncertainty":
            alpha_list = self.max_alpha - self.alpha_list[source_id][self.source_unlabeled_indices[source_id]].reshape(-1)
            uncertainty_list = np.exp(np.abs(self.AL_models[source_id].decision_function(
                self.X_source[source_id][self.source_unlabeled_indices[source_id]])).reshape(-1))
            return self.source_unlabeled_indices[source_id][np.argmin(alpha_list * uncertainty_list)]
        else:
            raise Exception("Error: shouldn't get here")


    def train_omega_ALModels(self):
        '''
            Calculate the omega which measures both proximity and reliability using the Active Learning models when doing Active Learning
        '''
        self.tau = np.zeros((self.n_domain, self.n_domain))
        for k in xrange(self.n_domain):
            for m in xrange(self.n_domain):
                if k != m:
                    self.tau[k][m] = np.exp(
                        self.beta_1 * self.AL_models[m].score(self.X_source[k][self.source_labeled_indices[k]],
                                                      self.Y_source_labeled[k]))
            self.tau[k] = (self.tau[k] / np.sum(self.tau[k]))
        self.tau = self.mu * np.eye(self.n_domain) + (1 - self.mu) * self.tau

        self.omega = self.source_constant_prox_weights.dot(self.tau)
        self.omega = self.omega / np.sum(self.omega)

    ######################################################################################################
    # Transfer Learning Classification Method
    ######################################################################################################

    def single_source_classify(self, x, source_id):
        '''
            Classify based on a single model trained on an individual source
        '''
        if not self.hasTrained:
            print "Warning: models are not trained"
            return None
        return self.models[source_id].predict(x)

    def multi_source_classify_PWMSTL(self, x):
        '''
            Classify based the proposed PW-MSTL method
        '''
        if not self.hasTrained:
            print "Warning: models are not trained"
            return None
        predict = 0.0
        for i in xrange(self.n_domain):
            predict += self.omega[i] * self.single_source_classify_PWMST(i, x)
        return np.sign(predict)

    def single_source_classify_PWMST(self, source_i, x):
        p_kk = self.models[source_i].decision_function(x)
        y_hat = np.sign(p_kk)
        if p_kk < self.b1:
            p_peer = 0.0
            for m in xrange(self.n_domain):
                if m != source_i:
                    p_peer += self.tau[source_i][m] * self.models[m].decision_function(x)
            y_peer = np.sign(p_peer)
            return y_peer
        else:
            return y_hat



