# Preference Learning Toolbox
# Copyright (C) 2018 Institute of Digital Games, University of Malta
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from sklearn import svm
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel
from sklearn.base import BaseEstimator
def prefs2graph(y):
    ids = np.max(y) + 1
    graph = np.zeros((ids,ids), dtype = int)
    for c in y:
        graph[c[0],c[1]] += 1
    return graph

def topological_sort(graph, check_unique = True):
    # Kahn's algorithm
    graph = graph.copy()
    S = np.where(graph.sum(axis=0) == 0)[0]
    S = S.tolist()
    L = []
    warn = False
    while len(S):
        if len(S) > 1:
            warn = True
        n = S.pop()
        L.append(n)
        out = np.flatnonzero(graph[n,:]).tolist()
        for m in out:
            graph[n,m] = 0
            if graph[:,m].sum() == 0:
                S.append(m)

    assert graph.sum() == 0, "Graph has cycles"
    if check_unique and warn:
        print("Warning: topological_sort(): Not enough preferences to get a unique total ranking")
    return L

def prefs2order(y, check_unique = True):
    """Warning: order may not be unique."""
    # FIXME How to detect that there is no unique order.
    return topological_sort(prefs2graph(y), check_unique = check_unique)

def order2rank(order): return np.argsort(order)

def utility2prefs(utility, minimum_distance_margin = 0):
    # Utility is assumed to be maximised.
    
    # pairs = matrix n x n matrix (where n = #objects)
    pairs = np.subtract.outer(utility, utility)
    # select ranks where difference between object ratings is > minimum distance margin (e.g. 0)
    # prefs contains the preferred object indexes, nons contains the non-preferred object indexes
    # n.b. this is only one way so no duplicates are created
    prefs, nons = np.where(pairs > minimum_distance_margin)
    prefs = np.transpose(np.vstack((prefs,nons)))
    return prefs

def check_prefs_data(X, y):
    y = np.asarray(y)
    ids = np.unique(y)
    # All ids in y appear in X and viceversa
    assert np.array_equal(ids, np.arange(len(X)))
    y_uniq = np.unique(y, axis=0)
    assert len(y) == len(y_uniq)
    y_contradiction = np.vstack((y, y[:,[1,0]]))
    y_uniq = np.unique(y_contradiction, axis=0)
    # Otherwise, it means there was already a contradiction
    assert len(y_contradiction) == len(y_uniq)
    # Check for cycles and check that we have enough preferences to get back a
    # total order
    prefs2order(y, check_unique = True)
    

class RankSVM():#svm.SVR
    """RankSVM algorithm implemented using the `scikit-learn` library.
    A Support Vector Machine (SVM) is a binary classifier that separates the input put samples linearly in a
    projected space. The decision boundary of the classifier is given by a linear combination of training samples
    (called support vectors) in the projected space. The projection in provided by the kernel function that the
    user must select. The support vector and weights are selected to satisfy a set of constrains derived from the
    input samples and a cost parameter which regulates the penalization of misclassified training samples.
    In PLT, the algorithm was implemented using the `scikit-learn` library. In this implementation, the quadratic
    programmer solver contained in LIBSVM is used. The RankSVM algorithm is a rank-based version of traditional
    SVM training algorithms. It uses the same solver as standard training algorithms for binary SVMs; the only
    difference lies in the set of constraints which are defined in terms of pairwise preferences between
    training samples.
    """

    def __init__(self, kernel="rbf", gamma=.1, degree=3, debug=False):
        """Initializes the RankSVM object.
        :param kernel: the kernel function mapping the input samples to the projected space (default
            :attr:`pyplt.util.enums.KernelType.RBF`).
        :type kernel: :class:`pyplt.util.enums.KernelType`, optional
        :param gamma: the kernel coefficient for the ‘rbf’, ‘poly’ and ‘sigmoid’ kernels. If gamma
            is set to ‘auto’ then 1/n_features will be used instead (default 'auto').
        :type gamma: float or 'auto', optional
        :param degree: the degree of the polynomial (‘poly’) kernel function (default 3).
        :type degree: float, optional
        :param debug: specifies whether or not to print notes to console for debugging (default False).
        :type debug: bool, optional
        :raises InvalidParameterValueException: if the user attempts to use a gamma value <= 0.0.
        """
        desc = "A Support Vector Machine (SVM) is a binary classifier that separates the input put samples " \
               "linearly in a projected space. The decision boundary of the classifier is given by a linear " \
               "combination of training samples (called support vectors) in the projected space. The projection " \
               "in provided by the kernel function that the user must select. The support vector and weights are " \
               "selected to satisfy a set of constrains derived from the input samples and a cost parameter which " \
               "regulates the penalization of misclassified training samples. In PLT, the algorithm was implemented " \
               "using the scikit-learn library. In this implementation, the quadratic programmer solver contained " \
               "in LIBSVM is used. The RankSVM algorithm is a rank-based version of traditional SVM training " \
               "algorithms. It uses the same solver as standard training algorithms for binary SVMs; the only " \
               "difference lies in the set of constraints which are defined in terms of pairwise preferences between " \
               "training samples."

        if gamma != 'auto':
            assert float(gamma) > 0.0

        self.kernel = kernel
        if self.kernel == "rbf" or self.kernel == "poly":
            self.gamma = gamma
        else:
            self.gamma = None
        if self.kernel == "poly":
            self.degree = degree
        else:
            self.degree = None
        self._r_svm = svm.OneClassSVM(kernel='precomputed', tol=1e-3, shrinking=True,
                                      cache_size=1000, nu=0.5)#, max_iter=5000

        self._train_X = None
        self._train_y = None
        self._debug = debug
    # def get_params(self, deep=True):
    #     # suppose this estimator has parameters "alpha" and "recursive"
    #     return {"kernel": self.kernel, "gamma": self.gamma , "degree": self.degree}

    # def set_params(self, **parameters):
    #     for parameter, value in parameters.items():
    #         if hasattr(self, parameter):
    #             setattr(self, parameter, value)
    #     return self
    def fit(self, X, y):
        """Train a RankSVM model on the given training data.
        :param X: the objects data to train the model on.
        :param Y: the pairwise rank data to train the model on.
         """
        y = np.asarray(y)
        check_prefs_data(X, y)
        
        # K = matrix of precomputed kernels (shape = n_ranks x n_ranks)
        # y_trans = array of +1s for each rank (shape = 1 x n_ranks)
        # print("precomputing kernels...")
        K, y_trans = self._transform_dataset(X, y)
                
        # print("precomputation of kernels complete.")
        self._train_X = X.copy()
        self._train_y = y.copy()

        # print("Starting training with RankSVM.")
        # training...
        self._r_svm.fit(K, y=y_trans)
        # print("Training complete.")

        # print("num of svs: ")
        # print(len(self._r_svm.support_))
        # print("sv indexes: ")
        # print(self._r_svm.support_)
        # ... e.g. [  2   3   4   7  14  15  18  19  20  21  23  26  28  29  30  31  33  34
        #   35  37  38  39  42  43  45  47  52  54  55  56  57  58  59  62  63  65
        #   66  67  68  69  71  72  73  74  76  80  81  82  83  84  85  86  87  88
        #   89  91  93  94  95  97  98  99 100 101 105 106 108 109 111 112 114 115
        #  119 120 121 122 123 125 126 127 129 130 131 133 135 136 137 139 140 142
        #  143 144 146 148 149 150 151 154 155 157 159 160 162 163 164 165 166 167
        #  168 169 170 171 173 174 175 176 177 178 179 181 182 183 184 187 188 189
        #  191 192 194 195 196 198 199 200 201 205]
        # ^ equivalent to IDS OF RANKS in the training ranks set

        if self._debug:
            # pref & non feature vectors of the ranks constituting the support vectors
            sv_ranks = self._train_y[self._r_svm.support_, :]
            sv_prefs = self._train_X[sv_ranks[:, 0], :]
            sv_nons = self._train_X[sv_ranks[:, 1], :]

#             for r in range(len(sv_ranks)):
#                 # print("sv_" + str(r) + ": " + str(sv_ranks[r, 0]) + " > " + str(sv_ranks[r, 1]))
#                 # print("alpha: " + str(self._r_svm.dual_coef_[0, r]))
#                 # print("pref_obj:")
#                 # print(sv_prefs[r, :])
#                 # print("non_pref_obj:")
#                 # print(sv_nons[r, :])

#         print("sv alphas: ")
#         print(self._r_svm.dual_coef_)

        return self

    def _transform_dataset(self, X, y):
        """Convert the data set for use by RankSVM prior to the training stage.
        The kernels of the training data are precomputed such that each row or column in the dataset corresponds
        to a rank (object pair) in train_ranks and thus each cell corresponds to a the value Qij in pg.3 of
        Herbrich et al. 1999 (Support Vector Learning for Ordinal Regression). This allows us to enforce the rank-based
        constraints of RankSVM, without having to modify the OneClassSVM algorithm. Additionally, a value of +1 is
        stored as the target class/label for each rank (object pair) in the training set.
        :param train_objects: the objects data to be converted.
        :type train_objects: `pandas.DataFrame`
        :param train_ranks: the pairwise rank data to be converted.
        :type train_ranks: `pandas.DataFrame`
        :param use_feats: a subset of the original features to be used when training (default None). If None, all
            original features are used.
        :type use_feats: list of str or None, optional
        :param progress_window: a GUI object (extending the `tkinter.Toplevel` widget) used to display a
            progress log and progress bar during the experiment execution (default None).
        :type progress_window: :class:`pyplt.gui.experiment.progresswindow.ProgressWindow`, optional
        :param exec_stopper: an abort flag object used to abort the execution before completion (default None).
        :type exec_stopper: :class:`pyplt.util.AbortFlag`, optional
        :return:
            * a tuple containing the precomputed kernel matrix K, the array of target classes/labels y_trans of
              shape (k,) (in this case, all +1s), and the `pandas.DataFrame` of training objects containing only the
              features specified by `use_feats` -- if execution is completed successfully.
            * None -- if aborted before completion by `exec_stopper`.
        :rtype: tuple (size 3)
        """
        
        # matrix-based calculation
        K = self._ranks_kernel_m(X, y)
        y_trans = np.ones(len(y), dtype=int)
        return K, y_trans

    def _ranks_kernel_m(self, X, X_i):
        """Pre-compute the n x n rank-based kernel matrix for the set of n training ranks.
        This method embeds the rank-based constraints that makes this implementation of SVM rank-based, without having
        to modify the algorithm itself. The values of each cell [i, j] in the output matrix corresponds to the value
        Qij in pg.3 of Herbrich et al. 1999 (Support Vector Learning for Ordinal Regression).
        :param X: the training objects.
        :type X: `pandas.DataFrame`
        :param X_i: array i of shape [n, 2] containing a copy the training ranks (i.e., pairs of object IDs).
        :type X_i: `numpy.ndarray`
        :param X_j: array j of shape [n, 2] containing another copy of the training ranks (i.e., pairs of object IDs).
        :type X_j: `numpy.ndarray`
        :return: the resulting kernel matrix K of shape [n, n].
        :rtype: `numpy.ndarray`
        """
        # pref & non feature vectors for rank x_i:
        X_i_prefs = X[X_i[:, 0], :]
        X_i_nons = X[X_i[:, 1], :]

        return np.subtract(np.subtract(np.add(self._kernel_base_m(X_i_prefs, X_i_prefs),
                                              self._kernel_base_m(X_i_nons, X_i_nons)),
                                       self._kernel_base_m(X_i_prefs, X_i_nons)),
                           self._kernel_base_m(X_i_nons, X_i_prefs))

    def _kernel_base_m(self, A, B, reshape_a=False, reshape_b=False):
        """Compute the kernel function (:meth:`self.kernel()`) on each corresponding pair of objects in A and B.
        Internally uses the corresponding kernel functions in `sklearn.metrics.pairwise`.
        :param A: matrix of shape [n_samples_A, n_features] containing the feature vectors of objects in A
        :type A: `numpy.ndarray`
        :param B: matrix of shape [n_samples_B, n_features] containing the feature vectors of objects in B
        :type B: `numpy.ndarray`
        :param reshape_a: specifies whether to reshape input A into an array of shape (1, -1) indicating
            a single sample (default: False).
        :type reshape_a: bool, optional
        :param reshape_b: specifies whether to reshape input B into an array of shape (1, -1) indicating
            a single sample (default: False).
        :type reshape_b: bool, optional
        :return: a matrix of shape [n_samples_A, n_samples_B] containing the float output of the kernel function
            for each AxB object pair.
        :rtype: `numpy.ndarray`
        """
        # sklearn tells us to "Reshape your data either using array.reshape(-1, 1) if your data has a single feature
        # or array.reshape(1, -1) if it contains a single sample."
        # when predicting A=input_object, our case is the latter (single sample).
        # print("A.shape = " + str(A.shape))
        # print("B.shape = " + str(B.shape))
        if reshape_a:
            A = A.reshape(1, -1)
            # print("A.shape after reshape = " + str(A.shape))
        if reshape_b:
            B = B.reshape(1, -1)
            # print("B.shape after reshape = " + str(B.shape))

        # convert gamma 'auto' into None for rbf_kernel
        gamma = self.gamma
        if gamma == 'auto':
            gamma = None

        # calculate the kernel
        if self.kernel == "linear":
            return linear_kernel(A, B)
        elif self.kernel == "poly":
            return polynomial_kernel(A, B, degree=self.degree, gamma=gamma, coef0=0)
            # TODO: make coef0 a user-defined parameter!!!
        elif self.kernel == "rbf":
            return rbf_kernel(A, B, gamma=gamma)
        else:
            raise ValueError("Unknown kernel")
        # TODO: do the same for any other kernels! (e.g. sigmoid)

    def predict(self, X):
        """Predict the output of a given set of input samples by running them through the learned RankSVM model.
        :param input_objects: array of shape [n_samples, n_feats] containing the input data corresponding
            to a set of (test) objects.
        :type input_objects: `numpy.ndarray`
        :param progress_window: a GUI object (extending the `tkinter.Toplevel` widget) used to display a progress
            log and progress bar during the experiment execution (default None).
        :type progress_window: :class:`pyplt.gui.experiment.progresswindow.ProgressWindow`, optional
        :param exec_stopper: an abort flag object used to abort the execution before completion
            (default None).
        :type exec_stopper: :class:`pyplt.util.AbortFlag`, optional
        :return:
            * a list containing the average predicted output resulting from running the learned model using the
              given input objects -- if execution is completed successfully.
            * None -- if aborted before completion by `exec_stopper`.
        :rtype: list of float (size 1)
        """
        sv_idx = self._r_svm.support_  # indices of the support vectors
        sv_coefs = self._r_svm.dual_coef_  # coefficients of the support vectors in the decision function

        # matrix-based calculation
        Alphas_i = sv_coefs[0, :]  # shape = [1, n_sv]
        # only where alpha_i != 0 !!!
        non_zeros = Alphas_i != 0  # these are the indexes of the rows in Alphas_i where the value is !=0
        sv_idx = sv_idx[non_zeros]
        Alphas_i = Alphas_i[non_zeros]  # same as Alphas_i[Alphas_i != 0]
        # reshape Alphas_i from (n_sv, ) to (1, n_sv) so that it works properly with np.matmul() later
        Alphas_i = Alphas_i.reshape(1, -1)

        X_i = self._train_y[sv_idx]  # shape = [n_sv, 2]
        pred_utility = np.matmul(Alphas_i, np.transpose(self._predict_kernel_subtraction_m2(X, X_i)))
        # print("req_utility.shape = " + str(req_utility.shape))
        # final result = [1, n_samples]

        # debug
        # print("number alphas EXCLUDING zeros: " + str(len(Alphas_i)))
        # print("number svs EXCLUDING alpha=zeros: " + str(len(sv_idx)))
        # print("reshpaed Alphas_i.shape = " + str(Alphas_i.shape))
        # print("X_i.shape = " + str(X_i.shape))
        # print("req_utility.shape = " + str(req_utility.shape))
        # print(req_utility)
        # print("summed req_utility.shape = " + str(req_utility.shape))

        return pred_utility.ravel()

    def _predict_kernel_subtraction_m2(self, X, X_i):
        """Compute the subtraction part of deriving the order of the given object with respect to the given support vectors.
        :param X: the objects used to train the SVM.
        :type X: `pandas.DataFrame`
        :param input_object: array of shape [n_samples, n_feats] containing the feature vectors of the
            set of (test) objects to be predicted.
        :type input_object: `numpy.ndarray`
        :param X_i: array of shape [n_support_vectors, 2] containing RankSVM model's support vectors (each in the
            form of a pair of object IDs).
        :type X_i: `numpy.ndarray`
        :return: matrix of shape [n_samples, n_support_vectors] containing the float results of the kernel
            subtraction for each support vector for each sample in the given (test) set of objects.
        :rtype: `numpy.ndarray`
        """
        # get the feature vectors (lists) of the actual objects from each rank (i.e. pair of object ids)
        # pref & non feature vectors for rank x_i:
        X_i_prefs = self._train_X[X_i[:, 0], :]
        X_i_nons = self._train_X[X_i[:, 1], :]
        # print("X_i_prefs.shape = " + str(X_i_prefs.shape))
        # print(X_i_prefs)
        # print("input_object.shape = " + str(input_object.shape))

        # returns shape = [n_rows * n_rows] aka [n_sv, n_sv] usually
        answer = np.subtract(self._kernel_base_m(X, X_i_prefs),
                             self._kernel_base_m(X, X_i_nons))
        # print("answer.shape = " + str(answer.shape))
        return answer

    def score(self, X, y_test):
        """An algorithm-specific approach to testing/validating the model using the given test data.
        :param objects: the objects data that the model was trained on.
        :type objects: `pandas.DataFrame`
        :param test_ranks: the pairwise rank data for the model to be tested/validated on.
        :type test_ranks: `pandas.DataFrame`
        :param use_feats: a subset of the original features to be used during the testing/validation
            process; if None, all original features are used (default None).
        :type use_feats: list of str or None, optional
        :param progress_window: a GUI object (extending the `tkinter.Toplevel` widget) used to display a
            progress log and progress bar during the experiment execution (default None).
        :type progress_window: :class:`pyplt.gui.experiment.progresswindow.ProgressWindow`, optional
        :param exec_stopper: an abort flag object used to abort the execution before completion
            (default None).
        :type exec_stopper: :class:`pyplt.util.AbortFlag`, optional
        :return:
            * the test/validation accuracy of the learned model -- if execution is completed successfully.
            * None -- if aborted before completion by `exec_stopper`.
        :rtype: float
        """
        check_prefs_data(X, y_test)
        
        test_prefs_obj = X[y_test[:, 0], :]
        test_nons_obj = X[y_test[:, 1], :]
        
        # print("test_prefs_obj.shape: " + str(test_prefs_obj.shape))  # shape = [n_test_ranks, n_feats]
        # print("test_nons_obj.shape: " + str(test_nons_obj.shape))  # shape = [n_test_ranks, n_feats]

        prefs_accuracies = self.predict(test_prefs_obj)
        nons_accuracies = self.predict(test_nons_obj)

        total_correct = np.greater(prefs_accuracies, nons_accuracies)
        total_correct = np.sum(total_correct)
        accuracy = float(total_correct) / float(len(y_test)) * 100

        return accuracy

'''
import pandas as pd

model = RankSVM(kernel="linear", debug = True)
X = np.random.rand(10,2)
# Linear utility functions
utility = np.inner([[0.6,0.4]], X).ravel()
print(np.argsort(-utility))

# Transform rating to pairwise preferences
y = utility2prefs(utility)
check_prefs_data(X,y)
order = prefs2order(y)

print(order)

print(prefs2order(y[::2,:]))

model.fit(X, y)
# Predict training
print(model.predict(X))
print(np.argsort(-model.predict(X)))
print(model.score(X, y))

# Example 2
model = RankSVM(kernel="rbf", debug = True)

import pandas as pd
X = pd.DataFrame(dict(feat1 = [3,10,2,1,4,9], feat2 = [83,88,81,91,91,58],
                      feat3 = [9935, 1574, 6742, 9691, 4657, 3924]))
X = X.values
rating = np.array([0.63, 0.51, 0.62, 0.43, 0.24, 0.88])

y = utility2prefs(rating)
check_prefs_data(X,y)
order = prefs2order(y)
print(order)
model.fit(X, y)
# Predict training
print(model.predict(X))
print(np.argsort(-model.predict(X)))
print(model.score(X, y))

'''