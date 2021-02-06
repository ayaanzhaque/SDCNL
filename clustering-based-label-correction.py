# Clustering Algorithms for Label Correction 

# This notebook contains the code to load and initialize all the clustering algorithms with the proper hyperparameters set.

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap.umap_ as umap

from tensorflow import keras
from tensorflow.keras import layers

from keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from keras.models import Model

# For subspace clustering
import warnings
import math
import progressbar
# import progressbar2
import spams
import time

from scipy import sparse
from scipy.linalg import orth
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import supervised
from sklearn import cluster
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.decomposition import sparse_encode
from sklearn.linear_model import orthogonal_mp
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state, check_array, check_symmetric
from sklearn.preprocessing import normalize

from kymatio import Scattering2D
from sklearn import cluster
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import normalize

features = pd.read_csv('features.csv', delimiter=',') # load the features after creating them
labels = pd.read_csv("../data/combined-set.csv")["is_suicide"]

# Dimensionality-Reduction Algorithms
 
# This section contains the algorithms used for dimensionality reduction with correct hyperparameters. The algorithms used are PCA, Deep Autoencoder, and UMAP.

# PCA
components = 2

pca_model = PCA(n_components = components)
low_dim_features = pca_model.fit_transform(features)


# Deep Autoencoder
encoding_dim = 2
input_size = 512 # based on embeddings

input_df = Input(shape=(input_size,))
encoded = Dense(encoding_dim, activation = 'relu')(input_df)
decoded = Dense(512, activation='sigmoid')(encoded)
autoencoder = Model(input_df, decoded)

# intermediate result
encoder = Model(input_df, encoded)

autoencoder.compile(optimizer='adam', loss='mse') # Mean-Squared Error Loss is used

autoencoder.summary()

autoencoder.fit(features, features,
                epochs=3000,
                batch_size=64,
                shuffle=True,
                validation_data=(features, features))

low_dim_features = encoder.predict(features)


#UMAP
reducer = umap.UMAP(
        n_neighbors=45,
        min_dist=0.7,
        n_components=2,
        metric='manhattan'
    )

low_dim_features = reducer.fit_transform(features)


# Clustering Algorithms
# This section contains the algorithms used for clustering with correct hyperparameters. The algorithms used are GMM, KMeans, and Subspace Clustering.

# GMM
gmm = GaussianMixture(n_components=2, covariance_type='full').fit(low_dim_features)
predictions = gmm.predict(low_dim_features)
probs = gmm.predict_proba(low_dim_features)


# KMeans
kmeans = KMeans(n_clusters=2, init='k-means++', n_init=100).fit(low_dim_features)
predictions = kmeans.predict(low_dim_features)


# Subspace Clustering
# The code for Subspace Clustering is adapted from here: The code is from here: https://github.com/ChongYou/subspace-clustering
# run'pip install spams'
# run'pip install kymatio'

class SelfRepresentation(BaseEstimator, ClusterMixin):
    """Base class for self-representation based subspace clustering.
    Parameters
    -----------
    n_clusters : integer, optional, default: 8
        Number of clusters in the dataset.
    affinity : string, optional, 'symmetrize' or 'nearest_neighbors', default 'symmetrize'
        The strategy for constructing affinity_matrix_ from representation_matrix_.
        If ``symmetrize``, then affinity_matrix_ is set to be
    		|representation_matrix_| + |representation_matrix_|^T.
		If ``nearest_neighbors``, then the affinity_matrix_ is the k nearest
		    neighbor graph for the rows of representation_matrix_
    random_state : int, RandomState instance or None, optional, default: None
        This is the random_state parameter for k-means. 
    n_init : int, optional, default: 10
        This is the n_init parameter for k-means. 
    n_jobs : int, optional, default: 1
        The number of parallel jobs to run.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
    Attributes
    ----------
    representation_matrix_ : array-like, shape (n_samples, n_samples)
        Self-representation matrix. Available only if after calling
        ``fit`` or ``fit_self_representation``.
    labels_ :
        Labels of each point. Available only if after calling ``fit``.
    """

    def __init__(self, n_clusters=8, affinity='symmetrize', random_state=None, n_init=20, n_jobs=1):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.random_state = random_state
        self.n_init = n_init
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Compute representation matrix, then apply spectral clustering
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
        """
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'], dtype=np.float64)
        time_base = time.time()
        
        self._self_representation(X)
        self.timer_self_representation_ = time.time() - time_base
        
        self._representation_to_affinity()
        self._spectral_clustering()
        self.timer_time_ = time.time() - time_base

        return self
	
    def fit_self_representation(self, X, y=None):
        """Compute representation matrix without apply spectral clustering.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
        """
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'], dtype=np.float64)
        time_base = time.time()
        
        self._self_representation(X)
        self.timer_self_representation_ = time.time() - time_base
        
        return self

    def _representation_to_affinity(self):
        """Compute affinity matrix from representation matrix.
        """
        normalized_representation_matrix_ = normalize(self.representation_matrix_, 'l2')
        if self.affinity == 'symmetrize':
            self.affinity_matrix_ = 0.5 * (np.absolute(normalized_representation_matrix_) + np.absolute(normalized_representation_matrix_.T))
        elif self.affinity == 'nearest_neighbors':
            neighbors_graph = kneighbors_graph(normalized_representation_matrix_, 3, 
		                                       mode='connectivity', include_self=False)
            self.affinity_matrix_ = 0.5 * (neighbors_graph + neighbors_graph.T)

    def _spectral_clustering(self):
        affinity_matrix_ = check_symmetric(self.affinity_matrix_)
        random_state = check_random_state(self.random_state)
        
        laplacian = sparse.csgraph.laplacian(affinity_matrix_, normed=True)
        _, vec = sparse.linalg.eigsh(sparse.identity(laplacian.shape[0]) - laplacian, 
                                     k=self.n_clusters, sigma=None, which='LA')
        embedding = normalize(vec)
        _, self.labels_, _ = cluster.k_means(embedding, self.n_clusters, 
                                             random_state=random_state, n_init=self.n_init)


def active_support_elastic_net(X, y, alpha, tau=1.0, algorithm='spams', support_init='knn', 
                               support_size=100, maxiter=40):
    """An active support based algorithm for solving the elastic net optimization problem
        min_{c} tau ||c||_1 + (1-tau)/2 ||c||_2^2 + alpha / 2 ||y - c X ||_2^2.
		
    Parameters
    -----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (1, n_features)
    alpha : float
    tau : float, default 1.0
    algorithm : string, default ``spams``
        Algorithm for computing solving the subproblems. Either lasso_lars or lasso_cd or spams
        (installation of spams package is required).
        Note: ``lasso_lars`` and ``lasso_cd`` only support tau = 1.
    support_init: string, default ``knn``
        This determines how the active support is initialized.
        It can be either ``knn`` or ``L2``.
    support_size: int, default 100
        This determines the size of the working set.
        A small support_size decreases the runtime per iteration while increase the number of iterations.
    maxiter: int default 40
        Termination condition for active support update.
		
    Returns
    -------
    c : shape n_samples
        The optimal solution to the optimization problem.
	"""
    n_samples = X.shape[0]

    if n_samples <= support_size:  # skip active support search for small scale data
        supp = np.arange(n_samples, dtype=int)  # this results in the following iteration to converge in 1 iteration
    else:    
        if support_init == 'L2':
            L2sol = np.linalg.solve(np.identity(y.shape[1]) * alpha + np.dot(X.T, X), y.T)
            c0 = np.dot(X, L2sol)[:, 0]
            supp = np.argpartition(-np.abs(c0), support_size)[0:support_size]
        elif support_init == 'knn':
            supp = np.argpartition(-np.abs(np.dot(y, X.T)[0]), support_size)[0:support_size]

    curr_obj = float("inf")
    for _ in range(maxiter):
        Xs = X[supp, :]
        if algorithm == 'spams':
            cs = spams.lasso(np.asfortranarray(y.T), D=np.asfortranarray(Xs.T), 
                             lambda1=tau*alpha, lambda2=(1.0-tau)*alpha)
            cs = np.asarray(cs.todense()).T
        else:
            cs = sparse_encode(y, Xs, algorithm=algorithm, alpha=alpha)
      
        delta = (y - np.dot(cs, Xs)) / alpha
		
        obj = tau * np.sum(np.abs(cs[0])) + (1.0 - tau)/2.0 * np.sum(np.power(cs[0], 2.0)) + alpha/2.0 * np.sum(np.power(delta, 2.0))
        if curr_obj - obj < 1.0e-10 * curr_obj:
            break
        curr_obj = obj
			
        coherence = np.abs(np.dot(delta, X.T))[0]
        coherence[supp] = 0
        addedsupp = np.nonzero(coherence > tau + 1.0e-10)[0]
        
        if addedsupp.size == 0:  # converged
            break

        # Find the set of nonzero entries of cs.
        activesupp = supp[np.abs(cs[0]) > 1.0e-10]  
        
        if activesupp.size > 0.8 * support_size:  # this suggests that support_size is too small and needs to be increased
            support_size = min([round(max([activesupp.size, support_size]) * 1.1), n_samples])
        
        if addedsupp.size + activesupp.size > support_size:
            ord = np.argpartition(-coherence[addedsupp], support_size - activesupp.size)[0:support_size - activesupp.size]
            addedsupp = addedsupp[ord]
        
        supp = np.concatenate([activesupp, addedsupp])
    
    c = np.zeros(n_samples)
    c[supp] = cs
    return c

  
def elastic_net_subspace_clustering(X, gamma=50.0, gamma_nz=True, tau=1.0, algorithm='lasso_lars', 
                                    active_support=True, active_support_params=None, n_nonzero=50):
    """Elastic net subspace clustering (EnSC) [1]. 
    Compute self-representation matrix C from solving the following optimization problem
    min_{c_j} tau ||c_j||_1 + (1-tau)/2 ||c_j||_2^2 + alpha / 2 ||x_j - c_j X ||_2^2 s.t. c_jj = 0,
    where c_j and x_j are the j-th rows of C and X, respectively.
	
	Parameter ``algorithm`` specifies the algorithm for solving the optimization problem.
	``lasso_lars`` and ``lasso_cd`` are algorithms implemented in sklearn, 
    ``spams`` refers to the same algorithm as ``lasso_lars`` but is implemented in 
	spams package available at http://spams-devel.gforge.inria.fr/ (installation required)
    In principle, all three algorithms give the same result.	
    For large scale data (e.g. with > 5000 data points), use any of these algorithms in
	conjunction with ``active_support=True``. It adopts an efficient active support 
	strategy that solves the optimization problem by breaking it into a sequence of 
    small scale optimization problems as described in [1].
    If tau = 1.0, the method reduces to sparse subspace clustering with basis pursuit (SSC-BP) [2].
    If tau = 0.0, the method reduces to least squares regression (LSR) [3].
	Note: ``lasso_lars`` and ``lasso_cd`` only support tau = 1.
    Parameters
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data to be clustered
    gamma : float
    gamma_nz : boolean, default True
        gamma and gamma_nz together determines the parameter alpha. When ``gamma_nz = False``, 
        alpha = gamma. When ``gamma_nz = True``, then alpha = gamma * alpha0, where alpha0 is 
        the largest number such that the solution to the optimization problem with alpha = alpha0
		is the zero vector (see Proposition 1 in [1]). Therefore, when ``gamma_nz = True``, gamma
        should be a value greater than 1.0. A good choice is typically in the range [5, 500].	
    tau : float, default 1.0
        Parameter for elastic net penalty term. 
        When tau = 1.0, the method reduces to sparse subspace clustering with basis pursuit (SSC-BP) [2].
        When tau = 0.0, the method reduces to least squares regression (LSR) [3].
    algorithm : string, default ``lasso_lars``
        Algorithm for computing the representation. Either lasso_lars or lasso_cd or spams 
        (installation of spams package is required).
        Note: ``lasso_lars`` and ``lasso_cd`` only support tau = 1.
    n_nonzero : int, default 50
        This is an upper bound on the number of nonzero entries of each representation vector. 
        If there are more than n_nonzero nonzero entries,  only the top n_nonzero number of
        entries with largest absolute value are kept.
    active_support: boolean, default True
        Set to True to use the active support algorithm in [1] for solving the optimization problem.
        This should significantly reduce the running time when n_samples is large.
    active_support_params: dictionary of string to any, optional
        Parameters (keyword arguments) and values for the active support algorithm. It may be
        used to set the parameters ``support_init``, ``support_size`` and ``maxiter``, see
        ``active_support_elastic_net`` for details. 
        Example: active_support_params={'support_size':50, 'maxiter':100}
        Ignored when ``active_support=False``
	
    Returns
    -------
    representation_matrix_ : csr matrix, shape: n_samples by n_samples
        The self-representation matrix.
	
    References
    -----------	
	[1] C. You, C.-G. Li, D. Robinson, R. Vidal, Oracle Based Active Set Algorithm for Scalable Elastic Net Subspace Clustering, CVPR 2016
	[2] E. Elhaifar, R. Vidal, Sparse Subspace Clustering: Algorithm, Theory, and Applications, TPAMI 2013
    [3] C. Lu, et al. Robust and efficient subspace segmentation via least squares regression, ECCV 2012
    """
    if algorithm in ('lasso_lars', 'lasso_cd') and tau < 1.0 - 1.0e-10:  
        warnings.warn('algorithm {} cannot handle tau smaller than 1. Using tau = 1'.format(algorithm))
        tau = 1.0
		
    if active_support == True and active_support_params == None:
        active_support_params = {}

    n_samples = X.shape[0]
    rows = np.zeros(n_samples * n_nonzero)
    cols = np.zeros(n_samples * n_nonzero)
    vals = np.zeros(n_samples * n_nonzero)
    curr_pos = 0
 
#     for i in progressbar.progressbar(range(n_samples)):
    for i in range(n_samples):
    #    if i % 1000 == 999:
    #        print('SSC: sparse coding finished {i} in {n_samples}'.format(i=i, n_samples=n_samples))
        y = X[i, :].copy().reshape(1, -1)
        X[i, :] = 0
        
        if algorithm in ('lasso_lars', 'lasso_cd', 'spams'):
            if gamma_nz == True:
                coh = np.delete(np.absolute(np.dot(X, y.T)), i)
                alpha0 = np.amax(coh) / tau  # value for which the solution is zero
                alpha = alpha0 / gamma
            else:
                alpha = 1.0 / gamma

            if active_support == True:
                c = active_support_elastic_net(X, y, alpha, tau, algorithm, **active_support_params)
            else:
                if algorithm == 'spams':
                    c = spams.lasso(np.asfortranarray(y.T), D=np.asfortranarray(X.T), 
                                    lambda1=tau * alpha, lambda2=(1.0-tau) * alpha)
                    c = np.asarray(c.todense()).T[0]
                else:
                    c = sparse_encode(y, X, algorithm=algorithm, alpha=alpha)[0]
        else:
          warnings.warn("algorithm {} not found".format(algorithm))
	    	  
        index = np.flatnonzero(c)
        if index.size > n_nonzero:
        #  warnings.warn("The number of nonzero entries in sparse subspace clustering exceeds n_nonzero")
          index = index[np.argsort(-np.absolute(c[index]))[0:n_nonzero]]
        rows[curr_pos:curr_pos + len(index)] = i
        cols[curr_pos:curr_pos + len(index)] = index
        vals[curr_pos:curr_pos + len(index)] = c[index]
        curr_pos += len(index)
        
        X[i, :] = y

#   affinity = sparse.csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples)) + sparse.csr_matrix((vals, (cols, rows)), shape=(n_samples, n_samples))
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))


class ElasticNetSubspaceClustering(SelfRepresentation):
    """Elastic net subspace clustering (EnSC) [1]. 
    This is a self-representation based subspace clustering method that computes
    the self-representation matrix C via solving the following elastic net problem
    min_{c_j} tau ||c_j||_1 + (1-tau)/2 ||c_j||_2^2 + alpha / 2 ||x_j - c_j X ||_2^2 s.t. c_jj = 0,
    where c_j and x_j are the j-th rows of C and X, respectively.
	
	Parameter ``algorithm`` specifies the algorithm for solving the optimization problem.
	``lasso_lars`` and ``lasso_cd`` are algorithms implemented in sklearn, 
    ``spams`` refers to the same algorithm as ``lasso_lars`` but is implemented in 
	spams package available at http://spams-devel.gforge.inria.fr/ (installation required)
    In principle, all three algorithms give the same result.	
    For large scale data (e.g. with > 5000 data points), use any of these algorithms in
	conjunction with ``active_support=True``. It adopts an efficient active support 
	strategy that solves the optimization problem by breaking it into a sequence of 
    small scale optimization problems as described in [1].
    If tau = 1.0, the method reduces to sparse subspace clustering with basis pursuit (SSC-BP) [2].
    If tau = 0.0, the method reduces to least squares regression (LSR) [3].
	Note: ``lasso_lars`` and ``lasso_cd`` only support tau = 1.
    Parameters
    -----------
    n_clusters : integer, optional, default: 8
        Number of clusters in the dataset.
    random_state : int, RandomState instance or None, optional, default: None
        This is the random_state parameter for k-means.
    affinity : string, optional, 'symmetrize' or 'nearest_neighbors', default 'symmetrize'
        The strategy for constructing affinity_matrix_ from representation_matrix_.		
    n_init : int, optional, default: 10
        This is the n_init parameter for k-means. 
    gamma : float
    gamma_nz : boolean, default True
        gamma and gamma_nz together determines the parameter alpha. If gamma_nz = False, then
        alpha = gamma. If gamma_nz = True, then alpha = gamma * alpha0, where alpha0 is the largest 
        number that the solution to the optimization problem with alpha = alpha0 is zero vector
        (see Proposition 1 in [1]). 
    tau : float, default 1.0
        Parameter for elastic net penalty term. 
        When tau = 1.0, the method reduces to sparse subspace clustering with basis pursuit (SSC-BP) [2].
        When tau = 0.0, the method reduces to least squares regression (LSR) [3].
    algorithm : string, default ``lasso_lars``
        Algorithm for computing the representation. Either lasso_lars or lasso_cd or spams 
        (installation of spams package is required).
        Note: ``lasso_lars`` and ``lasso_cd`` only support tau = 1.
    active_support: boolean, default True
        Set to True to use the active support algorithm in [1] for solving the optimization problem.
        This should significantly reduce the running time when n_samples is large.
    active_support_params: dictionary of string to any, optional
        Parameters (keyword arguments) and values for the active support algorithm. It may be
        used to set the parameters ``support_init``, ``support_size`` and ``maxiter``, see
        ``active_support_elastic_net`` for details. 
        Example: active_support_params={'support_size':50, 'maxiter':100}
        Ignored when ``active_support=False``
    n_nonzero : int, default 50
        This is an upper bound on the number of nonzero entries of each representation vector. 
        If there are more than n_nonzero nonzero entries,  only the top n_nonzero number of
        entries with largest absolute value are kept.
		
    Attributes
    ----------
    representation_matrix_ : array-like, shape (n_samples, n_samples)
        Self-representation matrix. Available only if after calling
        ``fit`` or ``fit_self_representation``.
    labels_ :
        Labels of each point. Available only if after calling ``fit``
    References
    -----------	
	[1] C. You, C.-G. Li, D. Robinson, R. Vidal, Oracle Based Active Set Algorithm for Scalable Elastic Net Subspace Clustering, CVPR 2016
	[2] E. Elhaifar, R. Vidal, Sparse Subspace Clustering: Algorithm, Theory, and Applications, TPAMI 2013
    [3] C. Lu, et al. Robust and efficient subspace segmentation via least squares regression, ECCV 2012
    """
    def __init__(self, n_clusters=8, affinity='symmetrize', random_state=None, n_init=20, n_jobs=1, gamma=50.0, gamma_nz=True, tau=1.0, 
                 algorithm='lasso_lars', active_support=True, active_support_params=None, n_nonzero=50):
        self.gamma = gamma
        self.gamma_nz = gamma_nz
        self.tau = tau
        self.algorithm = algorithm
        self.active_support = active_support
        self.active_support_params = active_support_params
        self.n_nonzero = n_nonzero

        SelfRepresentation.__init__(self, n_clusters, affinity, random_state, n_init, n_jobs)
    
    def _self_representation(self, X):
        self.representation_matrix_ = elastic_net_subspace_clustering(X, self.gamma, self.gamma_nz, 
                                                                      self.tau, self.algorithm, 
		                                                              self.active_support, self.active_support_params, 
		                                                              self.n_nonzero)
					

def sparse_subspace_clustering_orthogonal_matching_pursuit(X, n_nonzero=10, thr=1.0e-6):
    """Sparse subspace clustering by orthogonal matching pursuit (SSC-OMP)
    Compute self-representation matrix C by solving the following optimization problem
    min_{c_j} ||x_j - c_j X ||_2^2 s.t. c_jj = 0, ||c_j||_0 <= n_nonzero
    via OMP, where c_j and x_j are the j-th rows of C and X, respectively
    Parameters
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data to be clustered
    n_nonzero : int, default 10
        Termination condition for omp.
    thr : float, default 1.0e-5
        Termination condition for omp.	
    Returns
    -------
    representation_matrix_ : csr matrix, shape: n_samples by n_samples
        The self-representation matrix.
	
    References
    -----------			
    C. You, D. Robinson, R. Vidal, Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
    """	
    n_samples = X.shape[0]
    rows = np.zeros(n_samples * n_nonzero, dtype = int)
    cols = np.zeros(n_samples * n_nonzero, dtype = int)
    vals = np.zeros(n_samples * n_nonzero)
    curr_pos = 0

#     for i in progressbar.progressbar(range(n_samples)):
    for i in range(n_samples):
        residual = X[i, :].copy()  # initialize residual
        supp = np.empty(shape=(0), dtype = int)  # initialize support
        residual_norm_thr = np.linalg.norm(X[i, :]) * thr
        for t in range(n_nonzero):  # for each iteration of OMP  
            # compute coherence between residuals and X     
            coherence = abs( np.matmul(residual, X.T) )
            coherence[i] = 0.0
            # update support
            supp = np.append(supp, np.argmax(coherence))
            # compute coefficients
            c = np.linalg.lstsq( X[supp, :].T, X[i, :].T, rcond=None)[0]
            # compute residual
            residual = X[i, :] - np.matmul(c.T, X[supp, :])
            # check termination
            if np.sum(residual **2) < residual_norm_thr:
                break

        rows[curr_pos:curr_pos + len(supp)] = i
        cols[curr_pos:curr_pos + len(supp)] = supp
        vals[curr_pos:curr_pos + len(supp)] = c
        curr_pos += len(supp)

#   affinity = sparse.csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples)) + sparse.csr_matrix((vals, (cols, rows)), shape=(n_samples, n_samples))
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))
					

class SparseSubspaceClusteringOMP(SelfRepresentation):
    """Sparse subspace clustering by orthogonal matching pursuit (SSC-OMP). 
    This is a self-representation based subspace clustering method that computes
    the self-representation matrix C via solving the following problem
    min_{c_j} ||x_j - c_j X ||_2^2 s.t. c_jj = 0, ||c_j||_0 <= n_nonzero
    via OMP, where c_j and x_j are the j-th rows of C and X, respectively
    Parameters
    -----------
    n_clusters : integer, optional, default: 8
        Number of clusters in the dataset.
    affinity : string, optional, 'symmetrize' or 'nearest_neighbors', default 'symmetrize'
        The strategy for constructing affinity_matrix_ from representation_matrix_.
    random_state : int, RandomState instance or None, optional, default: None
        This is the random_state parameter for k-means. 
    n_init : int, optional, default: 10
        This is the n_init parameter for k-means. 
    n_nonzero : int, default 10
        Termination condition for omp.
    thr : float, default 1.0e-5
        Termination condition for omp.	
	
    Attributes
    ----------
    representation_matrix_ : array-like, shape (n_samples, n_samples)
        Self-representation matrix. Available only if after calling
        ``fit`` or ``fit_self_representation``.
    labels_ :
        Labels of each point. Available only if after calling ``fit``
    References
    -----------	
    C. You, D. Robinson, R. Vidal, Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
    """
    def __init__(self, n_clusters=8, affinity='symmetrize', random_state=None, n_init=10, n_jobs=1, n_nonzero=10, thr=1.0e-6):
        self.n_nonzero = n_nonzero
        self.thr = thr
        SelfRepresentation.__init__(self, n_clusters, affinity, random_state, n_init, n_jobs)
    
    def _self_representation(self, X):
        self.representation_matrix_ = sparse_subspace_clustering_orthogonal_matching_pursuit(X, self.n_nonzero, self.thr)


def least_squares_subspace_clustering(X, gamma=10.0, exclude_self=False):
    """Least squares subspace clustering. 
    Compute self-representation matrix C by solving the following optimization problem
        min_{c_j} ||c_j||_2^2 + gamma ||x_j - X c_j||_2^2 s.t. c_jj = 0  (*)
    Parameters
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data to be clustered
    gamma : float
        Parameter on noise regularization term
    exclude_self : boolean, default False
        When False, solves (*) without the constraint c_jj = 0
		
    Returns
    -------
    representation_matrix_ : shape n_samples by n_samples
        The self-representation matrix.
		
    References
    -----------	
    C. Lu, et al. Robust and efficient subspace segmentation via least squares regression, ECCV 2012
    """
    n_samples, n_features = X.shape
  
    if exclude_self == False:
        if n_samples < n_features:
            gram = np.matmul(X, X.T)
            return np.linalg.solve(gram + np.eye(n_sample) / gamma, gram).T
        else:
            tmp = np.linalg.solve(np.matmul(X.T, X) + np.eye(n_features) / gamma, X.T)
            return np.matmul(X, tmp).T
    else:
        if n_samples < n_features:
            D = np.linalg.solve(np.matmul(X, X.T) + np.eye(n_sample) / gamma, np.eye(n_sample))  
            # see Theorem 6 in https://arxiv.org/pdf/1404.6736.pdf
        else:
            tmp = np.linalg.solve(np.matmul(X.T, X) + np.eye(n_features) / gamma, X.T)
            D = eye(n_samples) - np.matmul(X, tmp)
        D = D / D.diagonal()[None,:]
        np.fill_diagonal(D, 0.0)
        return -1.0 * D.T


class LeastSquaresSubspaceClustering(SelfRepresentation):
    """Least squares subspace clustering.
	
	Parameters
    -----------
    n_clusters : integer, optional, default: 8
        Number of clusters in the dataset.
    affinity : string, optional, default 'symmetrize'
        This may be either 'symmetrize' or 'nearest_neighbors'.
    random_state : int, RandomState instance or None, optional, default: None
        This is the random_state parameter for k-means. 
    n_init : int, optional, default: 10
        This is the n_init parameter for k-means. 
    gamma : float
        Parameter on noise regularization term
    exclude_self : boolean, default False
        When False, solves (*) without the constraint c_jj = 0
	
    Attributes
    ----------
    representation_matrix_ : array-like, shape (n_samples, n_samples)
        Self-representation matrix. Available only if after calling
        ``fit`` or ``fit_self_representation``.
    labels_ :
        Labels of each point. Available only if after calling ``fit``
		
    References
    -----------	
    C. Lu, et al. Robust and efficient subspace segmentation via least squares regression, ECCV 2012
    """
    def __init__(self, n_clusters=8, affinity='symmetrize', random_state=None, n_init=None, n_jobs=1, gamma=10.0, exclude_self=False):
        self.gamma = gamma
        self.exclude_self = exclude_self
        SelfRepresentation.__init__(self, n_clusters, affinity, random_state, n_init, n_jobs)
    
    def _self_representation(self, X):
        self.representation_matrix_ = least_squares_subspace_clustering(X, self.gamma, self.exclude_self)

def dim_reduction(X, dim):
    """Dimension reduction by principal component analysis
		Let X^T = U S V^T be the SVD of X^T in which the singular values are
	in ascending order. The output Xp^T is the last `dim` rows of S * V^T.
  
    Parameters
    -----------
    X : array-like, shape (n_samples, n_features)
    dim: int
        Target dimension. 
		
    Returns
    -------
    Xp : shape (n_samples, dim)
        Dimension reduced data
	"""
    if dim == 0:
        return X

    w, v = np.linalg.eigh(X.T @ X)
  
    return X @ v[:, -dim:]

def clustering_accuracy(labels_true, labels_pred):
    """Clustering Accuracy between two clusterings.
    Clustering Accuracy is a measure of the similarity between two labels of
    the same data. Assume that both labels_true and labels_pred contain n 
    distinct labels. Clustering Accuracy is the maximum accuracy over all
    possible permutations of the labels, i.e.
    \max_{\sigma} \sum_i labels_true[i] == \sigma(labels_pred[i])
    where \sigma is a mapping from the set of unique labels of labels_pred to
    the set of unique labels of labels_true. Clustering accuracy is one if 
    and only if there is a permutation of the labels such that there is an
    exact match
    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.
    This metric is furthermore symmetric: switching ``label_true`` with
    ``label_pred`` will return the same score value. This can be useful to
    measure the agreement of two independent label assignments strategies
    on the same dataset when the real ground truth is not known.
    
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
    	A clustering of the data into disjoint subsets.
    labels_pred : array, shape = [n_samples]
    	A clustering of the data into disjoint subsets.
    
    Returns
    -------
    accuracy : float
       return clustering accuracy in the range of [0, 1]
    """
    labels_true, labels_pred = supervised.check_clusterings(labels_true, labels_pred)
    # value = supervised.contingency_matrix(labels_true, labels_pred, sparse=False)
    value = supervised.contingency_matrix(labels_true, labels_pred)
    [r, c] = linear_sum_assignment(-value)
    return value[r, c].sum() / len(labels_true)
  
 
def self_representation_loss(labels_true, representation_matrix):
    """Evaluation of self-representation error for self-expressive subspace clustering methods
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
    	A clustering of the data into disjoint subsets.
    representation_matrix : array, shape = [n_samples, n_samples]
    	Each row is a representation vector
    
    Returns
    -------
    loss : float
       return self_representation_loss in the range of [0, 1]
    """
    n_samples = labels_true.shape[0]
    loss = 0.0
    for i in range(n_samples):
        representation_vec = np.abs(representation_matrix[i, :])
        label = labels_true[i]
        loss += np.sum(representation_vec[labels_true != label]) / np.sum(representation_vec)
    
    return loss / n_samples

def gen_union_of_subspaces(ambient_dim, subspace_dim, num_subspaces, num_points_per_subspace, noise_level=0.0):
    """This funtion generates a union of subspaces under random model, i.e., 
    subspaces are independently and uniformly distributed in the ambient space,
    data points are independently and uniformly distributed on the unit sphere of each subspace
    Parameters
    -----------
    ambient_dim : int
        Dimention of the ambient space
    subspace_dim : int
        Dimension of each subspace (all subspaces have the same dimension)
    num_subspaces : int
        Number of subspaces to be generated
    num_points_per_subspace : int
        Number of data points from each of the subspaces
    noise_level : float
        Amount of Gaussian noise on data
		
    Returns
    -------
    data : shape (num_subspaces * num_points_per_subspace) by ambient_dim
        Data matrix containing points drawn from a union of subspaces as its rows
    label : shape (num_subspaces * num_points_per_subspace)
        Membership of each data point to the subspace it lies in
    """

    data = np.empty((num_points_per_subspace* num_subspaces, ambient_dim))
    label = np.empty(num_points_per_subspace * num_subspaces, dtype=int)
  
    for i in range(num_subspaces):
        basis = np.random.normal(size=(ambient_dim, subspace_dim))
        basis = orth(basis)
        coeff = np.random.normal(size=(subspace_dim, num_points_per_subspace))
        coeff = normalize(coeff, norm='l2', axis=0, copy=False)
        data_per_subspace = np.matmul(basis, coeff).T

        base_index = i*num_points_per_subspace
        data[(0+base_index):(num_points_per_subspace+base_index), :] = data_per_subspace
        label[0+base_index:num_points_per_subspace+base_index,] = i

    data += np.random.normal(size=(num_points_per_subspace * num_subspaces, ambient_dim)) * noise_level
  
    return data, label

# Run Spectral Clsutering

model_spectral = cluster.SpectralClustering(n_clusters=2,affinity='nearest_neighbors',n_neighbors=2, random_state=64)  # spectral clustering as baseline
# no dimensionality reduction is required for Subspace clustering
model_spectral.fit(features)
predictions = model_spectral.labels_
