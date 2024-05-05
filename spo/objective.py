import numpy as np
import numba as nb
from numba.experimental import jitclass
from .utils import np_type_f, np_type_i, nb_type_f, nb_type_i, FUNC_LOG, FUNC_EXP


################################################################################################
#
# Utility numba functions used in the Objective class 
# (refer to the latter for parameters annotations)
#
################################################################################################

@nb.jit(nb_type_f(nb_type_f[:,::1], nb_type_f[::1], nb_type_f), nopython=True, fastmath=True)
def _comp_val_LOG(X, w, eta):
    return - np.mean(np.log(np.dot(X, w) + eta))

@nb.jit(nb_type_f(nb_type_f[::1], nb_type_f), nopython=True, fastmath=True)
def _comp_conj_LOG(theta, eta):
    # 此处实际传输过来的为 lambda * theta
    n = theta.shape[0]
    return np.mean(np.log(n * theta) + 1 - n * eta * theta)

@nb.jit(nb_type_f[::1](nb_type_f[:,::1], nb_type_f[::1], nb_type_f), nopython=True, fastmath=True)
def _comp_dual_var_LOG(X, w, eta):
    return nb_type_f(1.0) / (np.dot(X, w) + eta) / X.shape[0]

@nb.jit(nb_type_f[::1](nb_type_f[::1]), nopython=True, fastmath=True)
def _comp_dual_neg_hess_LOG(theta):
    # 即为对上式求导，但注意上式中的传入的theta为lambda*\theta，而在此处传输的实际则为\theta，注意相关处理
    # 同时注意此处求得为负的hessian矩阵
    return 1. / theta**2 / theta.shape[0]


@nb.jit(nb_type_f(nb_type_f[:,::1], nb_type_f[::1], nb_type_f, nb_type_f), nopython=True, fastmath=True)
def _comp_val_EXP(X, w, eta, a):
    return -nb_type_f(1.) + np.mean(np.exp(-(a * np.dot(X, w) + eta)))

@nb.jit(nb_type_f(nb_type_f[::1], nb_type_f, nb_type_f), nopython=True, fastmath=True)
def _comp_conj_EXP(theta, eta, a):
    n = theta.shape[0]
    return - n * np.mean(theta * (np.log(n * theta / a) - 1  + eta)) / a - 1

@nb.jit(nb_type_f[::1](nb_type_f[:,::1], nb_type_f[::1], nb_type_f, nb_type_f), nopython=True, fastmath=True)
def _comp_dual_var_EXP(X, w, eta, a):
    return a * np.exp(- (a * np.dot(X, w) + eta)) / X.shape[0]

@nb.jit(nb_type_f[::1](nb_type_f[::1], nb_type_f, nb_type_f), nopython=True, fastmath=True)
def _comp_dual_neg_hess_EXP(theta, a, lam):
    return lam / a / theta


@nb.jit(nb_type_f[::1](nb_type_f[:,::1], nb_type_f[::1]), nopython=True, fastmath=True)
def _comp_grad(X, theta):
    return - np.dot(X.T, theta)

@nb.jit(nb_type_f[::1](nb_type_f[::1]), nopython=True, fastmath=True)
def _proj(w):
    c = np.sum(w)
    if c>0.:
        w[:] = w / c
    return w

spec = [
    ('eta', nb_type_f),
    ('a', nb_type_f),
    ('L_u', nb_type_f),
    ('L_dH', nb_type_f),
    ('func', nb.int32),
]



################################################################################################
#
# Objective class of utility functions
#
################################################################################################
@jitclass(spec)
class Objective(object):
    '''Sample average utility function

    f(w) = 1/n * sum_{j=1}^n u(z_j),   z_j = w^T X_j
    FUNC_LOG :        
        u(x) = log(eta+x)
    FUNC_EXP :
        u(x) = 1 - exp(-(a*x+eta))
    '''
    def __init__(self, n_samples, eta, func=FUNC_LOG, a=1.):
        '''Initialization.

        Parameters
        ----------
        n_samples : int
            The number of samples.
        eta : float
            The offset parameter.
        func : int, optional
            The utility function: 0 - log, 1 - exp.
        a : float
            The scale parameter for exponential utility.
        '''
        self.func = func        
        n_samples = n_samples
        if self.func == FUNC_LOG:
            self.a = 0.
            self.eta = eta
            self.L_u = 1./self.eta
            self.L_dH = 1./self.eta**2/n_samples
        else:
            self.a = a
            self.eta = eta
            self.L_u = self.a/np.exp(self.eta)
            self.L_dH = self.a**2/n_samples/np.exp(self.eta)
    

    def val(self, X, w):
        '''Compute the objective value.

        Parameters
        ----------        
        X : array, shape (n_samples, n_features)
            Training data of growth rates.
        w : array, shape (n_features, )
            The allocation vector.

        Returns
        -------
        The objective value.
        '''
        if self.func == FUNC_LOG:
            return _comp_val_LOG(X, w, self.eta)
        elif self.func == FUNC_EXP:
            return _comp_val_EXP(X, w, self.eta, self.a)
    
    
    def grad(self, X, w):
        '''Compute the gradient.

        Parameters
        ----------        
        X : array, shape (n_samples, n_features)
            Training data of growth rates.
        w : array, shape (n_features, )
            The allocation vector.

        Returns
        -------
        The gradient evaluated at w.
        '''
        theta = self.comp_dual_var(w, X)
        return _comp_grad(X, theta)
    
    
    def conj(self, theta):
        '''Compute the value of conjugate function:
        F*(theta) = 1/n *  sum_{j=1}^n f*(n * theta_j).

        Parameters
        ----------
        theta : array, shape (n_samples, )
            The dual vector.

        Returns
        -------
        The conjugate function evaluated at theta.
        '''
        if self.func == FUNC_LOG:
            return _comp_conj_LOG(theta, self.eta)
        elif self.func == FUNC_EXP:
            return _comp_conj_EXP(theta, self.eta, self.a)
    
    
    def comp_dual_var(self, X, w):
        '''Compute the dual variable.
        The dual variable equals to negative gradient of f(x) with respect to x.
        - lam * theta = 1/n * [f'(w^T X_1), ..., f'(w^T X_n)]^T

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data of growth rates.
        w : array, shape (n_features, )
            The allocation vector.

        Returns
        -------
        The dual vector - lam * theta.
        '''
        if self.func == FUNC_LOG:
            return _comp_dual_var_LOG(X, w, self.eta)
        elif self.func == FUNC_EXP:
            return _comp_dual_var_EXP(X, w, self.eta, self.a)
    

    def proj(self, w):
        '''Project primal variable to the constraint set of unit l1-norm ball.
        
        Parameters
        ----------
        w : array, shape (n_features, )
            The allocation vector.

        Returns
        -------
        The projected primal variable.
        '''
        return _proj(w)

    
    def comp_dual_neg_hess(self, theta, lam):
        '''Compute the dual negative Hessian with respect to z.
        
        Parameters
        ----------
        theta : array, shape (n_samples, )
            The dual vector.
        lam : float
            The l1 regularization parameter.

        Returns
        -------
        The negative Hessian.
        '''
        if self.func == FUNC_LOG:
            return _comp_dual_neg_hess_LOG(theta)
        elif self.func == FUNC_EXP:
            return _comp_dual_neg_hess_EXP(theta, self.a, lam)