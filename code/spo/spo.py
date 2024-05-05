from typing import Optional

import numpy as np
from numpy import linalg as LA
import numba as nb
from tqdm import tqdm

from .objective import Objective
from .utils import soft_thresholding, build_lambdas
from .utils import np_type_f, np_type_i, nb_type_f, nb_type_i, FUNC_LOG, FUNC_EXP


@nb.jit(nb_type_f(nb_type_f[::1], nb_type_f), nopython=True, fastmath=True) # python 加速
# nopython和object是numba的两种编译模式，前者编译的代码更快，但是可能会因为某些限制但是退化为object, 通过nopython=True可以阻止退化并抛出异常

def dual_scaling(XTtheta, lam_1):
    '''Dual scaling to project XTtheta onto feasible dual set.

    Parameters
    ----------
    XTtheta : array, float
        X^T theta.
    lam_1 : float
        The l1 regularization parameter.

    Returns
    -------
    dual_scale : float
        The dual scale such that theta/dual_scale is feasible.
    '''
    dual_scale = np.maximum(-lam_1, LA.norm(XTtheta, np.inf))
    return dual_scale


@nb.jit(nb.types.UniTuple(nb_type_f,3)(
    Objective.class_type.instance_type, nb_type_f[:,::1], nb_type_f[::1], nb_type_f[::1], 
    nb_type_f, nb_type_f), nopython=True, fastmath=True)
def dual_gap(obj, screened_X, screened_w, theta, dual_scale, lam_1):
    '''Feature screening.

    Parameters
    ----------
    obj : Objective
        The objective function class.
    screened_X : array, float
        The screened training data of growth rates. 
    screened_w : array, float
        The active primal variable.
    theta : array, float
        The dual variable.
    dual_scale : float
        The dual scale.
    lam_1 : float
        The l1 regularization parameter.

    Returns
    -------
    pval : float
        The value of primal objective.
    dval : float
        The value of dual objective.
    gap : float
        The dual gap.
    '''
    pval = obj.val(screened_X, screened_w)
    pval += lam_1 * LA.norm(screened_w, ord=1)
    dval = obj.conj(lam_1 * theta / dual_scale)     
    gap = np.maximum(pval - dval, nb_type_f(0.))

    return pval, dval, gap


@nb.jit(nb.types.Tuple((nb_type_f[::1], nb_type_i))(
    nb_type_f[::1], nb_type_f[::1], nb_type_f,
    nb_type_f[::1], nb_type_i, nb_type_i[::1]), nopython=True, fastmath=True)
def screening(w, XTcenter, r,
              norm2_X, n_active_features, disabled_features):
    '''Feature screening.

    Parameters
    ----------
    w : array, float
        The primal variable.
    XTcenter : array, float
        X^T Omega(theta).
    r : float
        The radius of the dual ball.
    norm2_X : array, float
        The norms of columns of X.
    n_active_features : int
        Number of active variables.    
    disabled_features : array, int
        A binary vector indicating if a feature is screened out.

    Returns
    -------
    w : array, float
        The updated primal variable.
    n_active_features : int
        Number of active variables after screening.
    
    Modified inplace:
        The array XTtheta and disabled_features.
    '''
    n_features = w.shape[0]

    # Safe rule for Feature level
    for j in range(n_features):

        if disabled_features[j] == 1:
            continue

        r_normX_j = r * np.sqrt(norm2_X[j])        
        if r_normX_j > 1.:
            continue

        if np.maximum(XTcenter[j], 0) + r_normX_j < 1.:
            w[j] = 0.
            disabled_features[j] = 1
            n_active_features -= 1
    return w, n_active_features    


@nb.jit(nb.types.Tuple((nb_type_f[::1], nb.boolean))(
    nb_type_f[::1], nb_type_f[::1], nb_type_f,
    nb_type_i[::1], nb_type_f, nb_type_f[::1], nb_type_f[::1]), nopython=True, fastmath=True)
def prox_gd(w, grad, L_dh,
            disabled_features, lam_1, w_old, w_old_old):
    """One step of proximal gradient descent.

    Parameters
    ----------
    w : array, float
        The primal variable.
    grad : array, float
        The gradient.    
    L_dh : float
        The upper bound of operator norm / Lipshitz constant.
    disabled_features : array, int
        A binary vector indicating if a feature is screened out.
    lam_1 : float
        The regularization parameter.
    w_old : array, float
        The primal variable in the last step.
    w_old_old : array, float
        The primal variable in the last two step.

    Returns
    -------
    w : array, float
        The updated primal variable.
    is_diff : boolean
        Whether the updated primal variable is different from the input.
    """      
    n_features = w.shape[0]
    is_diff = False

    # coordinate wise soft tresholding with nonnegative constraints
    thres = lam_1 / L_dh
    for j in range(n_features):
        if disabled_features[j] == 1:
            continue
        
        w_old_old[j] = w_old[j]
        w_old[j] = w[j]

        # ADMM for PGD
        # tmp = w[j] - grad[j] / L_dh
        # if tmp - thres > 0.:
        #     tmp = tmp - thres
        #     pass
        # else:
        #     xp = 1e-8
        #     z = tmp
        #     u = 0.
        #     iter = 0
        #     e_rel = 1.
        #     while iter<1e4 and e_rel>1e-4:
        #         iter += 1
        #         x = np.sign(z - u) * np.maximum(np.abs(z - u) - thres, nb_type_f(0.))                
        #         z = np.maximum(tmp + x + u, 0.)/2.0
        #         u = u + x - z  
        #         e_rel = np.abs((x - xp)/(xp+1e-8))
        #         xp = x
        #     tmp = x                
        # w[j] = tmp

        # 使得权重符合要求w大于0
        tmp = np.maximum(w[j] - (grad[j] / L_dh + thres), 0)
        if (w[j]-w_old_old[j])*(tmp-w[j])<0:
            if tmp-w[j]>0:
                w[j] = (w[j] + np.minimum(w_old_old[j], tmp)) / nb_type_f(2.)
            else:
                w[j] = (w[j] + np.maximum(w_old_old[j], tmp)) / nb_type_f(2.)
        elif tmp>w[j]>w_old_old[j]:
            w[j] = 2 * tmp - w[j]
        else:
            w[j] = tmp
        
        if is_diff is False and w[j]!=w_old[j]:
            is_diff = True        

    return w, is_diff  


@nb.jit(nb.types.UniTuple(nb_type_f,3)(
    Objective.class_type.instance_type, nb_type_f[:,::1], nb_type_f[::1], nb_type_f[::1],
    nb_type_f[::1], nb_type_f, 
    nb_type_f, nb_type_i, nb_type_f, nb_type_f, nb.boolean), nopython=True, fastmath=True)
def track_one_path(obj, X, w, theta0, 
             norm2_X, L_dh,        
             lam_1, max_iter, f, tol, screen):
    """Track one solution for a given value of lambda.

    Parameters
    ----------
    obj : Objective
        The objective function class.
    X : array, float
        Training data of growth rates. 
    w : array, float
        The initialized primal variable.
    theta0 : array, float
        The initialized dual variable.
    norm2_X : array, float
        The norms of columns of X.
    L_dh : float
        The upper bound of operator norm / Lipshitz constant.
    lam_1 : float
        The l1 regularization parameter.
    max_iter : int
        The maximum number of iterations.
    f : int
        The screening rule will be execute at each f pass on the data.
    tol : float
        The tolerance of terminate condition.
    screen : boolean
        Whether to screen features.

    Returns
    -------
    gaps : float
        The dual gaps at the end of the optimization for each lambda.
    n_active_features : int
        Number of active variables.
    n_iters : int
        The number of iterations taken by the block coordinate descent
        optimizer to reach the specified accuracy for each lambda.        
    """
    # _：单下划线，用作临时或无意义变量的名称。
    _, n_features = X.shape
    n_active_features = n_features
    
    final_pass = False
    gap = np.inf
    
    w_old = w.copy()
    w_old_old = w_old.copy()
    theta = theta0.copy()
    XTtheta = np.dot(X.T, theta)
    dual_scale = dual_scaling(XTtheta, lam_1)
    # 注意此处代入到二阶 hessian 矩阵的为 theta0/dual_scale。即 \theta_0，但是由公式可知，w=0时对应\theta约为0，故而可以代入。
    alpha = np.max(obj.comp_dual_neg_hess(theta0/dual_scale, lam_1))

    disabled_features = np.zeros(n_features, dtype=nb_type_i)
    
    for n_iter in range(max_iter):
        id_features = (disabled_features == 0)
        screened_w = w[id_features]
        screened_X = X[:, id_features]

        # Update dual variables
        theta[:] = obj.comp_dual_var(screened_X, screened_w)
        XTtheta[:] = np.dot(X.T, theta)

        # Screening
        if f != 0 and (n_iter % f == 0 or final_pass):   
            dual_scale = dual_scaling(XTtheta, lam_1)
            
            _, _, gap = dual_gap(obj, screened_X, screened_w, theta, dual_scale, lam_1)

            if gap <= tol or final_pass:
                final_pass = True
                break

            if screen:
                r = np.sqrt(2 * gap / alpha)
                # 注意此时的XTtheta实际为 lambda * dot(X.T,theta)，且\theta尚未被scale。
                XTcenter = XTtheta / dual_scale
                
                w, n_active_features = screening(w, XTcenter, r,
                        norm2_X, n_active_features, disabled_features)
            
            # The local Lipschitz constant of h's gradient.
            L_dh = LA.norm(X * np.sqrt(theta).reshape((-1,1)), ord=2) ** 2

        if final_pass:
            break

        w, is_diff = prox_gd(w, -XTtheta, L_dh,
            disabled_features, lam_1, w_old, w_old_old)
        
        if not is_diff:
            final_pass = True
    # 注意numpy数组为可变变量，即传递到函数中被改变后退出函数，其值也被相应修改的变量。
    w = obj.proj(w)

    return gap, n_active_features, n_iter


def spo_l1_path(X, func: int = 0, a: float = 1.0, 
    lambdas: Optional[list] = None, n_lambdas: int = 100, delta: float = 2.0,
    max_iter: int = int(1e5), tol: float = 1e-4, screen: bool = True, f: int = 30, verbose: bool = True):
    """Sparse portfolio optimization with proximal gradient descent

    The formulation reads:

    f(w) + lambda_1 norm(w, 1)
    where f(w) = - 1/n * sum_{j=1}^n u(w^TX_j).

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Training data of growth rates. 

    func : int, optional
        The utility function: 0 - log, 1 - exp.

    a : float, optional
        The risk aversion parameter for exp utility.

    lambdas : array, optional, shape (n_lambdas,)
        List of lambdas where to compute the models.

    n_lambdas : int, optional
        The number of lambdas.

    delta : float, optional
        The log spacing of lambdas.

    max_iter : int, optional
        The maximum number of iterations.

    tol : float, optional
        Prescribed accuracy on the duality gap.

    screen : boolean, optional
        Whether use screening rules or not.

    f : int, optional
        The screening rule will be execute at each f pass on the data


    Returns
    -------
    ws : array, shape (n_features, n_lambdas)
        Coefficients along the path.

    lambdas : array, shape (n_lambdas,)
        The list of lambdas.

    gaps : array, shape (n_lambdas,)
        The dual gaps at the end of the optimization for each lambda.

    n_iters : array, shape (n_lambdas,)
        The number of iterations taken by the block coordinate descent
        optimizer to reach the specified accuracy for each lambda.

    n_active_features : array, shape (n_lambdas,)
        Number of active variables.

    """
    n_samples, n_features = X.shape
    minX = np.min(X)
    # 此处用于给出异常，即`$X$`中元素必须满足大于0：此处`$X$`中每个元素指每天开始时股票的价格比上每天结束时股票价格的比值。
    if minX<=0.:
        raise ValueError('The growth rate X must be positive.') 
    X = X - minX
    # Use C-contiguous data to avoid unnecessary memory duplication.
    # ascontiguousarray 函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
    X = np.ascontiguousarray(X)

    # 此处为调动 jit 对程序进行加速，定义了数据类型 np_type_f
    obj = Objective(n_samples, minX, func, a)
    w_init = np.zeros(n_features, dtype=np_type_f)
    # 获得 dual vector - lam * theta
    theta = obj.comp_dual_var(X, w_init)
    XTtheta = np.dot(X.T, theta)
    
    if lambdas is None:
        lambdas = build_lambdas(XTtheta, n_lambdas, delta)

    n_lambdas = np_type_i(lambdas.shape[0])

    # Useful precomputation
    # X**2的shape与X相同，但每个元素均为原来X的平方
    # np.sum()，axis=0 意味对每列求和
    norm2_X = np.sum(X**2, axis=0) # 对X的每列求2-范数的平方
    # 在numpy中，向量`$\begin{bmatrix}1\\2\\3\end{bmatrix}$`和`$\begin{bmatrix}1&2&3\end{bmatrix}$`的`shape`均为`(3,)`，如果
    # 想得到`(3,1)`则调用`a.reshape(-1,1)`或者`np.reshape(a,(-1,1))`，如果想得到`(1,3)`，调用`a.reshape(-1,3)`或者
    # `np.reshape(a,(-1,3))`。
    L_dh = LA.norm(X * np.sqrt(theta).reshape((-1,1)), ord=2) ** 2

    # allocate 相应变量
    ws = np.zeros((n_features, n_lambdas), dtype=np_type_f)
    gaps = np.ones(n_lambdas, dtype=np_type_f)
    n_active_features = np.zeros(n_lambdas, dtype=np_type_i)
    n_iters = np.zeros(n_lambdas, dtype=np_type_i)

    # tqdm 是进度条，此处以lambda进展作为进度条件
    for t in tqdm(range(n_lambdas), disable=not verbose):
        # \：换行符。
        # track_one_path：对指定 lambda 进行算法
        gaps[t], n_active_features[t], \
        n_iters[t] = track_one_path(obj, X, w_init, theta, 
                norm2_X, L_dh,      
                lambdas[t], max_iter, f, tol, screen)

        ws[:, t] = w_init.copy()

    return ws, lambdas, gaps, n_iters, n_active_features


def spo_nw_min(X, func: int = 0, a: float = 1.0, 
    lambdas: Optional[list] = None, n_lambdas: int = 100, delta: float = 2.0,
    max_iter: int = int(1e5), tol: float = 1e-4, screen: bool = True, f: int = 30, verbose: bool = True,
    nw_min=5):
    """Modified version with nw_min requirement.

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data of growth rates. 

    func : int, optional
        The utility function: 0 - log, 1 - exp.

    a : float, optional
        The risk aversion parameter for exp utility.

    lambdas : ndarray, optional, shape (n_lambdas,)
        List of lambdas where to compute the models.

    n_lambdas : int, optional
        The number of lambdas.

    delta : float, optional
        The log spacing of lambdas.

    max_iter : int, optional
        The maximum number of iterations.

    tol : float, optional
        Prescribed accuracy on the duality gap.

    screen : boolean, optional
        Whether use screening rules or not.

    f : int, optional
        The screening rule will be execute at each f pass on the data


    Returns
    -------
    ws : array, shape (n_features, n_lambdas)
        Coefficients along the path.

    lambdas : ndarray, shape (n_lambdas,)
        The list of lambdas.

    gaps : array, shape (n_lambdas,)
        The dual gaps at the end of the optimization for each lambda.

    n_iters : array-like, shape (n_lambdas,)
        The number of iterations taken by the block coordinate descent
        optimizer to reach the specified accuracy for each lambda.

    n_active_features : array, shape (n_lambdas,)
        Number of active variables.

    """
    n_samples, n_features = X.shape
    minX = np.min(X)    
    if minX<=0.:
        raise ValueError('The growth rate X must be positive.') 
    X = X - minX
    # Use C-contiguous data to avoid unnecessary memory duplication.
    X = np.ascontiguousarray(X)

    obj = Objective(n_samples, minX, func, a)
    w_init = np.zeros(n_features, dtype=np_type_f)
    theta = obj.comp_dual_var(X, w_init)
    XTtheta = np.dot(X.T, theta)
    
    if lambdas is None:
        lambdas = build_lambdas(XTtheta, n_lambdas, delta)

    n_lambdas = np_type_i(lambdas.shape[0])

    # Useful precomputation
    norm2_X = np.sum(X**2, axis=0)
    L_dh = LA.norm(X * np.sqrt(theta).reshape((-1,1)), ord=2) ** 2
    
    ws = np.zeros((n_features, n_lambdas), dtype=np_type_f)
    gaps = np.ones(n_lambdas, dtype=np_type_f)
    n_active_features = np.zeros(n_lambdas, dtype=np_type_i)
    n_iters = np.zeros(n_lambdas, dtype=np_type_i)

    for t in tqdm(range(n_lambdas), disable=not verbose):
        gaps[t], n_active_features[t], \
        n_iters[t] = track_one_path(obj, X, w_init, theta, 
                norm2_X, L_dh,      
                lambdas[t], max_iter, f, tol, screen)

        ws[:, t] = w_init.copy()
        if np.sum(w_init>0.)>=nw_min:
            break

    return ws, lambdas, gaps, n_iters, n_active_features


def spo_nw_min_ex(X, func: int = 0, a: float = 1.0, 
    lambdas: Optional[list] = None, n_lambdas: int = 100, delta: float = 2.0,
    max_iter: int = int(1e5), tol: float = 1e-4, screen: bool = True, f: int = 30, verbose: bool = True,
    nw_min=5):
    """Modified version with first nw_min excluded.

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data of growth rates. 

    func : int, optional
        The utility function: 0 - log, 1 - exp.

    a : float, optional
        The risk aversion parameter for exp utility.

    lambdas : ndarray, optional, shape (n_lambdas,)
        List of lambdas where to compute the models.

    n_lambdas : int, optional
        The number of lambdas.

    delta : float, optional
        The log spacing of lambdas.

    max_iter : int, optional
        The maximum number of iterations.

    tol : float, optional
        Prescribed accuracy on the duality gap.

    screen : boolean, optional
        Whether use screening rules or not.

    f : int, optional
        The screening rule will be execute at each f pass on the data


    Returns
    -------
    ws : array, shape (n_features, n_lambdas)
        Coefficients along the path.

    lambdas : ndarray, shape (n_lambdas,)
        The list of lambdas.

    gaps : array, shape (n_lambdas,)
        The dual gaps at the end of the optimization for each lambda.

    n_iters : array-like, shape (n_lambdas,)
        The number of iterations taken by the block coordinate descent
        optimizer to reach the specified accuracy for each lambda.

    n_active_features : array, shape (n_lambdas,)
        Number of active variables.

    """
    n_samples, n_features = X.shape
    minX = np.min(X)    
    if minX<=0.:
        raise ValueError('The growth rate X must be positive.') 
    X = X - minX
    # Use C-contiguous data to avoid unnecessary memory duplication.
    X = np.ascontiguousarray(X)

    obj = Objective(n_samples, minX, func, a)
    w_init = np.zeros(n_features, dtype=np_type_f)
    theta = obj.comp_dual_var(X, w_init)
    XTtheta = np.dot(X.T, theta)
    
    if lambdas is None:
        lambdas = build_lambdas(XTtheta, n_lambdas, delta)

    n_lambdas = np_type_i(lambdas.shape[0])

    # Useful precomputation
    norm2_X = np.sum(X**2, axis=0)
    L_dh = LA.norm(X * np.sqrt(theta).reshape((-1,1)), ord=2) ** 2
    
    ws = np.zeros((n_features, n_lambdas), dtype=np_type_f)
    gaps = np.ones(n_lambdas, dtype=np_type_f)
    n_active_features = np.zeros(n_lambdas, dtype=np_type_i)
    n_iters = np.zeros(n_lambdas, dtype=np_type_i)

    nw = 0
    for t in tqdm(range(n_lambdas), disable=not verbose):
        # \:转义 / 续行符（折行）/ 反斜杠
        gaps[t], n_active_features[t], \
        n_iters[t] = track_one_path(obj, X, w_init, theta, 
                norm2_X, L_dh,      
                lambdas[t], max_iter, f, tol, screen)

        if t==0:
            nw = np.sum(w_init>0.)
            if nw<nw_min:
                X = X[:, w_init==0.] + minX          
                return w_init==0., X

        ws[:, t] = w_init.copy()
        if np.sum(w_init>0.)>=nw_min:
            break
    
    return ws, lambdas, gaps, n_iters, n_active_features