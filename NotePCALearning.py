Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
[GCC 5.4.0 20160609] on linux
Type "copyright", "credits" or "license()" for more information.
>>> import numpy as np
>>> X = np.array([[2.5,2.4],
	      [0.5,0.7],
	      [2.2,2.9],
	      [1.9,2.2],
	      [3.1,3.0],
	      [2.3,2.7],
	      [2.0,1.6],
	      [1.0,1.1],
	      [1.5,1.6],
	      [1.2,0.9]])
>>> X
array([[ 2.5,  2.4],
       [ 0.5,  0.7],
       [ 2.2,  2.9],
       [ 1.9,  2.2],
       [ 3.1,  3. ],
       [ 2.3,  2.7],
       [ 2. ,  1.6],
       [ 1. ,  1.1],
       [ 1.5,  1.6],
       [ 1.2,  0.9]])
>>> X.mean()
1.8649999999999998
>>> X[1,:]
array([ 0.5,  0.7])
>>> X[:,1]
array([ 2.4,  0.7,  2.9,  2.2,  3. ,  2.7,  1.6,  1.1,  1.6,  0.9])
>>> X[:,0]
array([ 2.5,  0.5,  2.2,  1.9,  3.1,  2.3,  2. ,  1. ,  1.5,  1.2])
>>> X[:,0].mean()
1.8199999999999998
>>> X[:,1].mean()
1.9100000000000001
>>> meanX1 = X[:,0].mean()
>>> meanX2 = X[:,1].mean()
>>> X[:,0] = X[:,0] - meanX1
>>> X
array([[ 0.68,  2.4 ],
       [-1.32,  0.7 ],
       [ 0.38,  2.9 ],
       [ 0.08,  2.2 ],
       [ 1.28,  3.  ],
       [ 0.48,  2.7 ],
       [ 0.18,  1.6 ],
       [-0.82,  1.1 ],
       [-0.32,  1.6 ],
       [-0.62,  0.9 ]])
>>> X[:,1] = X[:,1] - meanX2
>>> X
array([[ 0.68,  0.49],
       [-1.32, -1.21],
       [ 0.38,  0.99],
       [ 0.08,  0.29],
       [ 1.28,  1.09],
       [ 0.48,  0.79],
       [ 0.18, -0.31],
       [-0.82, -0.81],
       [-0.32, -0.31],
       [-0.62, -1.01]])
>>> X.shape
(10, 2)
>>> np.dot(X.T, X)
array([[ 5.416,  5.438],
       [ 5.438,  6.449]])
>>> help(np.cov)
Help on function cov in module numpy.lib.function_base:

cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None)
    Estimate a covariance matrix, given data and weights.
    
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element :math:`C_{ij}` is the covariance of
    :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
    of :math:`x_i`.
    
    See the notes for an outline of the algorithm.
    
    Parameters
    ----------
    m : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same form
        as that of `m`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : bool, optional
        Default normalization (False) is by ``(N - 1)``, where ``N`` is the
        number of observations given (unbiased estimate). If `bias` is True, then
        normalization is by ``N``. These values can be overridden by using the
        keyword ``ddof`` in numpy versions >= 1.5.
    ddof : int, optional
        If not ``None`` the default value implied by `bias` is overridden.
        Note that ``ddof=1`` will return the unbiased estimate, even if both
        `fweights` and `aweights` are specified, and ``ddof=0`` will return
        the simple average. See the notes for the details. The default value
        is ``None``.
    
        .. versionadded:: 1.5
    fweights : array_like, int, optional
        1-D array of integer freguency weights; the number of times each
        observation vector should be repeated.
    
        .. versionadded:: 1.10
    aweights : array_like, optional
        1-D array of observation vector weights. These relative weights are
        typically large for observations considered "important" and smaller for
        observations considered less "important". If ``ddof=0`` the array of
        weights can be used to assign probabilities to observation vectors.
    
        .. versionadded:: 1.10
    
    Returns
    -------
    out : ndarray
        The covariance matrix of the variables.
    
    See Also
    --------
    corrcoef : Normalized covariance matrix
    
    Notes
    -----
    Assume that the observations are in the columns of the observation
    array `m` and let ``f = fweights`` and ``a = aweights`` for brevity. The
    steps to compute the weighted covariance are as follows::
    
        >>> w = f * a
        >>> v1 = np.sum(w)
        >>> v2 = np.sum(w * a)
        >>> m -= np.sum(m * w, axis=1, keepdims=True) / v1
        >>> cov = np.dot(m * w, m.T) * v1 / (v1**2 - ddof * v2)
    
    Note that when ``a == 1``, the normalization factor
    ``v1 / (v1**2 - ddof * v2)`` goes over to ``1 / (np.sum(f) - ddof)``
    as it should.
    
    Examples
    --------
    Consider two variables, :math:`x_0` and :math:`x_1`, which
    correlate perfectly, but in opposite directions:
    
    >>> x = np.array([[0, 2], [1, 1], [2, 0]]).T
    >>> x
    array([[0, 1, 2],
           [2, 1, 0]])
    
    Note how :math:`x_0` increases while :math:`x_1` decreases. The covariance
    matrix shows this clearly:
    
    >>> np.cov(x)
    array([[ 1., -1.],
           [-1.,  1.]])
    
    Note that element :math:`C_{0,1}`, which shows the correlation between
    :math:`x_0` and :math:`x_1`, is negative.
    
    Further, note how `x` and `y` are combined:
    
    >>> x = [-2.1, -1,  4.3]
    >>> y = [3,  1.1,  0.12]
    >>> X = np.vstack((x,y))
    >>> print(np.cov(X))
    [[ 11.71        -4.286     ]
     [ -4.286        2.14413333]]
    >>> print(np.cov(x, y))
    [[ 11.71        -4.286     ]
     [ -4.286        2.14413333]]
    >>> print(np.cov(x))
    11.71

>>> np.cov(X)
array([[  1.80500000e-02,  -1.04500000e-02,  -5.79500000e-02,
         -1.99500000e-02,   1.80500000e-02,  -2.94500000e-02,
          4.65500000e-02,  -9.50000000e-04,  -9.50000000e-04,
          3.70500000e-02],
       [ -1.04500000e-02,   6.05000000e-03,   3.35500000e-02,
          1.15500000e-02,  -1.04500000e-02,   1.70500000e-02,
         -2.69500000e-02,   5.50000000e-04,   5.50000000e-04,
         -2.14500000e-02],
       [ -5.79500000e-02,   3.35500000e-02,   1.86050000e-01,
          6.40500000e-02,  -5.79500000e-02,   9.45500000e-02,
         -1.49450000e-01,   3.05000000e-03,   3.05000000e-03,
         -1.18950000e-01],
       [ -1.99500000e-02,   1.15500000e-02,   6.40500000e-02,
          2.20500000e-02,  -1.99500000e-02,   3.25500000e-02,
         -5.14500000e-02,   1.05000000e-03,   1.05000000e-03,
         -4.09500000e-02],
       [  1.80500000e-02,  -1.04500000e-02,  -5.79500000e-02,
         -1.99500000e-02,   1.80500000e-02,  -2.94500000e-02,
          4.65500000e-02,  -9.50000000e-04,  -9.50000000e-04,
          3.70500000e-02],
       [ -2.94500000e-02,   1.70500000e-02,   9.45500000e-02,
          3.25500000e-02,  -2.94500000e-02,   4.80500000e-02,
         -7.59500000e-02,   1.55000000e-03,   1.55000000e-03,
         -6.04500000e-02],
       [  4.65500000e-02,  -2.69500000e-02,  -1.49450000e-01,
         -5.14500000e-02,   4.65500000e-02,  -7.59500000e-02,
          1.20050000e-01,  -2.45000000e-03,  -2.45000000e-03,
          9.55500000e-02],
       [ -9.50000000e-04,   5.50000000e-04,   3.05000000e-03,
          1.05000000e-03,  -9.50000000e-04,   1.55000000e-03,
         -2.45000000e-03,   5.00000000e-05,   5.00000000e-05,
         -1.95000000e-03],
       [ -9.50000000e-04,   5.50000000e-04,   3.05000000e-03,
          1.05000000e-03,  -9.50000000e-04,   1.55000000e-03,
         -2.45000000e-03,   5.00000000e-05,   5.00000000e-05,
         -1.95000000e-03],
       [  3.70500000e-02,  -2.14500000e-02,  -1.18950000e-01,
         -4.09500000e-02,   3.70500000e-02,  -6.04500000e-02,
          9.55500000e-02,  -1.95000000e-03,  -1.95000000e-03,
          7.60500000e-02]])
>>> np.cov(np.dot(X.T,X))
array([[  2.42000000e-04,   1.11210000e-02],
       [  1.11210000e-02,   5.11060500e-01]])
>>> np.cov(X.T)
array([[ 0.60177778,  0.60422222],
       [ 0.60422222,  0.71655556]])
>>> 
