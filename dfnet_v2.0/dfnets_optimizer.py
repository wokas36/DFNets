from __future__ import print_function

import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
from cvxpy import *
import sys

np.set_printoptions(threshold=sys.maxsize)

def dfnets_coefficients_optimizer(mu, response, Kb, Ka, radius=0.85, show=0):
    
    '''
    This function finds polynomial coefficients (b,a) such that 
    the ARMA model "rARMA = polyval(wrev(b),mu)./polyval(wrev(a), mu)" 
    approximates the function response() at the points mu.
     
    REQUIRED INPUTS
    mu - the points where the response function is evaluated 
    response - the desired response function 
    Kb,Ka are the orders of the numerator and denominator respectively 
    
    OPTIONAL INPUTS
    radius allows to control the tradeoff between convergence speed (small)
        and accuracy (large). Should be below 1 if the standard iterative 
        implementation is used. With the conj. gradient implementation any 
        radius is allowed.
    '''
    
    if(mu.shape[0] == 1):
        mu = mu.conj().transpose()
    
    # -------------------------------------------------------------------------
    # Construct various utility matrices
    # -------------------------------------------------------------------------

    # N is the Vandermonde that will be used to evaluate the numerator.
    NM = np.zeros((len(mu),Kb+1))

    NM[:,0] = 1

    for k in range(1, Kb+1):
        NM[:,k] = NM[:,k-1] * mu

    # M is the Vandermonde that will be used to evaluate the denominator.
    MM = np.zeros((len(mu), Ka))

    MM[:,0] = mu

    for k in range(1, Ka):
        MM[:,k] = MM[:,k-1] * mu

    V = np.zeros((np.size(mu),Ka))

    for k in range(0, Ka):
        V[:,k] = mu**k

    ia = Variable(Ka,1)
    ib = Variable(Kb+1,1)
    
    reshape_mu = response(mu).reshape(response(mu).shape[0], 1)
    
    objective = Minimize(norm(reshape_mu + np.diag(response(mu))@MM@ia - NM@ib))
    #constraints = [max_entries(abs(V@ia)) <= radius]
    constraints = [norm((V@ia), 'inf') <= radius]
    
    prob = Problem(objective, constraints)
    result = prob.solve(verbose=True)
    
    a = ia.value
    b = ib.value
    
    # this is the achieved response
    rARMA = np.polyval(np.flipud(b),mu)/np.polyval(np.flipud(a), mu)

    # error
    error = norm(rARMA - response(mu))/norm(mu)
    
    return b, a, rARMA, error