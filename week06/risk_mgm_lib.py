import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
from bisect import bisect_left
import scipy

# Covariance estimation techniques
# Generate the exponential weights and covariance matrix.
def expWeights(n, lambd):
    weights = np.zeros(n)
    for i in range(1, n+1):
        weights[i-1] = (1-lambd) * (lambd**(i-1))
    normWeights = weights / np.sum(weights)
    return normWeights[::-1]

def expCovMat(data, weights):
    normData = data - data.mean()
    return np.dot(normData.T, np.diag(weights) @ normData)

# Non PSD fixes for correlation matrices
def cholPSD(root, a):
    n = a.shape[1]
    
    for j in range(n):
        s = 0.
        if j > 0:
            s = root[j, :j] @ root[j, :j].T
        diag = a[j, j] - s
        if diag <= 0 and diag >= -1e-5:
            diag = 0.
        root[j, j] = np.sqrt(diag)
        
        if root[j, j] == 0.:
            root[j, j:n-1] = 0.
        else:
            for i in range(j+1, n):
                s = root[i,:j] @ root[j,:j].T
                root[i,j] = (a[i,j] - s) / root[j,j]
    
    return root

def nearPSD(a, epsilon=1e-8):
    n = a.shape[1]
    invSD = None
    out = copy.deepcopy(a)
    diagSum = np.sum(np.isclose(1.0, np.diag(out)))
    
    if diagSum != n:
        invSD = np.diag(1. / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD
        
    eigvals, eigvecs = np.linalg.eigh(out)
    eigvals = np.maximum(eigvals, epsilon)
    T = 1.0 / (np.square(eigvecs) @ eigvals)
    T = np.diagflat(np.sqrt(T))
    l = np.diag(np.sqrt(eigvals))
    B = T @ eigvecs @ l
    out = B @ B.T

    if invSD != None:
        invSD = np.diag(1. / np.diag(invSD))
        out = invSD @ out @ invSD

    return out

def frobeniusNorm(matrix):
    return np.sqrt(np.square(matrix).sum())

def projection_u(matrix):
    res = copy.deepcopy(matrix)
    np.fill_diagonal(res, 1.0)
    return res

def projection_s(matrix, epsilon=1e-9):
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, epsilon)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

# Higham
def highamPSD(a, tol=1e-9):
    s = 0.0
    y = a
    prev_gamma = np.inf

    while True:
        r = y - s
        x = projection_s(r)
        s = x - r
        y = projection_u(x)
        gamma = frobeniusNorm(y - a)

        if abs(gamma - prev_gamma) < tol:  
            break
        prev_gamma = gamma

    return y

def isPSD(matrix):
    eigvals = np.linalg.eigvals(matrix)
    return np.all(eigvals >= 0)

# Simulation Methods
def multVarNormGen(cov, n=25000):
    root = np.full(cov.shape, 0.0)
    return cholPSD(root, cov) @ np.random.normal(size=(cov.shape[0], n))
def var(cov):
    return np.diag(cov)
def corr(cov):
    std = np.diag(1 / np.sqrt(var(cov)))
    return std @ cov @ std.T
def cov(var, cor):
    std = np.sqrt(var)
    return np.diag(std) @ cor @ np.diag(std).T


def simulationPCA(cov, percent, n=25000):
    eigvals, eigvecs = np.linalg.eigh(cov)

    sortedIndex = np.argsort(eigvals)[::-1]
    sortedEigvals = eigvals[sortedIndex]
    sortedEigvecs = eigvecs[:,sortedIndex]

    explain = sortedEigvals / sortedEigvals.sum()
    cumExplain = explain.cumsum()
    cumExplain[-1] = 1

    idx = bisect_left(cumExplain, percent)

    explainedVals = np.clip(sortedEigvals[:idx + 1], 0, np.inf)
    explainedVecs = sortedEigvecs[:, :idx + 1]

    B = explainedVecs @ np.diag(np.sqrt(explainedVals))
    r = np.random.randn(B.shape[1], n)
    return B @ r

def directSimulation(cov, n_samples=25000):
    B = cholPSD(cov)
    r = scipy.random.randn(len(B[0]), n_samples)
    return B @ r


# VaR calculation methods
# Given data and alpha, return the VaR
def calculateVar(data, u=0, alpha=0.05):
    return u-np.quantile(data, alpha)


def normalVar(data, u=0, alpha=0.05, n=10000):
    sigma = np.std(data)
    simulation_norm = np.random.normal(u, sigma, n)
    var_norm = calculateVar(simulation_norm, u, alpha)
    return var_norm

def ewcovNormalVar(data, u=0, alpha=0.05, n=10000, lambd=0.94):
    ew_cov = expCovMat(data, expWeights(len(data), lambd))
    sigma = np.sqrt(ew_cov)
    simuEW = np.random.normal(u, sigma, n)
    var = calculateVar(simuEW, u, alpha)
    return var

def tVar(data, u=0, alpha=0.05, n=10000):
    params = scipy.stats.t.fit(data, method="MLE")
    df, loc, scale = params
    simulation_t = scipy.stats.t(df, loc, scale).rvs(n)
    var_t = calculateVar(simulation_t, u, alpha)
    return var_t

def historicVar(data, u=0, alpha=0.05):
    return calculateVar(data, u, alpha)



# ES calculation
def calculateES(data, u=0, alpha=0.05):
    return abs(np.mean(data[data<-calculateVar(data, u, alpha)]))


# Other functions
# Input: price series
# Output: price returns
def calculateReturn(price, method='discrete'):
    returns = []
    for i in range(len(price)-1):
        returns.append(price[i+1]/price[i])
    returns = np.array(returns)
    if method == 'discrete':
        return returns - 1
    if method == 'log':
        return np.log(returns)


