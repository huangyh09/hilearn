# This class is a mixture of linear regression.

import numpy as np

class MixLinearRegression:
    def __init__(self, prior=None, coef_=None, intercept_=None, sigma=None):
        """an initialization of the model, which could be extended
        with initial settings."""
        self.prior = prior
        self.sigma = sigma
        self.coef_ = coef_
        self.intercept_ = intercept_

    def fit(self, X, Y, K=1, run_min=100, run_max=100000, gap_ratio=0.0001, 
        is_stochastic=False):
        """implement the training"""
        # check the shape of the input
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        if len(Y.shape) == 2 and Y.shape[1] == 1:
            Y = Y.reshape(-1)
        self.X = X
        self.Y = Y
        X1 = np.ones((X.shape[0], X.shape[1]+1))
        X1[:,1:] = X
        N, M = X1.shape
        R = np.random.dirichlet(np.ones(K), N)
        Prior = np.ones(K)
        Coef  = np.ones((M, K))
        Sigma = np.ones(K)
        for i in range(run_max):
            #M-step
            Prior = R.mean(axis=0)
            if is_stochastic:
                for n in range(R.shape[0]):
                    R[n,:] = np.random.multinomial(1, R[n,:], size=1)
            for j in range(K):
                Coef[:,j] = np.linalg.inv(np.dot((X1*R[:,j:j+1]).T, X1)).dot((X1*R[:,j:j+1]).T).dot(Y)
                Sigma[j]  = np.sqrt(np.sum((Y - X1.dot(Coef[:,j]))**2 * R[:,j]) / np.sum(R[:,j]))
            #E-step
            for j in range(K):
                R[:,j] = Prior[j] / (Sigma[j] * np.sqrt(2*np.pi)) * np.exp(-0.5*(Y - X1.dot(Coef[:,j]))**2 / Sigma[j]**2)
            R[:,:] = R / np.sum(R, axis=1).reshape(-1,1)
            # check convergence
            loglik = 0
            for j in range(K):
                loglik += Prior[j] / (Sigma[j] * np.sqrt(2*np.pi)) * np.exp(-0.5*(Y - X1.dot(Coef[:,j]))**2 / Sigma[j]**2)
            loglik = np.sum(np.log(loglik))

            if i >= run_min and (loglik - loglik_old) < gap_ratio*abs(loglik):
                break
            else:
                loglik_old = loglik

        self.coef_ = Coef[1:,:]
        self.intercept_ = Coef[0,:]
        self.loglik = loglik
        self.sigma  = Sigma
        self.prior  = Prior
        self.probs  = R
        self.bic    = -2*loglik + (K*(M+2)-1)*np.log(N)
        
    def predict(self, Xtest):
        """implement the prediction"""
        if len(Xtest.shape) == 1:
            Xtest = Xtest.reshape(-1,1)
        RV = np.zeros(Xtest.shape[0])
        for j in range(self.coef_.shape[1]):
            RV += self.prior[j] * (Xtest.dot(self.coef_[:,j]) + 
                self.intercept_[j])
        return RV.reshape(-1)

    def predict_label(self, Xtest, Ytest):
        K = self.prior.shape[0]
        P = np.zeros(K, N)
        for j in range(K):
            P[:,j] = self.prior[j] * 1.0 / (self.sigma[j] * np.sqrt(2*np.pi))
            P[:,j] *= np.exp(-0.5*(Xtest.dot(self.coef_[:,j]) + 
                self.intercept_[j] - Ytest)**2 / self.sigma[j]**2)
        P[:,:] = P / np.sum(R, axis=1).reshape(-1,1)
        return P

        