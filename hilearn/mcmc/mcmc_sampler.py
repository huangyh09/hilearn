import numpy as np

def Geweke_Z(X, first=0.1, last=0.5):
    N = X.shape[0]
    A = X[:int(first*N)]
    B = X[int(last*N):]
    if np.sqrt(np.var(A) + np.var(B)) == 0:
        Z = None
    else:
        Z = abs(A.mean() - B.mean()) / np.sqrt(np.var(A) + np.var(B))
    return Z

def mcmc_sampler(data, prior_pdf, likelihood, propose_pdf, 
                 propose_func, xx, param, adaptive_func=None, 
                 min_run=1000, max_rum=3000, gap=100):
    # initialization
    X_now = propose_func(xx, param)
    L_now = likelihood(X_now, data, param)
    P_now = prior_pdf(X_now) + L_now
    
    # MCMC running
    accept_num = 0
    L_all = np.zeros(max_rum)
    X_all = np.zeros((max_rum, len(X_now)))
    for m in range(max_rum):
        P_try, L_try = 0.0, 0.0
        Q_now, Q_try = 0.0, 0.0
        
        # step 1: propose a value
        X_try = propose_func(X_now, param)
        Q_now = propose_pdf(X_now, X_try, param)
        Q_try = propose_pdf(X_try, X_now, param)
        L_try = likelihood(X_try, data, param)
        P_try = prior_pdf(X_try) + L_try

        # step 2: accept or reject the proposal
        alpha = np.exp(min(P_try+Q_now-P_now-Q_try, 0))
        if alpha is None: 
            print("Warning: accept ratio alpha is none!")
        elif np.random.rand(1) < alpha:
            accept_num += 1
            X_now = X_try + 0.0
            P_now = P_try + 0.0
            L_now = L_try + 0.0
        L_all[m] = L_now
        X_all[m,:] = X_now
        
        # step 3. convergence diagnostics
        if m >= min_run and m % gap == 0:
            z_scores = np.zeros(X_all.shape[1])
            for k in range(X_all.shape[1]):
                z_scores[k] = Geweke_Z(X_all[:m, k])
            if sum(z_scores <= 2) == len(z_scores):
                L_all = L_all[:m]
                X_all = X_all[:m,:]
                break

        # step 4: adaptive MCMC
        if (adaptive_func is not None and 
            accept_num >= 10 and m % gap == 0):
            param = adaptive_func(X_all[:m,:], param)

    print("MCMC summary: %d acceptance in %d run (%.1f%%)." 
          %(accept_num, m, accept_num*100.0/m))
    return X_all, L_all, accept_num