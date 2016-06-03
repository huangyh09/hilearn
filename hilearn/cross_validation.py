# A general cross-validation

import numpy as np

class CrossValidation:
    def __init__(self, X, Y):
        """General cross-validation for both classification and regression.
        1) check the shape of X and Y are compatible with the input model;
        2) the model doesn't have memory on previous trainings, i.e., all 
        parameters are only based on current fit.
        """
        self.X, self.Y = X, Y
        
    def cv_regression(self, model, folds=3, shuffle=True):
        """run cross-validation for regression.
        For regression, make sure the input model has the following 
        functions: fit and predict
        """
        self.Y_pre = np.zeros(self.Y.shape[0])
        fold_len = int(self.Y.shape[0] / folds)
        idx_all = np.arange(self.Y.shape[0])
        if shuffle:
            np.random.shuffle(idx_all)
            
        for i in range(folds):
            if i < folds - 1:
                _idx = idx_all[i*fold_len : (i+1)*fold_len]
            else:
                _idx = idx_all[i*fold_len : self.Y.shape[0]]
            Xtest = self.X[_idx, :]
            Xtrain = np.delete(self.X, _idx, 0)
            Ytrain = np.delete(self.Y, _idx)
            
            model.fit(Xtrain, Ytrain)
            self.Y_pre[_idx] = model.predict(Xtest)
        return self.Y_pre
    
    def cv_classification(self, model, folds=3, shuffle=True):
        """Run cross-validation for classification.
        For classification, make sure the input model has the following 
        functions: fit, predict and predict_proba.
        """
        self.Ystate = np.zeros(self.Y.shape[0])
        self.Yscore = np.zeros(self.Y.shape[0])
        idx0 = np.where(self.Y == 0)[0]
        idx1 = np.where(self.Y == 1)[0]
        if shuffle:
            np.random.shuffle(idx0)
            np.random.shuffle(idx1)
        fold_len0 = int(idx0.shape[0]/folds)
        fold_len1 = int(idx1.shape[0]/folds)
        
        for i in range(folds):
            if i < folds - 1:
                _idx0 = idx0[i*fold_len0: (i+1)*fold_len0]
                _idx1 = idx1[i*fold_len1: (i+1)*fold_len1]
            else:
                _idx0 = idx0[i*fold_len0:]
                _idx1 = idx1[i*fold_len1:]
            _idx = np.append(_idx0, _idx1)
            Xtest = self.X[_idx, :]
            Xtrain = np.delete(self.X, _idx, 0)
            Ytrain = np.delete(self.Y, _idx)
            
            model.fit(Xtrain, Ytrain)
            self.Ystate[_idx] = model.predict(Xtest)
            self.Yscore[_idx] = model.predict_proba(Xtest)
        return self.Ystate, self.Yscore


    def plot_cv_regression(max_num=10000, outlier=0.01, line_on=True,
        extra_on=True):
        x, y = self.Y, self.Y_pre
        score = st.pearsonr(x, y)[0]
        np.random.seed(0)
        if len(x) > max_num:
            idx = np.random.permutation(len(x))[:max_num]
            x, y = x[idx], y[idx]
        outlier = int(len(x) * outlier)
        
        xy = np.vstack([x,y])
        z = st.gaussian_kde(xy)(xy)
        idx = z.argsort()
        idx1, idx2 = idx[outlier:], idx[:outlier]
        pl.set_cmap('Blues')
        pl.scatter(x[idx1], y[idx1], c=np.log(z[idx1]*9+1), edgecolor='', s=10)
        pl.scatter(x[idx2], y[idx2], c="k", edgecolor='', s=10, alpha=0.7)
        pl.grid(alpha=0.3)

        if line_on:
            clf = linear_model.LinearRegression()
            clf.fit(x.reshape(-1,1), y)
            xx = np.linspace(x.min(), x.max(), 1000).reshape(-1,1)
            yy = clf.predict(xx)
            pl.plot(xx, yy, "k--", label="R=%.3f" %score)

        if extra_on:
            pl.ylabel("predicted Y")
            pl.xlabel("observed Y")
            pl.xlim(x.min(),x.max())
            pl.ylim(y.min(),y.max())
            pl.legend(loc=2, fancybox=True, ncol=1)

            