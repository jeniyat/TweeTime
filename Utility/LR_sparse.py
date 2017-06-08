import numpy as np
import scipy as sp
import scipy.optimize

from scipy.sparse import *
from numpy.random import rand,normal
import decimal
decimal.getcontext().prec = 100

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

#TODO: version with sparse matrices?
class LR:
    def __init__(self, X, Y, l2=1.0):
        self.w0 = rand(X.shape[1])
        self.wStar = self.w0
        self.l2 = l2
        self.X = X
        self.Y = Y

    def NLL(self,w,X,Y):
        Y = np.squeeze(np.asarray(Y))
        return np.sum(np.log(1 + np.exp(-Y*X.dot(w)))) + self.l2 * np.dot(w.T, w)

    def GRAD(self,w,X,Y):
        Y = (Y+1.0) / 2
        Y = np.squeeze(np.asarray(Y))
        return np.squeeze(np.asarray(X.T.dot(((1.0 / (1.0 + np.exp(-X.dot(w).T))) - Y.T).T).T + 2 * self.l2 * w))
        #exp_ = np.exp(((-X.dot(w))))
        #print type(X)
        #print type(w)
        #return np.squeeze(np.asarray(X.T.dot(((1.0 / (1.0 + exp_)) - Y)) + 2 * self.l2 * w))

    def Train(self):
        for t in [1e-2, 1e-5]:
          (self.wStar, self.nll, self.status) = scipy.optimize.fmin_l_bfgs_b(lambda w,X,Y: self.NLL(w,X,Y), x0=self.w0, fprime = lambda w,X,Y: self.GRAD(w,X,Y), args = (self.X, self.Y), pgtol = t)
          print "nll:\t%s" % self.nll

    def Predict(self,X, w=None):
        if w == None:
            w = self.wStar
        return 1.0 / (1.0 + np.exp(-X.dot(w)))
        
    def CheckGrad(self):
        print "CHECKGRAD:"
        for i in range(10):
            w0 = rand(self.X.shape[1])
            print w0.shape
            print self.X.shape
            print scipy.optimize.check_grad(lambda w,X,Y: self.NLL(w,X,Y), lambda w,X,Y: self.GRAD(w,X,Y), w0, self.X, self.Y)

class LR_XR(LR):
    def __init__(self, X, Y, U, p_ex=0.5, l2=1.0, xr=10.0, temp=1.0):
        """ 
        Logistic regression with expectation regularization over class label distribution on unlabeled data.

        X -> labeled features
        Y -> labels
        U -> Unlabeled features
        p_exp -> expected proportion of positive examples in U
        l2 -> l2 regularization term
        """
        self.w0 = rand(X.shape[1])
        self.wStar = self.w0
        self.l2 = l2
        self.X = X
        self.Y = Y
        self.U = U
        self.p_ex = p_ex
        self.temp = temp
        self.xr = xr * self.X.shape[0]        #From Mann et. al. 2007: "we simply set \lambda = 10 * number of labeled examples"

    def XR_OBJ(self,w,X,Y,U):
        return self.NLL(w,X,Y) + self.xr * self.KL_TERM(w,X,Y,U)

    def XR_GRAD(self,w,X,Y,U):
        return self.GRAD(w,X,Y) + self.xr * self.KL_GRAD(w,X,Y,U)

    def KL_GRAD(self,w,X,Y,U):
        u_pred = self.Predict(U,w,temp=self.temp)
        q_wStar1  = np.sum(u_pred)
        q_wStar0 = np.sum(1.0 - u_pred)
        
        kl_grad = (((1.0 - self.p_ex)/q_wStar0) * u_pred * (1.0 - u_pred) 
                        - (self.p_ex /q_wStar1) * u_pred * (1.0 - u_pred))

        #kl_grad =  np.dot(U.T, kl_grad)
        kl_grad =  U.T.dot(kl_grad)
        kl_grad *= (1.0 / self.temp)
        return kl_grad

    def KL_TERM(self,w,X,Y,U):
        u_pred = self.Predict(U,w,temp=self.temp)
        p_em = np.sum(u_pred) / u_pred.shape[0]
        kl_div = self.p_ex * np.log(self.p_ex/p_em) + (1.0 - self.p_ex) * np.log((1.0 - self.p_ex)/(1.0 - p_em))
        return kl_div

    #These functions were used for incorporating temperature, but seemed to be mucking things up somehow....
#    def NLL_temp(self,w,X,Y):
    def NLL_old(self,w,X,Y):
        Y = (Y+1.0) / 2
        x_pred = self.Predict(X,w)
        #return -np.sum(Y * np.log(x_pred) + (1.0 - Y) * np.log(1.0 - x_pred)) + self.l2 * np.dot(w.T, w)
        #return np.sum(np.log(1 + np.exp(-Y*np.dot((1.0/self.temp)*w,X.T)))) + self.l2 * np.dot(w.T, w)
        return np.sum(np.log(1 + np.exp(-Y*np.dot(w,X.T)))) + self.l2 * np.dot(w.T, w)

#    def GRAD_temp(self,w,X,Y):
    def GRAD_old(self,w,X,Y):
        x_pred = self.Predict(X,w,temp=1.0)
        Y = (Y+1.0) / 2
        return np.dot(X.T, 
                      (x_pred - Y)) + 2 * self.l2 * w
#        return (1.0 / self.temp) * np.dot(X.T, 
#                                          (x_pred - Y)) + 2 * self.l2 * w

    def Predict(self,X, w=None, temp=1.0):
        if w == None:
            w = self.wStar
#        return 1.0 / (1.0 + np.exp(-np.dot(w, X.T)))
        #return 1.0 / (1.0 + np.exp(-(1.0/temp)*np.dot(w, X.T)))      #Incorporate temperature, (Mann et. al. 2007).
        return 1.0 / (1.0 + np.exp(-(1.0/temp)*X.dot(w)))      #Incorporate temperature, (Mann et. al. 2007).

    def Train(self):
        #for t in [10., 1., 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        for t in [1e-2, 1e-6]:
            (self.wStar, self.nll, self.status) = scipy.optimize.fmin_l_bfgs_b(lambda w,X,Y,U: self.XR_OBJ(w,X,Y,U), x0=self.wStar, fprime = lambda w,X,Y,U: self.XR_GRAD(w,X,Y,U), args = (self.X, self.Y, self.U), pgtol = t)
            print "nll:\t%s" % self.nll

    def CheckGrad(self):
        for i in range(10):
            w0 = rand(self.X.shape[1])
            print "CHECKGRAD:%s" % scipy.optimize.check_grad(lambda w,X,Y,U: self.XR_OBJ(w,X,Y,U), lambda w,X,Y,U: self.XR_GRAD(w,X,Y,U), w0, self.X, self.Y, self.U)
            print "CHECKGRAD_KL:%s" % scipy.optimize.check_grad(lambda w,X,Y,U: self.KL_TERM(w,X,Y,U), lambda w,X,Y,U: self.KL_GRAD(w,X,Y,U), w0, self.X, self.Y, self.U)

class XR2(LR_XR):
    def __init__(self, U1, U2, p_ex1=1.0, p_ex2=0.5, l2=1.0, xr1=10.0, xr2=10.0, temp=1.0):
        """ 
        U -> Unlabeled features
        """
        self.w0 = rand(U1.shape[1])
        self.wStar = self.w0
        self.l2 = l2
        self.U1 = U1
        self.U2 = U2
        self.p_ex1 = p_ex1
        self.p_ex2 = p_ex2
        self.temp = temp
        self.xr1 = xr1
        self.xr2 = xr2

    def XR2_OBJ(self,w,U1,U2):
        return self.xr1 * self.KL_TERM(w, self.U1, self.p_ex1) + self.xr2 * self.KL_TERM(w, self.U2, self.p_ex2) + self.l2 * np.dot(w.T, w)

    def XR2_GRAD(self,w,U1,U2):
        return self.xr2 * self.KL_GRAD(w, self.U1, self.p_ex1) + self.xr2 * self.KL_GRAD(w, self.U2, self.p_ex2) + 2 * self.l2 * w

    def KL_GRAD(self,w,U,p_ex):
        u_pred = self.Predict(U,w,temp=self.temp)
        q_wStar1  = np.sum(u_pred)
        q_wStar0 = np.sum(1.0 - u_pred)

        kl_grad = (((1.0 - p_ex)/q_wStar0) * u_pred * (1.0 - u_pred) 
                        - (p_ex /q_wStar1) * u_pred * (1.0 - u_pred))

        #kl_grad =  np.dot(U.T, kl_grad)
        kl_grad =  U.T.dot(kl_grad)
        kl_grad *= (1.0 / self.temp)
        return kl_grad

    def KL_TERM(self,w,U,p_ex):
        u_pred = self.Predict(U,w,temp=self.temp)
        p_em = np.sum(u_pred) / u_pred.shape[0]
        kl_div = p_ex * np.log(p_ex/p_em) + (1.0 - p_ex) * np.log((1.0 - p_ex)/(1.0 - p_em))
        return kl_div

    def CheckGrad(self):
        for i in range(10):
            w0 = rand(self.U1.shape[1])
            print "CHECKGRAD:%s" % scipy.optimize.check_grad(lambda w,U1,U2: self.XR2_OBJ(w,U1,U2), lambda w,U1,U2: self.XR2_GRAD(w,U1,U2), w0, self.U1, self.U2)
            print "CHECKGRAD_KL1:%s" % scipy.optimize.check_grad(lambda w,U,p_ex: self.KL_TERM(w,U,p_ex), lambda w,U,p_ex: self.KL_GRAD(w,U,p_ex), w0, self.U1, self.p_ex1)
            print "CHECKGRAD_KL2:%s" % scipy.optimize.check_grad(lambda w,U,p_ex: self.KL_TERM(w,U,p_ex), lambda w,U,p_ex: self.KL_GRAD(w,U,p_ex), w0, self.U2, self.p_ex2)

    def Train(self):
        #for t in [10., 1., 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        for t in [1e-2, 1e-6]:
            (self.wStar, self.nll, self.status) = scipy.optimize.fmin_l_bfgs_b(lambda w,U1,U2: self.XR2_OBJ(w,U1,U2), x0=self.wStar, fprime = lambda w,U1,U2: self.XR2_GRAD(w,U1,U2), args = (self.U1, self.U2), pgtol = t)
            print "nll:\t%s" % self.nll


#pred = np.dot(wStar,X_train.T)
#print np.sum((pred >= 0) == (Y_train >= 0)) / float(Y_train.shape[0])

def syntheticData(n,m):
    #X = np.zeros((n,m+1))
    X = lil_matrix((n,m+1))

    class1 = normal(0.0,10.0,m)
    #class2 = normal(0.0,10.0,m)
    class2 = normal(0.0,100.0,m)

    for i in range(n):
        if rand() > 0.5:
            X[i,1:] = normal(class1, 10.0)
            X[i,0] = 1.0
        else:
            X[i,1:] = normal(class2, 10.0)
            X[i,0] = -1.0
    X[:,-1] = 1 #bias term
    return X.tocsr()

if __name__ == "__main__":
    #Some fake data for testing purposes...
#    data = rand(10000,10)
    data = syntheticData(10000,10)

    X = data[:, 1:]
#    X[X > 0]  = 1.0
#    X[X <= 0] = 0.0
    Y = data[:, 0].todense()
    #Y = data[:, 0]

    #Only train on positive data...
#    X_train =X[:20,:][Y[:20] == 1,:]
#    X_test = X[20:,:]

    X_train =X[:20,:]
    X_test = X[20:,:]

#    Y_train = Y[:20][Y[:20] == 1]
#    Y_test = Y[20:]

    Y_train = Y[:20]
    Y_test = np.squeeze(np.asarray(Y[20:]))

    print "Y_train=%s" % Y_train
    print "Y_test=%s" % Y_test
    print "X_test"
    #print X_test

#    lr = LR(X_train, Y_train, l2=1.0)
    #lr = LR_XR(X_train, Y_train, X_test, p_ex=0.5, temp=10.0)
#    lr = LR_XR(X_train, Y_train, X_test, p_ex=0.5, temp=0.9, xr=100.0)
    #lr = LR_XR(X_train, Y_train, X_test, p_ex=0.5, temp=1.0, xr=1000.0)
    #lr = LR_XR(X_train, Y_train, X_test, p_ex=0.5, temp=1.0, xr=100.0)
#    lr = LR_XR(X_train, Y_train, X_test, p_ex=0.5, temp=1.0, xr=10.0)

    for i in range(1):
        print "test %s -----------------------------" % i
        #lr = LR_XR(X_train, Y_train, X_test, p_ex=0.5, temp=1.0, xr=10.0)
        #lr = LR_XR(X_train, Y_train, X_test, p_ex=0.5, temp=1.0, xr=10.0)
        lr = LR_XR(X_train, Y_train, X_test, p_ex=0.5, temp=0.1, xr=10.0)
        #lr = XR2(X_train, X_test, p_ex1=1.0, p_ex2=0.0, temp=1.0, l2=0.05)
        #lr = LR(X_train, Y_train, l2=1.0)
        lr.CheckGrad()
        lr.Train()

        print list(lr.wStar)
        print lr.nll
        print lr.status

        #print "xr_grad:\t%s" % list(lr.XR_GRAD(lr.wStar, lr.X, lr.Y, lr.U))
        #print "nll_grad:\t%s" % list(lr.GRAD(lr.wStar, lr.X, lr.Y))

        #pred = np.dot(wStar,X_test.T)
        pred = lr.Predict(X_test)
        print pred
        print "predShape: %s" % str(pred.shape)
        print "mean:%s" % np.mean(pred)
        print "std:%s" % np.std(pred)
        print "perc. pos:%s" % (np.sum(pred > 0.5) / float(pred.shape[0]))
        print "y pos:%s" % (np.sum(Y_test >= 0) / float(Y_test.shape[0]))
        print Y_test.shape
        print "shape: %s" % str(((pred >= 0.5) == (Y_test >= 0)).shape)
        print np.count_nonzero((pred >= 0.5) == (Y_test >= 0))
        print np.sum((pred >= 0.5) == (Y_test >= 0)) / float(Y_test.shape[0])
