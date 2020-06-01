import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import random
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.head(10)
y=df.iloc[0:1000,4].values
y=np.where(y=='Iris-setosa',0.0,1.0)
x = df.iloc[0:1000,[0,2]].values
#plt.scatter(x[:50, 0], x[:50,1], color='red', marker='o', label='setosa')
#plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='versicolor')
#plt.xlabel('petal length')
#plt.ylabel('sepal length')
#plt.legend(loc='upper left')
#plt.show
class Perceptron(object):
    """
    Parameters
    ------------
    eta : float
        学习率 (between 0.0 and 1.0)
    n_iter : int
        迭代次数
    Attributes
    -----------
    w_ : 1d-array
        权重
    errors_ : list
        误差
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, x, y):
        self.w_ = np.zeros(1 + x.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1.0, 0.0)

###########################
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(x, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()
###############################
##we test three groups pf random seeds
random.seed(0)
class tools():
    def rand(a,b):
        return (b-a)*random.random()+a
    def set_m(n,m):
        a=[];
        for i in range(n):
            a.append([0.0]*m)
        return a
    def sigmoid(x):
        return 1.0/(1.0+math.exp(-x))
    def sigmoid_derivate(x):
        return x*(1-x)
#####################
class bpnn():
    def _init_(self):
        self.inputn=0
        self.hiddenn=0
        self.output=0
        self.iw=[]
        self.ow=[]
        self.iwco=[]
        self.owco=[]
        self.i=[]
        self.h=[]
        self.o=[]
        self.n=0.05
    def setup(self,inn,outn,hiddenn):
        self.inputn=inn;
        self.outputn=outn;
        self.hiddenn=hiddenn;
        self.i=[0.0]*self.inputn
        self.o=[0.0]*self.outputn
        self.h=[0.0]*self.hiddenn
        self.iwco=tools.set_m(self.inputn,self.hiddenn)
        self.owco=tools.set_m(self.hiddenn,self.outputn)     
        self.iw=tools.set_m(self.inputn,self.hiddenn)
        self.ow=tools.set_m(self.hiddenn,self.outputn)
        for i in range(self.inputn):
            for j in range(self.hiddenn):
                self.iw[i][j]=tools.rand(0,1)
        for i in range(self.hiddenn):
            for j in range(self.outputn):
                self.ow[i][j]=tools.rand(0,1)
                
    def predict(self,n):
            self.i=n;
            for i in range(self.hiddenn):
                self.h[i]=0;
                for j in range(self.inputn):
                    self.h[i]=self.h[i]+self.i[j]*self.iw[j][i] 
                self.h[i]=tools.sigmoid(self.h[i])
                
            for i in range(self.outputn):
                self.o[i]=0;
                for j in range(self.hiddenn):
                    self.o[i]=self.o[i]+self.h[j]*self.ow[j][i] 
                self.o[i]=tools.sigmoid(self.o[i])
                if self.o[i]>0.5:
                    self.o[i]=1.0
                else:
                    self.o[i]=0.0
                    
    def update(self,n):
            g=[0.0]*self.outputn
            e=[0.0]*self.hiddenn
            self.n=0.05
            for i in range(self.outputn):
                y=self.o[i]
                g[i]= (n[i]-y)*y*(1-y)
                
            for i  in range(self.hiddenn):
                wg=0
                for j in range(self.outputn):
                    wg=wg+self.ow[i][j]*g[j]
                e[i]=self.h[i]*(1-self.h[i])*wg
                
            for i in range(self.hiddenn):
                for j in range(self.outputn):
                    self.ow[i][j]=self.ow[i][j]+self.n*g[j]*self.h[i]
            for i in range(self.inputn):
                for j in range(self.hiddenn):
                    self.iw[i][j]=self.iw[i][j]+self.n*e[j]*self.i[i]
                   
                
    def train(self,n,p):
            self.setup(len(n[0]),len(p[0]),int(math.sqrt(len(n[0])+len(p[0])))+5)
            self.errors_=[]
            for time in range(10):
                errors=0;
                diff=0;
                k=0;
                for i in n:
                    self.predict(i)
                    self.update(p[k])
                    print(self.o[:],p[k])
                    for j in range(self.outputn):
                        if p[k][j]!=self.o[j]:
                            errors+=1
                    k=k+1;
                self.errors_.append(errors)
                            
########################################################             
nn=bpnn()
y=y.reshape((len(y),1))
nn.train(x,y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()
