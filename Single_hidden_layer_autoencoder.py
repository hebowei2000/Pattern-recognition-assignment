import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import random
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y=df.iloc[0:1000,4].values
y=np.where(y=='Iris-setosa',0.0,1.0)
x = df.iloc[0:1000,[0,2]].values
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
#########################
class Sinautoencoder():
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
        self.n=0.018
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
                self.iw[i][j]=2*tools.rand(0,1)-1
        for i in range(self.hiddenn):
            for j in range(self.outputn):
                self.ow[i][j]=2*tools.rand(0,1)-1
                
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
                 #   self.o[i]=tools.sigmoid(self.o[i])
               
                    
    def update(self,n):
            g=[0.0]*self.outputn
            e=[0.0]*self.hiddenn
            self.n=0.03
            for i in range(self.outputn):
                y=self.o[i]
              #  g[i]= (n[i]-y)*y*(1-y)
                g[i]=n[i]-y
                
            for i  in range(self.hiddenn):
                wg=0
                for j in range(self.outputn):
                    wg=wg+self.ow[i][j]*g[j]
                    e[i]=self.h[i]*(1-self.h[i])*wg
                   # e[i]=wg
                
            for i in range(self.hiddenn):
                for j in range(self.outputn):
                    self.ow[i][j]=self.ow[i][j]+self.n*g[j]*self.h[i]
            for i in range(self.inputn):
                for j in range(self.hiddenn):
                    self.iw[i][j]=self.iw[i][j]+self.n*e[j]*self.i[i]
                   
                
    def train(self,n,p):
            self.setup(len(n[0]),len(p[0]),int(math.sqrt(len(n[0])+len(p[0])))+8)
            self.errors_=[]
            for time in range(30):
                errors=0;
                diff=0;
                k=0;
                for i in n:
                    self.predict(i)
                    self.update(p[k])
                    print(self.o[:],p[k])
                    for j in range(self.outputn):
                        if abs(p[k][j]-self.o[j])>0.2:
                            errors+=pow(p[k][j]-self.o[j],2)
                    k=k+1;
                errors = errors/len(n)
                self.errors_.append(errors)
###########################################
sac=Sinautoencoder()
y=x
sac.train(x,y)
plt.plot(range(1, len(sac.errors_) + 1), sac.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('averaged restruction error')
plt.show()
