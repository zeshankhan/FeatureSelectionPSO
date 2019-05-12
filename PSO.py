#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 18:01:39 2019

@author: alvi
"""

def error(X_train,Y_train,X_test,Y_test):
  from sklearn.metrics import mean_squared_error, r2_score
  from sklearn.svm import SVR
  svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
  svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1)
  y_rbf = svr_rbf.fit(X_train, Y_train).predict(X_test)
  y_poly = svr_poly.fit(X_train, Y_train).predict(X_test)
  mse_rbf=mean_squared_error(Y_test,y_rbf)
  mse_poly=mean_squared_error(Y_test,y_poly)
  error=(mse_rbf+mse_poly)/2
  return error
def PSO_update(particles,p_history,p_history_err,V,w=0.5,c1=1,c2=1,swarm=12,g=14,columns=0,leader=0):
  import random
  for i in range(swarm):
    min_err_g=0
    for j in range(g):
      if(p_history_err[j,i]<=p_history_err[min_err_g,i]):
        min_err_g=j
    r1=random.random()
    r2=random.random()
    cognetive=0
    social=0
    for j in range(columns):
      cognetive+=(p_history[min_err_g,i,j]-particles[i,j])**2
      social+=(particles[leader,j]-particles[i,j])**2
    w=0.5
    V[i]=w*V[i]+(c1*r1*cognetive)+(c2*r2*social)
    v=0
    while(v<V[i]):#for j in range(columns):
      col=random.randint(0, columns-1)
      #print(type(particles[i,col]),particles[i,col])
      particles[i,col]=(particles[i,col]+1)%2
      v+=1
  return particles,V
#Paritcle Swarm Optimization algorith
def PSO(X=None,Y=None,generations=14,swarm=12,w=0.5,c1=1,c2=1):
  import numpy as np
  columns=X.shape[1]
  particles=np.random.randint(2, size=(swarm,columns))
  p_history=np.zeros((generations,swarm,columns),int)
  p_history_err=np.zeros((generations,swarm),float)
  V=np.zeros((swarm),float)

  from sklearn.model_selection import train_test_split
  X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30)
  import copy
  err=np.zeros(swarm,float)
  print('Starting PSO with ',generations,' generations of ',swarm,' particles')
  for g in range(generations):
    i=0
    leader=0
    #Fitness
    #from sklearn.model_selection import cross_validate
    for i in range(swarm):
      x_train=copy.deepcopy(X_train)
      x_test=copy.deepcopy(X_test)
      for j in range(columns):
        if(particles[i,j]==0):
          x_train=x_train.drop(columns=[X_train.columns[j]])
          x_test=x_test.drop(columns=[X_train.columns[j]])
      #print('Computing Error of particle ',i)
      err[i]=error(x_train,Y_train,x_test,Y_test)
      #print('Error of particle number ',i,' is ',err[i],' and leader is particle number ',leader,' with error ',err[leader])
      if(err[i]<err[leader]):
        err[leader]=err[i]
        leader=i
    print("Leader: ",leader,'\t Error: ',err[leader],'\tSize: ',sum(particles[leader]))
    if(g==generations-1):
        return particles[leader],err[leader]
    #print(err)
    p_history[g]=particles
    p_history_err[g]=err
    particles,V=PSO_update(particles,p_history,p_history_err,V,w=0.5,c1=c1,c2=c2,swarm=swarm,g=g,columns=columns,leader=leader)
    print('Iteration '+str(g)+' Completed')
  print(particles.shape)