import numpy as np
import scipy.spatial.distance as distance
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter
import random

import sys #для записи в файл


#the number of columns and rows of matrix A
def n_col(A):
    return len(A[0])

def n_row(A):
    return len(A)

#создает матрицу случайных чисел [0,100]
def create_matrix(n,m):
    Matrix = np.ones((n,m))
    for i in range (n):
        for j in range (m):
            #Matrix[i][j] = random.randint(0,3)
            Matrix[i,j]=random.uniform(0, 100)
    return Matrix

def frob_norm(A):
    norm=0
    n=n_col(A)
    m=n_row(A)
    for i in range(m):
        for j in range(n):
            norm+=(A[i,j])**2
    return np.sqrt(norm)

def _NMF_step(X,W,H):
    N = n_row(W)#=n_row(X)
    d = n_col(W)
    p=n_col(H)#=n_col(X)

    W1=np.ones((N,d))
    for i in range(N):
        for j in range(d):
            W1[i, j] = W[i, j]
    H1=np.ones((d,p))
    for i in range(d):
        for j in range(p):
            H1[i, j] = H[i, j]

    for i in range(d):
        for j in range(p):
            #A = np.dot(np.transpose(W1), X)#B = np.dot(np.dot(np.transpose(W1), W1), H1)#H1[i, j] = H1[i, j] * (A[i, j] / B[i, j])
            A = np.dot(np.transpose(W1)[i], X[:,j])
            B = np.dot(np.dot(np.transpose(W1)[i], W1), H1[:,j])
            H1[i, j] = H1[i, j] * (A / B)

    for i in range(N):
        for j in range(d):
            #C = np.dot(X, np.transpose(H1))
            C = np.dot(X[i], np.transpose(H1)[:,j])
            #D = np.dot(W1, np.dot(H1, np.transpose(H1)))
            D = np.dot(W1[i], np.dot(H1, np.transpose(H1)[:,j]))
            W1[i, j] = W1[i, j] * (C / D)

    return W1,H1

def _NMF(X,d,num_of_iterations,num_of_start_points,file_name):
    """NMF factorization with d latent variables"""
    original_stdout = sys.stdout
    File_NMF = open(file_name, 'w')
    sys.stdout = File_NMF
    # with open('GridPointsValue','a') as file:
    print('number of iterations=' +str(num_of_iterations)+ '\n'+'number of start points='+str(num_of_start_points))
    print('rank od factorization='+str(d)+'\n')
    N=n_row(X)
    p=n_col(X)
    min_targ_func_value=0
    for num in range(num_of_start_points):
        W = create_matrix(N, d)
        H = create_matrix(d, p)
        #list_of_targ_func_values = [frob_norm(X - np.dot(W, H))]
        for _ in range(num_of_iterations):
            W, H = _NMF_step(X, W, H)
            #targ_func_value = frob_norm(X - np.dot(W, H))
            #list_of_targ_func_values.append(targ_func_value)
        targ_func_value = frob_norm(X - np.dot(W, H))
        if num==0 or (min_targ_func_value>targ_func_value):
            min_targ_func_value=targ_func_value
            W_main=W
            H_main=H

    print('MAIN RESULTS')
    print('X\n', X)
    print('W\n', np.around(W_main, decimals=2), '\nH\n', np.around(H_main, decimals=2),'\ntarg_func',min_targ_func_value)

    sys.stdout = original_stdout
    File_NMF.close()

    return W_main,H_main,min_targ_func_value

def jNMF(X,Y,d,num_of_iterations,num_of_start_points):
    """jointNMF factorization with d latent variables"""
    XY=np.hstack((X,Y))
    p=n_col(X)
    q=n_col(Y)
    #W,H=_NMF(XY, d, num_of_iterations, num_of_start_points, 'jNMF')
    W, H,min_targ_func_value = _NMF(XY, d, num_of_iterations, num_of_start_points, 'workNMF')
    [H_x,H_y]=np.split(H,[p],axis=1)
    original_stdout = sys.stdout
    File_NMF = open('jNMF', 'w')
    sys.stdout = File_NMF

    print('X\n', np.around(X, decimals=2),'\n Y\n',np.around(Y, decimals=2))
    print('W\n', np.around(W, decimals=2))
    print('H_x\n',np.around(H_x, decimals=2))
    print('H_y\n', np.around(H_y, decimals=2))
    print('targ_func_value=', min_targ_func_value)

    sys.stdout = original_stdout
    File_NMF.close()


def iNMF_step(X,Y,W,V_x,V_y,H_x,H_y,lam):

    N=n_row(W)
    d=n_col(W)
    p=n_col(H_x)
    q=n_col(H_y)

    for i in range(N):
        for j in range(d):
            A_x = np.dot(H_x, np.transpose(H_x[j]))#[:,j])
            A_y = np.dot(H_y, np.transpose(H_y[j]))#[:,j])
            S_x = W[i] + V_x[i]
            S_y = W[i] + V_y[i]
            U_x = np.dot(S_x, A_x)
            U_y = np.dot(S_y, A_y)
            B_x = np.dot(X[i], np.transpose(H_x[j]))
            B_y = np.dot(Y[i], np.transpose(H_y[j]))
            D_x = np.dot(V_x[i], A_x)
            D_y = np.dot(V_y[i], A_y)

            cW=(B_x+B_y)/(U_x+U_y)
            W[i,j]=W[i,j]*cW
            cVx=B_x/(U_x+lam*D_x)
            V_x[i,j]=V_x[i,j]*cVx
            cVy=B_y / (U_y + lam * D_y)
            V_y[i, j] = V_y[i, j] * cVy
    for i in range(d):
        for j in range(p):
            S_x = W + V_x
            T_x = np.dot(np.dot(np.transpose(V_x)[i], V_x), H_x[:,j])
            C_x = np.dot(np.dot(np.transpose(S_x)[i], S_x), H_x[:,j])
            E_x = np.dot(np.transpose(S_x)[i], X[:,j])
            cHx = E_x / (C_x + lam * T_x)
            H_x[i,j]=H_x[i,j]*cHx
    for i in range(d):
        for j in range(q):
            S_y = W + V_y
            T_y = np.dot(np.dot(np.transpose(V_y)[i], V_y), H_y[:,j])
            C_y = np.dot(np.dot(np.transpose(S_y)[i], S_y), H_y[:,j])
            E_y = np.dot(np.transpose(S_y)[i], Y[:,j])
            cHy = E_y / (C_y + lam * T_y)
            H_y[i, j] = H_y[i, j] * cHy

    return W,V_x,V_y,H_x,H_y

def targ_func_for_iNMF(X,Y,W,V_x,V_y,H_x,H_y,lam):
    targ_func_value = (frob_norm(X - np.dot(W + V_x, H_x))) ** 2 + (frob_norm(Y - np.dot(W + V_y, H_y))) ** 2 + lam * (
            (frob_norm(np.dot(V_x, H_x))) ** 2 + (frob_norm(np.dot(V_y, H_y))) ** 2)
    return targ_func_value

def iNMF(X,Y,d,lam,num_of_iterations,num_of_start_points):
    """NMF factorization with d latent variables"""
    original_stdout = sys.stdout
    File_iNMF = open('iNMF', 'w')
    sys.stdout = File_iNMF
    print('parameters:\n'+'lambda='+str(lam)+'\n rank of factorization='+str(d))

    if n_row(X)!=n_row(Y):
        print('warning! rows of X is not equal to rows of Y')
    N=n_row(X)
    p=n_col(X)
    q=n_col(Y)
    min_targ_func_value = 0
    for num in range(num_of_start_points):
        W = create_matrix(N, d)
        V_x = create_matrix(N, d)
        V_y = create_matrix(N, d)
        H_x = create_matrix(d, p)
        H_y = create_matrix(d, q)
        #list_of_aims = [targ_func_for_iNMF(X,Y,W,V_x,V_y,H_x,H_y,lam)]
        for _ in range(num_of_iterations):
            W, V_x, V_y, H_x, H_y = iNMF_step(X, Y, W, V_x, V_y, H_x, H_y, lam)
            #list_of_aims.append(targ_func_for_iNMF(X, Y, W, V_x, V_y, H_x, H_y, lam))
        #print('list of values of target function',list_of_aims,'\n')
        #for ii in range(len(list_of_aims)-1):
         #   if list_of_aims[ii]<list_of_aims[ii+1]:
         #       print('ofjofh')

        targ_func_value = targ_func_for_iNMF(X,Y,W,V_x,V_y,H_x,H_y,lam)

        if num==0 or (min_targ_func_value>targ_func_value):
            min_targ_func_value=targ_func_value
            W_main=W
            V_x_main=V_x
            V_y_main=V_y
            H_x_main=H_x
            H_y_main=H_y

    print('\n MAIN RESULTS')
    print('X\n', X)
    print('Y\n', Y)
    print('W\n', np.around(W_main, decimals=7),'\nV_x', np.around(V_x_main, decimals=7),'\nV_y\n',
          np.around(V_y_main, decimals=7),'\nH_x\n',np.around(H_x_main, decimals=7),
          '\nH_y\n',np.around(H_y_main, decimals=7), '\ntarg_func',
          min_targ_func_value)


    sys.stdout = original_stdout
    File_iNMF.close()

    return W,V_x,V_y,H_x,H_y


def main():
    matX = np.loadtxt('X')  # читаю данные из файла как матрицу
    matY = np.loadtxt('Y')  # читаю данные из файла как матрицу
    matXX = np.loadtxt('XX')
    N=len(matX) # the number os samples, i.e. rows of X and Y
    p=len(matX[0])
    q=len(matY[0])

    d=2 #quantity of latent variables (columns of W)
    lam=100
    #for i in range(len(aims_values)-1):
        #if aims_values[i]<aims_values[i+1]:
            #print('aiaiai')


    W,H,min_targ_func_value=_NMF(matXX,3,4,10,'NMF')
    print('W',W)
    print('H',H)

    print(matXX-np.dot(W,H))
    print(np.dot(W, H))

    print('iNMF')
    W,V_x,V_y,H_x,H_y=iNMF(matX, matY, d, lam, 7,7)


    print('W',np.around(W, decimals=3))
    print('V_x',np.around(V_x, decimals=3))
    print('H_x',np.around(H_x, decimals=3))
    print('V_y', np.around(V_y, decimals=3))
    print('H_y',np.around(H_y, decimals=3))

    print(np.dot(W+V_x,H_x))
    print(np.dot(W + V_y, H_y))

    print('jNMF')
    jNMF(matX, matY, d, 8, 4)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
