import numpy as np
from sklearn.cluster import KMeans 
import os
from sklearn.linear_model import LassoLars
 


def initialize_dictionary(all_elements, Dictionary_Size):
    
    k = Dictionary_Size
    descr_kmeans=KMeans(n_clusters=k, random_state=42)
    descr_kmeans.fit(all_elements.T)
    dictionary = descr_kmeans.cluster_centers_  

    return dictionary.T


def initialize_coefficients(x, b, a):
    coefficients=np.zeros((x.shape[1] , b.shape[1]))
    
    for j, b_j in enumerate(b.T):
        for i, x_i in enumerate(x.T):
            diff = np.linalg.norm(x_i-b_j)**2 
            nominator=np.exp(-a * diff)
            coefficients[i][j]=nominator
        coefficients[:,j]/=np.sum(coefficients[:,j])
    return coefficients.T


def obj_M(Lx, Lw, M, li=0.8, g = 0.8):
    A = np.linalg.norm(Lw-M@Lx)**2 + (li/g) * np.linalg.norm(M)**2
    return np.linalg.norm(Lw-M@Lx)**2, np.linalg.norm(M)**2, A


def update_M(Lx, Lw, li=0.8, g = 0.8): 
    C_1 = Lw@Lx.T #5x5
    C_2 = Lx@Lx.T  + (li/g) * np.identity(Lw.shape[0]) #5x5
    inv_C_2=np.linalg.inv(C_2) #5X5
    
    M=C_1 @ inv_C_2
    return M #5X5


def obj_Lx(X, Dx, M, Lw, Lx, g=0.8, li=0.8):
    A= np.linalg.norm(X-Dx@Lx)**2 + g * np.linalg.norm(Lw - M@Lx)**2 + li* np.linalg.norm(Lx)**2
    return np.linalg.norm(X-Dx@Lx)**2, np.linalg.norm(Lw - M@Lx)**2, np.linalg.norm(Lx)**2, A


def update_Lx(X, Dx, M, Lw, g=0.8, li=0.8):
    C_1 = np.matmul(Dx.T,Dx)  #5x5
    C_2 = g * np.matmul(M.T,M) #5x5
    C_a = C_1+C_2+ li*np.identity(Dx.shape[1]) #5x5
    inv_Ca = np.linalg.inv(C_a) #5x5


    C_3 = np.matmul(Dx.T,X) #5x28
    C_4 = g * np.matmul(M.T,Lw)#5x28
    C_b= C_3+C_4 #5x28

    Lx = np.matmul(inv_Ca,C_b)
    
    return Lx #5x28

def obj_Lw(W, Dw, M, Lx, Lw, li=0.01, gamma=0.7):
    A= np.linalg.norm(W - Dw@Lw)**2 + gamma * np.linalg.norm(M@Lx - Lw)**2 + li * np.linalg.norm(Lw, ord = 1)
    return np.linalg.norm(W - Dw@Lw)**2, np.linalg.norm(M@Lx - Lw)**2, np.linalg.norm(Lw, ord = 1), A


def lars_col_by_col(X, y , li=0.01):
    lasso = LassoLars(alpha=li)
    lasso.fit(X, y)
    return lasso.coef_

def update_Lw_step(W, Dw, M, Lx, li=0.01, gamma=0.7):

    Y = np.vstack([W, gamma * M @ Lx])
    X = np.vstack([Dw, gamma * np.identity(Dw.shape[1])])
    Lw = np.empty([Dw.shape[1], W.shape[1]])
    
    for i, y in enumerate(Y.T):
        lw_col = lars_col_by_col(X, y, li)
        Lw[:, i] = lw_col

    return Lw


def obj_dicts(X, Dx, Lx, W, Dw, Lw):
    return np.linalg.norm(X-Dx@Lx)**2,  np.linalg.norm(W-Dw@Lw)**2


def qcqp_eq_constraint_col(X, G, L):
    k = G.shape[1]
    G_new = np.empty(G.shape)
    for j in range(k):
        summary_term=  np.sum([G[:,i].reshape(G.shape[0], 1)\
                        @L[i,:].reshape(1, L.shape[1]) \
                            for i in range(k) if i!=j], axis=0)
        Y = X -summary_term
        G_new[:,j] = ( Y @ L[j].T ) / np.linalg.norm(Y @ L[j].T)
    return G_new


def update_Dx_Dw(X, Dx, Lx, W, Dw, Lw):
    Dx = qcqp_eq_constraint_col(X, Dx, Lx)
    Dw = qcqp_eq_constraint_col(W, Dw, Lw)
    return Dx, Dw


def optimization_loop(X, W, epochs):
    #Normalize W,Dw
    W/=47.18468020040863
    print(np.linalg.norm(W[:,5]))
    
    
    # slack variables
    dictionary_size = 5
    a_coeff = 0.5
    li= 0.00005
    g = 1
    
    # Initalization
    Dx = initialize_dictionary(X, dictionary_size)
    Dw = initialize_dictionary(W, dictionary_size)
    Lx = initialize_coefficients(X, Dx, a_coeff)
    Lw = initialize_coefficients(W, Dw, a_coeff)
    
 
    for i in range(epochs):
        print(f"----- Epoch {i} -----")
        
        # Update M
        if i > 0:
            print("Obj M before:", obj_M(Lx, Lw, M, li, g))
        M=update_M(Lx, Lw, li, g)
        # print(M)
        print("Obj M after:", obj_M(Lx, Lw, M, li, g))
        print()
        
        # Update Lx
        print("Obj Lx before:", obj_Lx(X, Dx, M, Lw, Lx, li, g))
        Lx = update_Lx(X, Dx, M, Lw, li, g)
        # print(Lx)
        print("Obj Lx after:", obj_Lx(X, Dx, M, Lw, Lx, li, g))
        print()
        
        # Update Lw
        print("Obj Lw before:", obj_Lw(W, Dw, M, Lx, Lw, li, g))
        Lw = update_Lw_step(W, Dw, M, Lx, li, g)
        # print(Lw)
        print("Obj Lw after:", obj_Lw(W, Dw, M, Lx, Lw, li, g))
        # print(Lw)
        print()
        
        # Update Dx, Dw
        print("Obj Dx, Dw before:", obj_dicts(X, Dx, Lx, W, Dw, Lw))
        Dx, Dw= update_Dx_Dw(X, Dx, Lx, W, Dw, Lw)
        print("Obj Dx, Dw after:",obj_dicts(X, Dx, Lx, W, Dw, Lw))
        print()
        print()
        print(np.linalg.norm(W[:,5]))
        
        # print(' ## Dw ## \n', Dw)
        
        # Normalize W,Dw
    print(np.linalg.norm(W[:,5]))
    print(np.linalg.norm(Dw[:,4]))
    
    W*=47.18468020040863
    Dw*=47.18468020040863    
    print(np.linalg.norm(W[:,5]))
    print(np.linalg.norm(Dw[:,4]))
        
    return Dx, Dw, Lx, Lw, M   
    
    