from sklearn.cluster import KMeans 
import numpy as np
from sklearn.linear_model import MultiTaskLasso
import cvxpy as cp


def initialize_dictionary(all_elements, Dictionary_Size):
    
    k = Dictionary_Size
    descr_kmeans=KMeans(n_clusters=k, random_state=0)
    descr_kmeans.fit(all_elements.T)
    dictionary = descr_kmeans.cluster_centers_  

    return dictionary.T


def initialize_coefficients(D, all_ellements, a):
    #X: (150, 500), d: (150X25), a: soothing factor
    coefficients=np.zeros((D.shape[1] , all_ellements.shape[1]))
    
    for j, sample in enumerate(all_ellements.T):
        for i, word in enumerate(D.T):
            diff = np.linalg.norm(sample-word)**2
            c_1=np.exp(-a * diff)
            coefficients[i][j]=c_1
    coefficients[:,j]/=np.sum(coefficients[:,j])
    return coefficients


def update_M(Lx, Lw): 
    C_1 = np.matmul(Lw,Lx.T)
    C_2 = np.matmul(Lx,Lx.T)  + 0.75 * np.identity(25)
    inv_C_2=np.linalg.inv(C_2)
    
    M=np.matmul(C_1, inv_C_2)
    return M


def update_Lx(X, W, Dx, Dw, M, Lw):
    #update Lw
    C_1 = np.matmul(Dx.T,Dx)  #25x25
    C_2 = 0.8 * np.matmul(M.T,M) #25x25
    C_a = C_1+C_2+0.6*np.identity(25)
    inv_Ca = np.linalg.inv(C_a)


    C_3 = np.matmul(Dx.T,X)
    C_4 = 0.8 * np.matmul(M.T,Lw)
    C_b= C_3+C_4

    Lx = np.matmul(inv_Ca,C_b)
    return Lx


def update_Lw(W, Dw, M, Lx, gamma=0.7, alpha=0.01):
    print(W.shape)
    print(M.shape)
    print(Lx.shape)
    Y = np.vstack([W, M @ Lx])
    X = np.vstack([Dw, gamma * np.identity(25)])
    lasso = MultiTaskLasso(alpha=alpha)
    lasso.fit(X, Y)
    # print(Y.shape, X.shape)
    Lw = lasso.coef_.T
    return Lw

def update_dictionaries(X, D, L, n):
    k = 25
    # (150x500) = (150x1)(1x500)
    for i in range(k):    
        Di = cp.Variable((n,1))
        # print("old: ",D[:10,i])
        objective =  cp.Minimize(cp.sum_squares(X - np.sum(D[:,j]@L[j,:].reshape(1,500) for j in range(k) if i!=i)+ Di@L[i,:].reshape(1,500)))# cp.Minimize(np.linalg.norm(X - Dx@Lx, "fro"))

        constraints = [cp.norm2(Di) <= 1]

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver='ECOS')
        except cp.error.SolverError:
            prob.solve(solver='SCS')
        
        D[:,i] = Di.value.reshape(n,)

    return D
