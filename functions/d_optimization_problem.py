from sklearn.cluster import KMeans 
import numpy as np
from sklearn.linear_model import MultiTaskLasso
import cvxpy as cp

#KMeans to initiate dictionaries
def initialize_dictionary(all_elements, Dictionary_Size):
    
    k = Dictionary_Size
    descr_kmeans=KMeans(n_clusters=k, random_state=0)
    descr_kmeans.fit(all_elements.T)
    dictionary = descr_kmeans.cluster_centers_  

    return dictionary.T

#Soft voting to initialize Lx (Zhang et al 2013 - 2.Soft Assignment Coding)
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


#Closed form solution Tian et al 2018
def update_M(Lx, Lw): 
    C_1 = np.matmul(Lw,Lx.T)
    C_2 = np.matmul(Lx,Lx.T)  + 0.75 * np.identity(25)
    inv_C_2=np.linalg.inv(C_2)
    
    M=np.matmul(C_1, inv_C_2)
    return M

#Closed form solution Tian et al 2018
def update_Lx(X, Dx, M, Lw):
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

    Y = np.vstack([W, gamma * M @ Lx])
    X = np.vstack([Dw, gamma * np.identity(25)])
    lasso = MultiTaskLasso(alpha=alpha)
    lasso.fit(X, Y)
    # print(Y.shape, X.shape)
    Lw = lasso.coef_.T
    return Lw


#QCQP with column by column update - Yang et al. METAFACE LEARNING FOR SPARSE REPRESENTATION BASED FACE RECOGNITION > 2. SPARSE REPRESENTATION BASED
# CLASSIFICATIION FOR FACE RECOGNITION > Step 3
X = np.random.uniform(0,1, [128, 300])
D = np.random.uniform(0,1, [128, 25])
Lx = np.random.uniform(0,1, [25, 300])
def update_dictionaries(X, D, L, n, no_samples):
    k = 25
    # (150x500) = (150x1)(1x500)
    for i in range(k):    
        Di = cp.Variable((n,1))
        # print("old: ",D[:10,i])
        objective =  cp.Minimize(cp.sum_squares(X - np.sum([D[:,j]@L[j,:].reshape(1,no_samples) 
                                                           for j in range(k) if i!=i]) + Di@L[i,:].reshape(1,no_samples)))
                                                 # cp.Minimize(np.linalg.norm(X - Dx@Lx, "fro"))
        
        constraints = [cp.norm2(Di) <= 1]

        prob = cp.Problem(objective, constraints) 
        try:
            prob.solve(solver='ECOS')
        except cp.error.SolverError:
            prob.solve(solver='SCS')

        print(prob.status) #infeasible_inaccurate
        D[:,i] = Di.value.reshape(n,) #AttributeError: 'NoneType' object has no attribute 'reshape'

    return D

Dx= update_dictionaries(X, D, Lx, 128, 300)
print(Dx.shape, np.max(Dx), np.min(Dx), np.linalg.norm(Dx - D))

def optimization_loop(X, Dx, Lx, W, Dw, Lw ):
    M = update_M(Lx, Lw)
    print("Initialized M", M.shape, np.max(M), np.min(M))
    Lx_new = update_Lx(X, Dx, M, Lw)
    print("Updated Lx", Lx_new.shape, np.max(Lx_new), np.min(Lx_new))
    Lw_new = update_Lw(W, Dw, M, Lx)
    print("Updated Lw", Lw_new.shape, np.max(Lw_new), np.min(Lw_new))
    Dx_new = update_dictionaries(X, Dx, Lx, 125)
    print("Updated Dx", Dx_new.shape, np.max(Dx_new), np.min(Dx_new))
    Dw_new = Dw # update_dictionaries(W, Dw, Lw, 640)
    print("Updated Dw", Dw_new.shape, np.max(Dw_new), np.min(Dw_new))
    M_new = update_M(Lx_new, Lw_new)
    print("Updated M", M_new.shape, np.max(M_new), np.min(M_new))
    print("dDx: ", np.linalg.norm(Dx_new-Dx))
    print("dDw: ", np.linalg.norm(Dw_new-Dw))
    print("dLx: ", np.linalg.norm(Lx_new-Lx))
    print("dLw: ", np.linalg.norm(Lw_new-Lw))
    print("dM: ", np.linalg.norm(M_new-M))


# W = np.load("data/X_W_arrays/W_space.npy")
# X = np.load("data/X_W_arrays/X_space.npy")
# Dx = np.load("data/dictionaries/Dx.npy")
# Dw = np.load("data/dictionaries/Dw.npy")
# Lx = np.load("data/dictionaries/Lx.npy")
# Lw= np.load("data/dictionaries/Lw.npy")
# M = update_M(Lx, Lw)

# print(np.load("data/X_W_arrays/W_space.npy").shape)
# print(np.load("data/X_W_arrays/X_space.npy").shape)
# print(np.load("data/dictionaries/Dx.npy").shape)
# print(np.load("data/dictionaries/Dw.npy").shape)
# print(np.load("data/dictionaries/Lx.npy").shape)
# print(np.load("data/dictionaries/Lw.npy").shape)

# M = update_M(Lx, Lw)
# print("Initialized M", M.shape, np.max(M), np.min(M))
# M_new = update_M(Lx_new, Lw_new)
# print("Updated M", M_new.shape, np.max(M_new), np.min(M_new))

# Dx_new = update_dictionaries(X, Dx, Lx, 125)

# optimization_loop(X, Dx, Lx, W, Dw, Lw )

# print(np.load('data/dictionaries/Dx.npy').shape)

