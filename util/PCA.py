import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import cupy as cp
import copy
from sklearn.decomposition import SparsePCA

class WeightQR():
    def __init__(self):
        pass

    def qr_basic(self, dataArray):
        Q, _ = cp.linalg.qr(dataArray)
        return Q

class WeightPCA():
    def __init__(self):
        pass

    def centerize(self, dataArray):
        mean = cp.mean(dataArray, axis=1)
        dataArray_new = copy.deepcopy(dataArray)
        for col in range(dataArray_new.shape[1]):
            dataArray_new[:,col] = dataArray_new[:,col] - mean
        return dataArray_new

    def print_intrinsic_dimension(self, values):
        index = 0
        changeFlag = False
        for i, l in enumerate(values):
            if l < 1e-7:
                index = i
                changeFlag = True
                break
        if not changeFlag: index = values.shape[0]
        intrinsic_index = 0
        values_sum = cp.sum(values)
        sum_cumul = 0
        for i, element in enumerate(values):
            sum_cumul += element
            if sum_cumul / values_sum > 0.99:
                intrinsic_index = i
                break
        print("99%-dimension : {}".format(intrinsic_index))
        print("real-dimension : {}".format(index))

    def print_eigenvalue(self, values, period=1):
        values_sum = cp.sum(values)
        print("lambda || cumulative sum || proportion of lambda || proportion of cumulative sum")
        sum_cumul = 0
        for i, element in enumerate(values):
            sum_cumul += element
            if i % period == 0:
                print("{:02d}. ".format(i), end="")
                print("{:.4f}".format(element), end=" || ")
                print("{:.4f}".format(sum_cumul), end=" || ")
                print("{:.4f}".format(element / values_sum), end=" || ")
                print("{:.4f}".format(sum_cumul / values_sum))
        
    def pca_basic(self, dataArray, isVerbose=False, period=1): #  just doing PCA(doing eigen composition about auto-correlation matrix)
        dataArray_centerized = self.centerize(dataArray)
        r = dataArray_centerized@cp.transpose(dataArray_centerized)
        values, basis = cp.linalg.eigh(r)
        values = values[::-1]
        basis = basis[:, ::-1]
        if isVerbose:
            print("="*20)
            self.print_intrinsic_dimension(values)
            self.print_eigenvalue(values, period=period)
            print("="*20) 
        return [values, basis]
    
    def pca_lowcost(self, dataArray, isVerbose=False, period=1):
        dataArray_centerized = self.centerize(dataArray)
        n_components = dataArray_centerized.shape[1]
        r = cp.transpose(dataArray_centerized)@dataArray_centerized
        values, projections = cp.linalg.eigh(r)
        values = values[::-1]
        projections = projections[:, ::-1]
        basisContainer = []
        for i in range(n_components):
            v = projections[:,i][:,None]
            basisContainer.append(cp.squeeze(dataArray_centerized@v) / cp.sqrt(values[i]))
        basis = cp.transpose(cp.array(basisContainer))
        if isVerbose:
            print("="*20)
            self.print_intrinsic_dimension(values)
            self.print_eigenvalue(values, period=period)
            print("="*20) 
        return [values, basis]
    
    def pca_proj(self, dataArray, isVerbose=False, period=1): # after doing PCA, project each data to principal subspace
        dataArray_centerized = self.centerize(dataArray)
        _, basis = self.pca_basic(dataArray, isVerbose=isVerbose, period=period)
        projections = cp.transpose(basis) @ dataArray_centerized
        return projections

    def pca_proj_lowcost(self, dataArray, isVerbose=False, period=1):
        dataArray_centerized = self.centerize(dataArray)
        r = cp.transpose(dataArray_centerized)@dataArray_centerized
        values, projections = cp.linalg.eigh(r)
        values = values[::-1]
        projections = projections[:, ::-1]
        if isVerbose:
            print("="*20)
            self.print_intrinsic_dimension(values)
            self.print_eigenvalue(values, period=period)
            print("="*20)
        return projections

class WeightSPCA():
    def __init__(self):
        pass

    # it seems we cannot get eigenvalues(variance) with SparsePCA...
    # so I do not define functions about print dimension information.

    def centerize(self, dataArray):
        mean = cp.mean(dataArray, axis=1)
        dataArray_new = copy.deepcopy(dataArray)
        for col in range(dataArray_new.shape[1]):
            dataArray_new[:,col] = dataArray_new[:,col] - mean
        return dataArray_new

    def pca_sparse(self, dataArray, alpha=0.25, isVerbose=False, period=1):
        dataArray_centerized = self.centerize(dataArray)
        n_components = dataArray.shape[1]
        transformer = SparsePCA(n_components=n_components, alpha=alpha, random_state=0)
        X = cp.transpose(dataArray_centerized)
        transformer.fit(X)
        basis = transformer.components_
        basis = cp.transpose(basis)
        basis = basis[:,::-1]
        return [np.zeros(n_components,), basis] # it seems we cannot get eigenvalues(variance) with SparsePCA...
    
    def pca_proj(self, dataArray, alpha=0.25, isVerbose=False, period=1):
        dataArray_centerized = self.centerize(dataArray)
        _, basis = self.pca_sparse(dataArray, isVerbose=isVerbose, period=period)    
        projections = cp.transpose(basis) @ dataArray_centerized
        return projections
        
class WeightKPCA():
    def __init__(self):
        pass

    def centerize(self, K):
        N = K.shape[0]
        one_n = cp.ones((N, N)) / N
        K_new = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        return K_new

    def print_intrinsic_dimension(self, values):
        index = 0
        changeFlag = False
        for i, l in enumerate(values):
            if l < 1e-7:
                index = i
                changeFlag = True
                break
        if not changeFlag: index = values.shape[0]
        print("real-dimension : {}".format(index))

    def print_eigenvalue(self, values, period=1):
        values_sum = cp.sum(values)
        print("lambda || cumulative sum || proportion of lambda || proportion of cumulative sum")
        sum_cumul = 0
        for i, element in enumerate(values):
            sum_cumul += element
            if i % period == 0:
                print("{:02d}. ".format(i), end="")
                print("{:.4f}".format(element), end=" || ")
                print("{:.4f}".format(sum_cumul), end=" || ")
                print("{:.4f}".format(element / values_sum), end=" || ")
                print("{:.4f}".format(sum_cumul / values_sum))

    def pca_kernel(self, K, isVerbose=False, period=1):
        K_centerized = self.centerize(K)
        values, basis = cp.linalg.eigh(K_centerized)
        # b = cp.array([1])
        # basisLen = cp.concatenate((b, cp.sqrt(values[1:])))
        values = abs(values)
        b = cp.array([1])
        basisLen = cp.concatenate((cp.sqrt(values[:-1]), b))
        basis = basis / basisLen
        # values = values[::-1]
        # basis = basis[:, ::-1]
        if isVerbose:
            print("="*20)
            self.print_intrinsic_dimension(values)
            self.print_eigenvalue(values, period=period)
            print("="*20)
        return [values, basis]
    
    def pca_proj_kernel(self, K, isVerbose=False, period=1):
        _, basis = self.pca_kernel(K, isVerbose=isVerbose, period=period)
        K_centerized = self.centerize(K)
        projections = cp.transpose(basis)@K_centerized
        return projections

class WeightRBFPCA(WeightKPCA):
    def __init__(self):
        pass

    # calculate kernel-matrix(K-matrix) based on RBF kernel function.
    def make_rbf_K_matrix(self, dataArray1, dataArray2, gamma):
        X1 = cp.transpose(dataArray1)
        X2 = cp.transpose(dataArray2)
        dims = X1.shape[1]
        rows = X1.shape[0]
        cols = X2.shape[0]
        out = cp.zeros((rows,cols))
        for dim in range(dims):
            out += cp.subtract.outer(X1[:,dim], X2[:,dim])**2
        K = cp.exp(-gamma * out)
        return K

    def rbf_kernel_pca(self, dataArray, gamma, isVerbose=False, period=1):
        K = self.make_rbf_K_matrix(dataArray, dataArray, gamma)
        values, basis = self.pca_kernel(K, isVerbose=isVerbose, period=period)
        return [values, basis]

    def rbf_kernel_pca_sum(self, dataArray1, dataArray2, gamma, isVerbose=False, period=1):
        dataArray = cp.concatenate((dataArray1, dataArray2), axis=1)
        values, basis = self.rbf_kernel_pca(dataArray, gamma, isVerbose=isVerbose, period=period)
        return [values, basis]
    
    def rbf_kernel_pca_karcher(self, dataArray1, dataArray2, gamma, isVerbose=False, period=1):
        n_components = dataArray1.shape[1]
        dataArray = cp.concatenate((dataArray1, dataArray2), axis=1)
        values, basis = self.rbf_kernel_pca(dataArray, gamma, isVerbose=isVerbose, period=period)
        return [values[:n_components], basis[:,:n_components]]
    
    def rbf_kernel_pca_projection(self, sumArray, dataArray, gamma, isVerbose=False, period=1):
        _, W = self.rbf_kernel_pca(sumArray, gamma, isVerbose=isVerbose, period=period)
        _, alphas = self.rbf_kernel_pca(dataArray, gamma, isVerbose=isVerbose, period=period)
        W = W
        alphas = alphas
        K = self.make_rbf_K_matrix(sumArray, dataArray, gamma)
        U, _ = cp.linalg.qr(cp.transpose(W) @ K @ alphas)
        return W@U