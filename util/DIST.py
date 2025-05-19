from abc import *
import pandas as pd
import cupy as cp
from . import SUBSPACE

smTool = SUBSPACE.SubspaceDiff()

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Dist(metaclass=ABCMeta):

    def _make_mask(self, idxContainer, sampleNum, trajNum):
        mask = []
        for i in range(sampleNum*trajNum):
            for idx in idxContainer:
                if i >=idx*sampleNum and i < (idx+1)*sampleNum : mask.append(i)
        return mask

    def save_K_matrix(self, K, prefix):
        columnNum = K.shape[1]
        columnContainer = []
        for i in range(columnNum):
            columnContainer.append("{:02d}".format(i))

        new_K = cp.asnumpy(K)
        df = pd.DataFrame(new_K, columns=columnContainer)
        makedirs("./dist")
        df.to_csv("./dist/" + prefix + ".csv", index=False)

    def load_K_matrix(self, prefix):
        df = pd.read_csv("./dist/" + prefix + ".csv")
        K = cp.asarray(df.values)
        return K
    
    def mask_K_matrix(self, K, idxContainer, trajNum):
        sampleNum = K.shape[0]
        sampleNum = int(sampleNum / trajNum)
        mask = self._make_mask(idxContainer, sampleNum, trajNum)
        K_new = K[mask] 
        K_new = K_new[:,mask]
        return K_new
    
    @abstractmethod
    def calc_dist(self):
        pass

    @abstractmethod
    def construct_K_matrix(self):
        pass

class GrassmannDist(Dist):
    def calc_dist(self, basis1, basis2):
        return smTool.calc_magnitude(basis1, basis2)

    def construct_K_matrix(self, basisContainer1, basisContainer2):
        rowNum = len(basisContainer1)
        colNum = len(basisContainer2)
        K = cp.zeros((rowNum, colNum))
        for i, basis_i in enumerate(basisContainer1):
            for j, basis_j in enumerate(basisContainer2):
                K[i,j] = self.calc_dist(basis_i, basis_j)
        return K
    
    def construct_K_matrix(self, basisContainer):
        rowNum = len(basisContainer)
        colNum = len(basisContainer)
        K = cp.zeros((rowNum, colNum))
        for i, basis_i in enumerate(basisContainer):
            for j, basis_j in enumerate(basisContainer):
                if j > i:
                    tmp = self.calc_dist(basis_i, basis_j)
                    K[i,j] = tmp
                    K[j,i] = tmp
        return K