import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import cupy as cp
from tqdm import tqdm
from . import AdjacentMag

class Layerwise1stMag(AdjacentMag.DoubleAdjacentMag):

    def __call__(self, csvFolder, layerNum, totalSampleNum):
        _layerNum = layerNum + 1
        if self.isConvLayer(_layerNum):
            result = self._pca_1stMag_layerwise_conv(csvFolder, layerNum, totalSampleNum=totalSampleNum)
        elif self.isFcLayer(_layerNum):
            result = self._pca_1stMag_layerwise_fc(csvFolder, layerNum, totalSampleNum=totalSampleNum)
        else:
            result = None
        return result

    def _pca_1stMag_layerwise_conv(self, csvFolder, layerNum, totalSampleNum=100):

        magContainer = []
        print("layer {}".format(layerNum))

        for i in tqdm(range(totalSampleNum - 1)):
            if i == 0: basis1, basis2 = self._get_adjacent_basis_conv(csvFolder, layerNum, i, isVerbose=True)
            else: basis1, basis2 = self._get_adjacent_basis_conv(csvFolder, layerNum, i)                
            magContainer.append(self.smTool.calc_magnitude(basis1, basis2))
        
        result = cp.array(magContainer).get()
        return result

    def _pca_1stMag_layerwise_fc(self, csvFolder, layerNum, totalSampleNum=100):

        magContainer = []
        print("layer {}".format(layerNum))

        for i in tqdm(range(totalSampleNum - 1)):
            if i == 0: basis1, basis2 = self._get_adjacent_basis_fc(csvFolder, layerNum, i, isVerbose=True)
            else: basis1, basis2 = self._get_adjacent_basis_fc(csvFolder, layerNum, i)                
            magContainer.append(self.smTool.calc_magnitude(basis1, basis2))
        
        result = cp.array(magContainer).get()
        return result
    
class Layerwise1stRBFMag(AdjacentMag.DoubleAdjacentRBFMag):

    def __call__(self, csvFolder, layerNum, totalSampleNum, gamma=5):
        _layerNum = layerNum + 1
        if self.isConvLayer(_layerNum):
            result = self._pca_1stMag_layerwise_conv(csvFolder, layerNum, gamma=gamma, totalSampleNum=totalSampleNum)
        elif self.isFcLayer(_layerNum):
            result = self._pca_1stMag_layerwise_fc(csvFolder, layerNum, gamma=gamma, totalSampleNum=totalSampleNum)
        else:
            result = None
        return result

    def _pca_1stRBFMag_layerwise_conv(self, csvFolder, layerNum, gamma=5, totalSampleNum=100):

        magContainer = []
        print("layer {}".format(layerNum))

        for i in tqdm(range(totalSampleNum - 1)):
            if i == 0: basis1, basis2 = self._get_adjacent_basis_conv(csvFolder, layerNum, i, gamma=gamma, isVerbose=True)
            else: basis1, basis2 = self._get_adjacent_basis_conv(csvFolder, layerNum, i, gamma=gamma)
            dataArray1, dataArray2 = self._get_adjacent_data_conv(csvFolder, layerNum, i)
            Kxy = self.pcaTool.make_rbf_K_matrix(dataArray1, dataArray2, gamma=gamma, centralizeFlag=False)
            magContainer.append(self.smTool.calc_rbf_magnitude(basis1, basis2, Kxy))
        
        result = cp.array(magContainer).get()
        return result

    def _pca_1stRBFMag_layerwise_fc(self, csvFolder, layerNum, gamma=5, totalSampleNum=100):

        magContainer = []
        print("layer {}".format(layerNum))

        for i in tqdm(range(totalSampleNum - 1)):
            if i == 0: basis1, basis2 = self._get_adjacent_basis_fc(csvFolder, layerNum, i, gamma=gamma, isVerbose=True)
            else: basis1, basis2 = self._get_adjacent_basis_fc(csvFolder, layerNum, i, gamma=gamma)                
            dataArray1, dataArray2 = self._get_adjacent_data_fc(csvFolder, layerNum, i)
            Kxy = self.pcaTool.make_rbf_K_matrix(dataArray1, dataArray2, gamma=gamma, centralizeFlag=False)
            magContainer.append(self.smTool.calc_rbf_magnitude(basis1, basis2, Kxy))
        
        result = cp.array(magContainer).get()
        return result

class Layerwise1stMag_decomposition(AdjacentMag.TripleAdjacentMag):

    def __call__(self, csvFolder, layerNum, totalSampleNum):
        _layerNum = layerNum + 1
        if self.isConvLayer(_layerNum):
            result = self._pca_1stMag_decomposition_conv(csvFolder, layerNum, totalSampleNum=totalSampleNum)
        elif self.isFcLayer(_layerNum):
            result = self._pca_1stMag_decomposition_fc(csvFolder, layerNum, totalSampleNum=totalSampleNum)
        else:
            result = None
        return result

    def _pca_1stMag_decomposition_conv(self, csvFolder, layerNum, totalSampleNum=100):

        alongContainer = []
        orthContainer = []
        print("layer {}".format(layerNum))

        for i in tqdm(range(totalSampleNum - 2)):
            if i == 0: basis1, basis2, basis3 = self._get_adjacent_basis_conv(csvFolder, layerNum, i, isVerbose=True)
            else: basis1, basis2, basis3 = self._get_adjacent_basis_conv(csvFolder, layerNum, i)                
            along, orth = self.smTool.calc_1st_magnitude_decomposed(basis1, basis2, basis3)
            alongContainer.append(along)
            orthContainer.append(orth)

        result1 = cp.array(alongContainer).get()
        result2 = cp.array(orthContainer).get()
        return [result1, result2]

    def _pca_1stMag_decomposition_fc(self, csvFolder, layerNum, totalSampleNum=100):

        alongContainer = []
        orthContainer = []
        print("layer {}".format(layerNum))

        for i in tqdm(range(totalSampleNum - 2)):
            if i == 0: basis1, basis2, basis3 = self._get_adjacent_basis_fc(csvFolder, layerNum, i, isVerbose=True)
            else: basis1, basis2, basis3 = self._get_adjacent_basis_fc(csvFolder, layerNum, i)                
            along, orth = self.smTool.calc_1st_magnitude_decomposed(basis1, basis2, basis3)
            alongContainer.append(along)
            orthContainer.append(orth)

        result1 = cp.array(alongContainer).get()
        result2 = cp.array(orthContainer).get()
        return [result1, result2]

class Layerwise2ndMag(AdjacentMag.TripleAdjacentMag):

    def __call__(self, csvFolder, layerNum, totalSampleNum):
        _layerNum = layerNum + 1
        if self.isConvLayer(_layerNum):
            result = self._pca_2ndMag_layerwise_conv(csvFolder, layerNum, totalSampleNum=totalSampleNum)
        elif self.isFcLayer(_layerNum):
            result = self._pca_2ndMag_layerwise_fc(csvFolder, layerNum, totalSampleNum=totalSampleNum)
        else:
            result = None
        return result

    def _pca_2ndMag_layerwise_conv(self, csvFolder, layerNum, totalSampleNum=100):

        magContainer = []
        print("layer {}".format(layerNum))

        for i in tqdm(range(totalSampleNum - 1)):
            if i == 0: basis1, basis2, basis3 = self._get_adjacent_basis_conv(csvFolder, layerNum, i, isVerbose=True)
            else: basis1, basis2, basis3 = self._get_adjacent_basis_conv(csvFolder, layerNum, i)                
            _, k = self.smTool.calc_karcher_subspace(basis1, basis3)
            magContainer.append(self.smTool.calc_magnitude(basis2, k))
        
        result = cp.array(magContainer).get()
        return result


    def _pca_2ndMag_layerwise_fc(self, csvFolder, layerNum, totalSampleNum=100):

        magContainer = []
        print("layer {}".format(layerNum))

        for i in tqdm(range(totalSampleNum - 1)):
            if i == 0: basis1, basis2, basis3 = self._get_adjacent_basis_fc(csvFolder, layerNum, i, isVerbose=True)
            else: basis1, basis2, basis3 = self._get_adjacent_basis_fc(csvFolder, layerNum, i)                
            _, k = self.smTool.calc_karcher_subspace(basis1, basis3)
            magContainer.append(self.smTool.calc_magnitude(basis2, k))
        
        result = cp.array(magContainer).get()
        return result
    
class Layerwise2ndRBFMag(AdjacentMag.TripleAdjacentRBFMag):

    def __call__(self, csvFolder, layerNum, totalSampleNum, gamma=5):
        _layerNum = layerNum + 1
        if self.isConvLayer(_layerNum):
            result = self._pca_2ndRBFMag_layerwise_conv(csvFolder, layerNum, gamma=gamma, totalSampleNum=totalSampleNum)
        elif self.isFcLayer(_layerNum):
            result = self._pca_2ndRBFMag_layerwise_fc(csvFolder, layerNum, gamma=gamma, totalSampleNum=totalSampleNum)
        else:
            result = None
        return result

    def _pca_2ndRBFMag_layerwise_conv(self, csvFolder, layerNum, gamma=5, totalSampleNum=100):

        magContainer = []
        print("layer {}".format(layerNum))

        for i in tqdm(range(totalSampleNum - 1)):
            if i == 0: _, basis2, _ = self._get_adjacent_basis_conv(csvFolder, layerNum, i, gamma=gamma, isVerbose=True)
            else: _, basis2, _ = self._get_adjacent_basis_conv(csvFolder, layerNum, i, gamma=gamma)
            dataArray1, dataArray2, dataArray3 = self._get_adjacent_data_conv(csvFolder, layerNum, i)
            _, m = self.pcaTool.rbf_kernel_pca_karcher(dataArray1, dataArray3, gamma=gamma)
            tmp = cp.concatenate((dataArray1, dataArray3), axis=1)    
            Kxy = self.pcaTool.make_rbf_K_matrix(dataArray2, tmp, gamma)
            magContainer.append(self.smTool.calc_rbf_magnitude(basis2, m, Kxy))
        
        result = cp.array(magContainer).get()
        return result


    def _pca_2ndRBFMag_layerwise_fc(self, csvFolder, layerNum, gamma=5, totalSampleNum=100):

        magContainer = []
        print("layer {}".format(layerNum))

        for i in tqdm(range(totalSampleNum - 1)):
            if i == 0: _, basis2, _ = self._get_adjacent_basis_fc(csvFolder, layerNum, i, gamma=gamma, isVerbose=True)
            else: _, basis2, _ = self._get_adjacent_basis_fc(csvFolder, layerNum, i, gamma=gamma)                
            dataArray1, dataArray2, dataArray3 = self._get_adjacent_data_fc(csvFolder, layerNum, i)
            _, m = self.pcaTool.rbf_kernel_pca_karcher(dataArray1, dataArray3, gamma=gamma)
            tmp = cp.concatenate((dataArray1, dataArray3), axis=1)    
            Kxy = self.pcaTool.make_rbf_K_matrix(dataArray2, tmp, gamma)
            magContainer.append(self.smTool.calc_rbf_magnitude(basis2, m, Kxy))
        
        result = cp.array(magContainer).get()
        return result

class Layerwise2ndMag_decomposition(AdjacentMag.TripleAdjacentMag):
    
    def __call__(self, csvFolder, layerNum, totalSampleNum):
        _layerNum = layerNum + 1
        if self.isConvLayer(_layerNum):
            result = self._pca_2ndMag_decomposition_conv(csvFolder, layerNum, totalSampleNum=totalSampleNum)
        elif self.isFcLayer(_layerNum):
            result = self._pca_2ndMag_decomposition_fc(csvFolder, layerNum, totalSampleNum=totalSampleNum)
        else:
            result = None
        return result

    def _pca_2ndMag_decomposition_conv(self, csvFolder, layerNum, totalSampleNum=100):

        alongContainer = []
        orthContainer = []
        print("layer {}".format(layerNum))

        for i in tqdm(range(totalSampleNum - 2)):
            if i == 0: basis1, basis2, basis3 = self._get_adjacent_basis_conv(csvFolder, layerNum, i, isVerbose=True)
            else: basis1, basis2, basis3 = self._get_adjacent_basis_conv(csvFolder, layerNum, i)                
            along, orth = self.smTool.calc_2nd_magnitude_decomposed(basis1, basis2, basis3)
            alongContainer.append(along)
            orthContainer.append(orth)

        result1 = cp.array(alongContainer).get()
        result2 = cp.array(orthContainer).get()
        return [result1, result2]

    def _pca_2ndMag_decomposition_fc(self, csvFolder, layerNum, totalSampleNum=100):

        alongContainer = []
        orthContainer = []
        print("layer {}".format(layerNum))

        for i in tqdm(range(totalSampleNum - 2)):
            if i == 0: basis1, basis2, basis3 = self._get_adjacent_basis_fc(csvFolder, layerNum, i, isVerbose=True)
            else: basis1, basis2, basis3 = self._get_adjacent_basis_fc(csvFolder, layerNum, i)                
            along, orth = self.smTool.calc_2nd_magnitude_decomposed(basis1, basis2, basis3)
            alongContainer.append(along)
            orthContainer.append(orth)

        result1 = cp.array(alongContainer).get()
        result2 = cp.array(orthContainer).get()
        return [result1, result2]
    
class Layerwise2ndRBFMag_decomposition(AdjacentMag.TripleAdjacentRBFMag):
    
    def __call__(self, csvFolder, layerNum, totalSampleNum, gamma=5):
        _layerNum = layerNum + 1
        if self.isConvLayer(_layerNum):
            result = self._pca_2ndRBFMag_decomposition_conv(csvFolder, layerNum, gamma=gamma, totalSampleNum=totalSampleNum)
        elif self.isFcLayer(_layerNum):
            result = self._pca_2ndRBFMag_decomposition_fc(csvFolder, layerNum, gamma=gamma, totalSampleNum=totalSampleNum)
        else:
            result = None
        return result

    def _pca_2ndRBFMag_decomposition_conv(self, csvFolder, layerNum, gamma=5, totalSampleNum=100):

        alongContainer = []
        orthContainer = []
        print("layer {}".format(layerNum))

        for i in tqdm(range(totalSampleNum - 2)):
            if i == 0: _, basis2, _ = self._get_adjacent_basis_conv(csvFolder, layerNum, i, isVerbose=True)
            else: _, basis2, _ = self._get_adjacent_basis_conv(csvFolder, layerNum, i)
            dataArray1, dataArray2, dataArray3 = self._get_adjacent_data_conv(csvFolder, layerNum, i)
            _, karcher_basis = self.pcaTool.rbf_kernel_pca_karcher(dataArray1, dataArray3, gamma)
            sumArray = cp.concatenate((dataArray1, dataArray3), axis=1)
            _, sum_basis = self.pcaTool.rbf_kernel_pca_sum(dataArray1, dataArray3, gamma)
            projection_basis = self.pcaTool.rbf_kernel_pca_projection(sumArray, dataArray2, gamma)
            K1 = self.pcaTool.make_rbf_K_matrix(sumArray, dataArray2, gamma)
            K2 = self.pcaTool.make_rbf_K_matrix(sumArray, sumArray, gamma)

            along = self.smTool.calc_rbf_magnitude(sum_basis, basis2, K1)
            orth = self.smTool.calc_rbf_magnitude(projection_basis, karcher_basis, K2)
            alongContainer.append(along)
            orthContainer.append(orth)

        result1 = cp.array(alongContainer).get()
        result2 = cp.array(orthContainer).get()
        return [result1, result2]

    def _pca_2ndRBFMag_decomposition_fc(self, csvFolder, layerNum, gamma=5, totalSampleNum=100):

        alongContainer = []
        orthContainer = []
        print("layer {}".format(layerNum))

        for i in tqdm(range(totalSampleNum - 2)):
            if i == 0: _, basis2, _ = self._get_adjacent_basis_fc(csvFolder, layerNum, i, isVerbose=True)
            else: _, basis2, _ = self._get_adjacent_basis_fc(csvFolder, layerNum, i)
            dataArray1, dataArray2, dataArray3 = self._get_adjacent_data_fc(csvFolder, layerNum, i)
            _, karcher_basis = self.pcaTool.rbf_kernel_pca_karcher(dataArray1, dataArray3, gamma)
            sumArray = cp.concatenate((dataArray1, dataArray3), axis=1)
            _, sum_basis = self.pcaTool.rbf_kernel_pca_sum(dataArray1, dataArray3, gamma)
            projection_basis = self.pcaTool.rbf_kernel_pca_projection(sumArray, dataArray2, gamma)
            K1 = self.pcaTool.make_rbf_K_matrix(sumArray, dataArray2, gamma)
            K2 = self.pcaTool.make_rbf_K_matrix(sumArray, sumArray, gamma)

            along = self.smTool.calc_rbf_magnitude(sum_basis, basis2, K1)
            orth = self.smTool.calc_rbf_magnitude(projection_basis, karcher_basis, K2)
            alongContainer.append(along)
            orthContainer.append(orth)

        result1 = cp.array(alongContainer).get()
        result2 = cp.array(orthContainer).get()
        return [result1, result2]
    
class Layerwise_geodesic_decomposition(AdjacentMag.SingleAdjacentMag):

    def __call__(self, csvFolder, layerNum, totalSampleNum):
        _layerNum = layerNum + 1
        if self.isConvLayer(_layerNum):
            result = self._pca_geodesic_decomposition_conv(csvFolder, layerNum, totalSampleNum=totalSampleNum)
        elif self.isFcLayer(_layerNum):
            result = self._pca_geodesic_decomposition_fc(csvFolder, layerNum, totalSampleNum=totalSampleNum)
        else:
            result = None
        return result

    def _pca_geodesic_decomposition_conv(self, csvFolder, layerNum, totalSampleNum=100):

        alongContainer = []
        orthContainer = []
        print("layer {}".format(layerNum))

        basis_start = self._get_adjacent_basis_conv(csvFolder, layerNum, 1, isVerbose=True)
        basis_end = self._get_adjacent_basis_conv(csvFolder, layerNum, totalSampleNum)
        
        H_geodesic_start_prime = self.grTool.logarithmic_mapping(basis_end, basis_start)
        H_geodesic_start = self.grTool.normalize(H_geodesic_start_prime)
        invertW = cp.linalg.inv(cp.transpose(H_geodesic_start)@H_geodesic_start)
        invertW = invertW.real

        projection_matrix = H_geodesic_start@invertW@cp.transpose(H_geodesic_start)
        orthogonal_projection_matrix = (cp.eye(H_geodesic_start.shape[0]) - H_geodesic_start@invertW@cp.transpose(H_geodesic_start))

        for i in tqdm(range(totalSampleNum)):
            basis = self._get_adjacent_basis_conv(csvFolder, layerNum, 1+1)
            H = self.grTool.logarithmic_mapping(basis_end, basis)
            orthContainer.append(self.grTool.calc_grassmannian_norm(orthogonal_projection_matrix@H))
            alongContainer.append(self.grTool.calc_grassmannian_norm(projection_matrix@H))

        result1 = cp.array(alongContainer).get()
        result2 = cp.array(orthContainer).get()
        return [result1, result2]

    def _pca_geodesic_decomposition_fc(self, csvFolder, layerNum, totalSampleNum=100):

        alongContainer = []
        orthContainer = []
        print("layer {}".format(layerNum))

        basis_start = self._get_adjacent_basis_fc(csvFolder, layerNum, 1, isVerbose=True)
        basis_end = self._get_adjacent_basis_fc(csvFolder, layerNum, totalSampleNum)
        
        H_geodesic_start_prime = self.grTool.logarithmic_mapping(basis_end, basis_start)
        H_geodesic_start = self.grTool.normalize(H_geodesic_start_prime)
        invertW = cp.linalg.inv(cp.transpose(H_geodesic_start)@H_geodesic_start)
        invertW = invertW.real

        projection_matrix = H_geodesic_start@invertW@cp.transpose(H_geodesic_start)
        orthogonal_projection_matrix = (cp.eye(H_geodesic_start.shape[0]) - H_geodesic_start@invertW@cp.transpose(H_geodesic_start))

        for i in tqdm(range(totalSampleNum)):
            basis = self._get_adjacent_basis_fc(csvFolder, layerNum, 1+1)
            H = self.grTool.logarithmic_mapping(basis_end, basis)
            orthContainer.append(self.grTool.calc_grassmannian_norm(orthogonal_projection_matrix@H))
            alongContainer.append(self.grTool.calc_grassmannian_norm(projection_matrix@H))

        result1 = cp.array(alongContainer).get()
        result2 = cp.array(orthContainer).get()
        return [result1, result2]