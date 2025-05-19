from tqdm import tqdm
from . import AdjacentMag
import copy

class Layerwise_grassmann_PCA(AdjacentMag.SingleAdjacentMag):
    
    def __call__(self, csvFolder, layerNum, totalSampleNum, loadFlag=False, prefix=None):
        _layerNum = layerNum + 1
        if self.isConvLayer(_layerNum):
            result = self._grassmann_pca_conv(csvFolder, layerNum, totalSampleNum=totalSampleNum, loadFlag=loadFlag, prefix=prefix)
        elif self.isFcLayer(_layerNum):
            result = self._grassmann_pca_fc(csvFolder, layerNum, totalSampleNum=totalSampleNum, loadFlag=loadFlag, prefix=prefix)
        else:
            result = None
        return result

    def _grassmann_pca_conv(self, csvFolder, layerNum, totalSampleNum=100, loadFlag=False, prefix=None):

        # calculate
        print("layer {}".format(layerNum))
        
        if not loadFlag:
            basisContainer = []
            for i in tqdm(range(totalSampleNum)):
                if i == 0: basis = self._get_adjacent_basis_conv(csvFolder, layerNum, i+1, isVerbose=True)
                else: basis = self._get_adjacent_basis_conv(csvFolder, layerNum, i+1)  
                basisContainer.append(basis)
            K = self.dsTool.construct_K_matrix(basisContainer)
            self.dsTool.save_K_matrix(K, prefix)
        else:
            K = self.dsTool.load_K_matrix(prefix)

        projections = self.kpcaTool.pca_proj_kernel(K)
        projections = projections.get()
        return [projections[0,:], projections[1,:], projections[2,:]]

    def _grassmann_pca_fc(self, csvFolder, layerNum, totalSampleNum=100, loadFlag=False, prefix=None):

        # calculate
        print("layer {}".format(layerNum))
        
        if not loadFlag:
            basisContainer = []
            for i in tqdm(range(totalSampleNum)):
                if i == 0: basis = self._get_adjacent_basis_fc(csvFolder, layerNum, i+1, isVerbose=True)
                else: basis = self._get_adjacent_basis_fc(csvFolder, layerNum, i+1)  
                basisContainer.append(basis)
            K = self.dsTool.construct_K_matrix(basisContainer)
            self.dsTool.save_K_matrix(K, prefix)
        else:
            K = self.dsTool.load_K_matrix(prefix)

        projections = self.kpcaTool.pca_proj_kernel(K)
        projections = projections.get()
        return [projections[0,:], projections[1,:], projections[2,:]]

class Layerwise_grassmann_PCA_plural(AdjacentMag.SingleAdjacentMag):
    
    def __call__(self, csvFolderList, layerNum, totalSampleNum, loadFlag=False, prefix=None, mask=None, reCenterizeFlag=True):
        _layerNum = layerNum + 1
        if self.isConvLayer(_layerNum):
            result = self._grassmann_pca_plural_conv(csvFolderList, layerNum, totalSampleNum=totalSampleNum, loadFlag=loadFlag, prefix=prefix, mask=mask, reCenterizeFlag=reCenterizeFlag)
        elif self.isFcLayer(_layerNum):
            result = self._grassmann_pca_plural_fc(csvFolderList, layerNum, totalSampleNum=totalSampleNum, loadFlag=loadFlag, prefix=prefix, mask=mask, reCenterizeFlag=reCenterizeFlag)
        else:
            result = None
        return result

    def _grassmann_pca_plural_conv(self, csvFolderList, layerNum, totalSampleNum=100, loadFlag=False, prefix=None, mask=None, reCenterizeFlag=True):

        # calculate
        print("layer {}".format(layerNum))
        trajNum = len(csvFolderList)
        
        if not loadFlag:
            basisContainer = []
            for csvIndex, csvFolder in enumerate(csvFolderList):
                for i in tqdm(range(totalSampleNum)):
                    if csvIndex == 0 and i == 0: basis = self._get_adjacent_basis_conv(csvFolder, layerNum, i+1, isVerbose=True)
                    else: basis = self._get_adjacent_basis_conv(csvFolder, layerNum, i+1)  
                    basisContainer.append(basis)

            K = self.dsTool.construct_K_matrix(basisContainer)
            self.dsTool.save_K_matrix(K, prefix)
        else:
            K = self.dsTool.load_K_matrix(prefix)

        # masking
        if mask is not None:
            K = self.dsTool.mask_K_matrix(K, mask, trajNum)
            trajNum = len(mask)

        # KPCA - projection
        projections = self.kpcaTool.pca_proj_kernel(K)

        if reCenterizeFlag:
            # re-centralize
            for i in range(trajNum):
                trajectory_center = copy.deepcopy(projections[:,i*totalSampleNum])
                for j in range(totalSampleNum):
                    sampleNum = i*totalSampleNum + j
                    projections[:,sampleNum] = projections[:,sampleNum] - trajectory_center

            # re-PCA - projection
            projections_new = self.pcaTool.pca_proj(projections)
            projections = projections_new

        projections = projections.get()
        return [projections[0,:], projections[1,:], projections[2,:]]
    
    def _grassmann_pca_plural_fc(self, csvFolderList, layerNum, totalSampleNum=100, loadFlag=False, prefix=None, mask=None, reCenterizeFlag=True):

        # calculate
        print("layer {}".format(layerNum))
        trajNum = len(csvFolderList)
        
        if not loadFlag:
            basisContainer = []
            for csvIndex, csvFolder in enumerate(csvFolderList):
                for i in tqdm(range(totalSampleNum)):
                    if csvIndex == 0 and i == 0: basis = self._get_adjacent_basis_fc(csvFolder, layerNum, i+1, isVerbose=True)
                    else: basis = self._get_adjacent_basis_fc(csvFolder, layerNum, i+1)  
                    basisContainer.append(basis)

            K = self.dsTool.construct_K_matrix(basisContainer)
            self.dsTool.save_K_matrix(K, prefix)
        else:
            K = self.dsTool.load_K_matrix(prefix)

        # masking
        if mask is not None:
            K = self.dsTool.mask_K_matrix(K, mask, trajNum)
            trajNum = len(mask)

        # KPCA - projection
        projections = self.kpcaTool.pca_proj_kernel(K)

        if reCenterizeFlag:
            # re-centralize
            for i in range(trajNum):
                trajectory_center = copy.deepcopy(projections[:,i*totalSampleNum])
                for j in range(totalSampleNum):
                    sampleNum = i*totalSampleNum + j
                    projections[:,sampleNum] = projections[:,sampleNum] - trajectory_center

            # re-PCA - projection
            projections_new = self.pcaTool.pca_proj(projections)
            projections = projections_new

        projections = projections.get()
        return [projections[0,:], projections[1,:], projections[2,:]]