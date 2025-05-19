import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import cupy as cp
from . import ParamIO
from util import PCA, SUBSPACE, DIST
from abc import *

class SingleAdjacentMag(metaclass=ABCMeta):
    def __init__(self):
        self.pcaTool = PCA.WeightPCA()
        self.kpcaTool = PCA.WeightKPCA()
        self.smTool = SUBSPACE.SubspaceDiff()
        self.grTool = SUBSPACE.Grassmannian()
        self.dsTool = DIST.GrassmannDist()

    # please define self.isConvLayer and self.isFcLayer
    # that class can know which layer is conv layer
    # and which layer is fc layer
    @abstractmethod
    def isConvLayer(self, layerNum):
        pass

    @abstractmethod
    def isFcLayer(self, layerNum):
        pass

    def _get_adjacent_basis_conv(self, csvFolder, layerNum, t, isVerbose=False):
        weight_layer1 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t), layerNum)
        out_channels = weight_layer1.shape[0]
        in_channels = weight_layer1.shape[1]
        ker_size = weight_layer1.shape[2] * weight_layer1.shape[3]
        
        if in_channels * ker_size < out_channels:
            if isVerbose:
                print("existing dimension(out_channels * ker_size) : {}".format(out_channels * ker_size))
                print("subspace dimension(in_channels) : {}".format(in_channels))
            weight_layer1 = cp.swapaxes(weight_layer1, 0, 1)
            dataArray1 = cp.transpose(cp.reshape(weight_layer1, (in_channels, out_channels * ker_size)))
            _, basis1 = self.pcaTool.pca_lowcost(dataArray1)
            basis1 = basis1[:,:in_channels]
        else:
            if isVerbose:
                print("existing dimension(in_channels * ker_size) : {}".format(in_channels * ker_size))
                print("subspace dimension(out_channels) : {}".format(out_channels))
            dataArray1 = cp.transpose(cp.reshape(weight_layer1, (out_channels, in_channels * ker_size)))
            _, basis1 = self.pcaTool.pca_lowcost(dataArray1)
            basis1 = basis1[:,:out_channels]
        
        return basis1
    
    def _get_adjacent_basis_fc(self, csvFolder, layerNum, t, isVerbose=False):
        weight_layer1 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t), layerNum)
        out_dims = weight_layer1.shape[0]
        in_dims = weight_layer1.shape[1]
        
        if in_dims < out_dims:
            if isVerbose:
                print("existing dimension(out_dims) : {}".format(out_dims))
                print("subspace dimension(in_dims) : {}".format(in_dims))
            dataArray1 = weight_layer1
            _, basis1 = self.pcaTool.pca_lowcost(dataArray1)
            basis1 = basis1[:,:in_dims]
        else:
            if isVerbose:
                print("existing dimension(in_dims) : {}".format(in_dims))
                print("subspace dimension(out_dims) : {}".format(out_dims))
            dataArray1 = cp.transpose(weight_layer1)
            _, basis1 = self.pcaTool.pca_lowcost(dataArray1)
            basis1 = basis1[:,:out_dims]
        
        return basis1

class DoubleAdjacentMag(metaclass=ABCMeta):
    def __init__(self):
        self.pcaTool = PCA.WeightPCA()
        self.kpcaTool = PCA.WeightKPCA()
        self.smTool = SUBSPACE.SubspaceDiff()
        self.grTool = SUBSPACE.Grassmannian()
        self.dsTool = DIST.GrassmannDist()

    # please define self.isConvLayer and self.isFcLayer
    # that class can know which layer is conv layer
    # and which layer is fc layer
    @abstractmethod
    def isConvLayer(self, layerNum):
        pass

    @abstractmethod
    def isFcLayer(self, layerNum):
        pass
    
    def _get_adjacent_basis_conv(self, csvFolder, layerNum, t, isVerbose=False):
        weight_layer1 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t), layerNum)
        weight_layer2 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t+1), layerNum)
        out_channels = weight_layer1.shape[0]
        in_channels = weight_layer1.shape[1]
        ker_size = weight_layer1.shape[2] * weight_layer1.shape[3]
        
        if in_channels * ker_size < out_channels:
            if isVerbose:
                print("existing dimension(out_channels * ker_size) : {}".format(out_channels * ker_size))
                print("subspace dimension(in_channels) : {}".format(in_channels))
            weight_layer1 = cp.swapaxes(weight_layer1, 0, 1)
            weight_layer2 = cp.swapaxes(weight_layer2, 0, 1)
            dataArray1 = cp.transpose(cp.reshape(weight_layer1, (in_channels, out_channels * ker_size)))
            dataArray2 = cp.transpose(cp.reshape(weight_layer2, (in_channels, out_channels * ker_size)))
            _, basis1 = self.pcaTool.pca_lowcost(dataArray1)
            _, basis2 = self.pcaTool.pca_lowcost(dataArray2)
            basis1 = basis1[:,:in_channels]
            basis2 = basis2[:,:in_channels]
        else:
            if isVerbose:
                print("existing dimension(in_channels * ker_size) : {}".format(in_channels * ker_size))
                print("subspace dimension(out_channels) : {}".format(out_channels))
            dataArray1 = cp.transpose(cp.reshape(weight_layer1, (out_channels, in_channels * ker_size)))
            dataArray2 = cp.transpose(cp.reshape(weight_layer2, (out_channels, in_channels * ker_size)))
            _, basis1 = self.pcaTool.pca_lowcost(dataArray1)
            _, basis2 = self.pcaTool.pca_lowcost(dataArray2)
            basis1 = basis1[:,:out_channels]
            basis2 = basis2[:,:out_channels]
        
        return [basis1, basis2]
    
    def _get_adjacent_basis_fc(self, csvFolder, layerNum, t, isVerbose=False):
        weight_layer1 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t), layerNum)
        weight_layer2 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t+1), layerNum)
        out_dims = weight_layer1.shape[0]
        in_dims = weight_layer1.shape[1]
        
        if in_dims < out_dims:
            if isVerbose:
                print("existing dimension(out_dims) : {}".format(out_dims))
                print("subspace dimension(in_dims) : {}".format(in_dims))
            dataArray1 = weight_layer1
            dataArray2 = weight_layer2
            _, basis1 = self.pcaTool.pca_lowcost(dataArray1)
            _, basis2 = self.pcaTool.pca_lowcost(dataArray2)
            basis1 = basis1[:,:in_dims]
            basis2 = basis2[:,:in_dims]
        else:
            if isVerbose:
                print("existing dimension(in_dims) : {}".format(in_dims))
                print("subspace dimension(out_dims) : {}".format(out_dims))
            dataArray1 = cp.transpose(weight_layer1)
            dataArray2 = cp.transpose(weight_layer2)
            _, basis1 = self.pcaTool.pca_lowcost(dataArray1)
            _, basis2 = self.pcaTool.pca_lowcost(dataArray2)
            basis1 = basis1[:,:out_dims]
            basis2 = basis2[:,:out_dims]
        
        return [basis1, basis2]

class TripleAdjacentMag(metaclass=ABCMeta):
    def __init__(self):
        self.pcaTool = PCA.WeightPCA()
        self.kpcaTool = PCA.WeightKPCA()
        self.smTool = SUBSPACE.SubspaceDiff()
        self.grTool = SUBSPACE.Grassmannian()
        self.dsTool = DIST.GrassmannDist()

    # please define self.isConvLayer and self.isFcLayer
    # that class can know which layer is conv layer
    # and which layer is fc layer
    @abstractmethod
    def isConvLayer(self, layerNum):
        pass

    @abstractmethod
    def isFcLayer(self, layerNum):
        pass

    def _get_adjacent_basis_conv(self, csvFolder, layerNum, t, isVerbose=False):
        weight_layer1 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t), layerNum)
        weight_layer2 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t+1), layerNum)
        weight_layer3 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t+2), layerNum)
        out_channels = weight_layer1.shape[0]
        in_channels = weight_layer1.shape[1]
        ker_size = weight_layer1.shape[2] * weight_layer1.shape[3]
        
        if in_channels * ker_size < out_channels:
            if isVerbose:
                print("existing dimension(out_channels * ker_size) : {}".format(out_channels * ker_size))
                print("subspace dimension(in_channels) : {}".format(in_channels))
            weight_layer1 = cp.swapaxes(weight_layer1, 0, 1)
            weight_layer2 = cp.swapaxes(weight_layer2, 0, 1)
            weight_layer3 = cp.swapaxes(weight_layer3, 0, 1)
            dataArray1 = cp.transpose(cp.reshape(weight_layer1, (in_channels, out_channels * ker_size)))
            dataArray2 = cp.transpose(cp.reshape(weight_layer2, (in_channels, out_channels * ker_size)))
            dataArray3 = cp.transpose(cp.reshape(weight_layer3, (in_channels, out_channels * ker_size)))
            _, basis1 = self.pcaTool.pca_lowcost(dataArray1)
            _, basis2 = self.pcaTool.pca_lowcost(dataArray2)
            _, basis3 = self.pcaTool.pca_lowcost(dataArray3)
            basis1 = basis1[:,:in_channels]
            basis2 = basis2[:,:in_channels]
            basis3 = basis3[:,:in_channels]
        else:
            if isVerbose:
                print("existing dimension(in_channels * ker_size) : {}".format(in_channels * ker_size))
                print("subspace dimension(out_channels) : {}".format(out_channels))
            dataArray1 = cp.transpose(cp.reshape(weight_layer1, (out_channels, in_channels * ker_size)))
            dataArray2 = cp.transpose(cp.reshape(weight_layer2, (out_channels, in_channels * ker_size)))
            dataArray3 = cp.transpose(cp.reshape(weight_layer3, (out_channels, in_channels * ker_size)))
            _, basis1 = self.pcaTool.pca_lowcost(dataArray1)
            _, basis2 = self.pcaTool.pca_lowcost(dataArray2)
            _, basis3 = self.pcaTool.pca_lowcost(dataArray3)
            basis1 = basis1[:,:out_channels]
            basis2 = basis2[:,:out_channels]
            basis3 = basis3[:,:out_channels]
        
        return [basis1, basis2, basis3]
    
    def _get_adjacent_basis_fc(self, csvFolder, layerNum, t, isVerbose=False):
        weight_layer1 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t), layerNum)
        weight_layer2 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t+1), layerNum)
        weight_layer3 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t+2), layerNum)
        out_dims = weight_layer1.shape[0]
        in_dims = weight_layer1.shape[1]
        
        if in_dims < out_dims:
            if isVerbose:
                print("existing dimension(out_dims) : {}".format(out_dims))
                print("subspace dimension(in_dims) : {}".format(in_dims))
            dataArray1 = weight_layer1
            dataArray2 = weight_layer2
            dataArray3 = weight_layer3
            _, basis1 = self.pcaTool.pca_lowcost(dataArray1)
            _, basis2 = self.pcaTool.pca_lowcost(dataArray2)
            _, basis3 = self.pcaTool.pca_lowcost(dataArray3)
            basis1 = basis1[:,:in_dims]
            basis2 = basis2[:,:in_dims]
            basis3 = basis3[:,:in_dims]
        else:
            if isVerbose:
                print("existing dimension(in_dims) : {}".format(in_dims))
                print("subspace dimension(out_dims) : {}".format(out_dims))
            dataArray1 = cp.transpose(weight_layer1)
            dataArray2 = cp.transpose(weight_layer2)
            dataArray3 = cp.transpose(weight_layer3)
            _, basis1 = self.pcaTool.pca_lowcost(dataArray1)
            _, basis2 = self.pcaTool.pca_lowcost(dataArray2)
            _, basis3 = self.pcaTool.pca_lowcost(dataArray3)
            basis1 = basis1[:,:out_dims]
            basis2 = basis2[:,:out_dims]
            basis3 = basis3[:,:out_dims]
        
        return [basis1, basis2, basis3]
    
class SingleAdjacentRBFMag(metaclass=ABCMeta):
    def __init__(self):
        self.pcaTool = PCA.WeightRBFPCA()
        self.kpcaTool = PCA.WeightKPCA()
        self.smTool = SUBSPACE.SubspaceDiff()
        self.grTool = SUBSPACE.Grassmannian()
        self.dsTool = DIST.GrassmannDist()

    # please define self.isConvLayer and self.isFcLayer
    # that class can know which layer is conv layer
    # and which layer is fc layer
    @abstractmethod
    def isConvLayer(self, layerNum):
        pass

    @abstractmethod
    def isFcLayer(self, layerNum):
        pass

    def _get_adjacent_data_conv(self, csvFolder, layerNum, t, isVerbose=False):
        weight_layer1 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t), layerNum)
        out_channels = weight_layer1.shape[0]
        in_channels = weight_layer1.shape[1]
        ker_size = weight_layer1.shape[2] * weight_layer1.shape[3]
        
        if in_channels * ker_size < out_channels:
            if isVerbose:
                print("existing dimension(out_channels * ker_size) : {}".format(out_channels * ker_size))
                print("subspace dimension(in_channels) : {}".format(in_channels))
            weight_layer1 = cp.swapaxes(weight_layer1, 0, 1)
            dataArray1 = cp.transpose(cp.reshape(weight_layer1, (in_channels, out_channels * ker_size)))
        else:
            if isVerbose:
                print("existing dimension(in_channels * ker_size) : {}".format(in_channels * ker_size))
                print("subspace dimension(out_channels) : {}".format(out_channels))
            dataArray1 = cp.transpose(cp.reshape(weight_layer1, (out_channels, in_channels * ker_size)))
        
        return dataArray1
    
    def _get_adjacent_data_fc(self, csvFolder, layerNum, t, isVerbose=False):
        weight_layer1 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t), layerNum)
        out_dims = weight_layer1.shape[0]
        in_dims = weight_layer1.shape[1]
        
        if in_dims < out_dims:
            if isVerbose:
                print("existing dimension(out_dims) : {}".format(out_dims))
                print("subspace dimension(in_dims) : {}".format(in_dims))
            dataArray1 = weight_layer1
        else:
            if isVerbose:
                print("existing dimension(in_dims) : {}".format(in_dims))
                print("subspace dimension(out_dims) : {}".format(out_dims))
            dataArray1 = cp.transpose(weight_layer1)
        
        return dataArray1

    def _get_adjacent_basis_conv(self, csvFolder, layerNum, t, gamma=5, isVerbose=False):
        weight_layer1 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t), layerNum)
        out_channels = weight_layer1.shape[0]
        in_channels = weight_layer1.shape[1]
        ker_size = weight_layer1.shape[2] * weight_layer1.shape[3]
        
        if in_channels * ker_size < out_channels:
            if isVerbose:
                print("existing dimension(out_channels * ker_size) : {}".format(out_channels * ker_size))
                print("subspace dimension(in_channels) : {}".format(in_channels))
            weight_layer1 = cp.swapaxes(weight_layer1, 0, 1)
            dataArray1 = cp.transpose(cp.reshape(weight_layer1, (in_channels, out_channels * ker_size)))
            _, basis1 = self.pcaTool.rbf_kernel_pca(dataArray1, gamma)
            basis1 = basis1[:,:in_channels]
        else:
            if isVerbose:
                print("existing dimension(in_channels * ker_size) : {}".format(in_channels * ker_size))
                print("subspace dimension(out_channels) : {}".format(out_channels))
            dataArray1 = cp.transpose(cp.reshape(weight_layer1, (out_channels, in_channels * ker_size)))
            _, basis1 = self.pcaTool.rbf_kernel_pca(dataArray1, gamma)
            basis1 = basis1[:,:out_channels]
        
        return basis1
    
    def _get_adjacent_basis_fc(self, csvFolder, layerNum, t, gamma=5, isVerbose=False):
        weight_layer1 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t), layerNum)
        out_dims = weight_layer1.shape[0]
        in_dims = weight_layer1.shape[1]
        
        if in_dims < out_dims:
            if isVerbose:
                print("existing dimension(out_dims) : {}".format(out_dims))
                print("subspace dimension(in_dims) : {}".format(in_dims))
            dataArray1 = weight_layer1
            _, basis1 = self.pcaTool.rbf_kernel_pca(dataArray1, gamma)
            basis1 = basis1[:,:in_dims]
        else:
            if isVerbose:
                print("existing dimension(in_dims) : {}".format(in_dims))
                print("subspace dimension(out_dims) : {}".format(out_dims))
            dataArray1 = cp.transpose(weight_layer1)
            _, basis1 = self.pcaTool.rbf_kernel_pca(dataArray1, gamma)
            basis1 = basis1[:,:out_dims]
        
        return basis1

class DoubleAdjacentRBFMag(metaclass=ABCMeta):
    def __init__(self):
        self.pcaTool = PCA.WeightRBFPCA()
        self.kpcaTool = PCA.WeightKPCA()
        self.smTool = SUBSPACE.SubspaceDiff()
        self.grTool = SUBSPACE.Grassmannian()
        self.dsTool = DIST.GrassmannDist()

    # please define self.isConvLayer and self.isFcLayer
    # that class can know which layer is conv layer
    # and which layer is fc layer
    @abstractmethod
    def isConvLayer(self, layerNum):
        pass

    @abstractmethod
    def isFcLayer(self, layerNum):
        pass

    def _get_adjacent_data_conv(self, csvFolder, layerNum, t, isVerbose=False):
        weight_layer1 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t), layerNum)
        weight_layer2 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t+1), layerNum)
        out_channels = weight_layer1.shape[0]
        in_channels = weight_layer1.shape[1]
        ker_size = weight_layer1.shape[2] * weight_layer1.shape[3]
        
        if in_channels * ker_size < out_channels:
            if isVerbose:
                print("existing dimension(out_channels * ker_size) : {}".format(out_channels * ker_size))
                print("subspace dimension(in_channels) : {}".format(in_channels))
            weight_layer1 = cp.swapaxes(weight_layer1, 0, 1)
            weight_layer2 = cp.swapaxes(weight_layer2, 0, 1)
            dataArray1 = cp.transpose(cp.reshape(weight_layer1, (in_channels, out_channels * ker_size)))
            dataArray2 = cp.transpose(cp.reshape(weight_layer2, (in_channels, out_channels * ker_size)))
        else:
            if isVerbose:
                print("existing dimension(in_channels * ker_size) : {}".format(in_channels * ker_size))
                print("subspace dimension(out_channels) : {}".format(out_channels))
            dataArray1 = cp.transpose(cp.reshape(weight_layer1, (out_channels, in_channels * ker_size)))
            dataArray2 = cp.transpose(cp.reshape(weight_layer2, (out_channels, in_channels * ker_size)))
        
        return [dataArray1, dataArray2]
    
    def _get_adjacent_data_fc(self, csvFolder, layerNum, t, isVerbose=False):
        weight_layer1 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t), layerNum)
        weight_layer2 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t+1), layerNum)
        out_dims = weight_layer1.shape[0]
        in_dims = weight_layer1.shape[1]
        
        if in_dims < out_dims:
            if isVerbose:
                print("existing dimension(out_dims) : {}".format(out_dims))
                print("subspace dimension(in_dims) : {}".format(in_dims))
            dataArray1 = weight_layer1
            dataArray2 = weight_layer2
        else:
            if isVerbose:
                print("existing dimension(in_dims) : {}".format(in_dims))
                print("subspace dimension(out_dims) : {}".format(out_dims))
            dataArray1 = cp.transpose(weight_layer1)
            dataArray2 = cp.transpose(weight_layer2)
        
        return [dataArray1, dataArray2]
    
    def _get_adjacent_basis_conv(self, csvFolder, layerNum, t, gamma=5, isVerbose=False):
        weight_layer1 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t), layerNum)
        weight_layer2 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t+1), layerNum)
        out_channels = weight_layer1.shape[0]
        in_channels = weight_layer1.shape[1]
        ker_size = weight_layer1.shape[2] * weight_layer1.shape[3]
        
        if in_channels * ker_size < out_channels:
            if isVerbose:
                print("existing dimension(out_channels * ker_size) : {}".format(out_channels * ker_size))
                print("subspace dimension(in_channels) : {}".format(in_channels))
            weight_layer1 = cp.swapaxes(weight_layer1, 0, 1)
            weight_layer2 = cp.swapaxes(weight_layer2, 0, 1)
            dataArray1 = cp.transpose(cp.reshape(weight_layer1, (in_channels, out_channels * ker_size)))
            dataArray2 = cp.transpose(cp.reshape(weight_layer2, (in_channels, out_channels * ker_size)))
            _, basis1 = self.pcaTool.rbf_kernel_pca(dataArray1, gamma)
            _, basis2 = self.pcaTool.rbf_kernel_pca(dataArray2, gamma)
            basis1 = basis1[:,:in_channels]
            basis2 = basis2[:,:in_channels]
        else:
            if isVerbose:
                print("existing dimension(in_channels * ker_size) : {}".format(in_channels * ker_size))
                print("subspace dimension(out_channels) : {}".format(out_channels))
            dataArray1 = cp.transpose(cp.reshape(weight_layer1, (out_channels, in_channels * ker_size)))
            dataArray2 = cp.transpose(cp.reshape(weight_layer2, (out_channels, in_channels * ker_size)))
            _, basis1 = self.pcaTool.rbf_kernel_pca(dataArray1, gamma)
            _, basis2 = self.pcaTool.rbf_kernel_pca(dataArray2, gamma)
            basis1 = basis1[:,:out_channels]
            basis2 = basis2[:,:out_channels]
        
        return [basis1, basis2]
    
    def _get_adjacent_basis_fc(self, csvFolder, layerNum, t, gamma=5, isVerbose=False):
        weight_layer1 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t), layerNum)
        weight_layer2 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t+1), layerNum)
        out_dims = weight_layer1.shape[0]
        in_dims = weight_layer1.shape[1]
        
        if in_dims < out_dims:
            if isVerbose:
                print("existing dimension(out_dims) : {}".format(out_dims))
                print("subspace dimension(in_dims) : {}".format(in_dims))
            dataArray1 = weight_layer1
            dataArray2 = weight_layer2
            _, basis1 = self.pcaTool.rbf_kernel_pca(dataArray1, gamma)
            _, basis2 = self.pcaTool.rbf_kernel_pca(dataArray2, gamma)
            basis1 = basis1[:,:in_dims]
            basis2 = basis2[:,:in_dims]
        else:
            if isVerbose:
                print("existing dimension(in_dims) : {}".format(in_dims))
                print("subspace dimension(out_dims) : {}".format(out_dims))
            dataArray1 = cp.transpose(weight_layer1)
            dataArray2 = cp.transpose(weight_layer2)
            _, basis1 = self.pcaTool.rbf_kernel_pca(dataArray1, gamma)
            _, basis2 = self.pcaTool.rbf_kernel_pca(dataArray2, gamma)
            basis1 = basis1[:,:out_dims]
            basis2 = basis2[:,:out_dims]
        
        return [basis1, basis2]

class TripleAdjacentRBFMag(metaclass=ABCMeta):
    def __init__(self):
        self.pcaTool = PCA.WeightRBFPCA()
        self.kpcaTool = PCA.WeightKPCA()
        self.smTool = SUBSPACE.SubspaceDiff()
        self.grTool = SUBSPACE.Grassmannian()
        self.dsTool = DIST.GrassmannDist()

    # please define self.isConvLayer and self.isFcLayer
    # that class can know which layer is conv layer
    # and which layer is fc layer
    @abstractmethod
    def isConvLayer(self, layerNum):
        pass

    @abstractmethod
    def isFcLayer(self, layerNum):
        pass

    def _get_adjacent_data_conv(self, csvFolder, layerNum, t, isVerbose=False):
        weight_layer1 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t), layerNum)
        weight_layer2 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t+1), layerNum)
        weight_layer3 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t+2), layerNum)
        out_channels = weight_layer1.shape[0]
        in_channels = weight_layer1.shape[1]
        ker_size = weight_layer1.shape[2] * weight_layer1.shape[3]
        
        if in_channels * ker_size < out_channels:
            if isVerbose:
                print("existing dimension(out_channels * ker_size) : {}".format(out_channels * ker_size))
                print("subspace dimension(in_channels) : {}".format(in_channels))
            weight_layer1 = cp.swapaxes(weight_layer1, 0, 1)
            weight_layer2 = cp.swapaxes(weight_layer2, 0, 1)
            weight_layer3 = cp.swapaxes(weight_layer3, 0, 1)
            dataArray1 = cp.transpose(cp.reshape(weight_layer1, (in_channels, out_channels * ker_size)))
            dataArray2 = cp.transpose(cp.reshape(weight_layer2, (in_channels, out_channels * ker_size)))
            dataArray3 = cp.transpose(cp.reshape(weight_layer3, (in_channels, out_channels * ker_size)))
        else:
            if isVerbose:
                print("existing dimension(in_channels * ker_size) : {}".format(in_channels * ker_size))
                print("subspace dimension(out_channels) : {}".format(out_channels))
            dataArray1 = cp.transpose(cp.reshape(weight_layer1, (out_channels, in_channels * ker_size)))
            dataArray2 = cp.transpose(cp.reshape(weight_layer2, (out_channels, in_channels * ker_size)))
            dataArray3 = cp.transpose(cp.reshape(weight_layer3, (out_channels, in_channels * ker_size)))
        
        return [dataArray1, dataArray2, dataArray3]
    
    def _get_adjacent_data_fc(self, csvFolder, layerNum, t, isVerbose=False):
        weight_layer1 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t), layerNum)
        weight_layer2 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t+1), layerNum)
        weight_layer3 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t+2), layerNum)
        out_dims = weight_layer1.shape[0]
        in_dims = weight_layer1.shape[1]
        
        if in_dims < out_dims:
            if isVerbose:
                print("existing dimension(out_dims) : {}".format(out_dims))
                print("subspace dimension(in_dims) : {}".format(in_dims))
            dataArray1 = weight_layer1
            dataArray2 = weight_layer2
            dataArray3 = weight_layer3
        else:
            if isVerbose:
                print("existing dimension(in_dims) : {}".format(in_dims))
                print("subspace dimension(out_dims) : {}".format(out_dims))
            dataArray1 = cp.transpose(weight_layer1)
            dataArray2 = cp.transpose(weight_layer2)
            dataArray3 = cp.transpose(weight_layer3)
        
        return [dataArray1, dataArray2, dataArray3]

    def _get_adjacent_basis_conv(self, csvFolder, layerNum, t, gamma=5, isVerbose=False):
        weight_layer1 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t), layerNum)
        weight_layer2 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t+1), layerNum)
        weight_layer3 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t+2), layerNum)
        out_channels = weight_layer1.shape[0]
        in_channels = weight_layer1.shape[1]
        ker_size = weight_layer1.shape[2] * weight_layer1.shape[3]
        
        if in_channels * ker_size < out_channels:
            if isVerbose:
                print("existing dimension(out_channels * ker_size) : {}".format(out_channels * ker_size))
                print("subspace dimension(in_channels) : {}".format(in_channels))
            weight_layer1 = cp.swapaxes(weight_layer1, 0, 1)
            weight_layer2 = cp.swapaxes(weight_layer2, 0, 1)
            weight_layer3 = cp.swapaxes(weight_layer3, 0, 1)
            dataArray1 = cp.transpose(cp.reshape(weight_layer1, (in_channels, out_channels * ker_size)))
            dataArray2 = cp.transpose(cp.reshape(weight_layer2, (in_channels, out_channels * ker_size)))
            dataArray3 = cp.transpose(cp.reshape(weight_layer3, (in_channels, out_channels * ker_size)))
            _, basis1 = self.pcaTool.rbf_kernel_pca(dataArray1, gamma)
            _, basis2 = self.pcaTool.rbf_kernel_pca(dataArray2, gamma)
            _, basis3 = self.pcaTool.rbf_kernel_pca(dataArray3, gamma)
            basis1 = basis1[:,:in_channels]
            basis2 = basis2[:,:in_channels]
            basis3 = basis3[:,:in_channels]
        else:
            if isVerbose:
                print("existing dimension(in_channels * ker_size) : {}".format(in_channels * ker_size))
                print("subspace dimension(out_channels) : {}".format(out_channels))
            dataArray1 = cp.transpose(cp.reshape(weight_layer1, (out_channels, in_channels * ker_size)))
            dataArray2 = cp.transpose(cp.reshape(weight_layer2, (out_channels, in_channels * ker_size)))
            dataArray3 = cp.transpose(cp.reshape(weight_layer3, (out_channels, in_channels * ker_size)))
            _, basis1 = self.pcaTool.rbf_kernel_pca(dataArray1, gamma)
            _, basis2 = self.pcaTool.rbf_kernel_pca(dataArray2, gamma)
            _, basis3 = self.pcaTool.rbf_kernel_pca(dataArray3, gamma)
            basis1 = basis1[:,:out_channels]
            basis2 = basis2[:,:out_channels]
            basis3 = basis3[:,:out_channels]
        
        return [basis1, basis2, basis3]
    
    def _get_adjacent_basis_fc(self, csvFolder, layerNum, t, gamma=5, isVerbose=False):
        weight_layer1 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t), layerNum)
        weight_layer2 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t+1), layerNum)
        weight_layer3 = ParamIO.makeDataArray(csvFolder + "/weights_epoch{:03d}.safetensors".format(t+2), layerNum)
        out_dims = weight_layer1.shape[0]
        in_dims = weight_layer1.shape[1]
        
        if in_dims < out_dims:
            if isVerbose:
                print("existing dimension(out_dims) : {}".format(out_dims))
                print("subspace dimension(in_dims) : {}".format(in_dims))
            dataArray1 = weight_layer1
            dataArray2 = weight_layer2
            dataArray3 = weight_layer3
            _, basis1 = self.pcaTool.rbf_kernel_pca(dataArray1, gamma)
            _, basis2 = self.pcaTool.rbf_kernel_pca(dataArray2, gamma)
            _, basis3 = self.pcaTool.rbf_kernel_pca(dataArray3, gamma)
            basis1 = basis1[:,:in_dims]
            basis2 = basis2[:,:in_dims]
            basis3 = basis3[:,:in_dims]
        else:
            if isVerbose:
                print("existing dimension(in_dims) : {}".format(in_dims))
                print("subspace dimension(out_dims) : {}".format(out_dims))
            dataArray1 = cp.transpose(weight_layer1)
            dataArray2 = cp.transpose(weight_layer2)
            dataArray3 = cp.transpose(weight_layer3)
            _, basis1 = self.pcaTool.rbf_kernel_pca(dataArray1, gamma)
            _, basis2 = self.pcaTool.rbf_kernel_pca(dataArray2, gamma)
            _, basis3 = self.pcaTool.rbf_kernel_pca(dataArray3, gamma)
            basis1 = basis1[:,:out_dims]
            basis2 = basis2[:,:out_dims]
            basis3 = basis3[:,:out_dims]
        
        return [basis1, basis2, basis3]
