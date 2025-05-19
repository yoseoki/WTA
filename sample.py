import sys, os
from calc_magnitude import *
from calc_projection import *
import Summerize as sum

def get_analysis_none_plural(plotPolicy, tool, csvFolderList, totalStepNum):

    for layerNum, layer in enumerate(plotPolicy.keys()):
        if plotPolicy[layer]:
            _ = tool(csvFolderList, layerNum+1, totalStepNum)
    return None, None

def get_analysis_single(plotPolicy, tool, csvFolder, totalStepNum):

    resultContainer = []
    labelContainer = []
    for layerNum, layer in enumerate(plotPolicy.keys()):
        if plotPolicy[layer]:
            resultContainer.append(tool(csvFolder, layerNum+1, totalStepNum))
            labelContainer.append(layer)
    return resultContainer, labelContainer

def get_analysis_single_plural(plotPolicy, tool, csvFolderList, totalStepNum):

    resultContainer = []
    labelContainer = []
    for layerNum, layer in enumerate(plotPolicy.keys()):
        if plotPolicy[layer]:
            resultContainer.append(tool(csvFolderList, layerNum+1, totalStepNum))
            labelContainer.append(layer)
    return resultContainer, labelContainer

def get_analysis_double(plotPolicy, tool, csvFolder, totalStepNum):

    resultContainer = []
    labelContainer = []
    for layerNum, layer in enumerate(plotPolicy.keys()):
        if plotPolicy[layer]:
            r1, r2 = tool(csvFolder, layerNum+1, totalStepNum)
            resultContainer.append((r1, r2))
            labelContainer.append(layer)
    return resultContainer, labelContainer

def get_analysis_triple(plotPolicy, tool, csvFolder, totalStepNum):

    resultContainer = []
    labelContainer = []
    for layerNum, layer in enumerate(plotPolicy.keys()):
        if plotPolicy[layer]:
            r1, r2, r3 = tool(csvFolder, layerNum+1, totalStepNum)
            resultContainer.append((r1, r2, r3))
            labelContainer.append(layer)
    return resultContainer, labelContainer

class VGG161stMag(Layerwise1stMag):
    def isConvLayer(self, layerNum):
        if layerNum < 14: return True
        else: return False

    def isFcLayer(self, layerNum):
        if layerNum == 14: return True
        else: return False

class VGG161stMagD(Layerwise1stMag_decomposition):
    def isConvLayer(self, layerNum):
        if layerNum < 14: return True
        else: return False

    def isFcLayer(self, layerNum):
        if layerNum == 14: return True
        else: return False

class VGG16GPCAD(Layerwise_grassmann_PCA_plural):
    def isConvLayer(self, layerNum):
        if layerNum < 14: return True
        else: return False

    def isFcLayer(self, layerNum):
        if layerNum == 14: return True
        else: return False

if __name__ == "__main__":

    whichNum = 1

    range1 = (whichNum == 1)
    range2 = (whichNum == 2)
    range3 = (whichNum == 3)
    range4 = (whichNum == 4)
    range5 = (whichNum == 5)

    plotPolicy = {"layer1" : range1,
                    "layer2" : range1,
                    "layer3" : range1,
                    "layer4" : range1,
                    "layer5" : range2,
                    "layer6" : range2,
                    "layer7" : range2,
                    "layer8" : range3,
                    "layer9" : range3,
                    "layer10" : range3,
                    "layer11" : range4,
                    "layer12" : range4,
                    "layer13" : range4,
                    "layer14" : range5,
                    "layer15" : range5,
                    "layer16" : range5}
    
    seedContainer = [6579, 7654, 3903, 9882, 4046, 2777, 8192, 2802, 7117]
    csvContainer = []
    for seed in seedContainer:
        csvContainer.append("VGG16__{:04d}".format(seed))

    totalStepNum = 100

    tool = VGG16GPCAD()
    visualize = sum.ProjPlot("VGG16")
    result, label= get_analysis_triple(plotPolicy, tool, csvContainer[:7], totalStepNum)
    visualize.plot_2d_plural(result, label, 7, seedContainer[:7], whichNum, dist="GRASSMANN")
    visualize.plot_3d_plural(result, label, 7, seedContainer[:7], whichNum, dist="GRASSMANN")
