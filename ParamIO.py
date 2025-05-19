import cupy as cp
from safetensors.torch import safe_open

def makeDataArray(path, layerNum):
    f = safe_open(path, framework="pt", device="cpu")
    weight = f.get_tensor("{:03d}".format(layerNum))
    weight_cp = cp.asarray(weight)
    return weight_cp