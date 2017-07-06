import PyTorch
import PyTorchHelpers
import numpy
def run(im_names,box):
    TorchModel = PyTorchHelpers.load_lua_class('pose-hg-demo/python/TorchModel.lua', 'TorchModel')
    TorchModel=TorchModel()
    result = PyTorch.DoubleTensor(1,16,2)
    result = TorchModel.predict(im_names,box)
    return result.asNumpyTensor()