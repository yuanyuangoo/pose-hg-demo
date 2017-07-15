import PyTorch
import PyTorchHelpers
def run(im,box):
    TorchModel = PyTorchHelpers.load_lua_class('pose-hg-demo/python/TorchModel.lua', 'TorchModel')
    TorchModel=TorchModel()
    result = PyTorch.DoubleTensor(len(box),16,2)
    result = TorchModel.predict(im,box)
    return result.asNumpyTensor()
