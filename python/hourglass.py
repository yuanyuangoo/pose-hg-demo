import PyTorch
import PyTorchHelpers
def run(im_names,box):
    TorchModel = PyTorchHelpers.load_lua_class('pose-hg-demo/python/TorchModel.lua', 'TorchModel')
    TorchModel=TorchModel()
    result=TorchModel.predict(im_names,box)

    print('adfafadfadfa')
    print(result)
