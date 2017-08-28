import PyTorch
import PyTorchHelpers
import numpy as np


def create():
    TorchModel = PyTorchHelpers.load_lua_class(
        'pose-hg-demo/python/TorchModel.lua', 'TorchModel')
    TorchModel = TorchModel()
    return TorchModel


def run(model, im, box):

    result = PyTorch.DoubleTensor(len(box), 16, 2)

    input = im.astype(np.double) / 255
    result = model.predict(input, box)
    # print result

    return result.asNumpyTensor()
