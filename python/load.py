import PyTorch
import PyTorchHelpers
TorchModel = PyTorchHelpers.load_lua_class('TorchModel.lua', 'TorchModel')
TorchModel=TorchModel()
TorchModel.predict(1)