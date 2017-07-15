require 'paths'
require 'torch'
paths.dofile('util.lua')
paths.dofile('../img.lua')

local TorchModel = torch.class('TorchModel')

function TorchModel:__init()
--self:buildModel(backend, imageSize, numClasses)
  self:loadmodel(i)
  a = loadAnnotations('test')

  self.imageSize = imageSize
  self.numClasses = numClasses
  self.backend = backend
end

function TorchModel:loadmodel(input)
  self.net=torch.load('./pose-hg-demo/umich-stacked-hourglass.t7')
end

function TorchModel:predict(input,box)
    print('input box')

    nsamples = box:size()[1]
    xlua.progress(0,nsamples)
    preds = torch.Tensor(nsamples,16,2)
    for i = 1,nsamples do
      -- Set up input image
      local im = input:transpose(3,1):transpose(2,3)

      local center = torch.Tensor(2)
      center[1]=box[i][1]+box[i][3]
      center[2]=box[i][2]+box[i][4]

      local scale = torch.max(center:ceil())/200

      local inp = crop(im, center:ceil(), scale, 0, 256)

      -- Get network output
      local out = self.net:forward(inp:view(1,3,256,256):cuda())
      cutorch.synchronize()
      local hm = out[#out][1]:float()
      hm[hm:lt(0)] = 0

      -- Get predictions (hm and img refer to the coordinate space)
      local preds_hm, preds_img = getPreds(hm, center, scale)
      preds[i]:copy(preds_img)

      xlua.progress(i,nsamples)

      -- Display the result
   --     preds_hm:mul(4) -- Change to input scale
   --     local dispImg = drawOutput(inp, hm, preds_hm[1])
  
   --   w = image.display{image=dispImg,win=w}
   --   sys.sleep(3)
   --   print(preds_img)
      collectgarbage()
    end
    return preds
end

