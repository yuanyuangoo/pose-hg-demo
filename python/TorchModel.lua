require 'paths'
paths.dofile('util.lua')
paths.dofile('../img.lua')   
local TorchModel = torch.class('TorchModel')

function TorchModel:__init()
--self:buildModel(backend, imageSize, numClasses)
  self:loadmodel(1)
  a = loadAnnotations('test')

  self.imageSize = imageSize
  self.numClasses = numClasses
  self.backend = backend
end

function TorchModel:loadmodel(input)
  self.net=torch.load('../umich-stacked-hourglass.t7')
end

function TorchModel:predict(input)
    idxs = torch.Tensor({695})
    nsamples = idxs:nElement()
    xlua.progress(0,nsamples)
    preds = torch.Tensor(nsamples,16,2)
    for i = 1,nsamples do
    -- Set up input image
    local im = image.load('../images/' .. a['images'][idxs[i]])
    local center = a['center'][idxs[i]]
    local scale = a['scale'][idxs[i]]
    local inp = crop(im, center, scale, 0, 256)

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
    preds_hm:mul(4) -- Change to input scale
    local dispImg = drawOutput(inp, hm, preds_hm[1])
--   w = image.display{image=dispImg,win=w}
--    sys.sleep(3)
    
    print(preds_img)
    collectgarbage()
    end
end