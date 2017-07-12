import math
import cv2 as cv
import np as np
# Coordinate transformation

def getTransform(center, scale, rot, res):
    h = 200 * scale
    t = np.eye(3)

    # Scaling
    t[1][1] = res / h
    t[2][2] = res / h

    # Translation
    t[1][3] = res * (-center[1] / h + .5)
    t[2][3] = res * (-center[2] / h + .5)

    # Rotation
    if rot != 0:
        rot = -rot
        r = np.eye(3)
        ang = rot * math.pi / 180
        s = math.sin(ang)
        c = math.cos(ang)
        r[1][1] = c
        r[1][2] = -s
        r[2][1] = s
        r[2][2] = c
        # Need to make sure rotation is around center
        t_ = np.eye(3)
        t_[1][3] = -res/2
        t_[2][3] = -res/2
        t_inv = np.eye(3)
        t_inv[1][3] = res/2
        t_inv[2][3] = res/2
        t = t_inv * r * t_ * t
    

    return t


def transform(pt, center, scale, rot, res, invert):
    # For managing coordinate transformations between the original image space
    # and the heatmap

    pt_ = np.ones(3)
    pt_[1] = pt[1]
    pt_[2] = pt[2]
    t = getTransform(center, scale, rot, res)
    if invert:
        t = np.inverse(t)
    
    new_point = (t*pt_)[0:2].astype(int)
    return new_point


#######################################-
# Cropping
#######################################-

def crop(img, center, scale, rot, res):
    # Crop def tailored to the needs of our system. Provide a center
    # and scale value and the image will be cropped and resized to the output
    # resolution determined by res. 'rot' will also rotate the image as needed.

    ul = transform((1,1), center, scale, 0, res, true)
    br = transform((res,res), center, scale, 0, res, true)

    pad = math.floor(np.linalg.norm((ul - br).astype(float))/2 - (br[1]-ul[1])/2)
    if rot != 0:
        ul = ul - pad
        br = br + pad
    
    ht, wd, channels = img.shape

    newDim = (channels, br[1] - ul[1], br[0] - ul[0])
    newImg = np.zeros(newDim[2],newDim[1],newDim[0])

    newX = (max(1, -ul[1]+1), min(br[1], wd) - ul[1])
    newY = (max(1, -ul[2]+1), min(br[2], ht) - ul[2])
    oldX = (max(1, ul[1]+1), min(br[1], wd))
    oldY = (max(1, ul[2]+1), min(br[2], ht))

    #if newDim[1] > 2:
    #    newImg:sub(1,newDim[1],newY[1],newY[2],newX[1],newX[2]):copy(img:sub(1,newDim[1],oldY[1],oldY[2],oldX[1],oldX[2]))
    newImg=np.resize(img,newDim)
    #else
    #    newImg:sub(newY[1],newY[2],newX[1],newX[2]):copy(img:sub(oldY[1],oldY[2],oldX[1],oldX[2]))

    #if rot != 0:
    #    newImg = image.rotate(newImg, rot * math.pi / 180, 'bilinear')
    #    newImg = 
    #    if newDim[0] > 2:
    #        newImg = newImg:sub(1,newDim[1],pad,newDim[2]-pad,pad,newDim[3]-pad)
    #    else
    #        newImg = newImg:sub(pad,newDim[1]-pad,pad,newDim[2]-pad)
        
    newImg=cv.scale(newImg,res,res)

    #newImg = image.scale(newImg,res,res)
    return newImg


def twoPointCrop(img, s, pt1, pt2, pad, res):
    center = (pt1 + pt2) / 2
    scale = max(20*s,np.linalg.norm(pt1 - pt2)) * .007
    scale = scale * pad
    angle = math.atan2(pt2[2]-pt1[2],pt2[1]-pt1[1]) * 180 / math.pi - 90
    return crop(img, center, scale, angle, res)


def compileImages(imgs, nrows, ncols, res):
    # Assumes the input images are all square/the same resolution
    totalImg = np.zeros(3,nrows*res,ncols*res)
    i=0
    for img in imgs:
        r = np.floor((i-1)/ncols) + 1
        c = ((i - 1) % ncols) + 1
     #   totalImg:sub(1,3,(r-1)*res+1,r*res,(c-1)*res+1,c*res):copy(imgs[i])
        totalImg[0:3,((r-1)*res+1):(r*res),((c-1)*res+1):(c*res)]=img
        i+=1
    
    return totalImg


#######################################-
# Non-maximum Suppression
#######################################-

# Set up max network for NMS
nms_window_size = 3
nms_pad = (nms_window_size - 1)/2
maxlayer = nn.Sequential()
if cudnn:
    maxlayer:add(cudnn.SpatialMaxPooling(nms_window_size, nms_window_size,1,1, nms_pad, nms_pad))
    maxlayer:cuda()
else
    maxlayer:add(nn.SpatialMaxPooling(nms_window_size, nms_window_size,1,1, nms_pad,nms_pad))

maxlayer:evaluate()

def local_maxes(hm, n, c, s, hm_idx)
    hm = np.Tensor(1,16,64,64):copy(hm):float()
    if hm_idx: hm = hm:sub(1,-1,hm_idx,hm_idx) 
    hm_dim = hm:size()
    max_out
    # First do nms
    if cudnn:
        hmCuda = np.CudaTensor(1, hm_dim[2], hm_dim[3], hm_dim[4])
        hmCuda:copy(hm)
        max_out = maxlayer:forward(hmCuda)
        cunp.synchronize()
    else
        max_out = maxlayer:forward(hm)
    

    nms = np.cmul(hm, np.eq(hm, max_out:float()):float())[1]
    # Loop through each heatmap retrieving top n locations, and their scores
    pred_coords = np.Tensor(hm_dim[2], n, 2)
    pred_scores = np.Tensor(hm_dim[2], n)
    for i = 1, hm_dim[2] do
        nms_flat = nms[i]:view(nms[i]:nElement())
        vals,idxs = np.sort(nms_flat,1,true)
        for j = 1,n do
            pt = {idxs[j] % 64, np.ceil(idxs[j] / 64) }
            pred_coords[i][j] = transform(pt, c, s, 0, 64, true)
            pred_scores[i][j] = vals[j]
        
    
    return pred_coords, pred_scores


#######################################-
# Drawing functions
#######################################-

def drawGaussian(img, pt, sigma)
    # Draw a 2D gaussian
    # Check that any part of the gaussian is in-bounds
    ul = {.floor(pt[1] - 3 * sigma), .floor(pt[2] - 3 * sigma)}
    br = {.floor(pt[1] + 3 * sigma), .floor(pt[2] + 3 * sigma)}
    # If not, return the image as is
    if (ul[1] > img:size(2) or ul[2] > img:size(1) or br[1] < 1 or br[2] < 1): return img 
    # Generate gaussian
    size = 6 * sigma + 1
    g = image.gaussian(size) # , 1 / size, 1)
    # Usable gaussian range
    g_x = {max(1, -ul[1]), min(br[1], img:size(2)) - max(1, ul[1]) + max(1, -ul[1])}
    g_y = {max(1, -ul[2]), min(br[2], img:size(1)) - max(1, ul[2]) + max(1, -ul[2])}
    # Image range
    img_x = {max(1, ul[1]), min(br[1], img:size(2))}
    img_y = {max(1, ul[2]), min(br[2], img:size(1))}
    assert(g_x[1] > 0 and g_y[1] > 0)
    img:sub(img_y[1], img_y[2], img_x[1], img_x[2]):add(g:sub(g_y[1], g_y[2], g_x[1], g_x[2]))
    img[img:gt(1)] = 1
    return img


def drawLine(img,pt1,pt2,width,color)
    # I'm sure there's a line drawing def somewhere in np,
    # but since I couldn't find it here's my basic implementation
    color = color or {1,1,1}
    m = np.dist(pt1,pt2)
    dy = (pt2[2] - pt1[2])/m
    dx = (pt2[1] - pt1[1])/m
    for j = 1,width do
        start_pt1 = np.Tensor({pt1[1] + (-width/2 + j-1)*dy, pt1[2] - (-width/2 + j-1)*dx})
        start_pt1:ceil()
        for i = 1,np.ceil(m) do
            y_idx = np.ceil(start_pt1[2]+dy*i)
            x_idx = np.ceil(start_pt1[1]+dx*i)
            if y_idx - 1 > 0 and x_idx -1 > 0 and y_idx < img:size(2) and x_idx < img:size(3):
                img:sub(1,1,y_idx-1,y_idx,x_idx-1,x_idx):fill(color[1])
                img:sub(2,2,y_idx-1,y_idx,x_idx-1,x_idx):fill(color[2])
                img:sub(3,3,y_idx-1,y_idx,x_idx-1,x_idx):fill(color[3])
            
         
    
    img[img:gt(1)] = 1

    return img


def colorHM(x)
    # Converts a one-channel grayscale image to a color heatmap image
    def gauss(x,a,b,c)
        return np.exp(-np.pow(np.add(x,-b),2):div(2*c*c)):mul(a)
    
    cl = np.zeros(3,x:size(1),x:size(2))
    cl[1] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
    cl[2] = gauss(x,1,.5,.3)
    cl[3] = gauss(x,1,.2,.3)
    cl[cl:gt(1)] = 1
    return cl



#######################################-
# Flipping functions
#######################################-

def shuffleLR(x)
    dim
    if x:nDimension() == 4:
        dim = 2
    else
        assert(x:nDimension() == 3)
        dim = 1
    

    matched_parts = {
        {1,6},   {2,5},   {3,4},
        {11,16}, {12,15}, {13,14}
    }

    for i = 1,#matched_parts do
        idx1, idx2 = unpack(matched_parts[i])
        tmp = x:narrow(dim, idx1, 1):clone()
        x:narrow(dim, idx1, 1):copy(x:narrow(dim, idx2, 1))
        x:narrow(dim, idx2, 1):copy(tmp)
    

    return x


def flip(x)
    require 'image'
    y = np.FloatTensor(x:size())
    for i = 1, x:size(1) do
        image.hflip(y[i], x[i]:float())
    
    return y:typeAs(x)

