import math
import numpy as np
import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F

def getPoolingKernel(kernel_size = 25):
    step = 1. / float(np.floor( kernel_size / 2.));
    x_coef = np.arange(step/2., 1. ,step)
    xc2 = np.hstack([x_coef,[1], x_coef[::-1]])
    kernel = np.outer(xc2.T,xc2)
    kernel = np.maximum(0,kernel)
    return kernel

def get_bin_weight_kernel_size_and_stride(patch_size, num_spatial_bins):
    bin_weight_stride = int(round(2.0 * math.floor(patch_size / 2) / float(num_spatial_bins + 1)))
    bin_weight_kernel_size = int(2 * bin_weight_stride - 1);
    return bin_weight_kernel_size, bin_weight_stride

class SIFTNet(nn.Module):
    def CircularGaussKernel(self,kernlen=21, circ = True):
        halfSize = kernlen / 2;
        r2 = float(halfSize*halfSize);
        sigma2 = 0.9 * r2;
        disq = 0;
        kernel = np.zeros((kernlen,kernlen))
        for y in range(kernlen):
            for x in range(kernlen):
                disq = (y - halfSize)*(y - halfSize) +  (x - halfSize)*(x - halfSize);
                kernel[y,x] = math.exp(-disq / sigma2)
                if circ and disq >= r2:
                    kernel[y,x] = 0.
        return kernel
    
    def __init__(self,
                 patch_size = 65, 
                 num_ang_bins = 8,
                 num_spatial_bins = 4,
                 clipval = 0.2,
                 rootsift = False,
                 masktype = 'CircularGauss'):
        super(SIFTNet, self).__init__()
        self.eps = 1e-10
        self.num_ang_bins = num_ang_bins
        self.num_spatial_bins = num_spatial_bins
        self.clipval = clipval
        self.rootsift = rootsift
        self.masktype = masktype
        if self.masktype == 'CircularGauss':
            self.gk = torch.from_numpy(self.CircularGaussKernel(kernlen=patch_size, circ=True).astype(np.float32))
        elif self.masktype == 'Gauss':
            self.gk = torch.from_numpy(self.CircularGaussKernel(kernlen=patch_size, circ=False).astype(np.float32))
        elif self.masktype == 'Uniform':
            self.gk = torch.ones(patch_size,patch_size).float() / float(patch_size*patch_size)
        else:
            raise ValueError(masktype, 'is unknown mask type')
            
        self.bin_weight_kernel_size, self.bin_weight_stride = get_bin_weight_kernel_size_and_stride(patch_size, num_spatial_bins)
        
        self.gx =  nn.Conv2d(1, 1, kernel_size=(1,3),  bias = False)
        self.gx.weight.data = torch.tensor(np.array([[[[-1, 0, 1]]]], dtype=np.float32))
        
        self.gy = nn.Conv2d(1, 1, kernel_size=(3,1),  bias = False)
        self.gy.weight.data = torch.from_numpy(np.array([[[[-1], [0], [1]]]], dtype=np.float32))
        
        self.pk = nn.Conv2d(1, 1, kernel_size=(self.bin_weight_kernel_size, self.bin_weight_kernel_size),
                            stride = (self.bin_weight_stride, self.bin_weight_stride),
                            bias = False)
        nw = getPoolingKernel(kernel_size = self.bin_weight_kernel_size)
        new_weights = np.array(nw.reshape((1, 1, self.bin_weight_kernel_size, self.bin_weight_kernel_size)))
        self.pk.weight.data = torch.from_numpy(new_weights.astype(np.float32))
        return

    def forward(self, x):
        gx = self.gx(F.pad(x, (1, 1, 0, 0), 'replicate'))
        gy = self.gy(F.pad(x, (0, 0, 1, 1), 'replicate'))
        mag = torch.sqrt(gx * gx + gy * gy + self.eps)
        ori = torch.atan2(gy,gx + self.eps)
        mag  = mag * self.gk.expand_as(mag).to(mag.device)
        o_big = (ori +2.0 * math.pi )/ (2.0 * math.pi) * float(self.num_ang_bins)
        bo0_big =  torch.floor(o_big)
        wo1_big = o_big - bo0_big
        bo0_big =  bo0_big %  self.num_ang_bins
        bo1_big = (bo0_big + 1) % self.num_ang_bins
        wo0_big = (1.0 - wo1_big) * mag
        wo1_big = wo1_big * mag
        ang_bins = []
        for i in range(0, self.num_ang_bins):
            ang_bins.append(self.pk((bo0_big == i).float() * wo0_big + (bo1_big == i).float() * wo1_big))
        ang_bins = torch.cat(ang_bins,1)
        ang_bins = ang_bins.view(ang_bins.size(0), -1)
        ang_bins = F.normalize(ang_bins, p=2)
        ang_bins = torch.clamp(ang_bins, 0.,float(self.clipval))
        ang_bins = F.normalize(ang_bins, p=2)
        if self.rootsift:
            ang_bins = F.normalize(torch.sqrt(F.normalize(ang_bins,p=1) + 1e-10), p=2)
        return ang_bins
