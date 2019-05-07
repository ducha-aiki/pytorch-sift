import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def getPoolingKernel(kernel_size = 25):
    half_size = float(kernel_size)/2.0
    xc2 = []
    for i in range(kernel_size):
        xc2.append(half_size - abs(float(i)+0.5-half_size))
    xc2 = np.array(xc2)
    kernel = np.outer(xc2.T,xc2)
    kernel = kernel/(half_size**2)
    return kernel

def get_bin_weight_kernel_size_and_stride(patch_size, num_spatial_bins):
    ks = 2*int(patch_size / (num_spatial_bins+1));
    stride= patch_size // num_spatial_bins
    pad = ks //4
    return ks, stride,pad

class SIFTNet(nn.Module):
    def CircularGaussKernel(self,kernlen=21, circ = True, sigma_type = 'hesamp'):
        halfSize = float(kernlen) / 2.;
        r2 = float(halfSize**2);
        if sigma_type == 'hesamp':
            sigma_mul_2 = 0.9 * r2;
        elif sigma_type == 'vlfeat':
            sigma_mul_2 = kernlen**2
        else:
            raise ValueError('Unknown sigma_type', sigma_type, 'try hesamp or vlfeat')
        disq = 0;
        kernel = np.zeros((kernlen,kernlen))
        for y in range(kernlen):
            for x in range(kernlen):
                disq = (y - halfSize+0.5)**2 +  (x - halfSize+0.5)**2;
                kernel[y,x] = math.exp(-disq / sigma_mul_2)
                if circ and (disq >= r2):
                    kernel[y,x] = 0.
        return kernel
    def __repr__(self):
            return self.__class__.__name__ + '(' + 'num_ang_bins=' + str(self.num_ang_bins) +\
             ', ' + 'num_spatial_bins=' + str(self.num_spatial_bins) +\
             ', ' + 'patch_size=' + str(self.patch_size) +\
             ', ' + 'rootsift=' + str(self.rootsift) +\
             ', ' + 'sigma_type=' + str(self.sigma_type) +\
             ', ' + 'mask_type=' + str(self.mask_type) +\
             ', ' + 'clipval=' + str(self.clipval) + ')'
    def __init__(self,
                 patch_size = 65, 
                 num_ang_bins = 8,
                 num_spatial_bins = 4,
                 clipval = 0.2,
                 rootsift = False,
                 mask_type = 'CircularGauss',
                 sigma_type = 'hesamp'):
        super(SIFTNet, self).__init__()
        self.eps = 1e-10
        self.num_ang_bins = num_ang_bins
        self.num_spatial_bins = num_spatial_bins
        self.clipval = clipval
        self.rootsift = rootsift
        self.mask_type = mask_type
        self.patch_size = patch_size
        self.sigma_type = sigma_type

        if self.mask_type == 'CircularGauss':
            self.gk = torch.from_numpy(self.CircularGaussKernel(kernlen=patch_size, circ=True, sigma_type=sigma_type).astype(np.float32))
        elif self.mask_type == 'Gauss':
            self.gk = torch.from_numpy(self.CircularGaussKernel(kernlen=patch_size, circ=False, sigma_type=sigma_type).astype(np.float32))
        elif self.mask_type == 'Uniform':
            self.gk = torch.ones(patch_size,patch_size).float() / float(patch_size*patch_size)
        else:
            raise ValueError(masktype, 'is unknown mask type')
            
        self.bin_weight_kernel_size, self.bin_weight_stride, self.pad = get_bin_weight_kernel_size_and_stride(patch_size, num_spatial_bins)
        self.gx =  nn.Conv2d(1, 1, kernel_size=(1,3),  bias = False)
        self.gx.weight.data = torch.tensor(np.array([[[[-1, 0, 1]]]], dtype=np.float32))
        
        self.gy = nn.Conv2d(1, 1, kernel_size=(3,1),  bias = False)
        self.gy.weight.data = torch.from_numpy(np.array([[[[-1], [0], [1]]]], dtype=np.float32))
        nw = getPoolingKernel(kernel_size = self.bin_weight_kernel_size)
        
        self.pk = nn.Conv2d(1, 1, kernel_size=(nw.shape[0], nw.shape[1]),
                            stride = (self.bin_weight_stride, self.bin_weight_stride),
                            padding = (self.pad , self.pad ),
                            bias = False)
        new_weights = np.array(nw.reshape((1, 1, nw.shape[0],nw.shape[1])))
        self.pk.weight.data = torch.from_numpy(new_weights.astype(np.float32))
        return

    def forward(self, x):
        gx = self.gx(F.pad(x, (1, 1, 0, 0), 'replicate'))
        gy = self.gy(F.pad(x, (0, 0, 1, 1), 'replicate'))
        mag = torch.sqrt(gx * gx + gy * gy + self.eps)
        ori = torch.atan2(gy, gx + self.eps)
        mag  = mag * self.gk.expand_as(mag).to(mag.device)
        o_big = (ori + 2.0 * math.pi )/ (2.0 * math.pi) * float(self.num_ang_bins)
        bo0_big_ =  torch.floor(o_big)
        wo1_big_ = o_big - bo0_big_
        bo0_big =  bo0_big_ %  self.num_ang_bins
        bo1_big = (bo0_big + 1) % self.num_ang_bins
        wo0_big = (1.0 - wo1_big_) * mag
        wo1_big = wo1_big_ * mag
        ang_bins = []
        for i in range(0, self.num_ang_bins):
            out = self.pk((bo0_big == i).float() * wo0_big + (bo1_big == i).float() * wo1_big)
            ang_bins.append(out)
        ang_bins = torch.cat(ang_bins,1)
        ang_bins = ang_bins.view(ang_bins.size(0), -1)
        ang_bins = F.normalize(ang_bins, p=2)
        ang_bins = torch.clamp(ang_bins, 0., float(self.clipval))
        ang_bins = F.normalize(ang_bins, p=2)
        if self.rootsift:
            ang_bins = torch.sqrt(F.normalize(ang_bins,p=1) + 1e-10)
        return ang_bins