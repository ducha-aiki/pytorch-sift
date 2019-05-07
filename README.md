This is an differentiable [pytorch](https://github.com/pytorch/pytorch) implementation of SIFT patch descriptor. It is very slow for describing one patch, but quite fast for batch. It can be used for descriptop-based learning shape of affine feature.

There are different implementations of the SIFT on the web. I tried to match [Michal Perdoch implementation](https://github.com/perdoch/hesaff/blob/master/siftdesc.cpp), which gives high quality features for image retrieval [CVPR2009](http://cmp.felk.cvut.cz/~chum/papers/perdoch-cvpr09.pdf). However, on planar datasets, it is inferior to [vlfeat implementation](http://www.vlfeat.org/sandbox/api/sift.html).
The main difference is gaussian weighting window parameters, so I have made a vlfeat-like version too.  MP version weights patch center much more and additionally crops everything outside the circular region.


![Michal Perdoch kernel](/img/mp_kernel.png)
![vlfeat kernel](/img/vlfeat_kernel.png)



```python

descriptor_mp_mode = SIFTNet(patch_size = 65,
                        sigma_type= 'hesamp',
                        masktype='CircularGauss')

descriptor_vlfeat_mode = SIFTNet(patch_size = 65,
                        sigma_type= 'vlfeat',
                        masktype='Gauss')

```
Results:

![hpatches mathing results](/img/hpatches-results.png)


```
VLFeat-SIFT - mAP 
    Easy      Hard      Tough      mean
--------  --------  ---------  --------
0.466584  0.203966  0.0935743  0.254708


OPENCV-SIFT - mAP 
   Easy     Hard      Tough     mean
-------  -------  ---------  -------
0.47788  0.20997  0.0967711  0.26154


PYTORCH-SIFT-MP-65 - mAP 
    Easy      Hard      Tough      mean
--------  --------  ---------  --------
0.430887  0.184834  0.0832707  0.232997


PYTORCH-SIFT-VLFEAT-65 - mAP 
    Easy      Hard      Tough      mean
--------  --------  ---------  --------
0.472563  0.202458  0.0910371  0.255353


```
    
Speed: 
- 0.00246 s per 65x65 patch - [numpy SIFT](https://github.com/ducha-aiki/numpy-sift)
- 0.00028 s per 65x65 patch - [C++ SIFT](https://github.com/perdoch/hesaff/blob/master/siftdesc.cpp)
- 0.00074 s per 65x65 patch - CPU, 256 patches per batch
- 0.00038 s per 65x65 patch - GPU (GM940, mobile), 256 patches per batch
- 0.00038 s per 65x65 patch - GPU (GM940, mobile), 256 patches per batch




If you use this code for academic purposes, please cite the following paper:

```
@InProceedings{AffNet2018,
    title = {Repeatability Is Not Enough: Learning Affine Regions via Discriminability},
    author = {Dmytro Mishkin, Filip Radenovic, Jiri Matas},
    booktitle = {Proceedings of ECCV},
    year = 2018,
    month = sep
}

```

