# interDance_API
The render code for interDance.  
The complete project code, including data preprocessing, training, inference and evaluating with our pretrained model, will be made publicly available.   
For more details, please visit our [webpage](https://inter-dance.github.io/).
## Getting started

This code was tested on `Ubuntu 20.04.1 LTS` and requires:

* Python 3.8
* CUDA capable GPU (one is enough)

### 1. environment
torch==2.0.0+cu118  
torchvision==0.15.0+cu118   
smplx==0.1.28   
pytorch3d     
pyrender==0.1.45        
trimesh==4.0.7  
tqdm==4.66.1    
### 2. Run
render smplx file
```shell 
python render.py --modir 'smplx pkl path' --mode smplx --fps 8 --outdir ./video_gt
```
render ver655 file
```shell 
python render.py --modir 'ver655 npy path' --mode ver655 --fps 8
```
