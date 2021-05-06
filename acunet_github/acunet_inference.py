# import argparse
from skimage.transform import resize
from skimage import img_as_bool
from skimage.exposure import rescale_intensity
import numpy as np
import torch
from model import *
import nibabel as nib
import sys

# path
# flair_path = "./input/pre/FLAIR.nii.gz"
# t1_path = "./input/pre/T1.nii.gz"
# out_path = "./output/result.nii.gz"
# brain_mask_path = "./output/flair_rbx_mask.nii.gz"

# path
flair_path = str(sys.argv[1])
t1_path = str(sys.argv[2])
out_path = str(sys.argv[3])
brain_mask_path = str(sys.argv[4])
model_type = str(sys.argv[5])


# read FLAIR, T1 and skull stripping mask
flairnii = nib.load(flair_path)
t1nii = nib.load(t1_path)

# nii to array, remove skull
brain_m = nib.load(brain_mask_path)
brain_m_data = brain_m.get_fdata()
flair = np.multiply(flairnii.get_fdata(), brain_m_data) 
t1 = np.multiply(t1nii.get_fdata(), brain_m_data) 

# record original shape and add channel axis
flair_ori_shape = flair.shape
flair = flair[:,: ,np.newaxis,:] #add channel dimension, H,W,C,S
t1 = t1[:,: ,np.newaxis,:] #add channel dimension, H,W,C,S

''' data manipulation ''' 
#croping, H,W,C,S
flair[flair < np.max(flair) * 0.1] = 0
x_projection = np.max(np.max(np.max(flair, axis=-1), axis=-1), axis=-1)
x_nonzero = np.nonzero(x_projection)
x_min = np.min(x_nonzero)
x_max = np.max(x_nonzero) + 1
y_projection = np.max(np.max(np.max(flair, axis=0), axis=-1), axis=-1)
y_nonzero = np.nonzero(y_projection)
y_min = np.min(y_nonzero)
y_max = np.max(y_nonzero) + 1
z_projection = np.max(np.max(np.max(flair, axis=0), axis=0), axis=0)
z_nonzero = np.nonzero(z_projection)
z_min = np.min(z_nonzero)
z_max = np.max(z_nonzero) + 1
flair = flair[x_min:x_max, y_min:y_max, :, z_min:z_max]
t1 = t1[x_min:x_max, y_min:y_max, :, z_min:z_max]
flair_crop_shape = flair.shape # record shape after cropping

# padding    
a = flair.shape[0]
b = flair.shape[1]
if a != b:
    diff = (max(a, b) - min(a, b)) / 2.0
# flair padding
if a > b: 
    padding = ((0, 0), (int(np.floor(diff)), int(np.ceil(diff))), (0, 0), (0, 0)) #padding width for mask
else:
    padding = ((int(np.floor(diff)), int(np.ceil(diff))), (0, 0), (0, 0), (0, 0)) #padding height for mask
flair = np.pad(flair, padding, mode="constant", constant_values=0)
# t1 padding
if a > b: 
    padding = ((0, 0), (int(np.floor(diff)), int(np.ceil(diff))), (0, 0), (0, 0)) #padding width for mask
else:
    padding = ((int(np.floor(diff)), int(np.ceil(diff))), (0, 0), (0, 0), (0, 0)) #padding height for mask
t1 = np.pad(t1, padding, mode="constant", constant_values=0)

# resize
size = 256
v_shape = flair.shape
out_shape = (size, size, v_shape[2], v_shape[3])
flair = resize(
    flair,
    output_shape=out_shape,
    order=2,
    mode="constant",
    cval=0,
    anti_aliasing=False,
)
t1 = resize(
    t1,
    output_shape=out_shape,
    order=2,
    mode="constant",
    cval=0,
    anti_aliasing=False,
)

temp = flair

# normalize
p10 = np.percentile(flair, 10)
p99 = np.percentile(flair, 99)
flair = rescale_intensity(flair, in_range=(p10, p99))
m = np.mean(flair, axis=(0, 1, 3))
s = np.std(flair, axis=(0, 1, 3))
flair = (flair - m) / s
#t1
p10 = np.percentile(t1, 10)
p99 = np.percentile(t1, 99)
t1 = rescale_intensity(t1, in_range=(p10, p99))
m = np.mean(t1, axis=(0, 1, 3))
s = np.std(t1, axis=(0, 1, 3))
t1 = (t1 - m) / s

# concatenate flair and t1 as input
x = np.concatenate((flair, t1), axis = 2)


# define device
# device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
device = torch.device("cpu")

with torch.set_grad_enabled(False):
    if model_type == 'acunet':
        net = ACUNet(in_channels=2, out_channels=1)
        state_dict = torch.load("/acunet.pt", map_location=device)
        
    
    net.load_state_dict(state_dict)
    net.eval()
    net.to(device)
    x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x.astype(np.float32))
    x = x.to(device)
    y_pred = net(x)
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = np.round(y_pred).astype(np.bool)
    y_pred = y_pred[:,0,:,:]
    y_pred = y_pred.transpose(1,2,0)

# de_resize, back to original size
pred_deresize = img_as_bool(resize(
    y_pred,
    output_shape= (v_shape[0], v_shape[1], v_shape[3]), 
    order=0,
    mode="constant",
    cval=0,
    anti_aliasing=False,
))

#de_pad
if a > b: 
    pred_depad = pred_deresize[:,int(np.floor(diff)):(pred_deresize.shape[1]-int(np.ceil(diff))),:]
else:
    pred_depad = pred_deresize[int(np.floor(diff)):(pred_deresize.shape[1]-int(np.ceil(diff))),:,:]
temp = np.zeros(flair_ori_shape)
temp[x_min:x_max, y_min:y_max, z_min:z_max] = pred_depad
temp = nib.Nifti1Image(temp, flairnii.affine)
temp.to_filename(out_path)


