import cv2
import os
import torch
import glob 

from depth_anything_v2.dpt import DepthAnythingV2
datadir = os.path.expanduser('~/datasets/umi_3dgs/rgb_20240424_pick_mapping_tmp/rgb')
image_paths = glob.glob(os.path.join(datadir, '*.png'))
print(len(image_paths))

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
import pdb; pdb.set_trace()

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread(image_paths[0])
depth = model.infer_image(raw_img) # HxW raw depth map in numpy

import pdb; pdb.set_trace()