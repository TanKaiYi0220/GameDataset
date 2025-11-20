import os

import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from src.gameData_loader import load_backward_velocity
from src.warp_module import ForwardWarpingNearest, BackwardWarpingNearest, ForwardWarpingNearestWithDepth
from src.utils import flow_to_image

import sys
sys.path.append('models/IFRNet')

from models.IFRNet import Model
from utils import warp

import numpy as np
import torch
from utils import read
from imageio import mimsave


model = Model().cuda().eval()
model.load_state_dict(torch.load('./models/IFRNet/checkpoints/IFRNet/IFRNet_Vimeo90K.pth'))

img0_np = read('./models/IFRNet/custom/image1.png')
img1_np = read('./models/IFRNet/custom/image2.png')

img0 = (torch.tensor(img0_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
img1 = (torch.tensor(img1_np.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
embt = torch.tensor(1/2).view(1, 1, 1, 1).float().cuda()

imgt_pred, up_flow0_1, up_flow1_1, up_mask_1 = model.inference(img0, img1, embt)

imgt_pred_np = (imgt_pred[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
up_flow0_1_np = flow_to_image(up_flow0_1[0].data.permute(1, 2, 0).cpu().numpy())
up_flow1_1_np = flow_to_image(up_flow1_1[0].data.permute(1, 2, 0).cpu().numpy())
up_mask_1_np = (up_mask_1[0, 0].data.cpu().numpy() * 255.0).astype(np.uint8)

# Save results
# BGR to RGB
cv2.imwrite('./models/IFRNet/custom/out_2x_test.png', cv2.cvtColor(imgt_pred_np, cv2.COLOR_RGB2BGR))

# Flow and mask
cv2.imwrite('./models/IFRNet/custom/out_2x_flow0.png', up_flow0_1_np)
cv2.imwrite('./models/IFRNet/custom/out_2x_flow1.png', up_flow1_1_np)
cv2.imwrite('./models/IFRNet/custom/out_2x_mask.png', up_mask_1_np)

# Warped images
img_0_warp = warp(img0, up_flow0_1)
img_1_warp = warp(img1, up_flow1_1)
cv2.imwrite('./models/IFRNet/custom/out_2x_img0_warp.png',
            cv2.cvtColor((img_0_warp[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR))
cv2.imwrite('./models/IFRNet/custom/out_2x_img1_warp.png',
            cv2.cvtColor((img_1_warp[0].data.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR))

images = [img0_np, imgt_pred_np, img1_np]
mimsave('./models/IFRNet/custom/out_2x_test.gif', images, fps=3)