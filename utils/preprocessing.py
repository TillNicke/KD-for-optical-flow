import torch
import torch.nn.functional as F
import numpy as np
import cv2
from math import ceil


def preprocessing_pwc(img_1, img_2):
	"""
	Preprocessing function for PWC net.
	img1: numpy.ndarray in shape of (H, W, C)
	img2: numpy.ndarray in shape of (H, W, C)

	return: torch.tensor in the shape of(1, 2 * C, H, W)
	"""

	img1 = img_1.numpy().copy()
	img2 = img_2.numpy().copy()

	H,W,C = img_1.shape

	divisor = 64
	H_ = int(ceil(H/divisor) * divisor)
	W_ = int(ceil(W/divisor) * divisor)

	img1 = cv2.resize(np.concatenate([img1,img1,img1], 2), (W_,H_))
	img2 = cv2.resize(np.concatenate([img2,img2,img2], 2), (W_,H_))

	images = [img1,img2]

	for _i, _inputs in enumerate(images):
		images[_i] = images[_i][:, :, ::-1]

		images[_i] = np.transpose(images[_i], (2, 0, 1))

		# Running into a "negative number" error in th line below, need to change that
		# The error suggested to use the copy() function as a workaround.
		images[_i] = torch.from_numpy(images[_i].copy())
		# Worked ¯\_(ツ)_/¯

		images[_i] = images[_i].expand(1, images[_i].size()[0], images[_i].size()[1], images[_i].size()[2])
		images[_i] = images[_i].float()

	return torch.cat(images,1)

def preprocessing_flownet(img_1, img_2):
	"""
	Preprocessing function for FlowNet2.

	img1: numpy.ndarray in shape of (H, W, C)
	img2: numpy.ndarray in shape of (H, W, C)

	return: torch.tensor in the	shape of (1, B, C, H, W)
	"""
    
    
	img1 = img_1.numpy().copy()
	img2 = img_2.numpy().copy()

	if img1.max() <= 1.0:
		img1 = img1 * 255
	if img2.max() <= 1.0:
		img2 = img2 * 255

	if img1.shape[2] == 1:
		img1 = np.concatenate([img1,img1,img1], 2)

	if img2.shape[2] == 1:
		img2 = np.concatenate([img2,img2,img2], 2)
    
	images = [img1, img2]
	images = np.array(images).transpose(3, 0, 1, 2)
	return torch.from_numpy(images.astype(np.float32)).unsqueeze(0)