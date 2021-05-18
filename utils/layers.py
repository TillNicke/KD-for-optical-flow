import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def correlation_layer(displace_range, feat_moving, feat_fixed):
	# tensor dimensionalities in comments are for an arbitrary choice of
	# displace_range = 11 & feat sizes of [1,24,80,78];
	# they clearly depend on the actual choice and only serve as numerical examples here.

	disp_hw = (displace_range -1 )//2
	# feat_mov: [1,24,80,78] -> 24 feature channels + spatial HW dims
	# feat_mov_unfold: [24,121,6240] -> mind chans, 11*11 = 121 displ steps, 6240 = 80*78 spatial positions
	feat_moving_unfold = F.unfold(feat_moving.transpose(1 ,0) ,(displace_range ,displace_range) ,padding=disp_hw)

	B ,C ,H ,W = feat_fixed.size()

	# feat_fixed: [24,1,6240] -> compute scalarproduct along feature dimension per broadcast + sum along 0
	# and reshape to [1,121,80,78]
	ssd_distance = ((feat_fixed.view(C ,1 ,-1) - feat_moving_unfold )**2).sum(0).view(1 ,displace_range**2 ,H ,W)
	# reshape the 4D tensor back to spatial dimensions
	return ssd_distance

def meanfield(ssd_distance ,img_fixed ,displace_range ,H ,W):

	# crnt_dev = ssd_distance.device


	cost = min_convolution(ssd_distance, displace_range, H, W)

	# probabilistic output: compute the contributions weights of every discrete displacement pair for all positions
	# along 121 possible xy displacements -> normalize to [0,1] with softmax:
	# in order to have the lowest SSD value as the "max" 1 value, multiply it with -10 beforehand

	# therefore apply the softmax along the displacement dimension
	# reshaping the cost tensor as follows: [1,121,H,W] -> [121, H*W] -> [H*W,121] : perform softmax along dim 1
	soft_cost = F.softmax(-10 *cost.view(displace_range**2 ,-1).t() ,1)

	# calculate displacement field (could be shorted when stacking x,y - but less intuitive)
	# idea: 1) construct a meshgrid of all discrete displacement pairs per position
	#       2) use broadcasting to get the weighted contributions of all displacement values (separated for x&y)
	#          (soft_cost) [H*W,121] x [1,121] (xs,ys) : sum along dim 1 -> [H*W] -> reshape [H,W]

	disp_hw = (displace_range -1 )//2
	xs ,ys = torch.meshgrid(torch.arange(-disp_hw ,disp_hw +1).float(),
						   torch.arange(-disp_hw ,disp_hw +1).float())

	disp_x = (soft_cost *xs.reshape(1 ,-1)).sum(1).data.reshape(H ,W).cpu()
	disp_y = (soft_cost *ys.reshape(1 ,-1)).sum(1).data.reshape(H ,W).cpu()

	# resample field to high resolution and add identity transform

	# this factor is used to resize the displacements to a valid range since the CNN architecture
	# downsamples the input images by a factor of 4
	scale_factor = 4

	# since so far only the position-wise displacements are computed, we also need to prepare the identity field
	# in order to warp the moving images correctly
	x ,y = torch.meshgrid(torch.arange(0 ,img_fixed.size(2)).float(),
						 torch.arange(0 ,img_fixed.size(3)).float())

	# before adding the displacements and the identity transform, the displacements need to be scaled according to
	# the original image dimensionality (so far computed on the 4x downsampled feature maps)
	disp_xy_up = scale_factor *F.interpolate(torch.stack((disp_x ,disp_y) ,0).cpu().unsqueeze(0),
											size=(img_fixed.size(2) ,img_fixed.size(3)),
											mode='bicubic')
	xi = x + disp_xy_up[0 ,0 ,: ,:]
	yi = y + disp_xy_up[0 ,1 ,: ,:]

	return soft_cost, disp_xy_up  # ,xi,yi

def min_convolution(ssd_distance, displace_range, H, W):
	# Prepare operators for smooth dense displacement space
	pad1 = nn.ReplicationPad2d(5)
	avg1 = nn.AvgPool2d(5 ,stride=1)
	max1 = nn.MaxPool2d(3 ,stride=1)
	pad2 = nn.ReplicationPad2d(6)
	# approximate min convolution / displacement compatibility

	# 1) switch dimensions in order to get per HW position the displacement with "highest correlation" by
	# means of lowest SSD
	# therefore, swap the dimensions of the tensor in order to make the pooling operations work along the
	# displacement search region: [1,121,80,78] -> [1,80,78,121] -> [1,80*78,11,11] = [1,6240,11,11]
	# using appropriate padding, -max(-x) u get the min value and 2x avg-pooling afterwards for quadratic smoothing
	ssd_minconv = avg1(avg1(-max1(-pad1(ssd_distance.permute(0 ,2 ,3 ,1).reshape(1 ,-1 ,displace_range ,displace_range)))))

	# 2)reconstruct the spatial arangement [1,121,80,78] and perform the spatial mean-field inference under valid padding

	ssd_minconv = ssd_minconv.permute(0 ,2 ,3 ,1).view(1 ,-1 ,H ,W)
	min_conv_cost = avg1(avg1(avg1(pad2(ssd_minconv))))

	return min_conv_cost


def warp(x, flo):
	"""
	warp an image/tensor (im2) back to im1, according to the optical flow

	x: [B, C, H, W] (im2)
	flo: [B, 2, H, W] flow

	"""
	B, C, H, W = x.size()
	# mesh grid
	xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
	yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
	xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
	yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
	grid = torch.cat((xx, yy), 1).float()

	if x.is_cuda:
		grid = grid.cuda()
	vgrid = Variable(grid) + flo

	# scale grid to [-1,1]
	vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
	vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

	vgrid = vgrid.permute(0, 2, 3, 1)
	output = nn.functional.grid_sample(x, vgrid)
	mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
	mask = nn.functional.grid_sample(mask, vgrid)

	# if W==128:
	# np.save('mask.npy', mask.cpu().data.numpy())
	# np.save('warp.npy', output.cpu().data.numpy())

	mask[mask < 0.9999] = 0
	mask[mask > 0] = 1

	return output * mask