import torch
from torch import nn
import torch.nn.functional as F


# This is epxerimental and needs to be adjusted later!
w,h = (320,256)
o_m = h//3
o_n = w//3
ogrid_xy = F.affine_grid(torch.eye(2,3).unsqueeze(0),(1,1,o_m,o_n)).view(1,1,-1,2)
disp_range = 0.25#0.25
displacement_width = 11#15#11#17
shift_xy = F.affine_grid(disp_range*torch.eye(2,3).unsqueeze(0),(1,1,displacement_width,displacement_width)).view(1,1,-1,2)

grid_size = 32#25#30
grid_xy = F.affine_grid(torch.eye(2,3).unsqueeze(0),(1,1,grid_size,grid_size)).view(1,-1,1,2)


class OBELISK2d(nn.Module):
	def __init__(self, chan = 16):

		super(OBELISK2d, self).__init__()
		channels = chan
		self.offsets = nn.Parameter(torch.randn(2 ,channels *2 ) *0.05)
		self.layer0 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=2, bias=False, padding=2)
		self.batch0 = nn.BatchNorm2d(4)

		self.layer1 = nn.Conv2d(channels *4, channels *4, 1, bias=False, groups=1)
		self.batch1 = nn.BatchNorm2d(channels *4)
		self.layer2 = nn.Conv2d(channels *4, channels *2, 3, bias=False, padding=1)
		self.batch2 = nn.BatchNorm2d(channels *2)
		self.layer3 = nn.Conv2d(channels *2, channels *1, 1)


	def forward(self, input_img):
		img_in = F.avg_pool2d(input_img ,3 ,padding=1 ,stride=2)
		img_in = F.relu(self.batch0(self.layer0(img_in)))
		# img_in = F.relu(self.batch0(self.layer0(img_in)))
		sampled = F.grid_sample(img_in ,ogrid_xy + self.offsets[0 ,:].view(1 ,-1 ,1 ,2)).view(1 ,-1 ,o_m ,o_n)
		sampled -= F.grid_sample(img_in ,ogrid_xy + self.offsets[1 ,:].view(1 ,-1 ,1 ,2)).view(1 ,-1 ,o_m ,o_n)

		x = F.relu(self.batch1(self.layer1(sampled)))
		x = F.relu(self.batch2(self.layer2(x)))
		features = self.layer3(x)
		return features


class deeds2d(nn.Module):
	def __init__(self):

		super(deeds2d, self).__init__()
		self.alpha = nn.Parameter(torch.Tensor([1 ,.1 ,1 ,1 ,.1 ,1])  )  # .cuda()

		self.pad1 = nn.ReplicationPad2d(3  )  # .cuda()
		self.avg1 = nn.AvgPool2d(3 ,stride=1  )  # .cuda()
		self.max1 = nn.MaxPool2d(3 ,stride=1  )  # .cuda()
		self.pad2 = nn.ReplicationPad2d(2  )  # .cuda()##

	def forward(self, feat00 ,feat50):

		# deeds correlation layer (slightly unrolled)
		deeds_cost = torch.zeros(1 ,grid_size**2 ,displacement_width, displacement_width)

		# print(deeds_cost.shape)
		xy8 = grid_size**2
		# i=1
		# print(grid_xy[:,i*grid_size:(i+1)*grid_size,:,:].shape)
		for i in range(grid_size):
			moving_unfold = F.grid_sample(feat50, grid_xy[: , i *grid_size:( i +1 ) *grid_size ,: ,:] + shift_xy
										  ,padding_mode='border')
			fixed_grid = F.grid_sample(feat00, grid_xy[: , i *grid_size:( i +1 ) *grid_size ,: ,:]) # grid_xy[:,i*xy8:(i+1)*xy8,:,:]
			deeds_cost[: , i *grid_size:( i +1 ) *grid_size ,: ,:] = self.alpha[1 ] +self.alpha[0 ] *torch.sum \
				(torch.pow(fixed_grid -moving_unfold ,2) ,1).view(1 ,-1 ,displacement_width ,displacement_width)


		# remove mean (not really necessary)
		# deeds_cost = deeds_cost.view(-1,displacement_width**3) - deeds_cost.view(-1,displacement_width**3).mean(1,keepdim=True)[0]
		# deeds_cost = deeds_cost.view(1,-1,displacement_width,displacement_width)
		# print(deeds_cost.shape)

		# approximate min convolution / displacement compatibility
		cost = self.avg1(self.avg1(-self.max1(-self.pad1(deeds_cost))))
		# grid-based mean field inference (one iteration)
		cost_permute = cost.permute(2 ,3 ,0 ,1).view(1 ,displacement_width**2 ,grid_size, grid_size)
		cost_avg = self.avg1(self.avg1(self.pad2(cost_permute))).permute(0 ,2 ,3 ,1).view(1 ,-1 ,displacement_width
																						  ,displacement_width)

		# second path
		cost = self.alpha[4 ] +self.alpha[2 ] *deeds_cost +self.alpha[3 ] *cost_avg
		cost = self.avg1(self.avg1(-self.max1(-self.pad1(cost))))

		# grid-based mean field inference (one iteration)
		cost_permute = cost.permute(2 ,3 ,0 ,1).view(1 ,displacement_width**2 ,grid_size ,grid_size)
		cost_avg = self.avg1(self.avg1(self.pad2(cost_permute))).permute(0 ,2 ,3 ,1).view(grid_size**2 ,displacement_width**2)
		# cost = alpha[4]+alpha[2]*deeds_cost+alpha[3]*cost.view(1,-1,displacement_width,displacement_width,displacement_width)
		# cost = avg1(avg1(-max1(-pad1(cost))))

		# probabilistic and continuous output
		cost_soft = F.softmax(-self.alpha[5 ] *cost_avg ,1)
		# pred_xyz = torch.sum(F.softmax(-5self.alpha[2]*cost_avg,1).unsqueeze(2)*shift_xyz.view(1,-1,3),1)
		pred_xy = torch.sum(cost_soft.unsqueeze(2 ) *shift_xy.view(1 ,-1 ,2) ,1)


		return cost_soft ,pred_xy