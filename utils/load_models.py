import argparse
import torch
import torch.nn as nn

from models.flownet2_pytorch.flownet2_mph import *

import warnings
warnings.filterwarnings('ignore')


def load_flownet2():
	flow = FlowNet2()
	state_dict = torch.load(
		"models/flownet2_pytorch/FlowNet2_checkpoint.pth.tar")
	flow.load_state_dict(state_dict['state_dict'])
	return flow

def load_pwcnet():
	pwc = PWCDCNet()
	state_dict = torch.load("models/pwc_net/pwc_net_chairs.pth.tar")
	pwc.load_state_dict(state_dict)
	pwc.eval()
	return pwc

def init_weights(m):
	if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
		nn.init.xavier_normal(m.weight)
		if m.bias is not None:
			nn.init.constant(m.bias, 0.0)
