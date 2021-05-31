import argparse
import torch
import torch.nn as nn

from models.flownet2.models import FlowNet2
from models.flownet2.utils import tools
from models.flownet2 import models, losses
from models.flownet2.utils.flow_utils import *

from models.pwc_net.models.PWCNet import PWCDCNet

def load_flownet2():
	parser = argparse.ArgumentParser()

	parser.add_argument('--start_epoch', type=int, default=1)
	parser.add_argument('--total_epochs', type=int, default=10000)
	parser.add_argument('--batch_size', '-b', type=int, default=8, help="Batch size")
	parser.add_argument('--train_n_batches', type=int, default=-1,
						help='Number of min-batches per epoch. If < 0, it will be determined by training_dataloader')
	parser.add_argument('--crop_size', type=int, nargs='+', default=[256, 256],
						help="Spatial dimension to crop training samples for training")
	parser.add_argument('--gradient_clip', type=float, default=None)
	parser.add_argument('--schedule_lr_frequency', type=int, default=0, help='in number of iterations (0 for no schedule)')
	parser.add_argument('--schedule_lr_fraction', type=float, default=10)
	parser.add_argument("--rgb_max", type=float, default=255.)

	parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
	parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
	parser.add_argument('--no_cuda', action='store_true')

	parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--name', default='run', type=str, help='a name to append to the save directory')
	parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

	parser.add_argument('--validation_frequency', type=int, default=5, help='validate every n epochs')
	parser.add_argument('--validation_n_batches', type=int, default=-1)
	parser.add_argument('--render_validation', action='store_true',
						help='run inference (save flows to file) and every validation_frequency epoch')

	parser.add_argument('--inference', action='store_true')
	parser.add_argument('--inference_visualize', action='store_true',
						help="visualize the optical flow during inference")
	parser.add_argument('--inference_size', type=int, nargs='+', default=[-1, -1],
						help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
	parser.add_argument('--inference_batch_size', type=int, default=1)
	parser.add_argument('--inference_n_batches', type=int, default=-1)
	parser.add_argument('--save_flow', action='store_true', help='save predicted flows to file')

	parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
	parser.add_argument('--log_frequency', '--summ_iter', type=int, default=1, help="Log every n batches")

	parser.add_argument('--skip_training', action='store_true')
	parser.add_argument('--skip_validation', action='store_true')

	parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
	parser.add_argument('--fp16_scale', type=float, default=1024.,
						help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

	tools.add_arguments_for_module(parser, models, argument_for_class='model', default='FlowNet2')

	tools.add_arguments_for_module(parser, losses, argument_for_class='loss', default='L1Loss')

	tools.add_arguments_for_module(parser, torch.optim, argument_for_class='optimizer', default='Adam', skip_params=['params'])

	args, unknown = parser.parse_known_args()
	flownet = FlowNet2(args)
	flownet.load_state_dict(torch.load("models/flownet2/weights/FlowNet2_checkpoint.pth.tar")['state_dict'])

	return flownet

def load_pwcnet():
	pwc = PWCDCNet()
	state_dict = torch.load("models/pwc_net/pwc_net_chairs.pth.tar")
	pwc.load_state_dict(state_dict)

	return pwc

def init_weights(m):
	if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
		nn.init.xavier_normal(m.weight)
		if m.bias is not None:
			nn.init.constant(m.bias, 0.0)
