import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from util.gen_sal_map import vid_sal_map

def KLDiv(MDN_outputs, fix_data):

	sal_map = gen_sal_map(*MDN_outputs)
	
	fix_map = vid_sal_map(fix_data)
	
	KLDiv_loss = nn.KLDivLoss()
	
	sal_map = Variable(torch.from_numpy(sal_map))
	fix_map = Variable(torch.from_numpy(fix_map))
	
	# Normalize by the sum to have a prob. distribution:
	if len(sal_map.size()) == 2:
		sal_map /= torch.sum(sal_map.contiguous().view(-1))
		fix_map /= torch.sum(fix_map.contiguous().view(-1))
		sal_map = sal_map.contiguous().view(1, *sal_map.size())
		fix_map = fix_map.contiguous().view(1, *fix_map.size())
	else:
		slmap_sum = torch.sum(sal_map.contiguous().view(sal_map.size(0),-1), dim=1)
		sal_map /= sal_map.contiguous().view(*slmap_sum.size(), 1, 1).expand_as(sal_map)
		fxmap_sum = torch.sum(fix_map.contiguous().view(fix_map.size(0),-1), dim=1)
		fix_map /= fix_map.contiguous().view(*fxmap_sum.size(), 1, 1).expand_as(fix_map)

	KLD = KLDiv_loss(torch.log(sal_map), fix_map).squeeze().data
	
	return KLD

	
def gen_sal_map(out_pi, out_mu_x, out_mu_y, out_sigma, out_corr):

	xGrid, yGrid = np.meshgrid(np.linspace(1, 112, 112), np.linspace(1, 112, 112))
	map_locations = torch.zeros(112*112, 2).cuda()
	xGrid = xGrid.reshape(112*112).astype(np.float32)
	yGrid = yGrid.reshape(112*112).astype(np.float32)
	map_locations[:,0] = torch.from_numpy(xGrid.copy()).cuda()
	map_locations[:,1] = torch.from_numpy(yGrid.copy()).cuda()
	del xGrid, yGrid
	
	if len(out_pi.size()) == 2:
		N, KMIX = out_pi.size()
	else
		N = 1
		KMIX = out_pi.size(0)

	map_locations = map_locations.expand(N, *map_locations.size())/112

	out_pi_all = out_pi.expand(112*112, *out_pi.size())
	out_pi_all = out_pi_all.contiguous().view(KMIX, N, 112*112)

	sal_results = torch.zeros(1, N, 112*112).cuda()

	# Generate saliency map from different gaussians in a loop to avoid memory overuse:
	for k in range(KMIX):
		sal_results = sal_results + out_pi_all[k,:,:].contiguous().view(
		    1, N, 112*112) *self._gaussian_distribution2d(out_mu_x[:,k].contiguous().view(N,1), 
		                              out_mu_y[:,k].contiguous().view(N,1), out_sigma[:,k].contiguous().view(N,1),
		                              out_corr[:,k].contiguous().view(N,1), map_locations)
	sal_results = sal_results/KMIX

	sal_results = sal_results.squeeze()
