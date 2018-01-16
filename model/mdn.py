import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.nn as nn


class MDN2D(nn.Module):
    def __init__(self, in_height=None, in_width=None, hidden_size=256, num_mixtures=16):
        super(MDN2D, self).__init__()
        
        # an image has three channels
        # for videos, we have:
        # many images (clips) with 1 channel. This ends up with a saliency map for each clip, which is what we want!
        # So, REPLACE 3 BY 1 IN THE COMPLETE NETWROK
        
        '''
        
        Input
        '''
        
        self.conv2fc = nn.Conv2d(3, hidden_size, (in_height, in_width))
        self.tanh = nn.Tanh()
        
        self.mu_out_x = nn.Linear(hidden_size, num_mixtures)
        self.mu_out_y = nn.Linear(hidden_size, num_mixtures)
        self.sigma_out = nn.Linear(hidden_size, num_mixtures)
        self.corr_out = torch.nn.Sequential(
            nn.Linear(hidden_size, num_mixtures),
            nn.Tanh()
            )
        self.pi_out = torch.nn.Sequential(
            nn.Linear(hidden_size, num_mixtures),
            nn.Softmax()
            )
        
    def forward(self, x):
        
        '''
        
        Input: a 4D Tensor. (N, C, H, W)
        Output:
            out_pi:    mixing coefficient
            out_mu:    mean of gassians
            out_sigma: variance of gaussians
            out_corr:  covariance of gaussians
        '''
        
        out = self.conv2fc(x)
        out = self.tanh(out)
        out = out.view(-1, self.num_flat_features(out))
        out_pi = self.pi_out(out)
        out_mu_x = self.mu_out_x(out)
        out_mu_y = self.mu_out_y(out)
        out_sigma = torch.exp(self.sigma_out(out))
        out_corr = self.corr_out(out)
        
        return (out_pi, out_mu_x, out_mu_y, out_sigma, out_corr)
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    
    
class MDN1D(nn.Module):
    def __init__(self, input_dim=256, hidden_size=256, num_mixtures=16):
        super(MDN1D, self).__init__()
        
        '''
        Input:
            - the last lstm hidden layer output (assumption: it has 256 neurons)
        '''
        
        self.lstm_fc = nn.Linear(input_dim, hidden_size)
        self.tanh = nn.Tanh()
        
        self.mu_out_x = nn.Linear(hidden_size, num_mixtures)
        self.mu_out_y = nn.Linear(hidden_size, num_mixtures)
        self.sigma_out = nn.Linear(hidden_size, num_mixtures)
        self.corr_out = torch.nn.Sequential(
            nn.Linear(hidden_size, num_mixtures),
            nn.Tanh()
            )
        self.pi_out = torch.nn.Sequential(
            nn.Linear(hidden_size, num_mixtures),
            nn.Softmax()
            )
        
    def forward(self, x):
        
        '''
        
        Input: a 4D Tensor. (N, C, H, W)
        Output:
            out_pi:    mixing coefficient
            out_mu:    mean of gassians
            out_sigma: variance of gaussians
            out_corr:  covariance of gaussians
        '''
        
        out = self.lstm_fc(x)
        out = self.tanh(out)
        out_pi = self.pi_out(out)
        out_mu_x = self.mu_out_x(out)
        out_mu_y = self.mu_out_y(out)
        out_sigma = torch.exp(self.sigma_out(out))
        out_corr = self.corr_out(out)
        
        return (out_pi, out_mu_x, out_mu_y, out_sigma, out_corr)

        
        
        
if __name__ == '__main__':

	# ==== Testing MDN1D (forward pass) ====
	x_data = np.float32(np.random.rand(1,256))
	x_data = Variable(torch.from_numpy(x_data))
	
	H, KMIX = 126, 10
	model = MDN1D(hidden_size=H, num_mixtures=KMIX)
	(out_pi, out_mu_x, out_mu_y, out_sigma, out_corr) = model(x_data)
	
	print('MDN1D output example: ', out_pi)
	
	
	# ===== Testing MDN2D (forward pass) =====
	x_data = np.float32(np.random.rand(1,3,32,32))
	x_data = Variable(torch.from_numpy(x_data))
	
	# defining the network
	h_in, w_in, H, KMIX = x_data.size(-2), x_data.size(-1), 256, 10 
	model = MDN2D(in_height=h_in, in_width=w_in, hidden_size=H, num_mixtures=KMIX)
	(out_pi, out_mu_x, out_mu_y, out_sigma, out_corr) = model(x_data)
	
	print('MDN2D output example: ', out_pi)
