"""
VidSal model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.autograd import Variable
import math
from model.mdn import MDN1D
from model.cnn3d import CNN3d

class VidSal(nn.Module):
    def __init__(self, model_path, hidden_size=256, LSTM_layers=1, freeze_cnn3d_weights =[True,True,True,True,True]):
        super(CNN3d, self).__init__()
        
        self.hidden_size = hidden_size
        self.LSTM_layers = LSTM_layers
        
        ## To 
        self.cnn3d = CNN3d(model_path,freeze_weights = freeze_cnn3d_weights)
        self.lstm = nn.LSTM(2048, self.hidden_size, self.LSTM_layers)
        self.hidden0 = self.init_hidden()
        
        self.mdn = MDN1D(input_dim=self.hidden_size)
        
        
    def forward(self, x):
        '''
        Note: Assumed batch size is 1 (1 video at a time, first dimension is # clips)!
        '''
        x = self.cnn3d(x)
        x = x.view(x.size(0), 1, -1)
        x, lasthidden = self.lstm(x, self.hidden0)
        x = x.view(x.size(0), -1)
        x = self.mdn(x)
        
        return x
    
    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        #(num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t=seq_len
        return (Variable(torch.zeros(self.LSTM_layers, 1, self.hidden_size)),
                Variable(torch.zeros(self.LSTM_layers, 1, self.hidden_size)))
    
        @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)