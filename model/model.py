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
        super(VidSal, self).__init__()
        
        self.Epoch_num = 0
        
        self.hidden_size = hidden_size
        self.LSTM_layers = LSTM_layers
        
        self.cnn3d = CNN3d(model_path,freeze_weights = freeze_cnn3d_weights)
        self.lstm = nn.LSTM(2048, self.hidden_size, self.LSTM_layers, batch_first=True)
        self.hidden0 = self.init_hidden()
        
        self.mdn = MDN1D(input_dim=self.hidden_size)
        
        
    def forward(self, x):
        '''
        Note: CNN3d loops over clips and gives feature map for the use of lstm
        Input dim (Ns, Nchan, Nf, H, W)
        '''
        #print('input size:',x.size())
        if len(x.size()) == 4:
            x = x.view(1, *x.size())
        
        # loop over clips
        nClips = x.size(2)-15
        #lstm_in = torch.cuda.FloatTensor(x.size(0), 0, 2048) #feature maps for lstm - dim = (Nseq,Ns,lstm_input_size)
        for i in range(nClips):
            clip = x[:,:,i:i+16,:,:]
            f = self.cnn3d(clip)
            f = f.view(f.size(0), 1, -1)
            #print("f size is ",f.size())
            if i == 0:
                lstm_in = f
            else:
                lstm_in = torch.cat((lstm_in,f), dim=1)
            
        #print('size after cnn3d:',lstm_in.size())
        
        x, lasthidden = self.lstm(lstm_in)#, self.hidden0)
        #print('after lstm size:', x.size())
        x = x.contiguous().view(-1, x.size(-1))
        #print('mdn input size:', x.size())
        x = self.mdn(x)

        
        return x
    
    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        #(num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t=seq_len
        return (Variable(torch.zeros(self.LSTM_layers, 1, self.hidden_size)).cuda(),
                Variable(torch.zeros(self.LSTM_layers, 1, self.hidden_size)).cuda())
    
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