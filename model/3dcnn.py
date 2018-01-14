"""
3dCNN
This code snippet is adopted from "https://github.com/kenshohara/video-classification-3d-cnn-pytorch"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv


class CNN3d(nn.Module):
    def __init__(self, shortcut_type='B', cardinality=32):
        super(CNN3d, self).__init__()

        '''
        To do: Create the ResNext101 layers here
        '''
        self.inplanes = 64
	## added for lstm
	self.hidden_size=256
	self.num_LSTM_Layers=1
        self.batch_size = ???
	##
        block = ResNeXtBottleneck
        # Convolutional layer 1
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, 3, shortcut_type, cardinality)
        self.layer2 = self._make_layer(block, 256, 4, shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(block, 512, 23, shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(block, 1024, 3, shortcut_type, cardinality, stride=2)
        # added for lstm
	self.lstm = nn.LSTM(1024,self.hidden_size,self.num_LSTM_Layers)
	self.hidden = self.init_hidden()
        ##

        self.loadweights()

    def forward(self, x):
        '''
        To do: write forward path here
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        ## added for lstm
        x = x.view(x.size(0), -1)
        x, self.hidden = self.lstm(x, self.hidden)
        ##

        return x

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def loadweights(self):
        print("Updating weights from ResNext...")
        state_dict = self.state_dict()
        pt_model = torch.load('resnext-101-kinetics.pth')
        pt_sd = pt_model['state_dict']

        for name, _ in state_dict.items():
            # print(type(state_dict[name]))
            # print(type(pt_sd['module.'+name]))
            assert state_dict[name].size() == pt_sd['module.' + name].size()
            state_dict[name] = pt_sd['module.' + name]
        self.load_state_dict(state_dict)
        print('weights have been updated')

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

    ## added for lstm
    def init_hidden()
        # the first is the hidden h
        # the second is the cell  c
        #(num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t=seq_len
        return (autograd.Variable(torch.zeros(self.num_LSTM_Layers, self.batch_size, self.hidden_size)),
                autograd.Variable(torch.zeros(self.num_LSTM_Layers, self.batch_size, self.hidden_size)))     
    ##

class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(mid_planes, mid_planes, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

return out
