from random import shuffle
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import math


class Solver(object):
    '''
    TO DO: have a measure for performance over time such as NSS or MSE to observe behavior on validation data
    '''

    default_adam_args = {"lr": 1e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={}):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.init_lr = self.optim_args['lr']
        self.oneDivSqrtTwoPI = 1.0 / math.sqrt(2.0*math.pi) # normalisation factor for gaussian.
        self.oneDivTwoPI = 1.0 / (2.0*math.pi)
        
        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []
        

    def _gaussian_distribution2d(self, out_mu_x, out_mu_y, out_sigma, out_corr, fix_data):
        #print(fix_data.size())
        nFrames, nFixs,_ = fix_data.size()
        KMIX = out_mu_x.size(1)
        # combine x and y mean values
        out_mu_xy = torch.cat((out_mu_x.unsqueeze(2), out_mu_y.unsqueeze(2)),2)
        # braodcast subtraction with mean and normalization to sigma
        fix_data = fix_data.expand(KMIX, *fix_data.size())
        out_mu_xy = out_mu_xy.expand(nFixs, *out_mu_xy.size())
        out_mu_xy = out_mu_xy.contiguous().view(fix_data.size())
        out_sigma = out_sigma.expand(nFixs, *out_sigma.size())
        out_sigma = out_sigma.contiguous().view(fix_data.size()[:-1])
        out_corr = out_corr.expand(nFixs, *out_corr.size())
        out_corr = out_corr.contiguous().view(fix_data.size()[:-1])

        result = (fix_data - out_mu_xy) 
        result = result[:,:,:,0]**2 + result[:,:,:,1]**2 - 2*out_corr*result.prod(3)
        result = result * torch.reciprocal(out_sigma**2)
        result = result * -0.5 * torch.reciprocal(1-out_corr**2)
        result = self.oneDivTwoPI * torch.reciprocal(out_sigma**2) * torch.reciprocal(torch.sqrt(1-out_corr**2)) * torch.exp(result)

        return result


    def mdn_loss_function(self, out_pi, out_mu_x, out_mu_y, out_sigma, out_corr, fix_data):
        '''
        input:
            - out_pi, out_mu_x, out_mu_y, out_sigma, out_corr : Gaussians parameters
            - fix_data : Ground truth fixation data
            
        Output:
            - Mean of negative log probability loss
        
        # the output of this function is the mean of the probability values. this differs from the original paper, in which the formula
        is taking the sum as oppose to the mean
        '''
        result = self._gaussian_distribution2d(out_mu_x, out_mu_y, out_sigma, out_corr, fix_data)
        nFix = result.size(-1)
        out_pi = out_pi.expand(nFix, *out_pi.size())
        out_pi = out_pi.contiguous().view(result.size())
        result = result * out_pi
        result = torch.sum(result, dim=0)
        result = - torch.log(result)

	# getting the mask to ignore fixation values with (0, 0) coordinate
        mask0 = fix_data[:,:,0] != 0
        mask1 = fix_data[:,:,1] != 0
        masks = mask0 + mask1
        mask = masks != 0
        KMIX = out_mu_x.size(1)
        mask = mask.expand(KMIX, *mask.size())
        #print(mask.size())
        return torch.mean(result[mask])



    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0, n_decay_epoch=None,decay_factor=0.1):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        self.n_decay_epoch = n_decay_epoch
        self.decay_factor = decay_factor
        optim = self.optim(filter(lambda p: p.requires_grad, model.parameters()), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')

        nIterations = num_epochs*iter_per_epoch


        for i in range(1,num_epochs+1):
            for j, (inputs, labels) in enumerate(train_loader, 1):
                # I don't know why the dataloader gives double tensors!
                inputs = inputs.float()
                labels = labels.float()
                it = i*iter_per_epoch + j
                inputs = inputs.squeeze(dim=0)
                labels = labels.squeeze(dim=0)
                inputs = Variable(inputs)
                labels = Variable(labels)
                if model.is_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                
                (out_pi, out_mu_x, out_mu_y, out_sigma, out_corr) = model(inputs)
                loss = self.mdn_loss_function(out_pi, out_mu_x, out_mu_y, out_sigma, out_corr, labels)
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                if it%log_nth==0:
                    print('[Iteration %i/%i] TRAIN loss: %f' % (it,nIterations,loss.data.cpu().numpy()))
                    self.train_loss_history.append(loss.data.cpu().numpy())

                    ####################################################### 
                    # To do: NSS score can be computed and reported here!
                    #
                    #
                    #######################################################
                    
                    # Validation set
                    val_losses = []
                    model.eval()   #Set model state to evaluation 
                    
                    for ii,(inputs, labels) in enumerate(val_loader, 1):
                        # I don't know why the dataloader gives double tensors!
                        inputs, labels = Variable(inputs.float().squeeze(dim=0)),Variable(labels.float().squeeze(dim=0))
                        
                        if model.is_cuda:
                            inputs, labels = inputs.cuda(), labels.cuda()

                        (out_pi, out_mu_x, out_mu_y, out_sigma, out_corr) = model.forward(inputs)
                        loss_val = self.mdn_loss_function(out_pi, out_mu_x, out_mu_y, out_sigma, out_corr, labels)
                        val_losses.append(loss_val.data.cpu().numpy())
                       
                    print('[Epoch %i/%i] TRAIN loss: /%f' % (i,num_epochs,loss.data.cpu().numpy()))
                    print('[Epoch %i/%i] VAL loss: %f' % (i,num_epochs,np.mean(val_losses)))
                    
                    model.train() #Set model state to training
                    
	    if self.n_decay_epoch is not None:
		optim = self.decay_lr(self, i, optim)
                        
        print('FINISH.')
        
    def decay_lr(self, epoch, optimizer):
        """Decay learning rate by a factor of decay_factor every n_decay_epoch epochs."""
        if epoch % lr_decay_epoch == 0:
            lr = self.init_lr * (self.decay_factor**(epoch // self.n_decay_epoch))
            print('Learning rate is set to {}'.format(lr))
            params = optimizer.state_dict()
            params['lr'] = lr
            optimizer.load_state_dict(params)

        return optimizer
        
        
        
        
if __name__ == '__main__':

    # ==== Testing MDN1D (forward and backward pass) ====
    FNUM = 10  # number of frames!
    inp_data = np.float32(np.random.rand(FNUM, 256))
    inp_data = Variable(torch.from_numpy(inp_data))
    fix_data = np.float32(np.random.rand(FNUM, 10, 2)) # FNUM frames, 10 fixation points
    fix_data = Variable(torch.from_numpy(fix_data), requires_grad=False)

    H, KMIX = 128, 10
    model = MDN1D(hidden_size=H, num_mixtures=KMIX)

    solver_obj = Solver()
    solver_obj.train(inp_data, fix_data)


    # ===== Testing MDN2D (forward and backward pass) =====
    FNUM = 10  # number of frames!
    inp_data = np.float32(np.random.rand(FNUM,3,32,32))
    inp_data = Variable(torch.from_numpy(inp_data))
    fix_data = np.float32(np.random.rand(FNUM, 10, 2)) # FNUM frames, 10 fixation points
    fix_data = Variable(torch.from_numpy(fix_data), requires_grad=False)

    # defining the network
    h_in, w_in, H, KMIX = inp_data.size(-2), inp_data.size(-1), 256, 10 
    model = MDN2D(in_height=h_in, in_width=w_in, hidden_size=H, num_mixtures=KMIX)

    solver_obj = Solver()
    solver_obj.train(inp_data, fix_data)
