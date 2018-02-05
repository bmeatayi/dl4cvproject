from random import shuffle
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import math
from random import *
import time

from util.KLDiv_measure import KLDiv


class Solver(object):

    default_adam_args = {"lr": 1e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={}, eval_measure='NSS'):
        
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.init_lr = self.optim_args['lr']
        self.oneDivSqrtTwoPI = 1.0 / math.sqrt(2.0*math.pi) # normalisation factor for gaussian.
        self.oneDivTwoPI = 1.0 / (2.0*math.pi)
        self.eval_measure = eval_measure
        
        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the NSS and the loss.
        """
        self.train_loss_history = []
        self.train_Eval_history = []
        self.val_Eval_history = []
        self.val_loss_history = []
        

    def _gaussian_distribution2d(self, out_mu_x, out_mu_y, out_sigma, out_corr, fix_data):
        
        #print('fixdata_size (GD):', fix_data.size())
        #fix_data = fix_data.view(fix_data.size(0)*fix_data.size(1), *fix_data.size()[2:])
        #print('out_mu_size (GD)', out_mu_x.size())
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
        result = torch.mean(result, dim=0)
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


    def NSS_score(self, out_pi, out_mu_x, out_mu_y, out_sigma, out_corr, fix_data):
        '''
        Input:
            - out_pi, out_mu_x, out_mu_y, out_sigma, out_corr : Gaussians parameters
            - fix_data : Ground truth fixation data

        Output:
            - NSS Score for each of the clips
        '''
        out_pi, out_mu_x, out_mu_y, out_sigma, out_corr,fix_data = out_pi.data, out_mu_x.data, out_mu_y.data, out_sigma.data, out_corr.data, fix_data.data

        #print('out_pi size:', out_pi.size())
        #print('fix_data size:', fix_data.size())
        xGrid, yGrid = np.meshgrid(np.linspace(1, 112, 112), np.linspace(1, 112, 112))
        map_locations = torch.zeros(112*112, 2).cuda()
        xGrid = xGrid.reshape(112*112).astype(np.float32)
        yGrid = yGrid.reshape(112*112).astype(np.float32)
        map_locations[:,0] = torch.from_numpy(xGrid.copy()).cuda()
        map_locations[:,1] = torch.from_numpy(yGrid.copy()).cuda()
        del xGrid, yGrid
        N, KMIX = out_pi.size()
        #KMIX = out_pi.size(1)
        #print('map_locations',map_locations.size())
        map_locations = map_locations.expand(N, *map_locations.size())/112
        #print('out_mu_x:', out_mu_x.size())
        #print('map_locations expanded',map_locations.size())
        out_pi_all = out_pi.expand(112*112, *out_pi.size())
        out_pi_all = out_pi_all.contiguous().view(KMIX, N, 112*112)
        #print('out_pi_all',out_pi_all.size())
        sal_results = torch.zeros(1, N, 112*112).cuda()
        
        # Generate saliency map from different gaussians in a loop to avoid memory overuse:
        for k in range(KMIX):
            sal_results = sal_results + out_pi_all[k,:,:].contiguous().view(
                1, N, 112*112) *self._gaussian_distribution2d(out_mu_x[:,k].contiguous().view(N,1), 
                                          out_mu_y[:,k].contiguous().view(N,1), out_sigma[:,k].contiguous().view(N,1),
                                          out_corr[:,k].contiguous().view(N,1), map_locations)
        sal_results = sal_results/KMIX

        sal_results = sal_results.squeeze()
        #print('fix_data:',fix_data.size())
        sal_results_mean = torch.mean(sal_results, dim=1)
        sal_results_mean = sal_results_mean.view(*sal_results_mean.size(),1)
        sal_results_std = torch.std(sal_results, dim=1)
        sal_results_std = sal_results_std.view(*sal_results_std.size(), 1)
        
        sal_results -= sal_results_mean 
        sal_results /= sal_results_std
        #print(sal_results_mean.size())
        #print('sal_res:', sal_results.size())
        sal_results  = sal_results.contiguous().view(sal_results.size(0),112,112)
        fix_data = fix_data.cpu().numpy().astype(np.float).transpose(0,2,1)
        nTrueFix = np.sum(fix_data!=0)
        mask = np.zeros((N,112,112),dtype=np.uint8)
        fix_data_xy = np.floor(fix_data*112).astype(np.uint8)
        #print('fix_data_xy size:',fix_data_xy.shape)
        for i in range(N):
            mask[i,fix_data_xy[i,0,:],fix_data_xy[i,1,:]]+=1
            mask[i,0,0]=0 #to remove the effect of zero fixations
        mask = torch.from_numpy(mask.astype(np.float32)).cuda()
        #print(mask.size(),sal_results.size())
        NSS = torch.sum(sal_results*mask)/nTrueFix
        

        return NSS


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
        #self._reset_histories()
        iter_per_epoch = len(train_loader)

        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')

        nIterations = num_epochs*iter_per_epoch
        it=0
        for i in range(num_epochs):
            for j, (inputs, labels) in enumerate(train_loader, 1):
                tic = time.clock()
                inputs = inputs.float()
                labels = labels.float()
                #it = i*iter_per_epoch + j
                it+=1
                inputs = inputs.squeeze(dim=0)
                labels = labels.squeeze(dim=0)
                inputs = Variable(inputs)
                labels = Variable(labels)
                if model.is_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                
                outputs = model(inputs)
                # Combine first two dimensions to concider batch size > 1
                if len(labels.size()) == 4:
                    labels = labels.view(labels.size(0)*labels.size(1), *labels.size()[2:])
                loss = self.mdn_loss_function(*outputs, labels)
                self.train_loss_history.append(loss.data.cpu().numpy())
                print('[Iteration %i/%i] TRAIN loss: %f' % (it,nIterations,loss.data.cpu().numpy()))
                toc=time.clock()
                print('This iteration took (training)', toc-tic, 'Seconds')
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                if it%log_nth==0:
                    tic = time.clock()
                    model.eval()   #Set model state to evaluation
                    
                    if self.eval_measure=='NSS':
                        train_NSS = self.NSS_score(*outputs, labels)
                        train_NSS = np.mean(train_NSS)
                        self.train_Eval_history.append(train_NSS)   
                        train_EvalM = train_NSS
                    else:
                        train_KLD = KLDiv(outputs, labels)
                        self.train_Eval_history.append(train_KLD)
                        train_EvalM = train_KLD
                        
                    

                    # Validation set
                    #val_losses = []
                    #val_NSS_Scores = []
                    
                    # Select random batch from the validation set:
                    rand_select = randint(1, len(val_loader))
                    for ii,(inputs, labels) in enumerate(val_loader, 1):
                        
                        inputs, labels = Variable(inputs.float().squeeze(dim=0)),Variable(labels.float().squeeze(dim=0))

                        if model.is_cuda:
                            inputs, labels = inputs.cuda(), labels.cuda()

                        outputs = model.forward(inputs)
                        if len(labels.size()) == 4:
                            labels = labels.view(labels.size(0)*labels.size(1), *labels.size()[2:])
                        loss_val = self.mdn_loss_function(*outputs, labels)
                        self.val_loss_history.append(loss_val.data.cpu().numpy())
                        if rand_select == ii:
                            
                            if self.eval_measure=='NSS':
                                val_NSS = self.NSS_score(*outputs, labels)
                                val_NSS = np.mean(val_NSS)
                                self.train_Eval_history.append(val_NSS)   
                                train_EvalM = val_NSS
                            else:
                                val_KLD = KLDiv(outputs, labels)
                                self.val_Eval_history.append(val_KLD)
                                val_EvalM = val_KLD
                                
                            print('[Epoch %i/%i] TRAIN KLD/loss: %f/%f' % (i+1, num_epochs, train_EvalM, loss.data.cpu().numpy()))
                            print('[Epoch %i/%i] VAL KLD/loss: %f/%f' % (i+1, num_epochs, val_EvalM, loss_val.data.cpu().numpy()))
                    toc=time.clock()
                    print('This iteration took (validation)', toc-tic, 'Seconds')
                    
                    model.train() #Set model state to training
                    
            model.Epoch_num += 1
            model.save('training_model.model')    #saves model after each epoch     
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
