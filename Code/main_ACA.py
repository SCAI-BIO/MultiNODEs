import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from scipy.stats import t, sem
import csv
import pandas as pd

class LatentODEfunc(nn.Module): # latent ODE with bottleneck structure
    #num_odelayers specifies the depth of the Neural ODE

    def __init__(self, latent_dim=4, nhidden_number=20, activation_ode = 'relu', num_odelayers=1):
        super(LatentODEfunc, self).__init__()
        #different activation functions
        if (activation_ode == 'none'):
            self.act = nn.Identity()
        elif(activation_ode == 'tanh'):
            self.act = nn.Tanh()
        elif(activation_ode == 'relu'):
            self.act = nn.ReLU()
        diff = nhidden_number-latent_dim
        #Implementing sequential Bottleneck structure of the Neural ODE (nhidden could be greater than 1)
        if (num_odelayers == 1):
            self.fc1 = nn.Linear(latent_dim, nhidden_number)
            self.fc2 = nn.Linear(nhidden_number, nhidden_number)
            self.fc3 = nn.Linear(nhidden_number, latent_dim)
        if (num_odelayers == 2):
            self.fc1 = nn.Linear(latent_dim, latent_dim+int(np.round(diff/2)))
            self.fc2 = nn.Linear(latent_dim+int(np.round(diff/2)), nhidden_number)
            self.fc2b = nn.Linear(nhidden_number,nhidden_number)
            self.fc2c = nn.Linear(nhidden_number,latent_dim+int(np.round(diff/2)))
            self.fc3 = nn.Linear(latent_dim+int(np.round(diff/2)), latent_dim)
        if (num_odelayers == 3):
            self.fc1 = nn.Linear(latent_dim, latent_dim+int(np.round(diff*1/3)))
            self.fc2 = nn.Linear(latent_dim+int(np.round(diff*1/3)), latent_dim+int(np.round(diff*2/3)))
            self.fc2b = nn.Linear(latent_dim+int(np.round(diff*2/3)),nhidden_number)
            self.fc2c = nn.Linear(nhidden_number,nhidden_number)
            self.fc2d = nn.Linear(nhidden_number,latent_dim+int(np.round(diff*2/3)))
            self.fc2e = nn.Linear(latent_dim+int(np.round(diff*2/3)),latent_dim+int(np.round(diff*1/3)))
            self.fc3 = nn.Linear(latent_dim+int(np.round(diff*1/3)), latent_dim)
        if (num_odelayers == 4):
            self.fc1 = nn.Linear(latent_dim, latent_dim+int(np.round(diff*1/4)))
            self.fc2 = nn.Linear(latent_dim+int(np.round(diff*1/4)), latent_dim+int(np.round(diff*2/4)))
            self.fc2b = nn.Linear(latent_dim+int(np.round(diff*2/4)),latent_dim+int(np.round(diff*3/4)))
            self.fc2c = nn.Linear(latent_dim+int(np.round(diff*3/4)),nhidden_number)
            self.fc2d = nn.Linear(nhidden_number,nhidden_number)
            self.fc2e = nn.Linear(nhidden_number,latent_dim+int(np.round(diff*3/4)))
            self.fc2f = nn.Linear(latent_dim+int(np.round(diff*3/4)),latent_dim+int(np.round(diff*2/4)))
            self.fc2g = nn.Linear(latent_dim+int(np.round(diff*2/4)),latent_dim+int(np.round(diff*1/4)))
            self.fc3 = nn.Linear(latent_dim+int(np.round(diff*1/4)), latent_dim)
        if (num_odelayers == 5):
            self.fc1 = nn.Linear(latent_dim, latent_dim+int(np.round(diff*1/5)))
            self.fc2 = nn.Linear(latent_dim+int(np.round(diff*1/5)), latent_dim+int(np.round(diff*2/5)))
            self.fc2b = nn.Linear(latent_dim+int(np.round(diff*2/5)),latent_dim+int(np.round(diff*3/5)))
            self.fc2c = nn.Linear(latent_dim+int(np.round(diff*3/5)),latent_dim+int(np.round(diff*4/5)))
            self.fc2d = nn.Linear(latent_dim+int(np.round(diff*4/5)),nhidden_number)
            self.fc2e = nn.Linear(nhidden_number,nhidden_number)
            self.fc2f = nn.Linear(nhidden_number,latent_dim+int(np.round(diff*4/5)))
            self.fc2g = nn.Linear(latent_dim+int(np.round(diff*4/5)),latent_dim+int(np.round(diff*3/5)))
            self.fc2h = nn.Linear(latent_dim+int(np.round(diff*3/5)),latent_dim+int(np.round(diff*2/5)))
            self.fc2i = nn.Linear(latent_dim+int(np.round(diff*2/5)),latent_dim+int(np.round(diff*1/5)))
            self.fc3 = nn.Linear(latent_dim+int(np.round(diff*1/5)), latent_dim)
        if (num_odelayers == 6):
            self.fc1 = nn.Linear(latent_dim, latent_dim+int(np.round(diff*1/6)))
            self.fc2 = nn.Linear(latent_dim+int(np.round(diff*1/6)), latent_dim+int(np.round(diff*2/6)))
            self.fc2b = nn.Linear(latent_dim+int(np.round(diff*2/6)),latent_dim+int(np.round(diff*3/6)))
            self.fc2c = nn.Linear(latent_dim+int(np.round(diff*3/6)),latent_dim+int(np.round(diff*4/6)))
            self.fc2d = nn.Linear(latent_dim+int(np.round(diff*4/6)),latent_dim+int(np.round(diff*5/6)))
            self.fc2e = nn.Linear(latent_dim+int(np.round(diff*5/6)),nhidden_number)
            self.fc2f = nn.Linear(nhidden_number,nhidden_number)
            self.fc2g = nn.Linear(nhidden_number,latent_dim+int(np.round(diff*5/6)))
            self.fc2h = nn.Linear(latent_dim+int(np.round(diff*5/6)),latent_dim+int(np.round(diff*4/6)))
            self.fc2i = nn.Linear(latent_dim+int(np.round(diff*4/6)),latent_dim+int(np.round(diff*3/6)))
            self.fc2j = nn.Linear(latent_dim+int(np.round(diff*3/6)),latent_dim+int(np.round(diff*2/6)))
            self.fc2k = nn.Linear(latent_dim+int(np.round(diff*2/6)),latent_dim+int(np.round(diff*1/6)))
            self.fc3 = nn.Linear(latent_dim+int(np.round(diff*1/6)), latent_dim)
        if (num_odelayers == 7):
            self.fc1 = nn.Linear(latent_dim, latent_dim+int(np.round(diff*1/7)))
            self.fc2 = nn.Linear(latent_dim+int(np.round(diff*1/7)), latent_dim+int(np.round(diff*2/7)))
            self.fc2b = nn.Linear(latent_dim+int(np.round(diff*2/7)),latent_dim+int(np.round(diff*3/7)))
            self.fc2c = nn.Linear(latent_dim+int(np.round(diff*3/7)),latent_dim+int(np.round(diff*4/7)))
            self.fc2d = nn.Linear(latent_dim+int(np.round(diff*4/7)),latent_dim+int(np.round(diff*5/7)))
            self.fc2e = nn.Linear(latent_dim+int(np.round(diff*5/7)),latent_dim+int(np.round(diff*6/7)))
            self.fc2f = nn.Linear(latent_dim+int(np.round(diff*6/7)),nhidden_number)
            self.fc2g = nn.Linear(nhidden_number,nhidden_number)
            self.fc2h = nn.Linear(nhidden_number,latent_dim+int(np.round(diff*6/7)))
            self.fc2i = nn.Linear(latent_dim+int(np.round(diff*6/7)),latent_dim+int(np.round(diff*5/7)))
            self.fc2j = nn.Linear(latent_dim+int(np.round(diff*5/7)),latent_dim+int(np.round(diff*4/7)))
            self.fc2k = nn.Linear(latent_dim+int(np.round(diff*4/7)),latent_dim+int(np.round(diff*3/7)))
            self.fc2l = nn.Linear(latent_dim+int(np.round(diff*3/7)),latent_dim+int(np.round(diff*2/7)))
            self.fc2m = nn.Linear(latent_dim+int(np.round(diff*2/7)),latent_dim+int(np.round(diff*1/7)))
            self.fc3 = nn.Linear(latent_dim+int(np.round(diff*1/7)), latent_dim)
        if (num_odelayers == 8):
            self.fc1 = nn.Linear(latent_dim, latent_dim+int(np.round(diff*1/8)))
            self.fc2 = nn.Linear(latent_dim+int(np.round(diff*1/8)), latent_dim+int(np.round(diff*2/8)))
            self.fc2b = nn.Linear(latent_dim+int(np.round(diff*2/8)),latent_dim+int(np.round(diff*3/8)))
            self.fc2c = nn.Linear(latent_dim+int(np.round(diff*3/8)),latent_dim+int(np.round(diff*4/8)))
            self.fc2d = nn.Linear(latent_dim+int(np.round(diff*4/8)),latent_dim+int(np.round(diff*5/8)))
            self.fc2e = nn.Linear(latent_dim+int(np.round(diff*5/8)),latent_dim+int(np.round(diff*6/8)))
            self.fc2f = nn.Linear(latent_dim+int(np.round(diff*6/8)),latent_dim+int(np.round(diff*7/8)))
            self.fc2g = nn.Linear(latent_dim+int(np.round(diff*7/8)),nhidden_number)
            self.fc2h = nn.Linear(nhidden_number,nhidden_number)
            self.fc2i = nn.Linear(nhidden_number,latent_dim+int(np.round(diff*7/8)))
            self.fc2j = nn.Linear(latent_dim+int(np.round(diff*7/8)),latent_dim+int(np.round(diff*6/8)))
            self.fc2k = nn.Linear(latent_dim+int(np.round(diff*6/8)),latent_dim+int(np.round(diff*5/8)))
            self.fc2l = nn.Linear(latent_dim+int(np.round(diff*5/8)),latent_dim+int(np.round(diff*4/8)))
            self.fc2m = nn.Linear(latent_dim+int(np.round(diff*4/8)),latent_dim+int(np.round(diff*3/8)))
            self.fc2n = nn.Linear(latent_dim+int(np.round(diff*3/8)),latent_dim+int(np.round(diff*2/8)))
            self.fc2o = nn.Linear(latent_dim+int(np.round(diff*2/8)),latent_dim+int(np.round(diff*1/8)))
            self.fc3 = nn.Linear(latent_dim+int(np.round(diff*1/8)), latent_dim)
       
        self.nfe = 0
        self.num_odelayers = num_odelayers

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        out = self.act(out)
        if (self.num_odelayers == 2):
            out = self.fc2b(out)
            out = self.act(out)
            out = self.fc2c(out)
            out = self.act(out)
        if (self.num_odelayers == 3):
            out = self.fc2b(out)
            out = self.act(out)
            out = self.fc2c(out)
            out = self.act(out)
            out = self.fc2d(out)
            out = self.act(out)
            out = self.fc2e(out)
            out = self.act(out)
        if (self.num_odelayers == 4):
            out = self.fc2b(out)
            out = self.act(out)
            out = self.fc2c(out)
            out = self.act(out)
            out = self.fc2d(out)
            out = self.act(out)
            out = self.fc2e(out)
            out = self.act(out)
            out = self.fc2f(out)
            out = self.act(out)
            out = self.fc2g(out)
            out = self.act(out)
        if (self.num_odelayers == 5):
            out = self.fc2b(out)
            out = self.act(out)
            out = self.fc2c(out)
            out = self.act(out)
            out = self.fc2d(out)
            out = self.act(out)
            out = self.fc2e(out)
            out = self.act(out)
            out = self.fc2f(out)
            out = self.act(out)
            out = self.fc2g(out)
            out = self.act(out)
            out = self.fc2h(out)
            out = self.act(out)
            out = self.fc2i(out)
            out = self.act(out)
        if (self.num_odelayers == 6):
            out = self.fc2b(out)
            out = self.act(out)
            out = self.fc2c(out)
            out = self.act(out)
            out = self.fc2d(out)
            out = self.act(out)
            out = self.fc2e(out)
            out = self.act(out)
            out = self.fc2f(out)
            out = self.act(out)
            out = self.fc2g(out)
            out = self.act(out)
            out = self.fc2h(out)
            out = self.act(out)
            out = self.fc2i(out)
            out = self.act(out)
            out = self.fc2j(out)
            out = self.act(out)
            out = self.fc2k(out)
            out = self.act(out)
        if (self.num_odelayers == 7):
            out = self.fc2b(out)
            out = self.act(out)
            out = self.fc2c(out)
            out = self.act(out)
            out = self.fc2d(out)
            out = self.act(out)
            out = self.fc2e(out)
            out = self.act(out)
            out = self.fc2f(out)
            out = self.act(out)
            out = self.fc2g(out)
            out = self.act(out)
            out = self.fc2h(out)
            out = self.act(out)
            out = self.fc2i(out)
            out = self.act(out)
            out = self.fc2j(out)
            out = self.act(out)
            out = self.fc2k(out)
            out = self.act(out)
            out = self.fc2l(out)
            out = self.act(out)
            out = self.fc2m(out)
            out = self.act(out)
        if (self.num_odelayers == 8):
            out = self.fc2b(out)
            out = self.act(out)
            out = self.fc2c(out)
            out = self.act(out)
            out = self.fc2d(out)
            out = self.act(out)
            out = self.fc2e(out)
            out = self.act(out)
            out = self.fc2f(out)
            out = self.act(out)
            out = self.fc2g(out)
            out = self.act(out)
            out = self.fc2h(out)
            out = self.act(out)
            out = self.fc2i(out)
            out = self.act(out)
            out = self.fc2j(out)
            out = self.act(out)
            out = self.fc2k(out)
            out = self.act(out)
            out = self.fc2l(out)
            out = self.act(out)
            out = self.fc2m(out)
            out = self.act(out)
            out = self.fc2n(out)
            out = self.act(out)
            out = self.fc2o(out)
            out = self.act(out)
        out = self.fc3(out)
        return out

class RecognitionRNN(nn.Module): 
    # No dropout, because last layer is Dropout layer
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1, activation_rnn='relu'):
        super(RecognitionRNN, self).__init__()
        if nhidden == 0:
            nhidden=1
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim)
        if (activation_rnn == 'none'):
            self.act = nn.Identity()
        elif(activation_rnn == 'tanh'):
            self.act = nn.Tanh()
        elif(activation_rnn == 'relu'):
            self.act = nn.ReLU()

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = self.act(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)

class Decoder(nn.Module): 

    def __init__(self, latent_dim=4, obs_dim=2, nhidden_number=20, dropout = 0, activation_dec='relu'):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, nhidden_number)
        self.fc2 = nn.Linear(nhidden_number, obs_dim)
        self.Dropout = nn.Dropout(dropout)
        self.activation_dec=activation_dec
        if (activation_dec == 'none'):
            self.act = nn.Identity()
        elif(activation_dec == 'tanh'):
            self.act = nn.Tanh()
        elif(activation_dec == 'relu'):
            self.act = nn.ReLU()
  
    def forward(self, z, test=False):
        out = self.fc1(z)
        if(test==True):
            if(self.activation_dec=='relu'):
                #out = self.Dropout(out) #It's convenient to use Dropout after activation, but in case of Relu before activation
                out = self.act(out)
            else:
                out = self.act(out)
                #out = self.Dropout(out)
        if(test==False):
            if(self.activation_dec=='relu'):
                out = self.Dropout(out) #It's convenient to use Dropout after activation, but in case of Relu before activation
                out = self.act(out)
            else:
                out = self.act(out)
                out = self.Dropout(out)
                
        out = self.fc2(out)
        return out

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

class Vaderlayer(nn.Module):
    # Implementation is a bit different than the one in the Github Repo
    # in TF, in repo A is a trained Variable, here A is the weights 
    # of this module
    def __init__(self, A_init):
        super(Vaderlayer, self).__init__()
        self.weights = nn.Parameter(torch.tensor(A_init))
    
    def forward(self, X, W):
        out = (1-W) * self.weights + X*W
        return out
    
class LSTM2outlayer(nn.Module):
    #Like in the RNN, the output of the LSTM has to be mapped back to the latent_dim
    def __init__(self, rnn_nhidden,target_dim):
        super(LSTM2outlayer, self).__init__()
        self.lin=nn.Linear(rnn_nhidden, target_dim)
        self.target_dim=target_dim
        
    def forward(self,h):
        out = self.lin(h)
        return out

def read_data(data, types_dict, missing_positions):

    #Sustitute NaN values by something (we assume we have the real missing value mask)

    true_miss_mask = np.ones([np.shape(data)[0],len(types_dict)])
    if missing_positions is not None:
        true_miss_mask[missing_positions[:,0]-1,missing_positions[:,1]-1] = 0 #Indexes in the csv start at 1
    data_masked = np.ma.masked_where(np.isnan(data.cpu()),data.cpu()) 
    #We need to fill the data depending on the given data...
    data_filler = []
    for i in range(len(types_dict)):
        if types_dict[i][0] == 'cat' or types_dict[i][0] == 'ordinal':
            aux = np.unique(data[:,i].cpu())
            if not np.isnan(aux[0]):
                data_filler.append(aux[0])  #Fill with the first element of the cat (0, 1, or whatever)
            else:
                data_filler.append(int(0))
        else:
            data_filler.append(0.0)
        
    data = data_masked.filled(data_filler)

    #Construct the data matrices
    data_complete = []
    for i in range(np.shape(data)[1]):
        
        if types_dict[i][0] == 'cat':
            #Get categories
            cat_data = [int(x) for x in data[:,i]]
            categories, indexes = np.unique(cat_data,return_inverse=True)
            #Transform categories to a vector of 0:n_categories
            new_categories = np.arange(int(types_dict[i][2]))
            cat_data = new_categories[indexes]
            #Create one hot encoding for the categories
            aux = np.zeros([np.shape(data)[0],len(new_categories)])
            aux[np.arange(np.shape(data)[0]),cat_data] = 1
            data_complete.append(aux)
            
        elif types_dict[i][0] == 'ordinal':
            #Get categories
            cat_data = [int(x) for x in data[:,i]]
            categories, indexes = np.unique(cat_data,return_inverse=True)
            #Transform categories to a vector of 0:n_categories
            new_categories = np.arange(int(types_dict[i][2]))
            cat_data = new_categories[indexes]
            #Create thermometer encoding for the categories
            aux = np.zeros([np.shape(data)[0],1+len(new_categories)])
            aux[:,0] = 1
            aux[np.arange(np.shape(data)[0]),1+cat_data] = -1
            aux = np.cumsum(aux,1)
            data_complete.append(aux[:,:-1])
            
        else:
            data_complete.append(np.transpose([data[:,i]]))
                    
    data = np.concatenate(data_complete,1)
    
        
    #Read Missing mask from csv (contains positions of missing values)
    n_samples = np.shape(data)[0]
    n_variables = len(types_dict)
    miss_mask = np.ones([np.shape(data)[0],n_variables])
    
    return data, types_dict, miss_mask, true_miss_mask, n_samples

class Statlayer(nn.Module):
    #Implementation of the Layer to represent the static baselinedata or the SNP Data
    def __init__(self,static,staticdata_onehot,s_dim,z_dim):
        super(Statlayer, self).__init__()

        #Computing the shape of the joint layer
        sha = staticdata_onehot.shape[1]
        y_dim = static.shape[1]
        
        self.log_pi_layer = nn.Linear(sha,s_dim)
        self.mean_layer = nn.Linear(sha+s_dim,z_dim)
        self.log_layer = nn.Linear(sha+s_dim,z_dim)
        self.y_layer = nn.Linear(z_dim,y_dim)
        
    def forward(self, staticdata_onehot, tau=1):
        #implement linear layer before hidden layer

        #Using GMM Prior after running modules through NN
        
        log_pi = self.log_pi_layer(staticdata_onehot)
        samples_s = torch.nn.functional.gumbel_softmax(log_pi,tau,hard=False)
        mean = self.mean_layer(torch.cat([staticdata_onehot,samples_s],dim=1))
        log = self.log_layer(torch.cat([staticdata_onehot,samples_s],dim=1))
        
        return samples_s, log_pi, mean, log

    def y_forward(self, b0):
        out = self.y_layer(b0)
        
        return out

class Statdecode(nn.Module):
    def __init__(self,static,static_types,s_dim,z_dim,device):
        super(Statdecode, self).__init__()
        
        self.mean_pz_layer = nn.Linear(s_dim,z_dim)
        
        self.linears = nn.ModuleList()
        self.z_dim=z_dim
        self.device=device
        
        sha = static.shape[1]
        
        for i in range(sha):
            if (static_types[i,0]=='real'):
                #first output dim is mean, second is logvar
                self.linears.append(nn.Linear(sha,static_types[i,1]*2))
            elif (static_types[i,0]=='cat'):
                self.linears.append(nn.Linear(sha,static_types[i,1]-1))


    def forward(self,y_latent,samples_s,staticdata_onehot,static_types,true_miss_mask, indices, tau=1,batch_norm=False,batchmean=None,batchvar=None,generation_prior=False,output_prior=False):
        
        #Computing the parameters of the p(z|s) distribution
        
        if samples_s is not None:
        
            mean_pz = self.mean_pz_layer(samples_s)
            log_var_pz = torch.zeros(size=[samples_s.shape[0],self.z_dim])
        
        #By generating from the prior you have to use the parameters of the prior distribution
        if(generation_prior==True):
            return mean_pz, log_var_pz
        
        if(output_prior==True):
            out = torch.zeros(size=y_latent.shape)
            
            for i in (range(y_latent.shape[1])):
                if static_types[i,0]=='real':
                    params = self.linears[i](y_latent)
                    mean = params[:,0]
                    logvar = params[:,1]
                    out[:,i] = torch.normal(mean=mean,std=torch.exp(0.5*logvar)) 
                else:
                    params = torch.cat([torch.zeros(y_latent.shape[0],1),self.linears[i](y_latent)],dim=1)
                    helpf = torch.nn.functional.gumbel_softmax(params,tau,hard=False)
    
                    out[:,i] = torch.from_numpy(np.argmax(helpf.detach().cpu().numpy(),1)).float()
            return out
                    
            
        else:
            #computing the output data
            
            out = torch.zeros(size=y_latent.shape)
            log_p_x = torch.zeros(size=y_latent.shape)
            
            batchind = 0
            onehotind = 0
            
            for i in(range(y_latent.shape[1])):
                if static_types[i,0]=='real':
                    params = self.linears[i](y_latent)
                    
                    #Compute loglik of the independent yd variables
                    
                    data = staticdata_onehot[indices,onehotind].clone()
                    
                    data[torch.isnan(data)]=0
                    
                    mean = params[:,0]
                    
                    logvar = params[:,1]
                    
                    if (batch_norm==True):
                        mean = np.sqrt(batchvar[batchind])*mean + batchmean[batchind]
                        #because of the logvar implementation of the variance, batchvar*var = exp(log(batchvar) + logvar)
                        logvar = batchvar[batchind] + logvar
                    
                    val = log_normal_pdf(data,mean,logvar).float().to(self.device)
                    miss = torch.from_numpy(true_miss_mask[indices,i]).float().to(self.device)
                    log_p_x[:,i] = val * miss
                    
                    out[:,i] = torch.normal(mean=mean,std=torch.exp(0.5*logvar))              
                    
                    batchind = batchind+1
                    
                    onehotind = onehotind + 1
                    
                elif static_types[i,0]=='cat':
                    
                    params = torch.cat([torch.zeros(y_latent.shape[0],1).float().to(self.device),self.linears[i](y_latent)],dim=1)
                    
                    eps = 1e-20 # to avoid log of 0
    
                    # Reconstruction Loss
                    
                    #compute with softmax sampling propability
                    x_prob = torch.nn.functional.softmax(input=params,dim=1)
                    
                    data = staticdata_onehot[:,onehotind:onehotind+static_types[i,1]].clone()
    
                    #value is not relevant
                    #data[torch.isnan(data)]=0
                    
                    #data has to be one hot encoded for the log_p_x
                    #data_onehot = pd.get_dummies(data).values
                    
                    #print(data.dtype)
                    #print(x_prob.dtype)
                    
                    data = data[indices,:].float()
                    
                    #data + torch.log(x_prob+eps)
                    
                    
                    log = torch.sum(data * torch.log(x_prob + eps), dim=1).float().to(self.device)
                    
                    miss = torch.from_numpy(true_miss_mask[indices,i]).float().to(self.device)
                    
                    log_p_x[:,i] = log*miss
                    
                    helpf = torch.nn.functional.gumbel_softmax(params,tau,hard=False)
    
                    out[:,i] = torch.from_numpy(np.argmax(helpf.detach().cpu().numpy(),1)).float()
                    
                    onehotind = onehotind+static_types[i,1]
                    
            return out, mean_pz, log_var_pz, log_p_x
    
def _initialize_imputation(X, W):
    # initializing of the A-variable for Vader, in essence same like in Vader repo
        # average per time point, variable
    if (type(X) == torch.Tensor):
        X = X.cpu().numpy()
    if (type(W) == torch.Tensor):
        W = W.cpu().numpy()
    W_A = np.sum(W, axis=0)
    A = np.sum(X * W, axis=0)
    A[W_A>0] = A[W_A>0] / W_A[W_A>0]
    # if not available, then average across entire variable
    for i in np.arange(A.shape[0]):
        for j in np.arange(A.shape[1]):
            if W_A[i,j] == 0:
                A[i,j] = np.sum(X[:,:,j]) / np.sum(W[:,:,j])
                W_A[i,j] = 1
    # if not available, then average across all variables
    A[W_A==0] = np.mean(X[W==1])
    return A.astype(np.float32)

def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))

def weighted_log_normal_pdf(x, mean, logvar, Wmul):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    #Has to be weighted by W
    return Wmul * (-.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar)))

def rec_loss2(x, mean, Wmul):
    return Wmul * (x-mean)**2

def rec_lossstat(x, mean):
    return (x-mean)**2

def normal_kl(mu1, lv1, mu2, lv2): #I am not sure, whether this implementation is correct, but it's the one from chen etal.
    #Multivariate normal_kl is sum of univariate normal KL, if Varianz is Ip
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl  

def batch_norm(staticdata_onehot, staticonehot_types, missing_onehot):
    #Batch Normalization for the Onehot encoded staticdata
    staticdata_norm = staticdata_onehot.copy()
    batchmean = []
    batchvar = []
    for i in range(staticdata_onehot.shape[1]):
        if (staticonehot_types[i,0]=='real'):
            newvec = []
            for j in range(missing_onehot.shape[0]):
                if missing_onehot[j,i]==1:
                    newvec.append(staticdata_onehot[j,i])
            newvec = np.array(newvec)
            mean=np.mean(newvec)
            var= np.var(newvec)
            staticdata_norm[:,i] = (staticdata_norm[:,i]-mean)/np.sqrt(var)
            batchmean.append(mean)
            batchvar.append(var)
    
    batchmean=np.array(batchmean)
    batchvar = np.array(batchvar)
    
    return staticdata_norm, batchmean, batchvar

def training(solver, nepochs, lr, train_dir, device, samp_trajs, samp_ts, latent_dim, nhidden, obs_dim, batchsize, activation_ode = 'relu',num_odelayers = 1, W_train = None, enctype = 'RNN',dectype = 'RNN', rnn_nhidden_enc=0.5,activation_rnn = 'relu',rnn_nhidden_dec=0.5,activation_dec = 'relu',dropout_dec = 0,static=None,staticdata_onehot=None,staticdata_types_dict=None,static_true_miss_mask=None,s_dim_static=0,z_dim_static=0,scaling_ELBO=1,batch_norm_static=False):
    
    #solver: char: 'Adjoint' or link to torch_ACA. Variable describes which solver is used
    #nepochs: int, number of epochs during training
    #lr: float, learning rate of the optimizer
    #train_dir: direction of saved training values
    #device: Using CPU or GPU
    #samp_trajs: Sampled trajectories / training data
    #samp_ts: Timestamps to samp_trajs
    #latent_dim: latent_dim of the ODE function, as percent of the obs_dim
    #nhidden: number of hidden layers of the ODE function in percent of absolute latent_dim or layer before(greater than 1 possible)
    #obs_dim: dimension of the output, number of variables
    #batchsize: in percent
    #activation_ode: char, values 'none', 'tanh', 'relu', specifies activation function of the ODE
    #num_odelayers: int, 1,2 or 3, specifies the number of the hidden ODE layers, default 1, number of units specified by nhidden, nhidden is here devided by number of ode_layers
    #W_train: Specifies the missing values of VADER
    #enctype: char: 'RNN' or 'LSTM 
    #dectype: char: 'RNN','LSTM',or 'FF', specifies the decoder type used
    #rnn_nhidden_enc: specifies, how many units has h of the RNN, fraction (in percent) of number of timesteps * number of features
    #activation_rnn: Optional: char, values,'none', 'relu', 'tanh', spcifies activation of the modules, if RNN is used
    #rnn_nhidden_dec: specifies, how many units has h of the RNN, fraction (in percent) of number of timesteps * number of features
    #activation_dec: Optional: char, values,'none', 'relu', 'tanh', spcifies activation of the modules, if RNN is used
    #dropout_dec: Optional: propability of dropout of decoder, if not LSTM dropout = 0, using no dropout
    #static: Optional: Staticdata, None means there is no static data
    #staticdata_onehot: Optional, OneHotencoded static data (if there are categorical data)
    #staticdata_types_dict: Optional, Dictionary with the types of the data
    #static_true_miss_mask: Optional, Mask, which has zero, where data is missing and one where data is observed
    #s_dim_static: Optional, Dimension of the sn of the Gaussian Mixture of the static data
    #z_dim_static: Optional, Dimension of the zn of the Gaussian Mixture of the static data
    #scaling_ELBO: Optional: Parameter to scale the two ELBOs of the static and the longitudinal data
    #batch_norm_static: Optional: specifies, whether batch normalization is used on static data or not    
    
    if W_train is not None:
        W = W_train
    else:
        W = np.ones(samp_trajs.shape, dtype=np.float32)
    W = torch.from_numpy(W).float().to(device)

    if solver == 'Adjoint':
        from torchdiffeq import odeint_adjoint as odeint
    elif solver == 'torchdiffeqpack':
        from TorchDiffEqPack.odesolver import odesolve as odeint
        
        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': 0.01})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-3})
        options.update({'t0': samp_ts.tolist()[0]})
        options.update({'t1': samp_ts.tolist()[-1]})
        options.update({'t_eval':samp_ts.tolist()})
    elif solver:
        import sys
        sys.path.insert(1, solver)
        from torch_ACA.odesolver import odesolve as odeint
        
        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': 0.01})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-3})
        options.update({'t0': samp_ts.tolist()[0]})
        options.update({'t1': samp_ts.tolist()[-1]})
        options.update({'t_eval':samp_ts.tolist()})
    else:
        from torchdiffeq import odeint
    
    #Initializer of Vader
    #Not sure, whether it's a good idea to use dropout before Vader
    A_init = _initialize_imputation(samp_trajs, W)
    vad = Vaderlayer(A_init).to(device)
    
    #making types list for onehot_encoding, necessary for Batch Normalization
    staticonehot_types=[]
    
    if staticdata_types_dict is not None:
        for i in range(staticdata_types_dict.shape[0]):
            for j in range(staticdata_types_dict[i,1]):
                staticonehot_types.append(staticdata_types_dict[i])
        staticonehot_types=np.array(staticonehot_types)
    
    missing_onehot = []

    if static_true_miss_mask is not None:
        for i in range(static_true_miss_mask.shape[1]):
            for j in range(staticdata_types_dict[i,1]):
                missing_onehot.append(static_true_miss_mask[:,i])
        missing_onehot=np.transpose(np.array(missing_onehot))
    
    #Computing the latent_dim
    latent_dim_max = obs_dim
    latent_dim_number = int(np.round(latent_dim*latent_dim_max)) 
    
    nhidden_number = int(np.round(nhidden*latent_dim_number))
    
    batchsize_number = int(np.round(batchsize*samp_trajs.shape[0]))
    
    #if statement, whether baselinedata is used or not + augmentation of the ODE
    if static is not None:
        modulefunc = Statlayer(static,staticdata_onehot,s_dim_static,z_dim_static).to(device)
        moduledec = Statdecode(static,staticdata_types_dict,s_dim_static,z_dim_static,device).to(device)
        func = LatentODEfunc(latent_dim_number+z_dim_static, nhidden_number, activation_ode = activation_ode, num_odelayers = num_odelayers).to(device)
    else:
        func = LatentODEfunc(latent_dim_number, nhidden_number, activation_ode = activation_ode, num_odelayers = num_odelayers).to(device)
    
    rnn_nhidden_max = obs_dim*len(samp_ts)
    rnn_nhidden_number = int(np.round(rnn_nhidden_enc*rnn_nhidden_max))
    
    params = list()
    if (enctype == 'LSTM'):
        rec = nn.LSTM(input_size = obs_dim, hidden_size = rnn_nhidden_number, num_layers = 1, bias = True, batch_first = True, dropout = 0, bidirectional = False).to(device)
        enc2out = LSTM2outlayer(rnn_nhidden_number,2*latent_dim_number).to(device)
        params = params + (list(enc2out.parameters()))
    else:
        rec = RecognitionRNN(latent_dim_number*2, obs_dim, rnn_nhidden_number, batchsize_number, activation_rnn).to(device)
    
    rnn_nhidden_dec_max = (obs_dim)*len(samp_ts)
    rnn_nhidden_dec_number = int(np.round(rnn_nhidden_dec*rnn_nhidden_dec_max))
    if (dectype == 'LSTM'):
        dec = nn.LSTM(input_size = latent_dim_number+z_dim_static, hidden_size = rnn_nhidden_dec_number, num_layers = 1, bias = True, batch_first = True, dropout = 0, bidirectional = False).to(device)
        dec2out = LSTM2outlayer(rnn_nhidden_dec_number,obs_dim).to(device)
        params =params + (list(dec2out.parameters()))
    elif(dectype == 'RNN'):
        dec = RecognitionRNN(obs_dim,latent_dim_number+z_dim_static,rnn_nhidden_dec_number, batchsize_number, activation_dec).to(device)
    else:
        dec = Decoder(latent_dim_number+z_dim_static, obs_dim, rnn_nhidden_dec_number, dropout_dec, activation_dec).to(device)
    
    if static is not None:
        params = params + (list(modulefunc.parameters()) + list(moduledec.parameters()) + list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()) + list(vad.parameters()))
    else:
        params = params + (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()) + list(vad.parameters()))
    
    optimizer = optim.Adam(params, lr=lr)
    loss_meter = RunningAverageMeter()
    
    #loading of training data
    if train_dir is not None:
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        ckpt_path = os.path.join(train_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path,map_location=torch.device('cpu'))
            func.load_state_dict(checkpoint['func_state_dict'])
            rec.load_state_dict(checkpoint['rec_state_dict'])
            dec.load_state_dict(checkpoint['dec_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            samp_trajs = checkpoint['samp_trajs']
            samp_ts = checkpoint['samp_ts']
            vad.load_state_dict(checkpoint['vad_state_dict'])
            if static is not None:
                modulefunc.load_state_dict(checkpoint['modulefunc_state_dict'])
                moduledec.load_state_dict(checkpoint['moduledec_state_dict'])
            if (enctype == 'LSTM'):
                enc2out.load_state_dict(checkpoint['enc2out_state_dict'])
            if (dectype == 'LSTM'):
                dec2out.load_state_dict(checkpoint['dec2out_state_dict'])
            print('Loaded ckpt from {}'.format(ckpt_path))

    try: #training part
        for itr in range(1, nepochs + 1):
            #randomly permute the indices of the trajs
            permutation = torch.randperm(samp_trajs.shape[0])
            
            tau = np.max([1.0 - (0.999/(nepochs-50))*itr,1e-3])
            
            
            #batchsize
            for i in range(0,samp_trajs.shape[0], batchsize_number):
                #if statement implemented for indices of batches
                #else is the case, when last batch isn't a "complete" one
                #then you have to complete the batch, f.e. with the first permutations
                if i + batchsize_number <= samp_trajs.shape[0]:
                    indices = permutation[i:i+batchsize_number]
                    
                else:
                    indices = permutation[i:samp_trajs.shape[0]]
                    indices = torch.cat((indices,permutation[0:i+batchsize_number-samp_trajs.size(0)]),0)
                
                optimizer.zero_grad()
                # backward in time to infer q(z_0)
                # Treat W as an indicator for nonmissingness (1: nonmissing; 0: missing)
                X = samp_trajs[indices,:,:]
                Wmul = W[indices,:,:]
                if ~np.all(W.cpu().numpy() == 1.0) and np.all(np.logical_or(W.cpu().numpy() == 0.0, W.cpu().numpy() == 1.0)):
                    #Wmul = torch.from_numpy(Wmul).float().to(device)
                    XW = vad.forward(X,Wmul)
                else:
                    XW = X
                
                if (enctype == 'LSTM'):
                    h_0 = torch.zeros(1,batchsize_number,rnn_nhidden_number).to(device)
                    c_0 = torch.zeros(1,batchsize_number,rnn_nhidden_number).to(device)
                    for t in reversed(range(XW.size(1))):
                        obs = XW[:, t:t+1, :] #t:t+1, because rec needs 3 dimensions
                        out, (h_0,c_0) = rec(obs,(h_0,c_0))
                        # out and h_0 are the same, because just one point is going through LSTM
                    out = out[:,0,:]
                    out = enc2out(out)
                else:
                    h = rec.initHidden().to(device)
                    for t in reversed(range(XW.size(1))):
                        obs = XW[:, t, :]
                        out, h = rec.forward(obs, h)
                qz0_mean, qz0_logvar = out[:, :latent_dim_number], out[:, latent_dim_number:]
                epsilon = torch.randn(qz0_mean.size()).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
                
                if static is not None: #b0 is the hidden dimension baseline data
                    
                    if (batch_norm_static == True):
                        staticdata_norm, batchmean, batchvar = batch_norm(staticdata_onehot[indices,:],staticonehot_types, missing_onehot[indices,:])
                        inp = torch.from_numpy(staticdata_norm).float().to(device)                    
                    
                    else:
                        inp = torch.from_numpy(staticdata_onehot[indices,:]).float().to(device)                    
                        batchmean = None
                        batchvar = None
                        
                    samples_s,log_pi, qz0_meanstat, qz0_logvarstat = modulefunc.forward(inp,tau)
                    
                    qz0_meanstat=qz0_meanstat.to(device)
                    qz0_logvarstat=qz0_logvarstat.to(device)
                    
                    epsilonstat = torch.randn(qz0_meanstat.size()).to(device)
                    b0 = epsilonstat * torch.exp(.5 * qz0_logvarstat) + qz0_meanstat
                    
                    y_latent = modulefunc.y_forward(b0)
                    
                    out, meanpz, logvarpz, log_p_x = moduledec(y_latent, samples_s, torch.from_numpy(staticdata_onehot).to(device),staticdata_types_dict, static_true_miss_mask, indices,tau=tau, batch_norm=batch_norm_static, batchmean=batchmean, batchvar=batchvar)
                    
                    meanpz = meanpz.to(device)
                    logvarpz = logvarpz.to(device)
                    
                    zinit = torch.cat((z0,b0),dim=1) #extending z0
                # forward in time and solve ode for reconstructions
                else:
                    zinit = z0
                
                
                if solver == 'Adjoint':
                    pred_z = odeint(func, zinit, samp_ts).permute(1, 0, 2)
                elif solver:
                    pred_z = odeint(func,zinit, options).permute(1,0,2)
                else:
                    pred_z = odeint(func, zinit, samp_ts).permute(1, 0, 2)
                
                if (dectype == 'LSTM'):
                    pred_x, (h,c) = dec(pred_z)
                    pred_x = dec2out(pred_x)
                elif(dectype == 'RNN'):
                    pred_x = torch.zeros(batchsize_number,len(samp_ts),obs_dim)
                    h = dec.initHidden().to(device)
                    for t in range(pred_z.size(1)):
                        pred_x[:,t,:], h = dec.forward(pred_z[:,t,:], h)
                else:
                    pred_x = dec(pred_z)
                
                pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
                analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                        pz0_mean, pz0_logvar).sum(-1)

                #Implement weighted mean squared error for Vader
                klavg = torch.mean(analytic_kl) #same like above
                # samp_trajs.size(1) is number of timesteps, samp_trajs.size(2) is number of variables
                
                rec_loss = rec_loss2(samp_trajs[indices,:,:],pred_x,Wmul)
                rec_lossavg = torch.sum(rec_loss)*samp_trajs.size(1)*samp_trajs.size(2) / (sum(sum(sum(Wmul))))
                # sum(sum(... is cardinality of W
                
                
                # compute loss
                if static is not None:
                    #log_pi from the encoder above
                    eps=1E-20
                    #KL(q(s|x)|p(s))
                    
                    #logits=log_pi, labels=pi_param
                    #because logits has to be transformed with softmax
                    pi_param = torch.nn.functional.softmax(log_pi,dim=1)
                    KL_s = torch.sum(pi_param * torch.log(pi_param + eps), dim=1) + torch.log(torch.tensor(float(s_dim_static)))
                    KL_s = KL_s.to(device)
                    
                    #meanpz, logvarpz, qz0_meanstat, qz0_logvarstat
                    #These two implementations of the multivariate KL divergence are equivalent, first one ist from torchdiffeq, second one from HI-VAE
                    analytic_kl_stat = normal_kl(qz0_meanstat, qz0_logvarstat,meanpz, logvarpz).sum(-1)
                    #KL_z_stat = -0.5*z_dim + 0.5*(torch.exp(qz0_logvarstat-logvarpz)+((meanpz - qz0_meanstat)**2.)/torch.exp(logvarpz) -qz0_logvarstat+logvarpz).sum(-1)                
                    
                    #Eq[log_p(x|y)]
                    loss_reconstruction_stat = log_p_x.sum(-1).to(device)
    
                    ELBO_stat = -torch.mean(loss_reconstruction_stat - analytic_kl_stat - KL_s,0)
    
                    #print(ELBO_stat)
                    #print(klavg + rec_lossavg)
                    #print(-torch.mean(loss_reconstruction_stat))
                    
                    long = (klavg + rec_lossavg)/(klavg + rec_lossavg + ELBO_stat)
                    
                    stat = (ELBO_stat)/(klavg + rec_lossavg + ELBO_stat)
                    
                    long_scaled=stat/(long+stat)*(klavg + rec_lossavg)
                    
                    stat_scaled=long/(long+stat)*(ELBO_stat)
                    
                    loss = long_scaled + scaling_ELBO *stat_scaled
                
                else:
                    loss=klavg + rec_lossavg
                # the grad_fn solver... can't find how to change it
                #print("hi")
                loss.backward()
                optimizer.step()
                loss_meter.update(loss.item())
                print(loss)
                #print(klavg + rec_lossavg)
                print('Iter: {}, running avg elbo: {:.4f}'.format(itr, -loss_meter.avg))
                
    except KeyboardInterrupt: #store training
        if train_dir is not None:
            ckpt_path = os.path.join(train_dir, 'ckpt.pth')
            if static is not None:
                if ((enctype == 'LSTM') & (dectype != 'LSTM')):
                    torch.save({
                        'func_state_dict': func.state_dict(),
                        'rec_state_dict': rec.state_dict(),
                        'dec_state_dict': dec.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'samp_trajs': samp_trajs,
                        'samp_ts': samp_ts,
                        'vad_state_dict': vad.state_dict(),
                        'modulefunc_state_dict': modulefunc.state_dict(),
                        'moduledec_state_dict': moduledec.state_dict(),
                        'enc2out_state_dict': enc2out.state_dict(),
                    }, ckpt_path)
                    print('Stored ckpt at {}'.format(ckpt_path))
                elif((enctype == 'LSTM') & (dectype == 'LSTM')):
                    torch.save({
                        'func_state_dict': func.state_dict(),
                        'rec_state_dict': rec.state_dict(),
                        'dec_state_dict': dec.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'samp_trajs': samp_trajs,
                        'samp_ts': samp_ts,
                        'vad_state_dict': vad.state_dict(),
                        'modulefunc_state_dict': modulefunc.state_dict(),
                        'moduledec_state_dict': moduledec.state_dict(),
                        'enc2out_state_dict': enc2out.state_dict(),
                        'dec2out_state_dict': dec2out.state_dict(),
                    }, ckpt_path)
                    print('Stored ckpt at {}'.format(ckpt_path))
                elif((enctype == 'RNN') & (dectype == 'LSTM')):
                    torch.save({
                        'func_state_dict': func.state_dict(),
                        'rec_state_dict': rec.state_dict(),
                        'dec_state_dict': dec.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'samp_trajs': samp_trajs,
                        'samp_ts': samp_ts,
                        'vad_state_dict': vad.state_dict(),
                        'modulefunc_state_dict': modulefunc.state_dict(),
                        'moduledec_state_dict': moduledec.state_dict(),
                        'dec2out_state_dict': dec2out.state_dict(),
                    }, ckpt_path)
                    print('Stored ckpt at {}'.format(ckpt_path))
                else:
                    torch.save({
                        'func_state_dict': func.state_dict(),
                        'rec_state_dict': rec.state_dict(),
                        'dec_state_dict': dec.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'samp_trajs': samp_trajs,
                        'samp_ts': samp_ts,
                        'vad_state_dict': vad.state_dict(),
                        'modulefunc_state_dict': modulefunc.state_dict(),
                        'moduledec_state_dict': moduledec.state_dict(),
                    }, ckpt_path)
                    print('Stored ckpt at {}'.format(ckpt_path))
            else:
                if ((enctype == 'LSTM') & (dectype != 'LSTM')):
                    torch.save({
                        'func_state_dict': func.state_dict(),
                        'rec_state_dict': rec.state_dict(),
                        'dec_state_dict': dec.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'samp_trajs': samp_trajs,
                        'samp_ts': samp_ts,
                        'vad_state_dict': vad.state_dict(),
                        'enc2out_state_dict': enc2out.state_dict(),
                    }, ckpt_path)
                    print('Stored ckpt at {}'.format(ckpt_path))
                elif((enctype == 'LSTM') & (dectype == 'LSTM')):
                    torch.save({
                        'func_state_dict': func.state_dict(),
                        'rec_state_dict': rec.state_dict(),
                        'dec_state_dict': dec.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'samp_trajs': samp_trajs,
                        'samp_ts': samp_ts,
                        'vad_state_dict': vad.state_dict(),
                        'enc2out_state_dict': enc2out.state_dict(),
                        'dec2out_state_dict': dec2out.state_dict(),
                    }, ckpt_path)
                    print('Stored ckpt at {}'.format(ckpt_path))
                elif((enctype == 'RNN') & (dec == 'LSTM')):
                    torch.save({
                        'func_state_dict': func.state_dict(),
                        'rec_state_dict': rec.state_dict(),
                        'dec_state_dict': dec.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'samp_trajs': samp_trajs,
                        'samp_ts': samp_ts,
                        'vad_state_dict': vad.state_dict(),
                        'dec2out_state_dict': dec2out.state_dict(),
                    }, ckpt_path)
                    print('Stored ckpt at {}'.format(ckpt_path))
                else:
                    torch.save({
                        'func_state_dict': func.state_dict(),
                        'rec_state_dict': rec.state_dict(),
                        'dec_state_dict': dec.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'samp_trajs': samp_trajs,
                        'samp_ts': samp_ts,
                        'vad_state_dict': vad.state_dict(),
                    }, ckpt_path)
                    print('Stored ckpt at {}'.format(ckpt_path))
    if train_dir is not None:
        ckpt_path = os.path.join(train_dir, 'ckpt.pth')
        if static is not None:
            if ((enctype == 'LSTM') & (dectype != 'LSTM')):
                torch.save({
                    'func_state_dict': func.state_dict(),
                    'rec_state_dict': rec.state_dict(),
                    'dec_state_dict': dec.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'samp_trajs': samp_trajs,
                    'samp_ts': samp_ts,
                    'vad_state_dict': vad.state_dict(),
                    'modulefunc_state_dict': modulefunc.state_dict(),
                    'moduledec_state_dict': moduledec.state_dict(),
                    'enc2out_state_dict': enc2out.state_dict(),
                }, ckpt_path)
                print('Stored ckpt at {}'.format(ckpt_path))
            elif((enctype == 'LSTM') & (dectype == 'LSTM')):
                torch.save({
                    'func_state_dict': func.state_dict(),
                    'rec_state_dict': rec.state_dict(),
                    'dec_state_dict': dec.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'samp_trajs': samp_trajs,
                    'samp_ts': samp_ts,
                    'vad_state_dict': vad.state_dict(),
                    'modulefunc_state_dict': modulefunc.state_dict(),
                    'moduledec_state_dict': moduledec.state_dict(),
                    'enc2out_state_dict': enc2out.state_dict(),
                    'dec2out_state_dict': dec2out.state_dict(),
                }, ckpt_path)
                print('Stored ckpt at {}'.format(ckpt_path))
            elif((enctype == 'RNN') & (dectype == 'LSTM')):
                torch.save({
                    'func_state_dict': func.state_dict(),
                    'rec_state_dict': rec.state_dict(),
                    'dec_state_dict': dec.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'samp_trajs': samp_trajs,
                    'samp_ts': samp_ts,
                    'vad_state_dict': vad.state_dict(),
                    'modulefunc_state_dict': modulefunc.state_dict(),
                    'moduledec_state_dict': moduledec.state_dict(),
                    'dec2out_state_dict': dec2out.state_dict(),
                }, ckpt_path)
                print('Stored ckpt at {}'.format(ckpt_path))
            else:
                torch.save({
                    'func_state_dict': func.state_dict(),
                    'rec_state_dict': rec.state_dict(),
                    'dec_state_dict': dec.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'samp_trajs': samp_trajs,
                    'samp_ts': samp_ts,
                    'vad_state_dict': vad.state_dict(),
                    'modulefunc_state_dict': modulefunc.state_dict(),
                    'moduledec_state_dict': moduledec.state_dict(),
                }, ckpt_path)
                print('Stored ckpt at {}'.format(ckpt_path))
        else:
            if ((enctype == 'LSTM') & (dectype != 'LSTM')):
                torch.save({
                    'func_state_dict': func.state_dict(),
                    'rec_state_dict': rec.state_dict(),
                    'dec_state_dict': dec.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'samp_trajs': samp_trajs,
                    'samp_ts': samp_ts,
                    'vad_state_dict': vad.state_dict(),
                    'enc2out_state_dict': enc2out.state_dict(),
                }, ckpt_path)
                print('Stored ckpt at {}'.format(ckpt_path))
            elif((enctype == 'LSTM') & (dectype == 'LSTM')):
                torch.save({
                    'func_state_dict': func.state_dict(),
                    'rec_state_dict': rec.state_dict(),
                    'dec_state_dict': dec.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'samp_trajs': samp_trajs,
                    'samp_ts': samp_ts,
                    'vad_state_dict': vad.state_dict(),
                    'enc2out_state_dict': enc2out.state_dict(),
                    'dec2out_state_dict': dec2out.state_dict(),
                }, ckpt_path)
                print('Stored ckpt at {}'.format(ckpt_path))
            elif((enctype == 'RNN') & (dectype == 'LSTM')):
                torch.save({
                    'func_state_dict': func.state_dict(),
                    'rec_state_dict': rec.state_dict(),
                    'dec_state_dict': dec.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'samp_trajs': samp_trajs,
                    'samp_ts': samp_ts,
                    'vad_state_dict': vad.state_dict(),
                    'dec2out_state_dict': dec2out.state_dict(),
                }, ckpt_path)
                print('Stored ckpt at {}'.format(ckpt_path))
            else:
                torch.save({
                    'func_state_dict': func.state_dict(),
                    'rec_state_dict': rec.state_dict(),
                    'dec_state_dict': dec.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'samp_trajs': samp_trajs,
                    'samp_ts': samp_ts,
                    'vad_state_dict': vad.state_dict(),
                }, ckpt_path)
                print('Stored ckpt at {}'.format(ckpt_path))

    return loss, rec_lossavg, klavg #first one is reconstruction loss

def reconRP(solver, train_dir, device,samp_trajs,samp_ts, latent_dim, nhidden, obs_dim, activation_ode = 'elu',num_odelayers=1,W_train=None, enctype = 'RNN',dectype = 'RNN', rnn_nhidden_enc=0.5,activation_rnn = 'relu',rnn_nhidden_dec=0.5,activation_dec = 'relu',static=None,staticdata_onehot=None,staticdata_types_dict=None,static_true_miss_mask=None,s_dim_static=0,z_dim_static=0,batch_norm_static=False, timemax = 1, negtime = False, timemin = None,num=2000, save_latent=False):

    #timemax: Maximum of the simulated time
    #negtime is boolean, whether time is backwards predicted or not
    #timemin: just necessary, if negtime = True, minimum time of simulations
    if solver == 'Adjoint':
        from torchdiffeq import odeint_adjoint as odeint
    elif solver:
        import sys
        sys.path.insert(1, solver)
        import torch_ACA
        from torch_ACA.odesolver import odesolve as odeint
        
        ts_pos = np.linspace(0., timemax, num=num)
        ts_pos = torch.from_numpy(ts_pos).float().to(device)
        
        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': 0.01})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-3})
        options.update({'t0': ts_pos.tolist()[0] })
        options.update({'t1': ts_pos.tolist()[-1] })
        options.update({'t_eval':ts_pos.tolist()})
    else:
        from torchdiffeq import odeint
    
    with torch.no_grad():
        # sample from trajectorys' approx. posterior
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        
                #Initializing of W
        if W_train is not None:
            W = W_train
        else:
            W = np.ones(samp_trajs.shape, dtype=np.float32)
        W = torch.from_numpy(W).float().to(device)
        
        ckpt_path = os.path.join(train_dir, 'ckpt.pth')
        #Initializer of Vader
        #Not sure, whether it's a good idea to use dropout before Vader
        A_init = _initialize_imputation(samp_trajs, W)
        vad = Vaderlayer(A_init).to(device)
        
        #making types list for onehot_encoding, necessary for Batch Normalization
        staticonehot_types=[]
    
        for i in range(staticdata_types_dict.shape[0]):
            for j in range(staticdata_types_dict[i,1]):
                staticonehot_types.append(staticdata_types_dict[i])
        staticonehot_types=np.array(staticonehot_types)
        
        missing_onehot = []
    
        for i in range(static_true_miss_mask.shape[1]):
            for j in range(staticdata_types_dict[i,1]):
                missing_onehot.append(static_true_miss_mask[:,i])
        missing_onehot=np.transpose(np.array(missing_onehot))
        
        #Computing the latent_dim
        latent_dim_max = obs_dim
        latent_dim_number = int(np.round(latent_dim*latent_dim_max)) 
    
        nhidden_number = int(np.round(nhidden*latent_dim_number))
        
        #if statement, whether baselinedata is used or not + augmentation of the ODE
        if static is not None :
            modulefunc = Statlayer(static,staticdata_onehot,s_dim_static,z_dim_static).to(device)
            moduledec = Statdecode(static,staticdata_types_dict,s_dim_static,z_dim_static,device).to(device)
            func = LatentODEfunc(latent_dim_number+z_dim_static, nhidden_number, activation_ode = activation_ode, num_odelayers = num_odelayers).to(device)
        else:
            func = LatentODEfunc(latent_dim_number, nhidden_number, activation_ode = activation_ode, num_odelayers = num_odelayers).to(device)
        
        rnn_nhidden_max = obs_dim*len(samp_ts)
        rnn_nhidden_number = int(np.round(rnn_nhidden_enc*rnn_nhidden_max))
        
        if (enctype == 'LSTM'):
            rec = nn.LSTM(input_size = obs_dim, hidden_size = rnn_nhidden_number, num_layers = 1, bias = True, batch_first = True, dropout = 0, bidirectional = False).to(device)
            enc2out = LSTM2outlayer(rnn_nhidden_number,2*latent_dim_number).to(device)
        else:
            rec = RecognitionRNN(latent_dim_number*2, obs_dim, rnn_nhidden_number, samp_trajs.size(0),activation_rnn).to(device)
        
        rnn_nhidden_dec_max = (obs_dim)*len(samp_ts)
        rnn_nhidden_dec_number = int(np.round(rnn_nhidden_dec*rnn_nhidden_dec_max))
        if (dectype == 'LSTM'):
            dec = nn.LSTM(input_size = latent_dim_number+z_dim_static, hidden_size = rnn_nhidden_dec_number, num_layers = 1, bias = True, batch_first = True, dropout = 0, bidirectional = False).to(device)
            dec2out = LSTM2outlayer(rnn_nhidden_dec_number,obs_dim).to(device)
        elif(dectype == 'RNN'):
            dec = RecognitionRNN(obs_dim,latent_dim_number+z_dim_static,rnn_nhidden_dec_number,samp_trajs.shape[0],activation_dec).to(device)
        else:
            dec = Decoder(latent_dim_number+z_dim_static, obs_dim, rnn_nhidden_dec_number,dropout=0,activation_dec=activation_dec).to(device)
        
        if os.path.exists(ckpt_path):
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path,map_location=torch.device('cpu'))
                func.load_state_dict(checkpoint['func_state_dict'])
                rec.load_state_dict(checkpoint['rec_state_dict'])
                dec.load_state_dict(checkpoint['dec_state_dict'])
                #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                #samp_trajs = checkpoint['samp_trajs']
                #samp_ts = checkpoint['samp_ts']
                vad.load_state_dict(checkpoint['vad_state_dict'])
                if static is not None:
                    modulefunc.load_state_dict(checkpoint['modulefunc_state_dict'])
                    moduledec.load_state_dict(checkpoint['moduledec_state_dict'])
                if (enctype == 'LSTM'):
                    enc2out.load_state_dict(checkpoint['enc2out_state_dict'])
                if (dectype == 'LSTM'):
                    dec2out.load_state_dict(checkpoint['dec2out_state_dict'])
                print('Loaded ckpt from {}'.format(ckpt_path))
    
        X = samp_trajs #not necessary, but consistent with training
        Wmul = W
        if ~np.all(W.cpu().numpy() == 1.0) and np.all(np.logical_or(W.cpu().numpy() == 0.0, W.cpu().numpy() == 1.0)):
            #Wmul = torch.from_numpy(Wmul).float().to(device)
            XW = vad.forward(X,Wmul)
        else:
            XW = X
        
        if (enctype == 'LSTM'):
            h_0 = torch.zeros(1,samp_trajs.shape[0],rnn_nhidden_number).to(device)
            c_0 = torch.zeros(1,samp_trajs.shape[0],rnn_nhidden_number).to(device)
            for t in reversed(range(XW.shape[1])):
                obs = XW[:, t:t+1, :] #t:t+1, because rec needs 3 dimensions
                out, (h_0,c_0) = rec(obs,(h_0,c_0))
            out = out[:,0,:]
            out = enc2out(out)
        else:
            h = rec.initHidden().to(device)
            for t in reversed(range(samp_trajs.size(1))):
                obs = XW[:, t, :]
                out, h = rec.forward(obs, h)
        qz0_mean, qz0_logvar = out[:, :latent_dim_number], out[:, latent_dim_number:]
        epsilon = torch.randn(qz0_mean.size()).to(device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        
        tau=1e-3
        
        if static is not None: #b0 is the hidden dimension baseline data
                    
            if (batch_norm_static == True):
                staticdata_norm, batchmean, batchvar = batch_norm(staticdata_onehot,staticonehot_types, missing_onehot)
                inp = torch.from_numpy(staticdata_norm).float().to(device)                    
            
            else:
                inp = torch.from_numpy(staticdata_onehot).float().to(device)                    
                batchmean = None
                batchvar = None
                
            samples_s,log_pi, qz0_meanstat, qz0_logvarstat = modulefunc.forward(inp,tau)
            epsilonstat = torch.randn(qz0_meanstat.size()).to(device)
            b0 = epsilonstat * torch.exp(.5 * qz0_logvarstat) + qz0_meanstat
            
            qz0_meanstat=qz0_meanstat.to(device)
            qz0_logvarstat=qz0_logvarstat.to(device)
            
            y_latent = modulefunc.y_forward(b0)
            
            out, meanpz, logvarpz, log_p_x = moduledec(y_latent, samples_s,torch.from_numpy(staticdata_onehot).to(device),staticdata_types_dict, static_true_miss_mask, indices=[x for x in range(static.shape[0])],tau=tau, batch_norm=batch_norm_static, batchmean=batchmean, batchvar=batchvar)
            
            meanpz = meanpz.to(device)
            logvarpz = logvarpz.to(device)
            
            zinit = torch.cat((z0,b0),dim=1) #extending z0   
            
            if save_latent:
                samples_s_np = samples_s.cpu().numpy()
                samples_s_df = pd.DataFrame(samples_s_np)
                samples_s_df.to_csv('mixture_latent.csv')
                
                b0_np = b0.cpu().numpy()
                b0_df = pd.DataFrame(b0_np)
                b0_df.to_csv('static_latent.csv')
                
                z0_np = z0.cpu().numpy()
                z0_df = pd.DataFrame(z0_np)
                z0_df.to_csv('longitudinal_latent.csv')
                
        else:
            zinit = z0
                
            if save_latent:
                z0_np = z0.cpu().numpy()
                z0_df = pd.DataFrame(z0_np)
                z0_df.to_csv('longitudinal_latent.csv')
        
        ts_pos = np.linspace(0., timemax, num=num)
        ts_pos = torch.from_numpy(ts_pos).float().to(device)
        
        if solver == 'Adjoint':
            zs_pos = odeint(func, zinit, ts_pos).permute(1, 0, 2)
        elif solver:
            zs_pos = odeint(func,zinit, options).permute(1,0,2)
        else:
            zs_pos = odeint(func, zinit, ts_pos).permute(1, 0, 2)
        
        if (dectype == 'LSTM'):
            pred_x, (h,c) = dec(zs_pos)
            pred_x = dec2out(pred_x)
        elif(dectype =='RNN'):
            pred_x = torch.zeros(samp_trajs.shape[0],len(ts_pos),obs_dim)
            h = dec.initHidden().to(device)
            for t in range(zs_pos.size(1)):
                pred_x[:,t,:], h = dec.forward(zs_pos[:,t,:], h)
        else:
            pred_x = dec(zs_pos)
        
        xs_pos=pred_x
        xs_pos = xs_pos.cpu().numpy()
        
        #if negtime:
        #    ts_neg = np.linspace(timemin, 0., num=2000)[::-1].copy()
        #    ts_neg = torch.from_numpy(ts_neg).float().to(device)
        #    zs_neg = odeint(func, zinit, ts_neg).permute(1, 0, 2)
        #    if (LSTM == True):
        #        xs_neg, (h,c) = dec(zs_neg)
        #    else:
        #        xs_neg = dec(zs_neg)
        #    xs_neg = torch.flip(xs_neg, dims=[0])
        #    xs_neg = xs_neg.cpu().numpy()
        #    return xs_pos, ts_pos, xs_neg, ts_neg
        #else:
        return xs_pos, ts_pos, out, qz0_mean, np.exp(0.5*qz0_logvar), epsilon, qz0_meanstat, np.exp(0.5*qz0_logvarstat),epsilonstat

def valloss(solver, train_dir, device,samp_trajs, samp_ts, latent_dim, nhidden, obs_dim,activation_ode = 'elu',num_odelayers=1, W_train=None, enctype = 'RNN',dectype = 'RNN', rnn_nhidden_enc=0.5,activation_rnn = 'relu',rnn_nhidden_dec=0.5,activation_dec = 'relu',static=None,staticdata_onehot=None,staticdata_types_dict=None,static_true_miss_mask=None,s_dim_static=0,z_dim_static=0,scaling_ELBO=1,batch_norm_static=False):

    #Doing simulation just on the original time stamps to compute reconstructionloss, important for hyperopt
    
    if solver == 'Adjoint':
        from torchdiffeq import odeint_adjoint as odeint
    
    elif solver == 'torchdiffeqpack':
        from TorchDiffEqPack.odesolver import odesolve

        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': 0.01})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-3})
        options.update({'t0': samp_ts.tolist()[0] })
        options.update({'t1': samp_ts.tolist()[-1] })
        options.update({'t_eval':samp_ts.tolist()})
    
    elif solver:
        import sys
        sys.path.insert(1, solver)
        import torch_ACA
        from torch_ACA.odesolver import odesolve as odeint
        
        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': 0.01})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-3})
        options.update({'t0': samp_ts.tolist()[0] })
        options.update({'t1': samp_ts.tolist()[-1] })
        options.update({'t_eval':samp_ts.tolist()})
    else:
        from torchdiffeq import odeint
    
    with torch.no_grad():
        # sample from trajectorys' approx. posterior
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        
                #Initializing of W
        if W_train is not None:
            W = W_train
        else:
            W = np.ones(samp_trajs.shape, dtype=np.float32)
        W = torch.from_numpy(W).float().to(device)
        
        ckpt_path = os.path.join(train_dir, 'ckpt.pth')
        #Initializer of Vader
        #Not sure, whether it's a good idea to use dropout before Vader
        A_init = _initialize_imputation(samp_trajs, W)
        vad = Vaderlayer(A_init).to(device)
        
        #making types list for onehot_encoding, necessary for Batch Normalization
        staticonehot_types=[]
    
        for i in range(staticdata_types_dict.shape[0]):
            for j in range(staticdata_types_dict[i,1]):
                staticonehot_types.append(staticdata_types_dict[i])
        staticonehot_types=np.array(staticonehot_types)
        
        missing_onehot = []
    
        for i in range(static_true_miss_mask.shape[1]):
            for j in range(staticdata_types_dict[i,1]):
                missing_onehot.append(static_true_miss_mask[:,i])
        missing_onehot=np.transpose(np.array(missing_onehot))
        
        #Computing the latent_dim
        latent_dim_max = obs_dim
        latent_dim_number = int(np.round(latent_dim*latent_dim_max)) 
    
        nhidden_number = int(np.round(nhidden*latent_dim_number))
        
        #if statement, whether baselinedata is used or not + augmentation of the ODE
        if static is not None:
            modulefunc = Statlayer(static,staticdata_onehot,s_dim_static,z_dim_static).to(device)
            moduledec = Statdecode(static,staticdata_types_dict,s_dim_static,z_dim_static,device).to(device)
            func = LatentODEfunc(latent_dim_number+z_dim_static, nhidden_number, activation_ode = activation_ode, num_odelayers = num_odelayers).to(device)
        else:
            func = LatentODEfunc(latent_dim_number, nhidden_number, activation_ode = activation_ode, num_odelayers = num_odelayers).to(device)
        
        rnn_nhidden_max = obs_dim*len(samp_ts)
        rnn_nhidden_number = int(np.round(rnn_nhidden_enc*rnn_nhidden_max))
        
        if (enctype == 'LSTM'):
            rec = nn.LSTM(input_size = obs_dim, hidden_size = rnn_nhidden_number, num_layers = 1, bias = True, batch_first = True, dropout = 0, bidirectional = False).to(device)
            enc2out = LSTM2outlayer(rnn_nhidden_number,2*latent_dim_number).to(device)
        else:
            rec = RecognitionRNN(latent_dim_number*2, obs_dim, rnn_nhidden_number, samp_trajs.size(0),activation_rnn).to(device)
        
        rnn_nhidden_dec_max = (obs_dim)*len(samp_ts)
        rnn_nhidden_dec_number = int(np.round(rnn_nhidden_dec*rnn_nhidden_dec_max))
        
        if (dectype == 'LSTM'):
            dec = nn.LSTM(input_size = latent_dim_number+z_dim_static, hidden_size = rnn_nhidden_dec_number, num_layers = 1, bias = True, batch_first = True, dropout = 0, bidirectional = False).to(device)
            dec2out = LSTM2outlayer(rnn_nhidden_dec_number,obs_dim).to(device)
        elif(dectype == 'RNN'):
            dec = RecognitionRNN(obs_dim,latent_dim_number+z_dim_static,rnn_nhidden_dec_number,samp_trajs.size(0),activation_dec).to(device)
        else:
            dec = Decoder(latent_dim_number+z_dim_static, obs_dim, rnn_nhidden_dec_number,dropout=0,activation_dec=activation_dec).to(device)

        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path,map_location=torch.device('cpu'))
            func.load_state_dict(checkpoint['func_state_dict'])
            rec.load_state_dict(checkpoint['rec_state_dict'])
            dec.load_state_dict(checkpoint['dec_state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #samp_trajs = checkpoint['samp_trajs']
            #samp_ts = checkpoint['samp_ts']
            vad.load_state_dict(checkpoint['vad_state_dict'])
            if static is not None :
                modulefunc.load_state_dict(checkpoint['modulefunc_state_dict'])
                moduledec.load_state_dict(checkpoint['moduledec_state_dict'])
            if (enctype == 'LSTM'):
                enc2out.load_state_dict(checkpoint['enc2out_state_dict'])
            if (dectype == 'LSTM'):
                dec2out.load_state_dict(checkpoint['dec2out_state_dict'])
                
            print('Loaded ckpt from {}'.format(ckpt_path))
        
        X = samp_trajs #not necessary, but consistent with training
        Wmul = W
        if ~np.all(W.cpu().numpy() == 1.0) and np.all(np.logical_or(W.cpu().numpy() == 0.0, W.cpu().numpy() == 1.0)):
            #Wmul = torch.from_numpy(Wmul).float().to(device)
            XW = vad.forward(X,Wmul)
        else:
            XW = X
        
        if (enctype == 'LSTM'):
            h_0 = torch.zeros(1,samp_trajs.shape[0],rnn_nhidden_number).to(device)
            c_0 = torch.zeros(1,samp_trajs.shape[0],rnn_nhidden_number).to(device)
            for t in reversed(range(XW.size(1))):
                obs = XW[:, t:t+1, :] #t:t+1, because rec needs 3 dimensions
                out, (h_0,c_0) = rec(obs,(h_0,c_0))
            out = out[:,0,:]
            out = enc2out(out)
        else:
            h = rec.initHidden().to(device)
            for t in reversed(range(samp_trajs.shape[1])):
                obs = XW[:, t, :]
                out, h = rec.forward(obs, h)
        qz0_mean, qz0_logvar = out[:, :latent_dim_number], out[:, latent_dim_number:]
        epsilon = torch.randn(qz0_mean.size()).to(device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        
        tau = 1e-3
            
        
        if static is not None: #b0 is the hidden dimension baseline data
                    
            if (batch_norm_static == True):
                staticdata_norm, batchmean, batchvar = batch_norm(staticdata_onehot,staticonehot_types, missing_onehot)
                inp = torch.from_numpy(staticdata_norm).float().to(device)                    
            
            else:
                inp = torch.from_numpy(staticdata_onehot).float().to(device)                    
                batchmean = None
                batchvar = None
                
            samples_s,log_pi, qz0_meanstat, qz0_logvarstat = modulefunc.forward(inp,tau)
            epsilonstat = torch.randn(qz0_meanstat.size()).to(device)
            b0 = epsilonstat * torch.exp(.5 * qz0_logvarstat) + qz0_meanstat
            
            qz0_meanstat=qz0_meanstat.to(device)
            qz0_logvarstat=qz0_logvarstat.to(device)
            
            y_latent = modulefunc.y_forward(b0)
            
            out, meanpz, logvarpz, log_p_x = moduledec(y_latent, samples_s, torch.from_numpy(staticdata_onehot).to(device),staticdata_types_dict, static_true_miss_mask, indices = [x for x in range(static.shape[0])],tau=tau, batch_norm=batch_norm_static, batchmean=batchmean, batchvar=batchvar)
   
            meanpz = meanpz.to(device)
            logvarpz = logvarpz.to(device)
    
            zinit = torch.cat((z0,b0),dim=1) #extending z0    

        else:
            zinit = z0

        if solver == 'Adjoint':
            zs_pos = odeint(func, zinit, samp_ts).permute(1, 0, 2)
        elif solver:
            zs_pos = odeint(func,zinit, options).permute(1,0,2)
        else:
            zs_pos = odeint(func, zinit, samp_ts).permute(1, 0, 2)
        
        if (dectype == 'LSTM'):
            pred_x, (h,c) = dec(zs_pos)
            pred_x = dec2out(pred_x)
        elif(dectype =='RNN'):
            pred_x = torch.zeros(samp_trajs.shape[0],len(samp_ts),obs_dim)
            h = dec.initHidden().to(device)
            for t in range(zs_pos.size(1)):
                pred_x[:,t,:], h = dec.forward(zs_pos[:,t,:], h)
        else:
            pred_x = dec(zs_pos)
        xs_pos = pred_x
        #Computing the losses for the positive xs_values
        pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
        analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                pz0_mean, pz0_logvar).sum(-1)
        
        #Implement weighted mean squared error for Vader
        klavg = torch.mean(analytic_kl) #same like above
        # samp_trajs.size(1) is number of timesteps, samp_trajs.size(2) is number of variables
        
        rec_loss = rec_loss2(samp_trajs,pred_x,Wmul)
        rec_lossavg = torch.sum(rec_loss)*samp_trajs.size(1)*samp_trajs.size(2) / (sum(sum(sum(Wmul))))

        # sum(sum(... is cardinality of W
        
        # compute loss
        if static is not None :
            #log_pi from the encoder above
            eps=1E-20
            #KL(q(s|x)|p(s))
            
            #logits=log_pi, labels=pi_param
            #because logits has to be transformed with softmax
            pi_param = torch.nn.functional.softmax(log_pi,dim=1)
            KL_s = torch.sum(pi_param * torch.log(pi_param + eps), dim=1) + torch.log(torch.tensor(float(s_dim_static)))
            KL_s = KL_s.to(device)
            
            #meanpz, logvarpz, qz0_meanstat, qz0_logvarstat
            #These two implementations of the multivariate KL divergence are equivalent, first one ist from torchdiffeq, second one from HI-VAE
            analytic_kl_stat = normal_kl(qz0_meanstat, qz0_logvarstat,meanpz, logvarpz).sum(-1)
            #KL_z_stat = -0.5*z_dim + 0.5*(torch.exp(qz0_logvarstat-logvarpz)+((meanpz - qz0_meanstat)**2.)/torch.exp(logvarpz) -qz0_logvarstat+logvarpz).sum(-1)                
            
            #Eq[log_p(x|y)]
            loss_reconstruction_stat = log_p_x.sum(-1).to(device)

            ELBO_stat = -torch.mean(loss_reconstruction_stat - analytic_kl_stat - KL_s,0)

            #print(ELBO_stat)
            #print(klavg + rec_lossavg)
            #print(-torch.mean(loss_reconstruction_stat))
            
            long = (klavg + rec_lossavg)/(klavg + rec_lossavg + ELBO_stat)
            
            stat = (ELBO_stat)/(klavg + rec_lossavg + ELBO_stat)
            
            long_scaled=stat/(long+stat)*(klavg + rec_lossavg)
            
            stat_scaled=long/(long+stat)*(ELBO_stat)
            
            loss = long_scaled + scaling_ELBO *stat_scaled
        
        else:
            
            loss=klavg + rec_lossavg
        
        
        xs_pos = xs_pos.cpu().numpy()

        return xs_pos, out, loss, klavg, rec_lossavg, loss_reconstruction_stat, analytic_kl_stat, KL_s
    
def visualizationSIR_eps_median(correctdata, correct_ts, xs_1, ts_1,xs_2,ts_2,xs_3,ts_3, varnumber,varnames, negtime,device,labels=None, legend=True):
    # take first trajectory for visualization
    # varnumber: number of plottet variable
    # varnames: names of the variables
    # gen: specifies, if data are generated files, changes description
    # num_of_visits: Specifies, how many visits are used for prediction
    alpha=0.05

    xs_pos = torch.from_numpy(xs_1).float().to(device)
    xs_pos = xs_1    
    
    xsp_mean = np.nanmedian(xs_1,axis=0)
    #xsp_std = xs_pos.std(dim=0)

    #KI of mean
    # KIxspu = np.nanpercentile(xs_pos,2.5,axis=0)
    # KIxspo = np.nanpercentile(xs_pos,97.5,axis=0)

    plt.figure()
    plt.plot(ts_1, xsp_mean[:, varnumber], 'darkorchid',
              label=labels[0])
    # plt.plot(ts_1, KIxspu[:, varnumber], 'darkorchid',
              # linestyle=':')
    # plt.plot(ts_1, KIxspo[:, varnumber], 'darkorchid',
              # linestyle=':')
    #plt.errorbar(ts_pos, xsp_mean[:, varnumber], yerr=xsp_std[:,varnumber], fmt = 'o', color = "red")

    xs_gen = torch.from_numpy(xs_2).float().to(device)
    xs_pos=xs_2
    
    xsp_mean = np.nanmedian(xs_2,axis=0)
    #xsp_std = xs_pos.std(dim=0)

    #KI of mean
    # KIxspu = np.nanpercentile(xs_pos,2.5,axis=0)
    # KIxspo = np.nanpercentile(xs_pos,97.5,axis=0)
    
    plt.plot(ts_2, xsp_mean[:, varnumber], '#ffa82e',
              label=labels[1])
    # plt.plot(ts_2, KIxspu[:, varnumber], '#ffa82e',
              # linestyle=':')
    # plt.plot(ts_2, KIxspo[:, varnumber], '#ffa82e',
              # linestyle=':')

    xs_pos = torch.from_numpy(xs_3).float().to(device)
    xs_pos=xs_3

    xsp_mean = np.nanmedian(xs_3,axis=0)
    #xsp_std = xs_pos.std(dim=0)

    #KI of mean
    # KIxspu = np.nanpercentile(xs_pos,2.5,axis=0)
    # KIxspo = np.nanpercentile(xs_pos,97.5,axis=0)
    plt.plot(ts_3, xsp_mean[:, varnumber], 'green',
              label=labels[2])
    # plt.plot(ts_3, KIxspu[:, varnumber], 'green',
              # linestyle=':')
    # plt.plot(ts_3, KIxspo[:, varnumber], 'green',
              # linestyle=':')

    plt.plot(correct_ts, correctdata[0,:,varnumber],
                label='theoretical SIR model', color='black')

    #plt.plot([], [], ':', color='darkorchid', label = '2.5% / 97.5% percentile')
    

    plt.xlabel('Time')
    plt.ylabel('People')
    if legend == True:
        plt.legend()
    plt.title(str(varnames[varnumber]))
    plt.savefig('./vistime.png', dpi=500)
    print('Saved visualization figure at {}'.format('./vis.png'))
    
def visualizationSIR_t_median(correct_ts, noisy, xs_1, ts_1,xs_2,ts_2,xs_3,ts_3, varnumber,varnames, negtime,device,labels=None, legend=True):
    # take first trajectory for visualization
    # varnumber: number of plottet variable
    # varnames: names of the variables
    # gen: specifies, if data are generated files, changes description
    # num_of_visits: Specifies, how many visits are used for prediction
    alpha=0.05

    xs_pos = torch.from_numpy(xs_1).float().to(device)
    xs_pos=xs_1
    
    xsp_mean = np.nanmedian(xs_pos,axis=0)

    #KI of mean
    #KIxspu = np.nanpercentile(xs_pos,2.5,axis=0)
    #KIxspo = np.nanpercentile(xs_pos,97.5,axis=0)

    plt.figure()
    plt.plot(ts_1, xsp_mean[:, varnumber], color = 'darkorchid',              label=labels[0])
    #plt.plot(ts_1, KIxspo[:,varnumber], color = 'darkorchid', linestyle = ':')
    #plt.plot(ts_1, KIxspu[:,varnumber], color = 'darkorchid', linestyle = ':')
    #plt.errorbar(ts_pos, xsp_mean[:, varnumber], yerr=xsp_std[:,varnumber], fmt = 'o', color = "red")

    xs_gen = torch.from_numpy(xs_2).float().to(device)
    xs_pos=xs_2
    
    xsp_mean = np.nanmedian(xs_pos,axis=0)

    #KI of mean
    #KIxspu = np.nanpercentile(xs_pos,2.5,axis=0)
    #KIxspo = np.nanpercentile(xs_pos,97.5,axis=0)
    
    plt.plot(ts_2, xsp_mean[:, varnumber], color='#ffa82e',
              label=labels[1])
    #plt.plot(ts_2, KIxspo[:,varnumber], color='#ffa82e', linestyle = ':')
    #plt.plot(ts_2, KIxspu[:,varnumber], color='#ffa82e', linestyle = ':')
    #plt.errorbar(ts_pos, xsp_mean[:, varnumber], yerr=xsp_std[:,varnumber], fmt = 'o', color = "red")

    xs_pos = torch.from_numpy(xs_3).float().to(device)
    xs_pos=xs_3

    xsp_mean = np.nanmedian(xs_pos,axis=0)

    #KI of mean
    #KIxspu = np.nanpercentile(xs_pos,2.5,axis=0)
    #KIxspo = np.nanpercentile(xs_pos,97.5,axis=0)
    plt.plot(ts_3, xsp_mean[:, varnumber], color = 'green',
              label=labels[2])
    #plt.plot(ts_3, KIxspo[:,varnumber], color = 'green', linestyle = ':')
    #plt.plot(ts_3, KIxspu[:,varnumber], color = 'green', linestyle = ':')
    #plt.errorbar(ts_pos, xsp_mean[:, varnumber], yerr=xsp_std[:,varnumber], fmt = 'o', color = "red")

    index_5 = np.linspace(0,199,5,dtype=int)
    index_10 = np.linspace(0,199,10,dtype=int)
    index_100 = np.linspace(0,199,100,dtype=int)

    plt.scatter(correct_ts[index_5], noisy[0,index_5,varnumber],                s = 7, linestyle = 'None', color='darkviolet')

    plt.scatter(correct_ts[index_10], noisy[1,index_10,varnumber],
                s = 7, linestyle = 'None', color='#ffbb5c')

    plt.scatter(correct_ts[index_100], noisy[2,index_100,varnumber],
                s = 7, linestyle = 'None',color='limegreen')
    
    plt.plot([], [], 'o', color='darkviolet', label = 'real sample t=5')
    plt.plot([], [], 'o', color='#ffbb5c', label = 'real sample t=10')
    plt.plot([], [], 'o', color='limegreen', label = 'real sample t=100')
    #plt.plot([], [], ':', color='darkorchid', label = '2.5% / 97.5% percentile')
    
    plt.xlabel('Time')
    plt.ylabel('People')
    if legend == True:
        plt.legend()
    plt.title(str(varnames[varnumber]))
    plt.savefig('./vistime.png', dpi=500)
    print('Saved visualization figure at {}'.format('./vis.png'))


def visualizationSIR_median(correctdata, correct_ts, samp_train, samp_ts,W, xs_1, ts_1,xs_2,ts_2,xs_3,ts_3, varnumber,varnames, negtime,device,labels=None, legend=True):
    # take first trajectory for visualization
    # varnumber: number of plottet variable
    # varnames: names of the variables
    # gen: specifies, if data are generated files, changes description
    # num_of_visits: Specifies, how many visits are used for prediction
    alpha=0.05

    xs_pos = torch.from_numpy(xs_1).float().to(device)
    xs_pos=xs_1
    
    xsp_mean = np.nanmedian(xs_pos,axis=0)

    #KI of mean
    #KIxspu = np.nanpercentile(xs_pos,2.5,axis=0)
    #KIxspo = np.nanpercentile(xs_pos,97.5,axis=0)

    plt.figure()
    plt.plot(ts_1, xsp_mean[:, varnumber], 'darkorchid',
              label=labels[0])
    #plt.plot(ts_1, KIxspo[:,varnumber], 'darkorchid', linestyle = ':')
    #plt.plot(ts_1, KIxspu[:,varnumber], 'darkorchid', linestyle = ':')
    #plt.errorbar(ts_pos, xsp_mean[:, varnumber], yerr=xsp_std[:,varnumber], fmt = 'o', color = "red")

    xs_gen = torch.from_numpy(xs_2).float().to(device)
    
    xs_pos=xs_2
    
    xsp_mean = np.nanmedian(xs_pos,axis=0)

    #KI of mean
    #KIxspu = np.nanpercentile(xs_pos,2.5,axis=0)
    #KIxspo = np.nanpercentile(xs_pos,97.5,axis=0)
    
    plt.plot(ts_2, xsp_mean[:, varnumber], '#ffa82e',
              label=labels[1])
    #plt.plot(ts_2, KIxspo[:,varnumber], '#ffa82e', linestyle = ':')    
    #plt.plot(ts_2, KIxspu[:,varnumber], '#ffa82e', linestyle = ':')    #plt.errorbar(ts_pos, xsp_mean[:, varnumber], yerr=xsp_std[:,varnumber], fmt = 'o', color = "red")

    xs_pos = torch.from_numpy(xs_3).float().to(device)

    xs_pos=xs_3

    xsp_mean = np.nanmedian(xs_pos,axis=0)

    #KI of mean
    #KIxspu = np.nanpercentile(xs_pos,2.5,axis=0)
    #KIxspo = np.nanpercentile(xs_pos,97.5,axis=0)
    
    plt.plot(ts_3, xsp_mean[:, varnumber], 'green',
              label=labels[2])
    #plt.plot(ts_3, KIxspo[:,varnumber], 'green', linestyle = ':')
    #plt.plot(ts_3, KIxspu[:,varnumber], 'green', linestyle = ':')
    #plt.errorbar(ts_pos, xsp_mean[:, varnumber], yerr=xsp_std[:,varnumber], fmt = 'o', color = "red")

    plt.plot(correct_ts, correctdata[0,:,varnumber],
                label='theoretical SIR model', color='black')

    plt.scatter(samp_ts, samp_train[:,:,varnumber],
                s = 7, linestyle = 'None', color='limegreen')

    plt.plot([], [], 'o', color='limegreen', label = 'real sample')    
    #plt.plot([], [], ':', color='darkorchid', label = '2.5% / 97.5% percentile')
    

    plt.xlabel('Time')
    plt.ylabel('People')
    if legend == True:
        plt.legend()
    plt.title(str(varnames[varnumber]))
    plt.savefig('./vistime.png', dpi=500)
    print('Saved visualization figure at {}'.format('./vis.png'))

def hypopt(solver, nepochs, lr, train_dir, device, samp_train,samp_test, samp_ts, latent_dim, nhidden, obs_dim, batchsize, activation_ode = 'relu',num_odelayers = 1, W_train = None,W_test=None, enctype = 'RNN',dectype = 'RNN', rnn_nhidden_enc=0.5,activation_rnn = 'relu',rnn_nhidden_dec=0.5,activation_dec = 'relu',dropout_dec = 0,static_train=None,static_test=None,staticdata_onehot_train=None,staticdata_onehot_test=None,staticdata_types_dict=None,static_true_miss_mask_train=None,static_true_miss_mask_test=None,s_dim_static=0,z_dim_static=0,scaling_ELBO=1,batch_norm_static=False):
    
    #Variables like in training, but
    #samp_train training data
    #samp_test test data
    #W_train, W_Test
    #modules_train, modules_test
    
    if W_train is not None:
        W = W_train
    else:
        W = np.ones(samp_train.shape, dtype=np.float32)
    W = torch.from_numpy(W).float().to(device)

    if solver == 'Adjoint':
        from torchdiffeq import odeint_adjoint as odeint
    
    elif solver == 'torchdiffeqpack':
        from TorchDiffEqPack.odesolver import odesolve

        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': 0.01})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-3})
        options.update({'t0': samp_ts.tolist()[0] })
        options.update({'t1': samp_ts.tolist()[-1] })
        options.update({'t_eval':samp_ts.tolist()})
    
    elif solver:
        import sys
        sys.path.insert(1, solver)
        import torch_ACA
        from torch_ACA.odesolver import odesolve as odeint
        
        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': 0.01})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-3})
        options.update({'t0': samp_ts.tolist()[0] })
        options.update({'t1': samp_ts.tolist()[-1] })
        options.update({'t_eval':samp_ts.tolist()})
    else:
        from torchdiffeq import odeint
    
    #Initializer of Vader
    #Not sure, whether it's a good idea to use dropout before Vader
    A_init = _initialize_imputation(samp_train, W)
    vad = Vaderlayer(A_init).to(device)
    
    staticonehot_types=[]
    
    if staticdata_types_dict is not None:
        for i in range(staticdata_types_dict.shape[0]):
            for j in range(staticdata_types_dict[i,1]):
                staticonehot_types.append(staticdata_types_dict[i])
        staticonehot_types=np.array(staticonehot_types)
    
    missing_onehot_train = []
    
    if static_true_miss_mask_train is not None:
        for i in range(static_true_miss_mask_train.shape[1]):
            for j in range(staticdata_types_dict[i,1]):
                missing_onehot_train.append(static_true_miss_mask_train[:,i])
        missing_onehot_train=np.transpose(np.array(missing_onehot_train))
    
    #Computing the latent_dim
    latent_dim_max = obs_dim
    latent_dim_number = int(np.round(latent_dim*latent_dim_max)) 
    
    if (latent_dim_number==0):
        latent_dim_number=1
    
    nhidden_number = int(np.round(nhidden*latent_dim_number))
    if (nhidden_number==0):
        nhidden_number=1
    
    #if statement, whether baselinedata is used or not + augmentation of the ODE
    if static_train is not None:
        modulefunc = Statlayer(static_train,staticdata_onehot_train,s_dim_static,z_dim_static).to(device)
        moduledec = Statdecode(static_train,staticdata_types_dict,s_dim_static,z_dim_static,device).to(device)
        func = LatentODEfunc(latent_dim_number+z_dim_static, nhidden_number, activation_ode = activation_ode, num_odelayers = num_odelayers).to(device)
    else:
        func = LatentODEfunc(latent_dim_number, nhidden_number, activation_ode = activation_ode, num_odelayers = num_odelayers).to(device)
    
    rnn_nhidden_max = obs_dim*len(samp_ts)
    rnn_nhidden_number = int(np.round(rnn_nhidden_enc*rnn_nhidden_max))
    
    if(rnn_nhidden_number==0):
        rnn_nhidden_number=1
    
    params = list()
    
    batchsize_number = int(np.round(batchsize*samp_train.shape[0]))
    
    if (enctype == 'LSTM'):
        rec = nn.LSTM(input_size = obs_dim, hidden_size = rnn_nhidden_number, num_layers = 1, bias = True, batch_first = True, dropout = 0, bidirectional = False).to(device)
        enc2out = LSTM2outlayer(rnn_nhidden_number,2*latent_dim_number).to(device)
        params = params + (list(enc2out.parameters()))
    else:
        rec = RecognitionRNN(latent_dim_number*2, obs_dim, rnn_nhidden_number, batchsize_number, activation_rnn).to(device)
    
    rnn_nhidden_dec_max = (obs_dim)*len(samp_ts)
    rnn_nhidden_dec_number = int(np.round(rnn_nhidden_dec*rnn_nhidden_dec_max))
    
    if(rnn_nhidden_dec_number==0):
        rnn_nhidden_dec_number=1
    
    if (dectype == 'LSTM'):
        dec = nn.LSTM(input_size = latent_dim_number+z_dim_static, hidden_size = rnn_nhidden_dec_number, num_layers = 1, bias = True, batch_first = True, dropout = 0, bidirectional = False).to(device)
        dec2out = LSTM2outlayer(rnn_nhidden_dec_number,obs_dim).to(device)
        params =params + (list(dec2out.parameters()))
    elif(dectype == 'RNN'):
        dec = RecognitionRNN(obs_dim,latent_dim_number+z_dim_static,rnn_nhidden_dec_number, batchsize_number, activation_dec).to(device)
    else:
        dec = Decoder(latent_dim_number+z_dim_static, obs_dim, rnn_nhidden_dec_number, dropout_dec, activation_dec).to(device)
    
    if static_train is not None:
        params = params + (list(modulefunc.parameters()) + list(moduledec.parameters()) + list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()) + list(vad.parameters()))
    else:
        params = params + (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()) + list(vad.parameters()))
    
    optimizer = optim.Adam(params, lr=lr)
    loss_meter = RunningAverageMeter()
    
    for itr in range(1, nepochs + 1):
        #randomly permute the indices of the trajs
        permutation = torch.randperm(samp_train.shape[0])
        
        tau = np.max([1.0 - (0.999/(nepochs-50))*itr,1e-3])
        
        #batchsize
        for i in range(0,samp_train.shape[0], batchsize_number):
            #if statement implemented for indices of batches
            #else is the case, when last batch isn't a "complete" one
            #then you have to complete the batch, f.e. with the first permutations
            if i + batchsize_number <= samp_train.shape[0]:
                indices = permutation[i:i+batchsize_number]
                
            else:
                indices = permutation[i:samp_train.shape[0]]
                indices = torch.cat((indices,permutation[0:i+batchsize_number-samp_train.size(0)]),0)
            
            optimizer.zero_grad()
            # backward in time to infer q(z_0)
            # Treat W as an indicator for nonmissingness (1: nonmissing; 0: missing)
            X = samp_train[indices,:,:]
            Wmul = W[indices,:,:]
            if ~np.all(W.cpu().numpy() == 1.0) and np.all(np.logical_or(W.cpu().numpy() == 0.0, W.cpu().numpy() == 1.0)):
                #Wmul = torch.from_numpy(Wmul).float().to(device)
                XW = vad.forward(X,Wmul)
            else:
                XW = X
            
            if (enctype == 'LSTM'):
                h_0 = torch.zeros(1,batchsize_number,rnn_nhidden_number).to(device)
                c_0 = torch.zeros(1,batchsize_number,rnn_nhidden_number).to(device)
                for t in reversed(range(XW.size(1))):
                    obs = XW[:, t:t+1, :] #t:t+1, because rec needs 3 dimensions
                    out, (h_0,c_0) = rec(obs,(h_0,c_0))
                    # out and h_0 are the same, because just one point is going through LSTM
                out = out[:,0,:]
                out = enc2out(out)
            else:
                h = rec.initHidden().to(device)
                for t in reversed(range(XW.size(1))):
                    obs = XW[:, t, :]
                    out, h = rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :latent_dim_number], out[:, latent_dim_number:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            
            if static_train is not None: #b0 is the hidden dimension baseline data
                
                if (batch_norm_static == True):
                    staticdata_norm, batchmean, batchvar = batch_norm(staticdata_onehot_train[indices,:],staticonehot_types, missing_onehot_train[indices,:])
                    inp = torch.from_numpy(staticdata_norm).float().to(device)                    
                
                else:
                    inp = torch.from_numpy(staticdata_onehot_train[indices,:]).float().to(device)                    
                    batchmean = None
                    batchvar = None
                    
                samples_s,log_pi, qz0_meanstat, qz0_logvarstat = modulefunc.forward(inp,tau)
                epsilonstat = torch.randn(qz0_meanstat.size()).to(device)
                b0 = epsilonstat * torch.exp(.5 * qz0_logvarstat) + qz0_meanstat
                
                qz0_meanstat=qz0_meanstat.to(device)
                qz0_logvarstat=qz0_logvarstat.to(device)
                
                y_latent = modulefunc.y_forward(b0)
                
                out, meanpz, logvarpz, log_p_x = moduledec(y_latent, samples_s, torch.from_numpy(staticdata_onehot_train).to(device),staticdata_types_dict, static_true_miss_mask_train, indices,tau=tau, batch_norm=batch_norm_static, batchmean=batchmean, batchvar=batchvar)
                
                meanpz = meanpz.to(device)
                logvarpz = logvarpz.to(device)
                
                zinit = torch.cat((z0,b0),dim=1) #extending z0
        # forward in time and solve ode for reconstructions
            else:
                zinit = z0
            
            if solver == 'Adjoint':
                pred_z= odeint(func, zinit, samp_ts).permute(1, 0, 2)
            elif solver:
                pred_z = odeint(func,zinit, options).permute(1,0,2)
            else:
                pred_z = odeint(func, zinit, samp_ts).permute(1, 0, 2)
            
            if (dectype == 'LSTM'):
                pred_x, (h,c) = dec(pred_z)
                pred_x = dec2out(pred_x)
            elif(dectype == 'RNN'):
                pred_x = torch.zeros(batchsize_number,len(samp_ts),obs_dim)
                h = dec.initHidden().to(device)
                for t in range(pred_z.size(1)):
                    pred_x[:,t,:], h = dec.forward(pred_z[:,t,:], h)
            else:
                pred_x = dec(pred_z)
        
            # compute loss
            pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
            analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                    pz0_mean, pz0_logvar).sum(-1)
            
            klavg = torch.mean(analytic_kl) #same like above
            # samp_trajs.size(1) is number of timesteps, samp_trajs.size(2) is number of variables
            
            rec_loss = rec_loss2(samp_train[indices,:,:],pred_x,Wmul)
            rec_lossavg = torch.sum(rec_loss)*samp_train.size(1)*samp_train.size(2) / (sum(sum(sum(Wmul))))

            if static_train is not None:
                #log_pi from the encoder above
                eps=1E-20
                #KL(q(s|x)|p(s))
                
                #logits=log_pi, labels=pi_param
                #because logits has to be transformed with softmax
                pi_param = torch.nn.functional.softmax(log_pi,dim=1)
                KL_s = torch.sum(pi_param * torch.log(pi_param + eps), dim=1) + torch.log(torch.tensor(float(s_dim_static)))
                KL_s = KL_s.to(device)
                
                #meanpz, logvarpz, qz0_meanstat, qz0_logvarstat
                #These two implementations of the multivariate KL divergence are equivalent, first one ist from torchdiffeq, second one from HI-VAE
                analytic_kl_stat = normal_kl(qz0_meanstat, qz0_logvarstat,meanpz, logvarpz).sum(-1)
                #KL_z_stat = -0.5*z_dim + 0.5*(torch.exp(qz0_logvarstat-logvarpz)+((meanpz - qz0_meanstat)**2.)/torch.exp(logvarpz) -qz0_logvarstat+logvarpz).sum(-1)                
                
                #Eq[log_p(x|y)]
                loss_reconstruction_stat = log_p_x.sum(-1).to(device)

                ELBO_stat = -torch.mean(loss_reconstruction_stat - analytic_kl_stat - KL_s,0)

                #print(ELBO_stat)
                #print(klavg + rec_lossavg)
                #print(-torch.mean(loss_reconstruction_stat))
                
                long = (klavg + rec_lossavg)/(klavg + rec_lossavg + ELBO_stat)
                
                stat = (ELBO_stat)/(klavg + rec_lossavg + ELBO_stat)
                
                long_scaled=stat/(long+stat)*(klavg + rec_lossavg)
                
                stat_scaled=long/(long+stat)*(ELBO_stat)
                
                loss = long_scaled + scaling_ELBO *stat_scaled
                
            else:
                loss=klavg + rec_lossavg

            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            #print(loss)
            #print('Iter: {}, running avg elbo: {:.4f}'.format(itr, -loss_meter.avg))
                
    if W_test is not None:
        W_test = W_test
    else:
        W_test = np.ones(samp_test.shape, dtype=np.float32)
    W_test = torch.from_numpy(W_test).float().to(device)
    
    missing_onehot_test = []
    
    if static_true_miss_mask_train is not None:
        for i in range(static_true_miss_mask_test.shape[1]):
            for j in range(staticdata_types_dict[i,1]):
                missing_onehot_test.append(static_true_miss_mask_test[:,i])
        missing_onehot_test=np.transpose(np.array(missing_onehot_test))
    
    X_test = samp_test #not necessary, but consistent with training

    if ~np.all(W_test.cpu().numpy() == 1.0) and np.all(np.logical_or(W_test.cpu().numpy() == 0.0, W_test.cpu().numpy() == 1.0)):
        #Wmul = torch.from_numpy(Wmul).float().to(device)
        XW_test = vad.forward(X_test,W_test)
    else:
        XW_test = X_test
    
    if (enctype == 'LSTM'):
        h_0 = torch.zeros(1,samp_test.shape[0],rnn_nhidden_number).to(device)
        c_0 = torch.zeros(1,samp_test.shape[0],rnn_nhidden_number).to(device)
        for t in reversed(range(XW_test.size(1))):
            obs = XW_test[:, t:t+1, :] #t:t+1, because rec needs 3 dimensions
            out, (h_0,c_0) = rec(obs,(h_0,c_0))
            # out and h_0 are the same, because just one point is going through LSTM
        out = out[:,0,:]
        out = enc2out(out)
    else:
        h = torch.zeros(samp_test.shape[0], rnn_nhidden_number).to(device)
        for t in reversed(range(samp_test.shape[1])):
            obs = XW_test[:, t, :]
            out, h = rec.forward(obs, h)
    qz0_mean, qz0_logvar = out[:, :latent_dim_number], out[:, latent_dim_number:]
    epsilon = torch.randn(qz0_mean.size()).to(device)
    z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
    
    tau = 1e-3
    
    if static_test is not None: #b0 is the hidden dimension baseline data
        
        if (batch_norm_static == True):
            staticdata_norm, batchmean, batchvar = batch_norm(staticdata_onehot_test,staticonehot_types, missing_onehot_test)
            inp = torch.from_numpy(staticdata_norm).float().to(device)                    
        
        else:
            inp = torch.from_numpy(staticdata_onehot_test).float().to(device)                    
            batchmean = None
            batchvar = None
            
        samples_s,log_pi, qz0_meanstat, qz0_logvarstat = modulefunc.forward(inp,tau)
        epsilonstat = torch.randn(qz0_meanstat.size()).to(device)
        b0 = epsilonstat * torch.exp(.5 * qz0_logvarstat) + qz0_meanstat
        
        qz0_meanstat=qz0_meanstat.to(device)
        qz0_logvarstat=qz0_logvarstat.to(device)
        
        y_latent = modulefunc.y_forward(b0)
        
        out, meanpz, logvarpz, log_p_x = moduledec(y_latent, samples_s, torch.from_numpy(staticdata_onehot_test).to(device),staticdata_types_dict, static_true_miss_mask_test, indices = [x for x in range(static_test.shape[0])],tau=tau, batch_norm=batch_norm_static, batchmean=batchmean, batchvar=batchvar)
        
        meanpz = meanpz.to(device)
        logvarpz = logvarpz.to(device)
        
        zinit = torch.cat((z0,b0),dim=1) #extending z0
    else:
        zinit = z0

    if solver == 'Adjoint':
        zs_pos= odeint(func, zinit, samp_ts).permute(1, 0, 2)
    elif solver:
        zs_pos = odeint(func,zinit, options).permute(1,0,2)
    else:
        zs_pos = odeint(func, zinit, samp_ts).permute(1, 0, 2)
    
    if (dectype == 'LSTM'):
        xs_pos, (h,c) = dec(zs_pos)
        xs_pos = dec2out(xs_pos)
    elif(dectype == 'RNN'):
        xs_pos = torch.zeros(samp_test.shape[0],len(samp_ts),obs_dim)
        h = torch.zeros(samp_test.shape[0], rnn_nhidden_dec_number)
        for t in range(pred_z.size(1)):
            xs_pos[:,t,:], h = dec.forward(zs_pos[:,t,:], h)
    else:
        xs_pos = dec(zs_pos, test=True)
    
    #Computing the losses for the positive xs_values

    rec_loss2train = rec_loss2(samp_test,xs_pos,W_test)
    rec_loss2train = torch.sum(rec_loss2train)*samp_test.size(1)*samp_test.size(2) / (sum(sum(sum(W_test))))
    
    rec_loss = rec_loss2train
    
    if static_test is not None:

        #Eq[log_p(x|y)]
        loss_reconstruction_stat = log_p_x.sum(-1)
        loss_rec_stat = -torch.mean(loss_reconstruction_stat)
    
        return rec_loss, loss_rec_stat, log_p_x
    else:
        return rec_loss

def generationprior(solver, train_dir, device,samp_trajs,samp_ts, latent_dim, nhidden, obs_dim, activation_ode = 'elu',num_odelayers=1, dectype = 'RNN',rnn_nhidden_dec=0.5,activation_dec = 'relu',static=None,staticdata_onehot=None,staticdata_types_dict=None,s_dim_static=0,z_dim_static=0,s_prob=None, timemax = 1, timemin = None,num=2000):
    
    #timemax: Maximum of the simulated time
    #negtime is boolean, whether time is backwards predicted or not
    #timemin: just necessary, if negtime = True, minimum time of simulations
    #s_prob: vector with propability of s during sampling
    
    if solver == 'Adjoint':
        from torchdiffeq import odeint_adjoint as odeint
    elif solver:
        import sys
        sys.path.insert(1, solver)
        import torch_ACA
        from torch_ACA.odesolver import odesolve as odeint
        
        ts_pos = np.linspace(0., timemax, num=num)
        ts_pos = torch.from_numpy(ts_pos).float().to(device)
        
        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': 0.01})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-3})
        options.update({'t0': ts_pos.tolist()[0] })
        options.update({'t1': ts_pos.tolist()[-1] })
        options.update({'t_eval':ts_pos.tolist()})
    else:
        from torchdiffeq import odeint
    
    with torch.no_grad():
        # sample from trajectorys' approx. posterior
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        
        
        ckpt_path = os.path.join(train_dir, 'ckpt.pth')
        
        #Computing the latent_dim
        latent_dim_max = obs_dim
        latent_dim_number = int(np.round(latent_dim*latent_dim_max)) 
    
        nhidden_number = int(np.round(nhidden*latent_dim_number))
        
        #if statement, whether baselinedata is used or not + augmentation of the ODE
        
        if static is not None :
            modulefunc = Statlayer(static,staticdata_onehot,s_dim_static,z_dim_static).to(device)
            moduledec = Statdecode(static,staticdata_types_dict,s_dim_static,z_dim_static,device).to(device)
            func = LatentODEfunc(latent_dim_number+z_dim_static, nhidden_number, activation_ode = activation_ode, num_odelayers = num_odelayers).to(device)
        else:
            func = LatentODEfunc(latent_dim_number, nhidden_number, activation_ode = activation_ode, num_odelayers = num_odelayers).to(device)
        
        rnn_nhidden_dec_max = (obs_dim)*len(samp_ts)
        rnn_nhidden_dec_number = int(np.round(rnn_nhidden_dec*rnn_nhidden_dec_max))
        if (dectype == 'LSTM'):
            dec = nn.LSTM(input_size = latent_dim_number+z_dim_static, hidden_size = rnn_nhidden_dec_number, num_layers = 1, bias = True, batch_first = True, dropout = 0, bidirectional = False).to(device)
            dec2out = LSTM2outlayer(rnn_nhidden_dec_number,obs_dim).to(device)
        elif(dectype == 'RNN'):
            dec = RecognitionRNN(obs_dim,latent_dim_number+z_dim_static,rnn_nhidden_dec_number,samp_trajs.shape[0],activation_dec).to(device)
        else:
            dec = Decoder(latent_dim_number+z_dim_static, obs_dim, rnn_nhidden_dec_number,dropout=0,activation_dec=activation_dec).to(device)
        
        if os.path.exists(ckpt_path):
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path,map_location=torch.device('cpu'))
                func.load_state_dict(checkpoint['func_state_dict'])
                dec.load_state_dict(checkpoint['dec_state_dict'])
                if static is not None:
                    moduledec.load_state_dict(checkpoint['moduledec_state_dict'])
                if (dectype == 'LSTM'):
                    dec2out.load_state_dict(checkpoint['dec2out_state_dict'])
                print('Loaded ckpt from {}'.format(ckpt_path))
        
        zinit = torch.from_numpy(pd.read_csv(train_dir+'/main_VirtualPPts.csv').to_numpy()).float()
        
        z0 = zinit[:,:-z_dim_static]
        
        b0 = zinit[:,-z_dim_static:] 
        
        y_latent = modulefunc.y_forward(b0)
        
        out = moduledec(y_latent=y_latent,samples_s=None,staticdata_onehot=None,static_types=staticdata_types_dict,true_miss_mask=None,indices=None, tau=1e-3,batch_norm=False,batchmean=None,batchvar=None,generation_prior=False,output_prior=True)

        #zinit = torch.cat((z0,b0),dim=1) 

        ts_pos = np.linspace(0., timemax, num=num)
        ts_pos = torch.from_numpy(ts_pos).float().to(device)
        if solver == 'Adjoint':
            zs_pos= odeint(func, zinit, ts_pos).permute(1, 0, 2)
        elif solver:
            zs_pos = odeint(func,zinit, options).permute(1,0,2)
        else:
            zs_pos = odeint(func, zinit, ts_pos).permute(1, 0, 2)
        
        if (dectype == 'LSTM'):
            pred_x, (h,c) = dec(zs_pos)
            pred_x = dec2out(pred_x)
        elif(dectype =='RNN'):
            pred_x = torch.zeros(samp_trajs.shape[0],len(ts_pos),obs_dim)
            h = dec.initHidden().to(device)
            for t in range(zs_pos.size(1)):
                pred_x[:,t,:], h = dec.forward(zs_pos[:,t,:], h)
        else:
            pred_x = dec(zs_pos)
        
        xs_pos=pred_x
        xs_pos = xs_pos.cpu().numpy()

        return xs_pos, ts_pos, out

def generationPosterior(solver, train_dir, device,samp_trajs,samp_ts, latent_dim, nhidden, obs_dim, activation_ode = 'elu',num_odelayers=1,W_train=None, enctype = 'RNN',dectype = 'RNN', rnn_nhidden_enc=0.5,activation_rnn = 'relu',rnn_nhidden_dec=0.5,activation_dec = 'relu',static=None,staticdata_onehot=None,staticdata_types_dict=None,static_true_miss_mask=None,s_dim_static=0,z_dim_static=0,batch_norm_static=False, timemax = 1, negtime = False, timemin = None,num=2000, sigmalong=1,sigmastat=1,N=1):

    #sigmafactorlong: Describes the sd of the epsilon of the reparameterization Trick of the longitudinal data, default = 1
    #sigmafactorstat: Describes the sd of the epsilon of the reparameterization Trick of the static data, default = 1
    #N: Describes the number of how often the complete set of patients is used (N=1 has 354 patients in PPMI)
    
    if solver == 'Adjoint':
        from torchdiffeq import odeint_adjoint as odeint
    
    elif solver == 'torchdiffeqpack':
        from TorchDiffEqPack.odesolver import odesolve

        ts_pos = np.linspace(0., timemax, num=num)
        ts_pos = torch.from_numpy(ts_pos).float().to(device)

        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': 0.01})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-3})
        options.update({'t0': ts_pos.tolist()[0] })
        options.update({'t1': ts_pos.tolist()[-1] })
        options.update({'t_eval':ts_pos.tolist()})
        
    elif solver:
        import sys
        sys.path.insert(1, solver)
        import torch_ACA
        from torch_ACA.odesolver import odesolve as odeint
        
        ts_pos = np.linspace(0., timemax, num=num)
        ts_pos = torch.from_numpy(ts_pos).float().to(device)
        
        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': 0.01})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-3})
        options.update({'t0': ts_pos.tolist()[0] })
        options.update({'t1': ts_pos.tolist()[-1] })
        options.update({'t_eval':ts_pos.tolist()})
    else:
        from torchdiffeq import odeint
    
    with torch.no_grad():
        # sample from trajectorys' approx. posterior
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        
                #Initializing of W
        if W_train is not None:
            W = W_train
        else:
            W = np.ones(samp_trajs.shape, dtype=np.float32)
        W = torch.from_numpy(W).float().to(device)
        
        ckpt_path = os.path.join(train_dir, 'ckpt.pth')
        #Initializer of Vader
        #Not sure, whether it's a good idea to use dropout before Vader
        A_init = _initialize_imputation(samp_trajs, W)
        vad = Vaderlayer(A_init).to(device)
        
        #making types list for onehot_encoding, necessary for Batch Normalization
        staticonehot_types=[]
        
        if staticdata_types_dict is not None:
            for i in range(staticdata_types_dict.shape[0]):
                for j in range(staticdata_types_dict[i,1]):
                    staticonehot_types.append(staticdata_types_dict[i])
            staticonehot_types=np.array(staticonehot_types)
        
        missing_onehot = []
        
        if static_true_miss_mask is not None:
            for i in range(static_true_miss_mask.shape[1]):
                for j in range(staticdata_types_dict[i,1]):
                    missing_onehot.append(static_true_miss_mask[:,i])
            missing_onehot=np.transpose(np.array(missing_onehot))
        
        #Computing the latent_dim
        latent_dim_max = obs_dim
        latent_dim_number = int(np.round(latent_dim*latent_dim_max)) 
    
        nhidden_number = int(np.round(nhidden*latent_dim_number))
        
        #if statement, whether baselinedata is used or not + augmentation of the ODE
        if static is not None:
            modulefunc = Statlayer(static,staticdata_onehot,s_dim_static,z_dim_static).to(device)
            moduledec = Statdecode(static,staticdata_types_dict,s_dim_static,z_dim_static,device).to(device)
            func = LatentODEfunc(latent_dim_number+z_dim_static, nhidden_number, activation_ode = activation_ode, num_odelayers = num_odelayers).to(device)
        else:
            func = LatentODEfunc(latent_dim_number, nhidden_number, activation_ode = activation_ode, num_odelayers = num_odelayers).to(device)
        
        rnn_nhidden_max = obs_dim*len(samp_ts)
        rnn_nhidden_number = int(np.round(rnn_nhidden_enc*rnn_nhidden_max))
        
        if (enctype == 'LSTM'):
            rec = nn.LSTM(input_size = obs_dim, hidden_size = rnn_nhidden_number, num_layers = 1, bias = True, batch_first = True, dropout = 0, bidirectional = False).to(device)
            enc2out = LSTM2outlayer(rnn_nhidden_number,2*latent_dim_number).to(device)
        else:
            rec = RecognitionRNN(latent_dim_number*2, obs_dim, rnn_nhidden_number, samp_trajs.size(0),activation_rnn).to(device)
        
        rnn_nhidden_dec_max = (obs_dim)*len(samp_ts)
        rnn_nhidden_dec_number = int(np.round(rnn_nhidden_dec*rnn_nhidden_dec_max))
        if (dectype == 'LSTM'):
            dec = nn.LSTM(input_size = latent_dim_number+z_dim_static, hidden_size = rnn_nhidden_dec_number, num_layers = 1, bias = True, batch_first = True, dropout = 0, bidirectional = False).to(device)
            dec2out = LSTM2outlayer(rnn_nhidden_dec_number,obs_dim).to(device)
        elif(dectype == 'RNN'):
            dec = RecognitionRNN(obs_dim,latent_dim_number+z_dim_static,rnn_nhidden_dec_number,samp_trajs.shape[0],activation_dec).to(device)
        else:
            dec = Decoder(latent_dim_number+z_dim_static, obs_dim, rnn_nhidden_dec_number,dropout=0,activation_dec=activation_dec).to(device)
        
        if os.path.exists(ckpt_path):
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path,map_location=torch.device('cpu'))
                func.load_state_dict(checkpoint['func_state_dict'])
                rec.load_state_dict(checkpoint['rec_state_dict'])
                dec.load_state_dict(checkpoint['dec_state_dict'])
                #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                #samp_trajs = checkpoint['samp_trajs']
                #samp_ts = checkpoint['samp_ts']
                vad.load_state_dict(checkpoint['vad_state_dict'])
                if static is not None:
                    modulefunc.load_state_dict(checkpoint['modulefunc_state_dict'])
                    moduledec.load_state_dict(checkpoint['moduledec_state_dict'])
                if (enctype == 'LSTM'):
                    enc2out.load_state_dict(checkpoint['enc2out_state_dict'])
                if (dectype == 'LSTM'):
                    dec2out.load_state_dict(checkpoint['dec2out_state_dict'])
                print('Loaded ckpt from {}'.format(ckpt_path))
                
        X = samp_trajs #not necessary, but consistent with training
        Wmul = W
        if ~np.all(W.cpu().numpy() == 1.0) and np.all(np.logical_or(W.cpu().numpy() == 0.0, W.cpu().numpy() == 1.0)):
            #Wmul = torch.from_numpy(Wmul).float().to(device)
            XW = vad.forward(X,Wmul)
        else:
            XW = X
        
        if (enctype == 'LSTM'):
            h_0 = torch.zeros(1,samp_trajs.shape[0],rnn_nhidden_number).to(device)
            c_0 = torch.zeros(1,samp_trajs.shape[0],rnn_nhidden_number).to(device)
            for t in reversed(range(XW.shape[1])):
                obs = XW[:, t:t+1, :] #t:t+1, because rec needs 3 dimensions
                out, (h_0,c_0) = rec(obs,(h_0,c_0))
            out = out[:,0,:]
            out = enc2out(out)
        else:
            h = rec.initHidden().to(device)
            for t in reversed(range(samp_trajs.size(1))):
                obs = XW[:, t, :]
                out, h = rec.forward(obs, h)
        qz0_mean, qz0_logvar = out[:, :latent_dim_number], out[:, latent_dim_number:]
        
        #Making N representations of the Original data
        qz0_mean = torch.cat([qz0_mean]*N,dim=0)
        qz0_logvar = torch.cat([qz0_logvar]*N,dim=0)
        #epsilon = torch.randn(qz0_mean.size()).to(device)
        epsilon = torch.normal(mean=torch.zeros(qz0_mean.size()),std=torch.zeros(qz0_mean.size())+sigmalong)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        
        tau=1e-3
        
        if static is not None: #b0 is the hidden dimension baseline data
                    
            if (batch_norm_static == True):
                staticdata_norm, batchmean, batchvar = batch_norm(staticdata_onehot,staticonehot_types, missing_onehot)
                inp = torch.from_numpy(staticdata_norm).float().to(device)                    
            
            else:
                inp = torch.from_numpy(staticdata_onehot).float().to(device)                    
                batchmean = None
                batchvar = None
                
            samples_s,log_pi, qz0_meanstat, qz0_logvarstat = modulefunc.forward(inp,tau)
            
            #Making N representations of the Original data
            qz0_meanstat = torch.cat([qz0_meanstat]*N,dim=0)
            qz0_logvarstat = torch.cat([qz0_logvarstat]*N,dim=0)
            
            qz0_meanstat=qz0_meanstat.to(device)
            qz0_logvarstat=qz0_logvarstat.to(device)
            
            #epsilonstat = torch.randn(qz0_meanstat.size()).to(device)
            
            epsilonstat = torch.normal(mean=torch.zeros(qz0_meanstat.size()),std=torch.zeros(qz0_meanstat.size())+sigmastat)
            
            b0 = epsilonstat * torch.exp(.5 * qz0_logvarstat) + qz0_meanstat
            
            y_latent = modulefunc.y_forward(b0)
            
            out, meanpz, logvarpz, log_p_x = moduledec(y_latent, torch.from_numpy(np.concatenate([samples_s]*N,axis=0)).to(device),torch.cat([torch.from_numpy(staticdata_onehot).to(device)]*N,dim=0),staticdata_types_dict, np.concatenate([static_true_miss_mask]*N,axis=0), indices=[x for x in range(torch.cat([static]*N,dim=0).shape[0])],tau=tau, batch_norm=batch_norm_static, batchmean=batchmean, batchvar=batchvar)
            
            meanpz = meanpz.to(device)
            logvarpz = logvarpz.to(device)
            
            zinit = torch.cat((z0,b0),dim=1) #extending z0   
        
        else:
            zinit = z0

        ts_pos = np.linspace(0., timemax, num=num)
        ts_pos = torch.from_numpy(ts_pos).float().to(device)
        if solver == 'Adjoint':
            zs_pos= odeint(func, zinit, ts_pos).permute(1, 0, 2)
        elif solver:
            zs_pos = odeint(func,zinit, options).permute(1,0,2)
        else:
            zs_pos = odeint(func, zinit, ts_pos).permute(1, 0, 2)
            
        if (dectype == 'LSTM'):
            pred_x, (h,c) = dec(zs_pos)
            pred_x = dec2out(pred_x)
        elif(dectype =='RNN'):
            pred_x = torch.zeros(samp_trajs.shape[0]*N,len(ts_pos),obs_dim)
            h = torch.zeros(samp_trajs.shape[0]*N, rnn_nhidden_dec_number)
            for t in range(zs_pos.size(1)):
                pred_x[:,t,:], h = dec.forward(zs_pos[:,t,:], h)
        else:
            pred_x = dec(zs_pos)
        
        xs_pos=pred_x
        xs_pos = xs_pos.cpu().numpy()
        
        #if negtime:
        #    ts_neg = np.linspace(timemin, 0., num=2000)[::-1].copy()
        #    ts_neg = torch.from_numpy(ts_neg).float().to(device)
        #    zs_neg = odeint(func, zinit, ts_neg).permute(1, 0, 2)
        #    if (LSTM == True):
        #        xs_neg, (h,c) = dec(zs_neg)
        #    else:
        #        xs_neg = dec(zs_neg)
        #    xs_neg = torch.flip(xs_neg, dims=[0])
        #    xs_neg = xs_neg.cpu().numpy()
        #    return xs_pos, ts_pos, xs_neg, ts_neg
        #else:

        return xs_pos, ts_pos, out, qz0_mean, np.exp(0.5*qz0_logvar), epsilon, qz0_meanstat, np.exp(0.5*qz0_logvarstat),epsilonstat

def hypopt_early(solver, nepochs, lr, train_dir, device, samp_train,samp_test, samp_ts, latent_dim, nhidden, obs_dim, batchsize, activation_ode = 'relu',num_odelayers = 1, W_train = None,W_test=None, enctype = 'RNN',dectype = 'RNN', rnn_nhidden_enc=0.5,activation_rnn = 'relu',rnn_nhidden_dec=0.5,activation_dec = 'relu',dropout_dec = 0,static_train=None,static_test=None,staticdata_onehot_train=None,staticdata_onehot_test=None,staticdata_types_dict=None,static_true_miss_mask_train=None,static_true_miss_mask_test=None,s_dim_static=0,z_dim_static=0,scaling_ELBO=1,batch_norm_static=False,min_delta=0,patience=50):
    
    #Variables like in training, but
    #samp_train training data
    #samp_test test data
    #W_train, W_Test
    #modules_train, modules_test
    #min_delta is the value, when test error is seen as non decreasing
    #patience: number of epochs to seach, when the test error gets minimized
    
    if W_train is not None:
        W = W_train
    else:
        W = np.ones(samp_train.shape, dtype=np.float32)
    W = torch.from_numpy(W).float().to(device)

    if solver == 'Adjoint':
        from torchdiffeq import odeint_adjoint as odeint
        
    elif solver == 'torchdiffeqpack':
        from TorchDiffEqPack.odesolver import odesolve

        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': 0.01})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-3})
        options.update({'t0': samp_ts.tolist()[0] })
        options.update({'t1': samp_ts.tolist()[-1] })
        options.update({'t_eval':samp_ts.tolist()})
    
    elif solver:
        import sys
        sys.path.insert(1, solver)
        import torch_ACA
        from torch_ACA.odesolver import odesolve as odeint
        
        options = {}
        options.update({'method': 'Dopri5'})
        options.update({'h': 0.01})
        options.update({'rtol': 1e-3})
        options.update({'atol': 1e-3})
        options.update({'t0': samp_ts.tolist()[0] })
        options.update({'t1': samp_ts.tolist()[-1] })
        options.update({'t_eval':samp_ts.tolist()})
    else:
        from torchdiffeq import odeint
    
    #Initializer of Vader
    #Not sure, whether it's a good idea to use dropout before Vader
    A_init = _initialize_imputation(samp_train, W)
    vad = Vaderlayer(A_init).to(device)
    
    staticonehot_types=[]
    
    if staticdata_types_dict is not None:
        for i in range(staticdata_types_dict.shape[0]):
            for j in range(staticdata_types_dict[i,1]):
                staticonehot_types.append(staticdata_types_dict[i])
        staticonehot_types=np.array(staticonehot_types)
    
    missing_onehot_train = []
    
    if static_true_miss_mask_train is not None:
        for i in range(static_true_miss_mask_train.shape[1]):
            for j in range(staticdata_types_dict[i,1]):
                missing_onehot_train.append(static_true_miss_mask_train[:,i])
        missing_onehot_train=np.transpose(np.array(missing_onehot_train))
    
    #Computing the latent_dim
    latent_dim_max = obs_dim
    latent_dim_number = int(np.round(latent_dim*latent_dim_max)) 
    
    if (latent_dim_number==0):
        latent_dim_number=1
    
    nhidden_number = int(np.round(nhidden*latent_dim_number))
    if (nhidden_number==0):
        nhidden_number=1
    
    #if statement, whether baselinedata is used or not + augmentation of the ODE
    if static_train is not None:
        modulefunc = Statlayer(static_train,staticdata_onehot_train,s_dim_static,z_dim_static).to(device)
        moduledec = Statdecode(static_train,staticdata_types_dict,s_dim_static,z_dim_static,device).to(device)
        func = LatentODEfunc(latent_dim_number+z_dim_static, nhidden_number, activation_ode = activation_ode, num_odelayers = num_odelayers).to(device)
    else:
        func = LatentODEfunc(latent_dim_number, nhidden_number, activation_ode = activation_ode, num_odelayers = num_odelayers).to(device)
    
    rnn_nhidden_max = obs_dim*len(samp_ts)
    rnn_nhidden_number = int(np.round(rnn_nhidden_enc*rnn_nhidden_max))
    
    if(rnn_nhidden_number==0):
        rnn_nhidden_number=1
    
    params = list()
    
    batchsize_number = int(np.round(batchsize*samp_train.shape[0]))
    
    if (enctype == 'LSTM'):
        rec = nn.LSTM(input_size = obs_dim, hidden_size = rnn_nhidden_number, num_layers = 1, bias = True, batch_first = True, dropout = 0, bidirectional = False).to(device)
        enc2out = LSTM2outlayer(rnn_nhidden_number,2*latent_dim_number).to(device)
        params = params + (list(enc2out.parameters()))
    else:
        rec = RecognitionRNN(latent_dim_number*2, obs_dim, rnn_nhidden_number, batchsize_number, activation_rnn).to(device)
    
    rnn_nhidden_dec_max = (obs_dim)*len(samp_ts)
    rnn_nhidden_dec_number = int(np.round(rnn_nhidden_dec*rnn_nhidden_dec_max))
    
    if(rnn_nhidden_dec_number==0):
        rnn_nhidden_dec_number=1
    
    if (dectype == 'LSTM'):
        dec = nn.LSTM(input_size = latent_dim_number+z_dim_static, hidden_size = rnn_nhidden_dec_number, num_layers = 1, bias = True, batch_first = True, dropout = 0, bidirectional = False).to(device)
        dec2out = LSTM2outlayer(rnn_nhidden_dec_number,obs_dim).to(device)
        params =params + (list(dec2out.parameters()))
    elif(dectype == 'RNN'):
        dec = RecognitionRNN(obs_dim,latent_dim_number+z_dim_static,rnn_nhidden_dec_number, batchsize_number, activation_dec).to(device)
    else:
        dec = Decoder(latent_dim_number+z_dim_static, obs_dim, rnn_nhidden_dec_number, dropout_dec, activation_dec).to(device)
    
    if static_train is not None:
        params = params + (list(modulefunc.parameters()) + list(moduledec.parameters()) + list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()) + list(vad.parameters()))
    else:
        params = params + (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()) + list(vad.parameters()))
    
    if W_test is not None:
        W_test = W_test
    else:
        W_test = np.ones(samp_test.shape, dtype=np.float32)
    W_test = torch.from_numpy(W_test).float().to(device)
    
    missing_onehot_test = []

    if static_true_miss_mask_test is not None:
        for i in range(static_true_miss_mask_test.shape[1]):
            for j in range(staticdata_types_dict[i,1]):
                missing_onehot_test.append(static_true_miss_mask_test[:,i])
        missing_onehot_test=np.transpose(np.array(missing_onehot_test))
    
    X_test = samp_test #not necessary, but consistent with training
    
    optimizer = optim.Adam(params, lr=lr)
    loss_meter = RunningAverageMeter()
    
    #early stopping validated only on the reconstruction loss of the longitudinal data (like the hyperparameters)
    rec_loss_test = [] #list for the reconstruction losses
    best_epoch = 1 #best_epoch
    no_improve = 0
    
    for itr in range(1, nepochs + 1):
        #randomly permute the indices of the trajs
        permutation = torch.randperm(samp_train.shape[0])
        
        tau = np.max([1.0 - (0.999/(nepochs-50))*itr,1e-3])
        
        #batchsize
        for i in range(0,samp_train.shape[0], batchsize_number):
            #if statement implemented for indices of batches
            #else is the case, when last batch isn't a "complete" one
            #then you have to complete the batch, f.e. with the first permutations
            if i + batchsize_number <= samp_train.shape[0]:
                indices = permutation[i:i+batchsize_number]
                
            else:
                indices = permutation[i:samp_train.shape[0]]
                indices = torch.cat((indices,permutation[0:i+batchsize_number-samp_train.size(0)]),0)
            
            optimizer.zero_grad()
            # backward in time to infer q(z_0)
            # Treat W as an indicator for nonmissingness (1: nonmissing; 0: missing)
            X = samp_train[indices,:,:]
            Wmul = W[indices,:,:]
            if ~np.all(W.cpu().numpy() == 1.0) and np.all(np.logical_or(W.cpu().numpy() == 0.0, W.cpu().numpy() == 1.0)):
                #Wmul = torch.from_numpy(Wmul).float().to(device)
                XW = vad.forward(X,Wmul)
            else:
                XW = X
            
            if (enctype == 'LSTM'):
                h_0 = torch.zeros(1,batchsize_number,rnn_nhidden_number).to(device)
                c_0 = torch.zeros(1,batchsize_number,rnn_nhidden_number).to(device)
                for t in reversed(range(XW.size(1))):
                    obs = XW[:, t:t+1, :] #t:t+1, because rec needs 3 dimensions
                    out, (h_0,c_0) = rec(obs,(h_0,c_0))
                    # out and h_0 are the same, because just one point is going through LSTM
                out = out[:,0,:]
                out = enc2out(out)
            else:
                h = rec.initHidden().to(device)
                for t in reversed(range(XW.size(1))):
                    obs = XW[:, t, :]
                    out, h = rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :latent_dim_number], out[:, latent_dim_number:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            
            if static_train is not None : #b0 is the hidden dimension baseline data
                
                if (batch_norm_static == True):
                    staticdata_norm, batchmean, batchvar = batch_norm(staticdata_onehot_train[indices,:],staticonehot_types, missing_onehot_train[indices,:])
                    inp = torch.from_numpy(staticdata_norm).float().to(device)                    
                
                else:
                    inp = torch.from_numpy(staticdata_onehot_train[indices,:]).float().to(device)                    
                    batchmean = None
                    batchvar = None
                    
                samples_s,log_pi, qz0_meanstat, qz0_logvarstat = modulefunc.forward(inp,tau)
                epsilonstat = torch.randn(qz0_meanstat.size()).to(device)
                b0 = epsilonstat * torch.exp(.5 * qz0_logvarstat) + qz0_meanstat
                
                qz0_meanstat=qz0_meanstat.to(device)
                qz0_logvarstat=qz0_logvarstat.to(device)
                
                y_latent = modulefunc.y_forward(b0)
                
                out, meanpz, logvarpz, log_p_x = moduledec(y_latent, samples_s, torch.from_numpy(staticdata_onehot_train).to(device),staticdata_types_dict, static_true_miss_mask_train, indices,tau=tau, batch_norm=batch_norm_static, batchmean=batchmean, batchvar=batchvar)
                
                meanpz = meanpz.to(device)
                logvarpz = logvarpz.to(device)
                
                zinit = torch.cat((z0,b0),dim=1) #extending z0
        # forward in time and solve ode for reconstructions
            else:
                zinit = z0
            if solver == 'Adjoint':
                pred_z= odeint(func, zinit, samp_ts).permute(1, 0, 2)
            elif solver:
                pred_z = odeint(func,zinit, options).permute(1,0,2)
            else:
                pred_z = odeint(func, zinit, samp_ts).permute(1, 0, 2)
            
            if (dectype == 'LSTM'):
                pred_x, (h,c) = dec(pred_z)
                pred_x = dec2out(pred_x)
            elif(dectype == 'RNN'):
                pred_x = torch.zeros(batchsize_number,len(samp_ts),obs_dim)
                h = dec.initHidden().to(device)
                for t in range(pred_z.size(1)):
                    pred_x[:,t,:], h = dec.forward(pred_z[:,t,:], h)
            else:
                pred_x = dec(pred_z)
        
            # compute loss
            pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
            analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                    pz0_mean, pz0_logvar).sum(-1)
            
            klavg = torch.mean(analytic_kl) #same like above
            # samp_trajs.size(1) is number of timesteps, samp_trajs.size(2) is number of variables
            
            rec_loss = rec_loss2(samp_train[indices,:,:],pred_x,Wmul)
            rec_lossavg = torch.sum(rec_loss)*samp_train.size(1)*samp_train.size(2) / (sum(sum(sum(Wmul))))

            if static_train is not None :
                #log_pi from the encoder above
                eps=1E-20
                #KL(q(s|x)|p(s))
                
                #logits=log_pi, labels=pi_param
                #because logits has to be transformed with softmax
                pi_param = torch.nn.functional.softmax(log_pi,dim=1)
                KL_s = torch.sum(pi_param * torch.log(pi_param + eps), dim=1) + torch.log(torch.tensor(float(s_dim_static)))
                KL_s = KL_s.to(device)
                
                #meanpz, logvarpz, qz0_meanstat, qz0_logvarstat
                #These two implementations of the multivariate KL divergence are equivalent, first one ist from torchdiffeq, second one from HI-VAE
                analytic_kl_stat = normal_kl(qz0_meanstat, qz0_logvarstat,meanpz, logvarpz).sum(-1)
                #KL_z_stat = -0.5*z_dim + 0.5*(torch.exp(qz0_logvarstat-logvarpz)+((meanpz - qz0_meanstat)**2.)/torch.exp(logvarpz) -qz0_logvarstat+logvarpz).sum(-1)                
                
                #Eq[log_p(x|y)]
                loss_reconstruction_stat = log_p_x.sum(-1).to(device)

                ELBO_stat = -torch.mean(loss_reconstruction_stat - analytic_kl_stat - KL_s,0)

                #print(ELBO_stat)
                #print(klavg + rec_lossavg)
                #print(-torch.mean(loss_reconstruction_stat))
                
                long = (klavg + rec_lossavg)/(klavg + rec_lossavg + ELBO_stat)
                
                stat = (ELBO_stat)/(klavg + rec_lossavg + ELBO_stat)
                
                long_scaled=stat/(long+stat)*(klavg + rec_lossavg)
                
                stat_scaled=long/(long+stat)*(ELBO_stat)
                
                loss = long_scaled + scaling_ELBO *stat_scaled
                
            else:
                loss=klavg + rec_lossavg

            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            #print(loss)
            #print('Iter: {}, running avg elbo: {:.4f}'.format(itr, -loss_meter.avg))
                
    
        if ~np.all(W_test.cpu().numpy() == 1.0) and np.all(np.logical_or(W_test.cpu().numpy() == 0.0, W_test.cpu().numpy() == 1.0)):
            #Wmul = torch.from_numpy(Wmul).float().to(device)
            XW_test = vad.forward(X_test,W_test)
        else:
            XW_test = X_test
        
        if (enctype == 'LSTM'):
            h_0 = torch.zeros(1,samp_test.shape[0],rnn_nhidden_number).to(device)
            c_0 = torch.zeros(1,samp_test.shape[0],rnn_nhidden_number).to(device)
            for t in reversed(range(XW_test.size(1))):
                obs = XW_test[:, t:t+1, :] #t:t+1, because rec needs 3 dimensions
                out, (h_0,c_0) = rec(obs,(h_0,c_0))
                # out and h_0 are the same, because just one point is going through LSTM
            out = out[:,0,:]
            out = enc2out(out)
        else:
            h = torch.zeros(samp_test.shape[0], rnn_nhidden_number).to(device)
            for t in reversed(range(samp_test.shape[1])):
                obs = XW_test[:, t, :]
                out, h = rec.forward(obs, h)
        qz0_mean, qz0_logvar = out[:, :latent_dim_number], out[:, latent_dim_number:]
        epsilon = torch.randn(qz0_mean.size()).to(device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        
        tau = 1e-3
        
        if static_test is not None: #b0 is the hidden dimension baseline data
            
            if (batch_norm_static == True):
                staticdata_norm, batchmean, batchvar = batch_norm(staticdata_onehot_test,staticonehot_types, missing_onehot_test)
                inp = torch.from_numpy(staticdata_norm).float().to(device)                    
            
            else:
                inp = torch.from_numpy(staticdata_onehot_test).float().to(device)                    
                batchmean = None
                batchvar = None
                
            samples_s,log_pi, qz0_meanstat, qz0_logvarstat = modulefunc.forward(inp,tau)
            epsilonstat = torch.randn(qz0_meanstat.size()).to(device)
            b0 = epsilonstat * torch.exp(.5 * qz0_logvarstat) + qz0_meanstat
            
            qz0_meanstat=qz0_meanstat.to(device)
            qz0_logvarstat=qz0_logvarstat.to(device)
            
            y_latent = modulefunc.y_forward(b0)
            
            out, meanpz, logvarpz, log_p_x = moduledec(y_latent, samples_s, torch.from_numpy(staticdata_onehot_test).to(device),staticdata_types_dict, static_true_miss_mask_test, indices = [x for x in range(static_test.shape[0])],tau=tau, batch_norm=batch_norm_static, batchmean=batchmean, batchvar=batchvar)
            
            meanpz = meanpz.to(device)
            logvarpz = logvarpz.to(device)
            
            zinit = torch.cat((z0,b0),dim=1) #extending z0
        else:
            zinit = z0
    
        if solver == 'Adjoint':
            zs_pos= odeint(func, zinit, samp_ts).permute(1, 0, 2)
            zs_pos = odeint(func,zinit, options).permute(1,0,2)
        else:
            zs_pos = odeint(func, zinit, samp_ts).permute(1, 0, 2)
        
        if (dectype == 'LSTM'):
            xs_pos, (h,c) = dec(zs_pos)
            xs_pos = dec2out(xs_pos)
        elif(dectype == 'RNN'):
            xs_pos = torch.zeros(samp_test.shape[0],len(samp_ts),obs_dim)
            h = torch.zeros(samp_test.shape[0], rnn_nhidden_dec_number)
            for t in range(pred_z.size(1)):
                xs_pos[:,t,:], h = dec.forward(zs_pos[:,t,:], h)
        else:
            xs_pos = dec(zs_pos, test=True)
        
        #Computing the losses for the positive xs_values
    
        rec_loss2test = rec_loss2(samp_test,xs_pos,W_test)
        rec_loss2test = torch.sum(rec_loss2test)*samp_test.size(1)*samp_test.size(2) / (sum(sum(sum(W_test))))
        
        rec_loss_test.append(rec_loss2test)
        
        #if condition if new test loss, better than the best one
        if rec_loss2test < rec_loss_test[best_epoch-1]:
            no_improve = 0
            best_epoch=itr
        else:
            no_improve = no_improve + 1
        
        #break loop if no improvement of test loss is 
        if no_improve == patience:
            break
        
        #early stopping
        
    if static_test is not None :

        #Eq[log_p(x|y)]
        loss_reconstruction_stat = log_p_x.sum(-1)
        loss_rec_stat = -torch.mean(loss_reconstruction_stat)
    
        return rec_loss_test[best_epoch-1], loss_rec_stat, log_p_x, best_epoch
    else:
        return rec_loss_test[best_epoch-1], best_epoch
