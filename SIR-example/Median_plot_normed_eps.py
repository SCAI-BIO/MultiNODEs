import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import processing as pr

import main as m

node_dir='/home/pwendland/Dokumente/GitHub/mnode/NeuralODE_code/SIR/SIR_synth_new'
node_dir='C:/Users/wendl/Documents/Python/Neural_ODE/SIR_synth_new'


rpnode_n3000_eps075_t10 = torch.load(node_dir+'/Reconstruction_n3000_eps075_t10_normed.pth')

rpnode_n3000_eps05_t10 = torch.load(node_dir+'/Reconstruction_n3000_eps05_t10_normed.pth')

rpnode_n3000_eps100_t10 = torch.load(node_dir+'/Reconstruction_n3000_eps100_t10_normed.pth')

genposteriornode_n3000_eps075_t10 = torch.load(node_dir+'/Generationposterior_n3000_eps075_t10_normed.pth')

genposteriornode_n3000_eps05_t10 = torch.load(node_dir+'/Generationposterior_n3000_eps05_t10_normed.pth')

genposteriornode_n3000_eps100_t10 = torch.load(node_dir+'/Generationposterior_n3000_eps100_t10_normed.pth')

rp_n3000_eps075_t10 = pd.read_csv(node_dir + '/SIR-data_correct_n3000_eps075.csv',sep=',',header=0,
                   index_col=0, engine='python')

rp_n3000_eps05_t10 = pd.read_csv(node_dir + '/SIR-data_correct_n3000_eps05.csv',sep=',',header=0,
                   index_col=0, engine='python')

rp_n3000_eps100_t10 = pd.read_csv(node_dir + '/SIR-data_correct_n3000_eps100.csv',sep=',',header=0,
                   index_col=0, engine='python')

correct = pd.read_csv(node_dir + '/SIR_correct.csv',sep=',',header=0,
                   index_col=0, engine='python')

n=1
values=correct.values
var=3
timesteps = np.linspace(0,40,num=200)
tover = timesteps/200 #relative values of time
temps = len(timesteps)

correctdata = np.reshape(values, (n, var, temps))
correctdata = np.swapaxes(correctdata, 1, 2)
correctdata = correctdata[:1,:,:]



device = 'cpu'

n = len(rp_n3000_eps05_t10) #number of persons
#timesteps equal to months
timesteps = np.linspace(0,40,num=200)
tover = timesteps/200 #relative values of time
temps = len(timesteps)
var = 3

values = rp_n3000_eps05_t10.values
varnames = rp_n3000_eps05_t10.columns
IDs = rp_n3000_eps05_t10.index

dataset = np.reshape(values, (n, var, temps))
dataset = np.swapaxes(dataset, 1, 2)
index = np.linspace(0,199,10,dtype=int)
dataset_05 = dataset[:1,tuple(index),:]
#dataset has 3 indices, first one is person, second one is visits, third one is variable

X_train, W_train = pr.weighter(dataset)

samp_trajs = X_train
samp_ts = tover
samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
samp_ts = torch.from_numpy(samp_ts).float().to(device)

samp_trajs_n3000_eps005_t10 = samp_trajs[:1,tuple(index),:]
W_train_n3000_eps005_t10 = W_train[:1,tuple(index),:]

n = len(rp_n3000_eps075_t10) #number of persons
#timesteps equal to months
timesteps = np.linspace(0,40,num=200)
tover = timesteps/200 #relative values of time
temps = len(timesteps)
var = 3

values = rp_n3000_eps075_t10.values
varnames = rp_n3000_eps075_t10.columns
IDs = rp_n3000_eps075_t10.index

dataset = np.reshape(values, (n, var, temps))
dataset = np.swapaxes(dataset, 1, 2)
dataset_075 = dataset[:1,tuple(index),:]
#dataset has 3 indices, first one is person, second one is visits, third one is variable

X_train, W_train = pr.weighter(dataset)

samp_trajs = X_train
samp_ts = tover
samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
samp_ts = torch.from_numpy(samp_ts).float().to(device)

samp_trajs_n3000_eps001_t10 = samp_trajs[:1,tuple(index),:]
W_train_n3000_eps001_t10 = W_train[:1,tuple(index),:]

n = len(rp_n3000_eps100_t10) #number of persons
#timesteps equal to months
timesteps = np.linspace(0,40,num=200)
tover = timesteps/200 #relative values of time
temps = len(timesteps)
var = 3

values = rp_n3000_eps100_t10.values
varnames = rp_n3000_eps100_t10.columns
IDs = rp_n3000_eps100_t10.index

dataset = np.reshape(values, (n, var, temps))
dataset = np.swapaxes(dataset, 1, 2)
dataset_100 = dataset[:1,tuple(index),:]
#dataset has 3 indices, first one is person, second one is visits, third one is variable

X_train, W_train = pr.weighter(dataset)

samp_trajs = X_train
samp_ts = tover
samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
samp_ts = torch.from_numpy(samp_ts).float().to(device)

samp_trajs_n3000_eps01_t10 = samp_trajs[:1,tuple(index),:]
W_train_n3000_eps01_t10 = W_train[:1,tuple(index),:]


beta=0.2
gamma=1/10
static = np.empty((1000,2))
static[:] = np.array((beta,gamma))

static = torch.from_numpy(static)

static_types=np.array([['real',1],['real',1]],dtype=np.object)

static_missing = None

negtime = False


xs_pos_posterior_n3000_eps05_t10 = genposteriornode_n3000_eps05_t10['Generation_values'][:,:,:]*1000
ts_pos_posterior_n3000_eps05_t10 = genposteriornode_n3000_eps05_t10['Generation_time']

xs_pos_rec_n3000_eps05_t10 = rpnode_n3000_eps05_t10['Recon_values'][:,:,:]*1000
ts_pos_rec_n3000_eps05_t10 = rpnode_n3000_eps05_t10['Recon_time']

xs_pos_posterior_n3000_eps075_t10 = genposteriornode_n3000_eps075_t10['Generation_values'][:,:,:]*1000
ts_pos_posterior_n3000_eps075_t10 = genposteriornode_n3000_eps075_t10['Generation_time']

xs_pos_rec_n3000_eps075_t10 = rpnode_n3000_eps075_t10['Recon_values'][:,:,:]*1000
ts_pos_rec_n3000_eps075_t10 = rpnode_n3000_eps075_t10['Recon_time']

xs_pos_posterior_n3000_eps100_t10 = genposteriornode_n3000_eps100_t10['Generation_values'][:,:,:]*1000
ts_pos_posterior_n3000_eps100_t10 = genposteriornode_n3000_eps100_t10['Generation_time']

xs_pos_rec_n3000_eps100_t10 = rpnode_n3000_eps100_t10['Recon_values'][:,:,:]*1000
ts_pos_rec_n3000_eps100_t10 = rpnode_n3000_eps100_t10['Recon_time']


samp_ts=samp_ts*200

dataset=dataset

varnames = ["Susceptible","Infected","Removed"]

#correctdata=correctdata[:,tuple(index),:]


for i in range(3):

    varnumber = i


    #if i == 0:
    #    legend=True
    #else:
    #    legend=False
        
    legend=True

    vis = m.visualizationSIR_eps_median(correctdata ,samp_ts,xs_pos_posterior_n3000_eps05_t10,ts_pos_posterior_n3000_eps05_t10, xs_pos_posterior_n3000_eps075_t10, ts_pos_posterior_n3000_eps075_t10,xs_pos_posterior_n3000_eps100_t10, ts_pos_posterior_n3000_eps100_t10,varnumber,varnames, negtime,device,labels=["eps=50%","eps=75%","eps=100%"],legend=legend)

    plt.savefig('./Median_posterior_eps_var_normed_' + str(i) + '.png', dpi=500)

    plt.savefig('./Median_posterior_eps/Median_posterior_eps_var_normed_' + varnames[i] + '.png', dpi=500)

