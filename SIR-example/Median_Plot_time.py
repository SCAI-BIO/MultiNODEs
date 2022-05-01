import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import processing as pr

import main as m

node_dir='/home/pwendland/Dokumente/GitHub/masterarbeit-philipp/NeuralODE_code/SIR/SIR_synth_new'
node_dir='C:/Users/wendl/Documents/Python/Neural_ODE/SIR_synth_new'


rpnode_n3000_eps005_t10 = torch.load(node_dir+'/Reconstruction_n3000_eps005_t10_normed.pth')

rpnode_n3000_eps005_t5 = torch.load(node_dir+'/Reconstruction_n3000_eps005_t5_normed.pth')

rpnode_n3000_eps005_t100 = torch.load(node_dir+'/Reconstruction_n3000_eps005_t100_normed.pth')

# rpnode_n3000_eps2_t10 = torch.load(node_dir+'/Reconstruction_n3000_eps2_t10.pth')

# rpnode_n3000_eps05_t10 = torch.load(node_dir+'/Reconstruction_n3000_eps05_t10.pth')

#rpnode_n7000_eps1_t10 = torch.load(node_dir+'/Reconstruction_n7000_eps1_t10.pth')

#rpnode_n2100_eps1_t10 = torch.load(node_dir+'/Reconstruction_n2100_eps1_t10.pth')


genposteriornode_n3000_eps005_t10 = torch.load(node_dir+'/Generationposterior_n3000_eps005_t10_normed.pth')

genposteriornode_n3000_eps005_t5 = torch.load(node_dir+'/Generationposterior_n3000_eps005_t5_normed.pth')

genposteriornode_n3000_eps005_t100 = torch.load(node_dir+'/Generationposterior_n3000_eps005_t100_normed.pth')

# genposteriornode_n3000_eps2_t10 = torch.load(node_dir+'/Generationposterior_n3000_eps2_t10.pth')

# genposteriornode_n3000_eps05_t10 = torch.load(node_dir+'/Generationposterior_n3000_eps05_t10.pth')

#genposteriornode_n7000_eps1_t10 = torch.load(node_dir+'/Generationposterior_n7000_eps1_t10.pth')

#genposteriornode_n2100_eps1_t10 = torch.load(node_dir+'/Generationposterior_n2100_eps1_t10.pth')

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

rp = pd.read_csv(node_dir + '/SIR-data_correct_n3000_eps005.csv',sep=',',header=0,
                   index_col=0, engine='python')

device = 'cpu'

n = len(rp) #number of persons
#timesteps equal to months
timesteps = np.linspace(0,40,num=200)
tover = timesteps/200 #relative values of time
temps = len(timesteps)
var = 3

values = rp.values
varnames = rp.columns
IDs = rp.index

dataset = np.reshape(values, (n, var, temps))
dataset = np.swapaxes(dataset, 1, 2)
dataset = dataset[:,:,:]
#dataset has 3 indices, first one is person, second one is visits, third one is variable

X_train, W_train = pr.weighter(dataset)

samp_trajs = X_train
samp_ts = tover
samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
samp_ts = torch.from_numpy(samp_ts).float().to(device)

samp_trajs = samp_trajs[:1,:,:]
W_train = W_train[:1,:,:]

samp_trajs=samp_trajs/1000

negtime = False

xs_pos_posterior_n3000_eps005_t10 = genposteriornode_n3000_eps005_t10['Generation_values'][:,:,:]*1000
ts_pos_posterior_n3000_eps005_t10 = genposteriornode_n3000_eps005_t10['Generation_time']

xs_pos_rec_n3000_eps005_t10 = rpnode_n3000_eps005_t10['Recon_values'][:,:,:]*1000
ts_pos_rec_n3000_eps005_t10 = rpnode_n3000_eps005_t10['Recon_time']

xs_pos_posterior_n3000_eps005_t5 = genposteriornode_n3000_eps005_t5['Generation_values'][:,:,:]*1000
ts_pos_posterior_n3000_eps005_t5 = genposteriornode_n3000_eps005_t5['Generation_time']

xs_pos_rec_n3000_eps005_t5 = rpnode_n3000_eps005_t5['Recon_values'][:,:,:]*1000
ts_pos_rec_n3000_eps005_t5 = rpnode_n3000_eps005_t5['Recon_time']

xs_pos_posterior_n3000_eps005_t100 = genposteriornode_n3000_eps005_t100['Generation_values'][:,:,:]*1000
ts_pos_posterior_n3000_eps005_t100 = genposteriornode_n3000_eps005_t100['Generation_time']

xs_pos_rec_n3000_eps005_t100 = rpnode_n3000_eps005_t100['Recon_values'][:,:,:]*1000
ts_pos_rec_n3000_eps005_t100 = rpnode_n3000_eps005_t100['Recon_time']


samp_ts=samp_ts*200

dataset=dataset

varnames = ["Susceptible","Infected","Removed"]

index_5 = np.linspace(0,199,5,dtype=int)
index_10 = np.linspace(0,199,10,dtype=int)
index_100 = np.linspace(0,199,100,dtype=int)


for i in range(3):


    if i == 0:
        legend=True
    else:
        legend=False

    varnumber = i

    vis = m.visualizationSIR_t_median(samp_ts,dataset, xs_pos_posterior_n3000_eps005_t5, ts_pos_posterior_n3000_eps005_t5,xs_pos_posterior_n3000_eps005_t10, ts_pos_posterior_n3000_eps005_t10,xs_pos_posterior_n3000_eps005_t100,ts_pos_posterior_n3000_eps005_t100,varnumber,varnames, negtime,device,labels=["t=5","t=10","t=100"], legend=legend)

    plt.savefig('./Median_posterior_t_var_normed_' + str(i) + '.png', dpi=500)

    plt.savefig('./Median_posterior/Median_posterior_t_var_normed_' + varnames[i] + '.png', dpi=500)



