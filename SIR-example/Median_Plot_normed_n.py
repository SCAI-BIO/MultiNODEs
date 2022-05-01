import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import processing as pr

import main as m

node_dir='/home/pwendland/Dokumente/GitHub/masterarbeit-philipp/NeuralODE_code/SIR/SIR_synth_new'
node_dir='C:/Users/wendl/Documents/Python/Neural_ODE/SIR_synth_new'


#rpnode_n3000_eps005_t10 = torch.load(node_dir+'/Reconstruction_n3000_eps005_t10_normed.pth')

#rpnode_n3000_eps005_t5 = torch.load(node_dir+'/Reconstruction_n3000_eps005_t5_normed.pth')

#rpnode_n3000_eps005_t100 = torch.load(node_dir+'/Reconstruction_n3000_eps005_t100_normed.pth')

# rpnode_n3000_eps2_t10 = torch.load(node_dir+'/Reconstruction_n3000_eps2_t10.pth')

rpnode_n3000_eps005_t10 = torch.load(node_dir+'/Reconstruction_n3000_eps005_t10_normed.pth')

rpnode_n7000_eps005_t10 = torch.load(node_dir+'/Reconstruction_n7000_eps005_t10_normed.pth')

rpnode_n2100_eps005_t10 = torch.load(node_dir+'/Reconstruction_n2100_eps005_t10_normed.pth')


#genposteriornode_n3000_eps005_t10 = torch.load(node_dir+'/Generationposterior_n3000_eps005_t10_normed.pth')

#genposteriornode_n3000_eps005_t5 = torch.load(node_dir+'/Generationposterior_n3000_eps005_t5_normed.pth')

#genposteriornode_n3000_eps005_t100 = torch.load(node_dir+'/Generationposterior_n3000_eps005_t100_normed.pth')

# genposteriornode_n3000_eps2_t10 = torch.load(node_dir+'/Generationposterior_n3000_eps2_t10.pth')

genposteriornode_n3000_eps005_t10 = torch.load(node_dir+'/Generationposterior_n3000_eps005_t10_normed.pth')

genposteriornode_n7000_eps005_t10 = torch.load(node_dir+'/Generationposterior_n7000_eps005_t10_normed.pth')

genposteriornode_n2100_eps005_t10 = torch.load(node_dir+'/Generationposterior_n2100_eps005_t10_normed.pth')

rp_n3000_eps1_t10 = pd.read_csv(node_dir + '/SIR-data_correct_n3000_eps005.csv',sep=',',header=0,
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

n = len(rp_n3000_eps1_t10) #number of persons
#timesteps equal to months
timesteps = np.linspace(0,40,num=200)
tover = timesteps/200 #relative values of time
temps = len(timesteps)
var = 3

values = rp_n3000_eps1_t10.values
varnames = rp_n3000_eps1_t10.columns
IDs = rp_n3000_eps1_t10.index

dataset = np.reshape(values, (n, var, temps))
dataset = np.swapaxes(dataset, 1, 2)

index = np.linspace(0,199,10,dtype=int)
dataset_3000 = dataset[:1,tuple(index),:]
#dataset has 3 indices, first one is person, second one is visits, third one is variable

X_train, W_train = pr.weighter(dataset)

samp_trajs = X_train
samp_ts = tover
samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
samp_ts = torch.from_numpy(samp_ts).float().to(device)

W_train_n3000_eps1_t10 = W_train[:1,tuple(index),:]

beta=0.2
gamma=1/10
static = np.empty((1000,2))
static[:] = np.array((beta,gamma))

static = torch.from_numpy(static)

static_types=np.array([['real',1],['real',1]],dtype=np.object)

static_missing = None

samp_ts=samp_ts*200

xs_pos_posterior_n3000_eps005_t10 = genposteriornode_n3000_eps005_t10['Generation_values'][:,:,:]*1000
ts_pos_posterior_n3000_eps005_t10 = genposteriornode_n3000_eps005_t10['Generation_time']

xs_pos_rec_n3000_eps005_t10 = rpnode_n3000_eps005_t10['Recon_values'][:,:,:]*1000
ts_pos_rec_n3000_eps005_t10 = rpnode_n3000_eps005_t10['Recon_time']

xs_pos_posterior_n2100_eps005_t10 = genposteriornode_n2100_eps005_t10['Generation_values'][:,:,:]*1000
ts_pos_posterior_n2100_eps005_t10 = genposteriornode_n2100_eps005_t10['Generation_time']

xs_pos_rec_n2100_eps005_t10 = rpnode_n2100_eps005_t10['Recon_values'][:,:,:]*1000
ts_pos_rec_n2100_eps005_t10 = rpnode_n2100_eps005_t10['Recon_time']

xs_pos_posterior_n7000_eps005_t10 = genposteriornode_n7000_eps005_t10['Generation_values'][:,:,:]*1000
ts_pos_posterior_n7000_eps005_t10 = genposteriornode_n7000_eps005_t10['Generation_time']

xs_pos_rec_n7000_eps005_t10 = rpnode_n7000_eps005_t10['Recon_values'][:,:,:]*1000
ts_pos_rec_n7000_eps005_t10 = rpnode_n7000_eps005_t10['Recon_time']


varnames = ["Susceptible","Infected","Removed"]
negtime=False

index = np.linspace(0,199,10,dtype=int)
#correctdata=correctdata[:,tuple(index),:]


for i in range(3):

    varnumber = i

    #if i == 0:
    #    legend=True
    #else:
    #    legend=False
    legend=True

    vis = m.visualizationSIR_median(correctdata, samp_ts,dataset_3000, samp_ts[index],W_train_n3000_eps1_t10, xs_pos_posterior_n2100_eps005_t10, ts_pos_posterior_n2100_eps005_t10,xs_pos_posterior_n3000_eps005_t10, ts_pos_rec_n3000_eps005_t10,xs_pos_posterior_n7000_eps005_t10,ts_pos_rec_n7000_eps005_t10,varnumber,varnames, negtime,device,labels=["n=100","n=1000","n=5000"],legend=legend)

    plt.savefig('./Median_posterior_n_var_normed_' + str(i) + '.png', dpi=500)

    plt.savefig('./Median_posterior_n/Median_posterior_n_var_normed_' + varnames[i] + '.png', dpi=500)



