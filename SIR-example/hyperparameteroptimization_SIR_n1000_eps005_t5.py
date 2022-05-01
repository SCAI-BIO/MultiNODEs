import os
import numpy as np
import torch
import pandas as pd
import processing as pr

from sklearn.model_selection import KFold

import optuna

import main as m

def time_objective(trial):
    
    max_epochs=10000
    
    lr =  trial.suggest_uniform('lr',0.0001,0.01)
    
    latent_dim =  trial.suggest_uniform('latent_dim',0,4)
    
    nhidden =  trial.suggest_uniform('nhidden',0,4)
    
    batchsize =  trial.suggest_uniform('batchsize',0,1)
    
    activation_ode =  trial.suggest_categorical('activation_ode',['tanh','relu','none'])
    
    num_odelayers =  trial.suggest_int('num_odelayers',1,6,1)
    
    enctype =  trial.suggest_categorical('enctype',['LSTM','RNN'])
    
    if (enctype =='RNN'):
        activation_rnn =  trial.suggest_categorical('activation_rnn',['tanh','relu','none'])
    else:
        activation_rnn = None
    
    activation_dec =  trial.suggest_categorical('activation_dec',['tanh','relu','none'])
    
    dropout_dec =  trial.suggest_uniform('dropout_dec',0,1)
    
    rnn_nhidden_enc =  trial.suggest_uniform('rnn_nhidden_enc',0,1)
    
    rnn_nhidden_dec =  trial.suggest_uniform('rnn_nhidden_dec',0,1)
    
    s_dim_static = trial.suggest_int('s_dim_static',1,6,1)
    
    z_dim_static = trial.suggest_int('z_dim_static',1,6,1)
    
    scaling_ELBO = trial.suggest_uniform('scaling_ELBO',0,2)
    
    batch_norm_static=False  
    
    avg_loss = 0
    
    n_fold = 5
    kfold = KFold(n_splits=n_fold, shuffle=True) 
    
    fold = 0
    
    dectype = 'orig'
  
    train = np.asarray(samp_trajs[:750,:,:])
    val = np.asarray(samp_trajs[750:,:,:])
    
    train_wei = np.asarray(W_train[:750,:,:])
    val_wei = np.asarray(W_train[750:,:,:])
    
    static_train = np.asarray(static[:750,:])
    static_test = np.asarray(static[750:,:])
    
    staticdata_onehot_train = np.asarray(staticdata_onehot[:750,:])
    staticdata_onehot_test = np.asarray(staticdata_onehot[750:,:])
    
    static_true_miss_mask_train = np.asarray(static_true_miss_mask[:750,:])
    static_true_miss_mask_test = np.asarray(static_true_miss_mask[750:,:])
    
    train = torch.from_numpy(train).float().to(device)
    val = torch.from_numpy(val).float().to(device)
    
    static_train = torch.from_numpy(static_train).float().to(device)
    static_test = torch.from_numpy(static_test).float().to(device)
    
    rec_loss, loss_rec_stat, log_p_x, epochs = m.hypopt_early(solver, max_epochs, lr, train_dir, device, train, val, samp_ts, latent_dim, nhidden, obs_dim, batchsize, activation_ode, num_odelayers, W_train = train_wei,W_test=val_wei, enctype = enctype,dectype = dectype, rnn_nhidden_enc=rnn_nhidden_enc,activation_rnn = activation_rnn,rnn_nhidden_dec=rnn_nhidden_dec,activation_dec = activation_dec,dropout_dec = dropout_dec,static_train=static_train,static_test=static_test,staticdata_onehot_train=staticdata_onehot_train,staticdata_onehot_test=staticdata_onehot_test,staticdata_types_dict=staticdata_types_dict,static_true_miss_mask_train=static_true_miss_mask_train,static_true_miss_mask_test=static_true_miss_mask_test,s_dim_static=s_dim_static,z_dim_static=z_dim_static,scaling_ELBO=scaling_ELBO,batch_norm_static=batch_norm_static,patience=100)
    
    rec_loss = rec_loss.item()
    
    trial.set_user_attr("nepochs",epochs)
    
    #print(trial.number)
    return rec_loss

if __name__ == "__main__":

    solver = '/home/pwendland/Downloads/torch_ACA-dense_state2/'

    train_dir = '../../train'
    
    # check, whether cuda is available or not
    gpu = 0
    device = torch.device('cuda:' + str(gpu)
                          if torch.cuda.is_available() else 'cpu')
    
    # initiate model
    
    mypath = '/home/bio/groupshare/pwendland/masterarbeit-philipp/NeuralODE_code/Analyse/SIR'
    #mypath = r'C:\Users\User\Documents\GitHub\masterarbeit-philipp\NeuralODE_code\Analyse\SIR'
    #------ Raw Data -------
    
    varSIR = pd.read_csv(mypath + '/SIR_n1000_eps005.csv',
                       sep=',',header=0, 
                       index_col=0, engine='python')
    
    n = len(varSIR) #number of persons
    #timesteps equal to months
    timesteps = np.linspace(0, 40, 200)
    
    tover = timesteps/200 #relative values of time
    temps = len(timesteps)
    var = 3
    
    values = varSIR.values
    varnames = varSIR.columns
    IDs = varSIR.index
    
    dataset = np.reshape(values, (n, var, temps))
    dataset = np.swapaxes(dataset, 1, 2)
    #dataset has 3 indices, first one is person, second one is visits, third one is variable
    
    X_train, W_train = pr.weighter(dataset)
    #zeros are the missing values, W_train describes, whether a value is missing or not
    
    #logFolder = '/Performance/logs/TimeDependent/DataPred/ODENN/bmode_data/'
    
    beta=0.2
    gamma=1/10
    static = np.empty((1000,2))
    static[:] = np.array((beta,gamma))
    
    static = torch.from_numpy(static)
    
    static_types=np.array([['real',1],['real',1]],dtype=np.object)
    
    static_missing = None
    
    staticdata_onehot, staticdata_types_dict, static_miss_mask, static_true_miss_mask, static_n_samples = m.read_data(static,static_types,static_missing)
    
    samp_trajs = X_train
    
    index = np.linspace(0,199,5,dtype=int)
    samp_trajs = samp_trajs[0:1000,tuple(index),:]
    
    W_train = W_train[0:1000,tuple(index),:]
    
    samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
    samp_ts = torch.from_numpy(index/199*40).float().to(device)
    
    obs_dim = var
    
    samp_trajs = samp_trajs/1000
    
    static = static.numpy()    
    
    print("Connecting to URL {} to load study {}".format(os.environ['DBURL'], os.environ['STUDY']))
    study = optuna.load_study(sampler=optuna.samplers.TPESampler(),study_name=os.environ['STUDY'], storage=os.environ['DBURL'])
    
    
    #n_trial is the stopping criterion, n_jobs is the number of parallel used cpus/gpus
    study.optimize(time_objective, n_trials=1, n_jobs=1)
    
    print("Number of finished trials: ", len(study.trials))
    
    print("Best trial:")
    trial = study.best_trial
    
    print("  rec_loss: ", trial.value)
    
    
    
    #
    #with h5py.File(mypath+params_name+'pred.h5', 'w') as hf:
    #
    #    hf.create_dataset("pred",  data=pred)
    #    hf.create_dataset("X_test",  data=X_test)
    #    hf.create_dataset("W_test",  data=W_test)
    #    
    #    hf.close()
    
    
