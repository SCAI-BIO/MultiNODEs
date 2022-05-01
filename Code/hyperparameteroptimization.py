import os
import numpy as np
import torch
import pandas as pd
import processing as pr

from sklearn.model_selection import KFold

import optuna

import main as m


def time_objective(trial):
    
    max_epochs = 5000
    
    lr =  trial.suggest_uniform('lr',0.0001,0.01)
    
    latent_dim =  trial.suggest_uniform('latent_dim',1,2)
    
    nhidden =  trial.suggest_uniform('nhidden',0,10)
    
    batchsize =  trial.suggest_uniform('batchsize',0,1)
    
    activation_ode =  trial.suggest_categorical('actvation_ode',['tanh','relu','none'])
    
    num_odelayers =  trial.suggest_int('num_odelayers',1,8,1)
    
    enctype =  trial.suggest_categorical('enctype',['LSTM','RNN'])
    
    if (enctype =='RNN'):
        activation_rnn =  trial.suggest_categorical('actvation_rnn',['tanh','relu','none'])
    else:
        activation_rnn = None
    
    activation_dec =  trial.suggest_categorical('actvation_dec',['tanh','relu','none'])
    
    dropout_dec =  trial.suggest_uniform('dropout_dec',0,1)
    
    rnn_nhidden_enc =  trial.suggest_uniform('rnn_nhidden_enc',0,1)
    
    rnn_nhidden_dec =  trial.suggest_uniform('rnn_nhidden_dec',0,1)
    
    s_dim_static = trial.suggest_int('s_dim_static',1,6,1)
    
    z_dim_static = trial.suggest_int('z_dim_static',1,6,1)
    
    scaling_ELBO = trial.suggest_uniform('scaling_ELBO',0,2)
    
    batch_norm_static=True    
    
    avg_loss = 0
    
    n_fold = 5
    kfold = KFold(n_splits=n_fold, shuffle=True) 
    
    fold = 0
    
    dectype = 'orig'
    
    nepochs = 0
  
    for train_idx, val_idx in kfold.split(X_train):
    
        #print(train_idx)
        
        fold +=1
    
        train = np.asarray([X_train[x] for x in range(len(X_train)) if x in train_idx])
        val = np.asarray([X_train[x] for x in range(len(X_train)) if x in val_idx])
        
        train_wei = np.asarray([W_train[x] for x in range(len(W_train)) if x in train_idx])
        val_wei = np.asarray([W_train[x] for x in range(len(W_train)) if x in val_idx])
        
        static_train = np.asarray([static[x] for x in range(len(static)) if x in train_idx])
        static_test = np.asarray([static[x] for x in range(len(static)) if x in val_idx])
        
        staticdata_onehot_train = np.asarray([staticdata_onehot[x] for x in range(len(staticdata_onehot)) if x in train_idx])
        staticdata_onehot_test = np.asarray([staticdata_onehot[x] for x in range(len(staticdata_onehot)) if x in val_idx])
        
        static_true_miss_mask_train = np.asarray([static_true_miss_mask[x] for x in range(len(static_true_miss_mask)) if x in train_idx])
        static_true_miss_mask_test = np.asarray([static_true_miss_mask[x] for x in range(len(static_true_miss_mask)) if x in val_idx])
        
        train = torch.from_numpy(train).float().to(device)
        val = torch.from_numpy(val).float().to(device)
        
        static_train = torch.from_numpy(static_train).float().to(device)
        static_test = torch.from_numpy(static_test).float().to(device)
        
        rec_loss, loss_rec_stat, log_p_x, epochs = m.hypopt_early(solver,max_epochs, lr, train_dir, device, train, val, samp_ts, latent_dim, nhidden, obs_dim, batchsize, activation_ode, num_odelayers, W_train = train_wei,W_test=val_wei, enctype = enctype,dectype = dectype, rnn_nhidden_enc=rnn_nhidden_enc,activation_rnn = activation_rnn,rnn_nhidden_dec=rnn_nhidden_dec,activation_dec = activation_dec,dropout_dec = dropout_dec,static_train=static_train,static_test=static_test,staticdata_onehot_train=staticdata_onehot_train,staticdata_onehot_test=staticdata_onehot_test,staticdata_types_dict=staticdata_types_dict,static_true_miss_mask_train=static_true_miss_mask_train,static_true_miss_mask_test=static_true_miss_mask_test,s_dim_static=s_dim_static,z_dim_static=z_dim_static,scaling_ELBO=scaling_ELBO,batch_norm_static=batch_norm_static,patience=100)

        avg_loss += rec_loss.item()
        
        nepochs += epochs
    
    obj_loss = np.round(avg_loss / n_fold, 2)
    
    nepochs = np.round(nepochs / n_fold,2) 
    
    trial.set_user_attr("nepochs",nepochs)
    
    #print(trial.number)
    return obj_loss

if __name__ == "__main__":
    
    solver = 'Adjoint' #using of adjoint method
    #adjoint = '/home/bio/groupshare/torch_ACA-dense_state2/'

    train_dir = '../../train'
    
    # check, whether cuda is available or not
    gpu = 0
    device = torch.device('cuda:' + str(gpu)
                          if torch.cuda.is_available() else 'cpu')
    
    # initiate model
    
    mypath = '../../data'
    train_dir = '/home/bio/groupshare/pwendland/masterarbeit-philipp/NeuralODE_code'
    mypath = train_dir
    
    #------ Raw Data -------
    
    varPPMI = pd.read_csv(mypath + '/longitudinal.csv',
                       sep=',',header=0, 
                       index_col=0, engine='python')
    
    n = len(varPPMI) #number of persons
    #timesteps equal to months
    timesteps = np.array([0, 3, 6, 9, 12, 18, 24, 30, 36, 42, 48, 54])
    tover = timesteps/54 #relative values of time
    temps = len(timesteps)
    var = 25
    
    values = varPPMI.values
    varnames = varPPMI.columns
    IDs = varPPMI.index
    
    dataset = np.reshape(values, (n, var, temps))
    dataset = np.swapaxes(dataset, 1, 2)
    #dataset has 3 indices, first one is person, second one is visits, third one is variable
    
    X_train, W_train = pr.weighter(dataset)
    
    #X_train = X_train[:,:2,:]
    #W_train = W_train[:,:2,:]
    #tover = tover[:2]
    #zeros are the missing values, W_train describes, whether a value is missing or not
    
    #logFolder = '/Performance/logs/TimeDependent/DataPred/ODENN/bmode_data/'
    
    biolog = pd.read_csv(mypath + '/Biological_VIS00.csv',
                       sep=',',header=None, 
                       engine='python')
    biologval = torch.from_numpy(biolog.values).float().to(device)
    
    patdem = pd.read_csv(mypath + '/PatDemo_VIS00.csv',
                       sep=',',header=None, 
                       engine='python')
    patdemoval = torch.from_numpy(patdem.values).float().to(device)
    
    patpd = pd.read_csv(mypath + '/PatPDHist_VIS00.csv',
                       sep=',',header=None, 
                       engine='python')
    patpdval = torch.from_numpy(patpd.values).float().to(device)
    
    stalone = pd.read_csv(mypath + '/stalone_VIS00BL_nofill.csv',
                       sep=',',header=None, 
                       engine='python')
    staloneval = torch.from_numpy(stalone.values).float().to(device)
    
    modules = dict([('biolog',biologval),('patdemo',patdemoval),('patpd',patpdval),('stalone',staloneval)])
    
    static = torch.cat((biologval,patdemoval,patpdval,staloneval),dim=1)
    
    
    biologtypes = pd.read_csv(mypath + '/Biological_VIS00_types.csv',
                       sep=',',header=0, 
                       engine='python')
    
    patdemtypes = pd.read_csv(mypath + '/PatDemo_VIS00_types.csv',
                       sep=',',header=0, 
                       engine='python')
    
    patpdtypes = pd.read_csv(mypath + '/PatPDHist_VIS00_types.csv',
                       sep=',',header=0, 
                       engine='python')
    
    stalonetypes = pd.read_csv(mypath + '/stalone_VIS00_types.csv',
                       sep=',',header=0, 
                       engine='python')
    
    static_types=np.concatenate([biologtypes.values,patdemtypes.values,patpdtypes.values,stalonetypes.values],axis=0)
    
    biologmissing = pd.read_csv(mypath + '/Biological_VIS00_missing.csv',
                       sep=',',header=None, 
                       engine='python')
    biologmissing=biologmissing.values
    biologmissing[:,1] = biologmissing[:,1]
    
    patdemmissing = pd.read_csv(mypath + '/PatDemo_VIS00_missing.csv',
                       sep=',',header=None, 
                       engine='python')
    patdemmissing = patdemmissing.values
    patdemmissing[:,1] = patdemmissing[:,1]+biologval.shape[1]
    
    patpdmissing = pd.read_csv(mypath + '/PatPDHist_VIS00_missing.csv',
                       sep=',',header=None, 
                       engine='python')
    patpdmissing = patpdmissing.values
    patpdmissing[:,1] = patpdmissing[:,1]+biologval.shape[1]+patdemoval.shape[1]
    
    
    stalonemissing = pd.read_csv(mypath + '/stalone_VIS00_missing.csv',
                       sep=',',header=None, 
                       engine='python')
    stalonemissing = stalonemissing.values
    stalonemissing[:,1] = stalonemissing[:,1]+biologval.shape[1] + patdemoval.shape[1] + patpdval.shape[1]
    
    static_missing = np.concatenate([biologmissing,patdemmissing,patpdmissing,stalonemissing])
    
    staticdata_onehot, staticdata_types_dict, static_miss_mask, static_true_miss_mask, static_n_samples = m.read_data(static,static_types,static_missing)
    
    samp_trajs = X_train
    samp_ts = tover
    samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
    samp_ts = torch.from_numpy(samp_ts).float().to(device)
    
    obs_dim = var
    
    static = static.numpy()
    
    print("Connecting to URL {} to load study {}".format(os.environ['DBURL'], os.environ['STUDY']))
    study = optuna.load_study(sampler=optuna.samplers.TPESampler(),study_name=os.environ['STUDY'], storage=os.environ['DBURL'])
    
    
    #n_trial is the stopping criterion, n_jobs is the number of parallel used cpus/gpus
    study.optimize(time_objective, n_trials=1, n_jobs=1)
    
    print("Number of finished trials: ", len(study.trials))
    
    print("Best trial:")
    trial = study.best_trial
    
    print("  rec_loss: ", trial.value)
    
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
        file = open(train_dir+'/tries_lat.txt', 'a')
        file.write("    {}: {}".format(key, value))
        file.write("\n")
    file.write("rec_loss: "+str(trial.value))
    file.write("\n"+"------------"+"\n")
    file.close()
    
    dic = dict(trial.params)
    dic['value'] = trial.value
    
    df = pd.DataFrame.from_dict(data=dic,orient='index').to_csv(train_dir + '/tries_lat.csv',header=False)
    
    
    #
    #with h5py.File(mypath+params_name+'pred.h5', 'w') as hf:
    #
    #    hf.create_dataset("pred",  data=pred)
    #    hf.create_dataset("X_test",  data=X_test)
    #    hf.create_dataset("W_test",  data=W_test)
    #    
    #    hf.close()
    
    
