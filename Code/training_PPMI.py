import numpy as np
import torch
import pandas as pd
import processing as pr

import main as m


solver = 'Adjoint' #using of adjoint method
train_dir = '/home/pwendland/Dokumente/GitHub/masterarbeit-philipp/NeuralODE_code'

# check, whether cuda is available or not
gpu = 0
device = torch.device('cuda:' + str(gpu)
                      if torch.cuda.is_available() else 'cpu')

# initiate model
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
#zeros are the missing values, W_train describes, whether a value is missing or not

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

#ec': 'none', 'actvation_ode': 'none', 'actvation_rnn': 'none', 'batch_norm_static': 'False', 'batchsize': 0.6400910825235765, 'dectype': 'RNN', 'dropout_dec': 0.4951111384939868, 'enctype': 'RNN', 'latent_dim': 1.8837462054054002, 'lr': 0.0015680290621642827, 'nepochs': 1900, 'nhidden': 4.584466407303982, 'num_odelayers': 1, 'rnn_nhidden_dec': 0.6974659448314735, 'rnn_nhidden_enc': 0.49866725741363976, 's_dim_static': 6, 'scaling_ELBO': 0.9675595125571386, 'z_dim_static': 3}
activation_dec='none'
activation_ode='none'
activation_rnn='none'
batch_norm_static = False
batchsize = (0.6400910825235765*274)/354
dectype='orig'
dropout_dec = 0.4951111384939868
enctype = 'RNN'
latent_dim = 1.8837462054054002
lr = 0.0015680290621642827
nepochs = 1900
nhidden = 4.584466407303982
num_odelayers = 1
rnn_nhidden_dec = 0.6974659448314735
rnn_nhidden_enc = 0.49866725741363976
s_dim_static = 6
scaling_ELBO = 0.9675595125571386
z_dim_static = 3

loss,rec_loss,kl_loss= m.training(solver, nepochs, lr, train_dir, device, samp_trajs, samp_ts, latent_dim, nhidden, obs_dim, batchsize, activation_ode = activation_ode,num_odelayers = num_odelayers, W_train = W_train, enctype = enctype,dectype = dectype, rnn_nhidden_enc=rnn_nhidden_enc,activation_rnn = activation_rnn,rnn_nhidden_dec=rnn_nhidden_dec,activation_dec = activation_dec,dropout_dec = dropout_dec,static=static,staticdata_onehot=staticdata_onehot,staticdata_types_dict=staticdata_types_dict,static_true_miss_mask=static_true_miss_mask,s_dim_static=s_dim_static,z_dim_static=z_dim_static,scaling_ELBO=scaling_ELBO,batch_norm_static=batch_norm_static)

#limits of simulated time
timemax = 1
timemin = 0

#describes the level of noise in the latent room (to reconstruct the original data set sigmalong=1 and sigmastat=2)
sigmalong=2
sigmastat=2
#describes how often a complete population is generated
N=1

xs_pos_rec, ts_pos, out_rec, qz0_mean_rec, qz0_sigma_rec, epsilon_rec, qz0_meanstat_rec, qz0sigma_rec,epsilonstat_rec = m.reconRP(solver, train_dir, device,samp_trajs,samp_ts, latent_dim, nhidden, obs_dim, activation_ode = activation_ode,num_odelayers=num_odelayers,W_train=W_train, enctype = enctype,dectype = dectype, rnn_nhidden_enc=rnn_nhidden_enc,activation_rnn = activation_rnn,rnn_nhidden_dec=rnn_nhidden_dec,activation_dec = activation_dec,static=static,staticdata_onehot=staticdata_onehot,staticdata_types_dict=staticdata_types_dict,static_true_miss_mask=static_true_miss_mask,s_dim_static=s_dim_static,z_dim_static=z_dim_static,batch_norm_static=batch_norm_static, timemax = 1, negtime = False, timemin = None,num=2000)

xs_pos_prior, ts_pos_prior, out_prior = m.generationprior(solver, train_dir, device,samp_trajs,samp_ts, latent_dim, nhidden, obs_dim, activation_ode = activation_ode,num_odelayers=num_odelayers, dectype = dectype,rnn_nhidden_dec=rnn_nhidden_dec,activation_dec = activation_dec,static=static,staticdata_onehot=staticdata_onehot,staticdata_types_dict=staticdata_types_dict,s_dim_static=s_dim_static,z_dim_static=z_dim_static,s_prob=None, timemax = 1, timemin = None,num=2000)

xs_pos_posterior, ts_pos_posterior, out_posterior, qz0_mean_posterior, logsigma_posterior, epsilon_posterior, qz0_meanstat_posterior, logsigmastat_posterior,epsilonstat_posterior = m.generationPosterior(solver, train_dir, device,samp_trajs,samp_ts, latent_dim, nhidden, obs_dim, activation_ode = activation_ode,num_odelayers=num_odelayers,W_train=W_train, enctype = enctype,dectype = dectype, rnn_nhidden_enc=rnn_nhidden_enc,activation_rnn = activation_rnn,rnn_nhidden_dec=rnn_nhidden_dec,activation_dec = activation_dec,static=static,staticdata_onehot=staticdata_onehot,staticdata_types_dict=staticdata_types_dict,static_true_miss_mask=static_true_miss_mask,s_dim_static=s_dim_static,z_dim_static=z_dim_static,batch_norm_static=batch_norm_static, timemax = 1, negtime = False, timemin = None,num=2000, sigmalong=sigmalong,sigmastat=sigmastat,N=N)

varnamessave = ["MedicalHistory_WGTKG","MedicalHistory_HTCM","MedicalHistory_TEMPC","MedicalHistory_SYSSUP","MedicalHistory_DIASUP","MedicalHistory_HRSUP","MedicalHistory_SYSSTND","MedicalHistory_DIASTND","MedicalHistory_HRSTND","NonMotor_DVT_TOTAL_RECALL","NonMotor_DVS_LNS","NonMotor_BJLOT","NonMotor_ESS","NonMotor_GDS","NonMotor_QUIP","NonMotor_RBD","NonMotor_SCOPA","NonMotor_SFT","NonMotor_STA","NonMotor_STAI.State","NonMotor_STAI.Trait","UPDRS_UPDRS1","UPDRS_UPDRS2","UPDRS_UPDRS3","SA_CSF_CSF.Alpha.synuclein"]
varnamesstatic = ["CSF_Abeta.42_VIS00","CSF_p.Tau181P_VIS00","CSF_Total.tau_VIS00","CSF_tTau.Abeta_VIS00","CSF_pTau.Abeta_VIS00","CSF_pTau.tTau_VIS00","Biological_ALDH1A1..rep.1._VIS00","Biological_ALDH1A1..rep.2._VIS00","Biological_GAPDH..rep.1._VIS00","Biological_GAPDH..rep.2._VIS00","Biological_HSPA8..rep.1._VIS00","Biological_HSPA8..rep.2._VIS00","Biological_LAMB2..rep.1._VIS00","Biological_LAMB2..rep.2._VIS00","Biological_PGK1..rep.1._VIS00","Biological_PGK1..rep.2._VIS00","Biological_PSMC4..rep.1._VIS00","Biological_PSMC4..rep.2._VIS00","Biological_SKP1..rep.1._VIS00","Biological_SKP1..rep.2._VIS00","Biological_UBE2K..rep.1._VIS00","Biological_UBE2K..rep.2._VIS00","Biological_Serum.IGF.1_VIS00","PatDemo_HISPLAT","PatDemo_RAINDALS","PatDemo_RAASIAN","PatDemo_RABLACK","PatDemo_RAWHITE","PatDemo_RANOS","PatDemo_EDUCYRS","PatDemo_HANDED","PatDemo_GENDER","PatPDHist_BIOMOMPD","PatPDHist_BIODADPD","PatPDHist_FULSIBPD","PatPDHist_MAGPARPD","PatPDHist_PAGPARPD","PatPDHist_MATAUPD","PatPDHist_PATAUPD","SA_Imaging_VIS00","SA_Enrollment_Age","SA_CADD_filtered_impact_scores_VIS00","SA_Polygenetic_risk_scores_VIS00"]

torch.save({'Recon_values': xs_pos_rec,
            'Recon_time': ts_pos,
            'varnames': varnamessave,
            'orig_time': samp_ts,
            'static_values': out_rec,
            'varnamesstatic':varnamesstatic,
            'W_train': W_train}
           ,train_dir + '/Reconstruction_new.pth')

torch.save({'Generation_values': xs_pos_prior,
            'Generation_time':ts_pos_prior,
            'varnames': varnamessave,
            'orig_time': samp_ts,
            'static_values': out_prior,
            'varnamesstatic':varnamesstatic}
           ,train_dir + '/Generationprior_new.pth')

torch.save({'Generation_values': xs_pos_posterior,
            'Generation_time':ts_pos_posterior,
            'varnames': varnamessave,
            'orig_time': samp_ts,
            'static_values': out_posterior,
            'varnamesstatic':varnamesstatic}
           ,train_dir + '/Generationposterior_new.pth')


