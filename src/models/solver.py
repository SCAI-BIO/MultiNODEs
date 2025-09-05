import imp
import os
import warnings
import numpy as np
import torch
from networks import *
from utils import RunningAverageMeter
warnings.filterwarnings('ignore')


class Solver(object):
    def __init__(self, config, dataloader, f_train=False):

        self.config = config

        self.device = torch.device('cuda:{}'.format(config.GPU[0])) if config.GPU else torch.device('cpu')
        self.dataloader = dataloader

        # Steps time are the same
        self.T = self.dataloader.dataset.get_T().to(self.device)

        # Some modifications to run inmeadiately after training
        self.f_train = f_train  # Just used for validation after training
        if self.f_train:
            self.config.epoch_init = self.config.num_epochs

        self.build_model()

    def build_model(self):

        A_init = self.initialize_imputation()
        self.Imp_Layer = VaderLayer(A_init).to(self.device)

        # config.latent_dim is the latent_dim of the ODE function,
        #   as percentage of config.n_long_var

        l_dim_number = int(np.round(self.config.latent_dim * self.config.n_long_var))
        nhidden_number = int(np.round(self.config.nhidden * l_dim_number))
        self.batch_size = self.config.batch_size
        self.l_dim_number = l_dim_number

        # ODE and Static NN if Static data is available
        if self.config.static_data:

            self.S_Enc = StatLayerEnc(
                self.config.s_vals_dim, self.config.s_onehot_dim,
                self.config.s_dim_static, self.config.z_dim_static).to(self.device)

            _, static_types, _ = self.dataloader.dataset.get_static()
            self.S_Dec = StatDecoder(
                self.config.s_vals_dim, static_types,
                self.config.s_dim_static, self.config.z_dim_static).to(self.device)

            self.static_types = static_types
            if self.config.batch_norm_static:
                self.s_onehot_types = self.dataloader.dataset.s_onehot_types

            self.ODE = LatentODEFunc(self.config,
                latent_dim=l_dim_number + self.config.z_dim_static,
                nhidden_number=nhidden_number).to(self.device)
        else:
            self.ODE = LatentODEFunc(self.config, latent_dim=l_dim_number,
                nhidden_number=nhidden_number).to(self.device)

        rnn_nhidden_max = self.config.n_long_var * len(self.T)
        rnn_nhidden_enc_number = int(np.round(self.config.rnn_nhidden_enc * rnn_nhidden_max))
        rnn_nhidden_dec_number = int(np.round(self.config.rnn_nhidden_dec * rnn_nhidden_max))
        self.rnn_nhidden_enc_number = rnn_nhidden_enc_number

        # Longitudinal Encoder
        if self.config.type_enc == 'LSTM':
            self.L_Enc = LSTMEncoder(self.config.n_long_var,
                rnn_nhidden_enc_number, 2 * l_dim_number).to(self.device)
        else:
            self.L_Enc = RecognitionRNN(2 * l_dim_number,
                self.config.n_long_var, rnn_nhidden_enc_number,
                self.config.act_rnn).to(self.device)

        # Longitudinal Decoder
        inp_dec = l_dim_number + self.config.z_dim_static if self.config.static_data else l_dim_number
        if self.config.type_dec == 'LSTM':
            self.L_Dec = LSTMDecoder(inp_dec, rnn_nhidden_dec_number,
                self.config.n_long_var).to(self.device)
        elif self.config.type_dec == 'RNN':
            self.L_Dec = RecognitionRNN(self.config.n_long_var, inp_dec,
                rnn_nhidden_dec_number, self.config.act_dec).to(self.device)
        else:
            self.L_Dec = Decoder(self.config, inp_dec,
                rnn_nhidden_dec_number).to(self.device)

        if 'train' in self.config.mode:
            self.get_optimizer()
            self.loss = RunningAverageMeter()

        print('Models were built')

        # It does not matter if it is for continue training
        #   or for loading the model to generate predictions
        if self.config.epoch_init != 1 or self.config.from_best:
            self.load_models()
        else:
            self.best_loss = 10e5

    def get_optimizer(self):
        params = list(self.Imp_Layer.parameters()) + list(self.ODE.parameters()) + \
            list(self.L_Enc.parameters()) + list(self.L_Dec.parameters())
        if self.config.static_data:
            params = params + list(self.S_Enc.parameters()) + list(self.S_Dec.parameters())

        self.optimizer = torch.optim.Adam(params, self.config.lr)

    def load_models(self, best=False):

        if self.config.from_best:
            weights = torch.load(os.path.join(
                self.config.save_path_models, 'Best.pth'))
            self.config.epoch_init = weights['Epoch']
            epoch = self.config.epoch_init            
        else:
            epoch = self.config.epoch_init
            weights = torch.load(os.path.join(
                self.config.save_path_models, 'Ckpt_%d.pth'%(epoch)))

        self.best_loss = weights['Loss']
        self.Imp_Layer.load_state_dict(weights['IL'])
        self.ODE.load_state_dict(weights['ODE'])
        self.L_Enc.load_state_dict(weights['L_Enc'])
        self.L_Dec.load_state_dict(weights['L_Dec'])
        if 'train' in self.config.mode:
            self.optimizer.load_state_dict(weights['Opt'])

        if self.config.static_data:
            self.S_Enc.load_state_dict(weights['S_Enc'])
            self.S_Dec.load_state_dict(weights['S_Dec'])
        
        print('Models have loaded from epoch:', epoch)

    def save(self, epoch, loss, best=False):

        weights = {}
        weights['IL'] = self.Imp_Layer.state_dict()
        weights['ODE'] = self.ODE.state_dict()
        weights['L_Enc'] = self.L_Enc.state_dict()
        weights['L_Dec'] = self.L_Dec.state_dict()
        weights['Opt'] = self.optimizer.state_dict()

        if self.config.static_data:
            weights['S_Enc'] = self.S_Enc.state_dict()
            weights['S_Dec'] = self.S_Dec.state_dict()

        weights['Loss'] = loss
        if best:
            weights['Epoch'] = epoch
            torch.save(weights, 
                os.path.join(self.config.save_path_models, 'Best.pth'))
        else:
            torch.save(weights, 
                os.path.join(self.config.save_path_models, 'Ckpt_%d.pth'%(epoch)))

        print('Models have been saved')

    def initialize_imputation(self):

        # Initialize of the A-Variable for VADER
        X, W = self.dataloader.dataset.get_XW()
        W_A = torch.sum(W, 0)
        A = torch.sum(X * W, 0)
        A[W_A>0] = A[W_A>0] / W_A[W_A>0]
        A[W_A>0] = A[W_A>0] / W_A[W_A>0]
        # if not available, then average across entire variable
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if W_A[i, j] == 0:
                    A[i, j] = torch.sum(X[:, :, j]) / torch.sum(W[:, :, j])
                    W_A[i, j] = 1
        # if not available, then average across all variables
        A[W_A==0] = torch.mean(X[W==1])
        return A

    def rec_loss2(self, x, pred, Wmul):
        return Wmul * (x - pred)**2

    def normal_kl(self, mu1, lv1, mu2, lv2):
        # I am not sure, whether this implementation is correct,
        #   but it's the one from chen etal.
        # Multivariate normal_kl is sum of univariate normal KL, if Varianz is Ip
        v1 = torch.exp(lv1)
        v2 = torch.exp(lv2)
        lstd1 = lv1 / 2.
        lstd2 = lv2 / 2.

        kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2)**2.) / (2. * v2)) - .5
        return kl

    def onehot_batch_norm(self, s_onehot, s_types, s_miss):
        # Batch Normalization for the Onehot encoded static data
        s_data_norm = s_onehot.clone()
        b_mean, b_var = [], []

        for i in range(s_onehot.shape[1]):
            if s_types[i, 0] == 'real':
                n_vec = []
                for j in range(s_miss.shape[0]):
                    if s_miss[j, i] == 1:
                        n_vec.append(s_onehot[j, i])

                n_vec = torch.stack(n_vec)
                mean = torch.mean(n_vec)
                var = torch.var(n_vec)
                s_data_norm[:, i] = (s_data_norm[:, i] - mean) / torch.sqrt(var)
                b_mean.append(mean)
                b_var.append(var)
        b_mean = torch.stack(b_mean).to(s_onehot.device)
        b_var = torch.stack(b_var).to(s_onehot.device)

        return s_data_norm, b_mean, b_var
