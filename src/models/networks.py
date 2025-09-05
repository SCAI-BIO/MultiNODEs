import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import log_normal_pdf

# ========================================
# ======== ENCODERS AND DECODERS =========
# ========================================

class VaderLayer(nn.Module):
    def __init__(self, A_init):
        super(VaderLayer, self).__init__()

        self.b = nn.Parameter(A_init)

    def forward(self, X, W):
        # X is the data and w is the indicator function
        # Handle missing values section of the main text
        return (1 - W) * self.b + X * W 


class RecognitionRNN(nn.Module):
    # When is used as encoder obs_dim is the number of longitudinal variables
    #   and latent_dim the Z dim

    # When is used as decoder latent_dim is the number of longitudinal variables
    #   because at the end we want to have the same number of variables and
    #   obs_dim is the latent dimension

    # in both cases nhidden is the hidden state´s dimension between input and output

    def __init__(self, latent_dim, obs_dim, nhidden, act):
        super(RecognitionRNN, self).__init__()

        if nhidden == 0:
            nhidden = 1
        self.nhidden = nhidden
        self.latent_dim = latent_dim

        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim)

        if act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'relu':
            self.act = nn.ReLU()
        else:  # act == 'none'
            self.act = nn.Identity()

    def internal_forward(self, x, h):

        combined = torch.cat((x, h), dim=1)
        h = self.act(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def forward(self, data, time_dim=10, encoder=False):
        # If the network isn´t an encoder it must be a decoder
        # When used as decoder, time_dim should be ajusted according
        # to the desired time dimension
        h = torch.zeros(data.size(0), self.nhidden).to(data.device)

        if encoder:
            # data here is XW it means the longitudinal data
            # for reconRP data can be smaller but in the train script
            # we will change this dimension
            for t in reversed(range(data.size(1))):
                long_v = data[:, t, :]
                out, h = self.internal_forward(long_v, h)
                # out here is Z
        else:
            # data here is pred_z
            out = torch.zeros(data.size(0), time_dim, self.latent_dim)
            for t in range(data.size(1)):
                z = data[:, t, :]
                out[:, t, :], h = self.internal_forward(z, h)
                # out here is pred_x

        return out


class LSTMEncoder(nn.Module):
    def __init__(self, i_size, h_size, target_size):
        super(LSTMEncoder, self).__init__()

        self.lstm = nn.LSTM(i_size, h_size, num_layers=1, bias=True,
                            batch_first=True, dropout=0, bidirectional=False)
        self.lin = nn.Linear(h_size, target_size)
        self.h_size = h_size

    def forward(self, XW):

        h = torch.zeros(1, XW.size(0), self.h_size).to(XW.device)
        c = torch.zeros(1, XW.size(0), self.h_size).to(XW.device)

        for t in reversed(range(XW.size(1))):
            long_v = XW[:, t:t+1, :]  # longitudinal variable in time t
            out, (h, c) = self.lstm(long_v, (h, c))
            # out and h_0 are the same, because just one point is going through LSTM

        out = out[:, 0, :]
        out = self.lin(out)  # out is pred_z_long
        return out


class LSTMDecoder(nn.Module):
    def __init__(self, i_size, h_size, target_size):
        super(LSTMDecoder, self).__init__()

        self.lstm = nn.LSTM(i_size, h_size, num_layers=1, bias=True,
                            batch_first=True, dropout=0, bidirectional=False)
        self.lin = nn.Linear(h_size, target_size)

    def forward(self, z):
        out, _ = self.lstm(z)
        out = self.lin(out)  # out is pred_x
        return out


class Decoder(nn.Module):
    def __init__(self, config, latent_dim, nhidden_number):
        super(Decoder, self).__init__()

        if config.act_dec == 'tanh':
            self.act = nn.Tanh()
        elif config.act_dec == 'relu':
            self.act = nn.ReLU()
        else:  # config.act_dec == 'none'
            self.act = nn.Identity()

        self.act_dec = config.act_dec

        self.fc1 = nn.Linear(latent_dim, nhidden_number)
        self.fc2 = nn.Linear(nhidden_number, config.n_long_var)
        self.drop = nn.Dropout(config.drop_dec)

    def forward(self, z):
        out = self.fc1(z)

        # It's convenient to use dropout after activation, but 
        # in case of Relu before activation
        if self.act_dec == 'relu':
            out = self.act(self.drop(out))
        else:
            out = self.drop(self.act(out))

        out = self.fc2(out)  # out is pred_x
        return out


# ========================================
# ================= ODE ==================
# ========================================


class LatentODEFunc(nn.Module):
    # Latent ODE with bottleneck structure
    # num_odelayers specifies the depth of the Neural ODE
    def __init__(self, config, latent_dim=4, nhidden_number=20):
        super (LatentODEFunc, self).__init__()

        num_ode_layers = config.num_ode_layers

        if config.act_ode == 'tanh':
            self.act = nn.Tanh()
        elif config.act_ode == 'relu':
            self.act = nn.ReLU()
        else:  # config.act_ode == 'none'
            self.act = nn.Identity()

        diff = nhidden_number - latent_dim

        layers_enc = []
        layers_dec = []

        # Encoder y decoder parts of the ODE function
        # input and output are just the opposite
        # at the end we reverse the dec list to be aligned
        # with the encoder one
        for i in range(num_ode_layers):

            if num_ode_layers == 1:
                layers_enc.append(nn.Linear(latent_dim, nhidden_number))
                layers_dec.append(nn.Linear(nhidden_number, latent_dim))
            else:
                fact_i = i/num_ode_layers
                fact_o = (i+1)/num_ode_layers

                c_i = latent_dim + int(np.round(diff*fact_i))
                c_o = latent_dim + int(np.round(diff*fact_o))

                if fact_o == 1:
                    layers_enc.append(nn.Linear(c_i, nhidden_number))
                    layers_dec.append(nn.Linear(nhidden_number, c_i))
                else:
                    layers_enc.append(nn.Linear(c_i, c_o))
                    layers_dec.append(nn.Linear(c_o, c_i))

            layers_enc.append(self.act)
            layers_dec.append(self.act)

        layers_enc.append(nn.Linear(nhidden_number, nhidden_number))
        layers_enc.append(self.act)
        layers_dec = layers_dec[:-1]  #remove the last act layer
        layers_dec.reverse()
        layers = layers_enc + layers_dec  # concat enc + dec

        self.model = nn.Sequential(*layers)
        self.nfe = 0

    def forward(self, t, x):

        self.nfe += 1
        out = self.model(x)
        return out


# ========================================
# ============= STATIC MODELS ============
# ========================================


class StatLayerEnc(nn.Module):
    def __init__(self, s_vals_dim, onehot_dim, sn_g_dim, zn_g_dim):
        super(StatLayerEnc, self).__init__()

        self.log_pi_layer = nn.Linear(onehot_dim, sn_g_dim)
        self.mean_layer = nn.Linear(onehot_dim + sn_g_dim, zn_g_dim)
        self.log_layer = nn.Linear(onehot_dim + sn_g_dim, zn_g_dim)
        self.y_layer = nn.Linear(zn_g_dim, s_vals_dim)

    def y_forward(self, z_static):
        return self.y_layer(z_static)

    def forward(self, static_data, tau=1):
        # Using GMM Prior after running modules through NN
        log_pi = self.log_pi_layer(static_data)
        samples_s = F.gumbel_softmax(log_pi, tau, hard=False)
        mean = self.mean_layer(torch.cat((static_data, samples_s), dim=1))
        log = self.log_layer(torch.cat((static_data, samples_s), dim=1))
        return samples_s, log_pi, mean, log


class StatDecoder(nn.Module):
    def __init__(self, s_vals_dim, static_types, sn_g_dim, zn_g_dim):
        super(StatDecoder, self).__init__()

        self.mean_pz_layer = nn.Linear(sn_g_dim, zn_g_dim)

        self.layers = nn.ModuleList()
        self.zn_g_dim = zn_g_dim
        for i in range(s_vals_dim):
            if static_types[i, 0] == 'real':
                # first output dim is mean and second is logvar
                self.layers.append(nn.Linear(s_vals_dim, static_types[i, 1]*2))
            elif static_types[i, 0] == 'cat':
                self.layers.append(nn.Linear(s_vals_dim, static_types[i, 1]-1))

    def forward(self, y_latent, samples_s, static_onehot, static_types,
                static_true_miss_mask, tau=1, batch_norm=False,
                batch_mean=None, batch_var=None, gen_prior=False, out_prior=False):

        # Computing the parameters of the p(z|s) distribution
        if samples_s is not None:
            mean_pz = self.mean_pz_layer(samples_s)
            log_var_pz = torch.zeros(samples_s.shape[0], self.zn_g_dim).to(samples_s.device)

        # By generating from the prior you have to use the parameters
        # of the prior distribution
        if gen_prior:
            return mean_pz, log_var_pz
        elif out_prior:
            out = torch.zeros(y_latent.shape)

            for i in range(y_latent.shape[1]):
                params = self.layers[i](y_latent)
                if static_types[i, 0] == 'real':
                    mean = params[:, 0]
                    logvar = params[:, 1]
                    out[:, i] = torch.normal(mean, torch.exp(0.5*logvar))
                else:
                    zeros = torch.zeros(y_latent.shape[0], 1).to(y_latent.device)
                    params = torch.cat([zeros, params], dim=1)
                    helpf = F.gumbel_softmax(params, tau, hard=False)
                    out[:, i] = torch.argmax(helpf.detach(), 1).float()

            return out
        else:
            # Computing the output data
            out = torch.zeros(y_latent.shape)
            log_p_x = torch.zeros(y_latent.shape)

            batch_id, onehot_id = 0, 0

            for i in range(y_latent.shape[1]):
                params = self.layers[i](y_latent)

                if static_types[i, 0] == 'real':

                    # data = static_onehot[indices, onehot_id].clone()
                    data = static_onehot[:, onehot_id].clone()
                    data[torch.isnan(data)] = 0

                    mean = params[:, 0]
                    logvar = params[:, 1]

                    if batch_norm:
                        mean = torch.sqrt(batch_var[batch_id]) * mean + batch_mean[batch_id]
                        # because of the logvar implementation of the variance,
                        # batchvar*var = exp(log(batchvar) + logvar)
                        logvar = batch_var[batch_id] + logvar
                    val = log_normal_pdf(data, mean, logvar)
                    miss = static_true_miss_mask[:, i].float()

                    log_p_x[:, i] = val * miss

                    out[:, i] = torch.normal(mean, torch.exp(0.5*logvar))
                    batch_id +=1
                    onehot_id += 1
                else:
                    zeros = torch.zeros(y_latent.shape[0], 1).to(y_latent.device)
                    params = torch.cat((zeros, params), 1)
                    eps = 1e-20  # to avoid log of 0

                    # Reconstruction loss
                    # Compute with softmax sampling probability
                    x_prob = F.softmax(params, dim=1)
                    data = static_onehot[:, onehot_id:onehot_id + static_types[i, 1]].clone()
                    data = data.float()

                    # data = data + torch.log(x_prob + eps)

                    log = torch.sum(data * torch.log(x_prob + eps), dim=1).float()
                    miss = static_true_miss_mask[:, i].float()
                    log_p_x[:, i] = log * miss

                    helpf = F.gumbel_softmax(params, tau, hard=False)
                    out[:, i] = torch.argmax(helpf.detach(), 1).float()
                    onehot_id += static_types[i, 1]

            return out, mean_pz, log_var_pz, log_p_x
