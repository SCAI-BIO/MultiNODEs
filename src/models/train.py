import warnings
import numpy as np
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
from collections import OrderedDict
from utils import print_current_losses
from solver import Solver
from validation import *
warnings.filterwarnings('ignore')

# On one hand, config has all the options for build the model 
# and run it. The most important ones are the following

# config.solver describes which solver is used
# config.num_epochs describes the number of epochs for training
# config.batch_size in percentage at the beggining and then change to int according
#   to percentage * len(dataset)
# config.batch_norm_static specifies, whether batch normalization is used
#   on static data or not
# config.lr describes the optimizer´s learning rate
# config.save_path_models is the path to save the models
# config.n_long_var is the number of longitudinal variables.
#   Therefore it is also the output dimension of the model
# config.latent_dim is the latent_dim of the ODE function,
#   as percentage of config.n_long_var
# config.nhidden is the number of hidden unit of the  ODE function
#   in percentage of absolute latent_dim (greater than 1 possible)
# config.act_ode specifies activation funtion of the ODE solver
# config.num_ode_layers specifies the number of the hidden ODE layers
# config.type_enc describes the encoder to be used
# config.type_dec describes the decoder to be used
# config.rnn_nhidden_enc specifies, how many units has h of the encoder RNN,
#   fraction (in percent) of number of timesteps * number of features
# config.rnn_nhidden_dec specifies, how many units has h of the decoder RNN,
#   fraction (in percent) of number of timesteps * number of features
# config.act_rnn specifies the RNN´s activation
# config.drop_dec is the probability of dropout of decoder
# config.static_data if false, no static data is used
# config.s_dim_static Dimension of the sn of the Gaussian Mixture 
#   of the static data
# config.z_dim_static Dimension of the zn of the Gaussian Mixture
#   of the static data
# config.scaling_ELBO Parameter to scale the two ELBOs of the static
#   and the longitudinal data

# On the other hand, data has the following inputs

# X_train is the train data (sampled trajectories)
# W_train specifies the missing values of VADER
# t_train are the Timestamps related to X_train
# static_onehot if there are categorical data
#   if None means no static data is used
# static_types variable with the types of data
#   if None means no static data is used
# static_true_miss_mask which has 0, where data is missing
#   and 1 where data is observed
#   if None means no static data is used

class Train(Solver):
    def __init__(self, config, dataloader, f_train=False):
        super(Train, self).__init__(config, dataloader, f_train)

        self.run()
        if self.config.mode == 'train_full':
            Validation(config, dataloader, f_train=True)

    def run(self):

        if self.config.solver == 'Adjoint':
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        method = self.config.method_solver
        rtol = self.config.rtol
        atol = self.config.atol

        global_steps = 0
        total_time = time.time()

        # Epoch_init begins in 1
        for epoch in range(self.config.epoch_init, self.config.num_epochs + 1):

            desc_bar = '[Iter: %d] Epoch: %d/%d' % (
                global_steps, epoch, self.config.num_epochs)

            progress_bar = tqdm(enumerate(self.dataloader),
                                unit_scale=True,
                                total=len(self.dataloader),
                                desc=desc_bar)

            tau = np.max([1.0 - (0.999/(self.config.num_epochs - 50)) *
                         (epoch), 1e-3])

            epoch_time_init = time.time()

            # Training along dataset            
            for iter, data in progress_bar:
                global_steps += 1

                X, W = data[0].to(self.device), data[1].to(self.device)

                self.optimizer.zero_grad()

                if ~torch.all(W == 1.0):
                    XW = self.Imp_Layer(X, W)
                else:
                    XW = X.clone()

                # Get Z0_long
                if self.config.type_enc == 'LSTM':
                    out = self.L_Enc(XW)
                else:
                    out = self.L_Enc(XW, encoder=True)

                qz0_mean, qz0_logvar = out[:, :self.l_dim_number], out[:, self.l_dim_number:]
                eps = torch.randn(qz0_mean.size()).to(self.device)
                z0_long = eps * torch.exp(.5 * qz0_logvar) + qz0_mean

                # Static Data process
                if self.config.static_data:
                    S_OneHot = data[2].to(self.device)
                    S_Types = self.static_types
                    S_True_MMask = data[3].to(self.device)

                    if self.config.batch_norm_static:
                        S_Types_OneHot = self.s_onehot_types
                        S_OneHot_MMask = data[4].to(self.device)
                        s_inp_enc, s_batch_mean, s_batch_var = self.onehot_batch_norm(
                            S_OneHot, S_Types_OneHot, S_OneHot_MMask)
                    else:
                        s_inp_enc = S_OneHot.clone()
                        s_batch_mean, s_batch_var = None, None

                    s_samples, log_pi, qz0_mean_stat, qz0_logvar_stat = self.S_Enc(
                        s_inp_enc, tau)
                    eps_stat = torch.randn(qz0_mean_stat.size()).to(self.device)
                    z0_stat = eps_stat * torch.exp(.5 * qz0_logvar_stat) + qz0_mean_stat

                    # Reconstruct Static data
                    y_latent = self.S_Enc.y_forward(z0_stat)
                    # Indices is a range in this way because we are using just some
                    # data per each iteration in train

                    out, mean_pz, logvar_pz, log_p_x = self.S_Dec(
                        y_latent, s_samples, S_OneHot, S_Types, S_True_MMask,
                        tau=tau, batch_norm=self.config.batch_norm_static,
                        batch_mean=s_batch_mean, batch_var=s_batch_var)
                    z_init = torch.cat((z0_long, z0_stat), dim=1)  # Extending z0
                else:
                    z_init = z0_long

                # ODE Solver
                pred_z = odeint(self.ODE, z_init, self.T, rtol=rtol,
                                atol=atol, method=method).permute(1, 0, 2)

                if self.config.type_dec == 'RNN':
                    pred_x = self.L_Dec(pred_z, time_dim=len(self.T))
                else:
                    pred_x = self.L_Dec(pred_z)

                pz0_mean = pz0_logvar = torch.zeros(z0_long.size()).to(self.device)

                analytic_kl = self.normal_kl(
                    qz0_mean, qz0_logvar,
                    pz0_mean, pz0_logvar).sum(-1)

                # Implement weighted mean squared error for Vader
                KL_avg = torch.mean(analytic_kl)
                rec_loss = self.rec_loss2(X, pred_x, W)
                rec_loss = torch.sum(rec_loss)*X.size(1)*X.size(2) / (sum(sum(sum(W))))

                if self.config.static_data:

                    eps = 1e-20
                    # KL(q(s|x)|p(s))
                    # logits=log_pi, labels=pi_param
                    # because logits has to be transformed with softmax
                    pi_param = F.softmax(log_pi,dim=1)
                    KL_s = torch.sum(pi_param * torch.log(pi_param + eps), dim=1) \
                        + torch.log(torch.tensor(float(self.config.s_dim_static)))

                    # meanpz, logvarpz, qz0_meanstat, qz0_logvarstat
                    analytic_kl_stat = self.normal_kl(
                        qz0_mean_stat, qz0_logvar_stat,
                        mean_pz, logvar_pz).sum(-1)

                    rec_loss_stat = log_p_x.sum(-1).to(self.device)

                    ELBO_stat = -torch.mean(rec_loss_stat - analytic_kl_stat - KL_s, 0)

                    long = (KL_avg + rec_loss) / (KL_avg + rec_loss + ELBO_stat)
                    stat = ELBO_stat / (KL_avg + rec_loss + ELBO_stat)

                    long_scaled = stat / (long + stat) *(KL_avg + rec_loss)
                    stat_scaled = long / (long + stat) * ELBO_stat
                    loss = long_scaled + self.config.scaling_ELBO * stat_scaled
                else:
                    loss = KL_avg + rec_loss

                loss.backward()
                self.optimizer.step()
                self.loss.update(loss.item())

                if (iter + 1) % self.config.print_freq == 0 or (iter + 1) == len(self.dataloader):
                    losses = OrderedDict()
                    if self.config.static_data:
                        losses['L_Scaled'] = long_scaled.item()
                        losses['S_Scaled'] = self.config.scaling_ELBO * stat_scaled.item()
                        losses['Total'] = loss.item()
                    else:
                        losses['L_Rec'] = rec_loss.item()
                        losses['L_KL'] = KL_avg.item()
                        losses['Total'] = loss.item()

                    losses['Avg_ELBO'] = -self.loss.avg
                    progress_bar.set_postfix(**losses)

            t_epoch = time.time() - epoch_time_init
            t_total = time.time() - total_time
            print_current_losses(epoch, global_steps, losses,
                t_epoch, t_total, self.config.save_path_losses,
                s_excel=True)

            if (epoch) % self.config.save_freq == 0:
                self.save(epoch, loss.item())
            if loss.item() < self.best_loss:
                self.save(epoch, loss.item(), best=True)
                self.best_loss = loss.item()

        self.save(epoch, -self.loss.avg)
