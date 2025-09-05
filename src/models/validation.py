import os
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from solver import Solver
warnings.filterwarnings('ignore')


class Validation(Solver):
    def __init__(self, config, dataloader, f_train=False):
        super(Validation, self).__init__(config, dataloader, f_train)

        self.method = self.config.method_solver
        self.rtol = self.config.rtol
        self.atol = self.config.atol

        # Get data. We do not use a dataloader here, intead just the dataset
        X, W = self.dataloader.dataset.get_XW()
        self.X, self.W = X.to(self.device), W.to(self.device)
        # It is necessary to modify the generation time for the prediction
        # in the solver.py script
        self.Gen_T = torch.linspace(self.config.time_min, self.config.time_max,
                                    self.config.time_steps).to(self.device)
        self.var_names_save, self.var_names_static = self.dataloader.dataset.get_var_names()
        self.tau = 1e-3

        self.Imp_Layer.eval()
        self.ODE.eval()
        self.L_Enc.eval()
        self.L_Dec.eval()
        if self.config.static_data:
            self.S_Enc.eval()
            self.S_Dec.eval()

        if f_train or self.config.mode == 'only_rec':
            print('=========================')
            print('=====Reconstrucction=====')
            print('=========================')
            self.reconstruction()
        if f_train or self.config.mode == 'only_prior':
            print('=========================')
            print('=====Generation Prior====')
            print('=========================')
            self.generation_prior()
        if f_train or self.config.mode == 'only_posterior':
            print('=========================')
            print('==Generation Posterior===')
            print('=========================')
            self.generation_posterior()

    @torch.no_grad()
    def long_encoder(self):

        if ~torch.all(self.W == 1.0):
            XW = self.Imp_Layer(self.X, self.W)
        else:
            XW = self.X.clone()

        # Get Z0_long
        if self.config.type_enc == 'LSTM':
            out = self.L_Enc(XW)
        else:
            out = self.L_Enc(XW, encoder=True)
        
        qz0_mean, qz0_logvar = out[:, :self.l_dim_number], out[:, self.l_dim_number:]

        return qz0_mean, qz0_logvar

    @torch.no_grad()
    def stat_encoder(self):

        static_data = self.dataloader.dataset.get_static()
        self.S_OneHot = static_data[0].to(self.device)
        # It's a dictionary therefore it doesn't a device
        self.S_Types = static_data[1]
        self.S_True_MMask = static_data[2].to(self.device)

        if self.config.batch_norm_static:
            static_data = self.dataloader.dataset.get_onehot_static()
            # It's a dictionary therefore it doesn't a device
            S_Types_OneHot = static_data[0]
            S_OneHot_MMask = static_data[1].to(self.device)
            s_inp_enc, self.s_batch_mean, self.s_batch_var = self.onehot_batch_norm(
                self.S_OneHot, S_Types_OneHot, S_OneHot_MMask)
        else:
            s_inp_enc = self.S_OneHot.clone()
            self.s_batch_mean, self.s_batch_var = None, None
            
        s_samples, _, qz0_mean_stat, qz0_logvar_stat = self.S_Enc(
            s_inp_enc, self.tau)

        return s_samples, qz0_mean_stat, qz0_logvar_stat

    @torch.no_grad()
    def base_decoder(self, z_init):

        if self.config.solver == 'Julia':
            from diffeqpy import ode
            a=1
        else:
            if self.config.solver == 'Adjoint':
                from torchdiffeq import odeint_adjoint as odeint
            else:
                from torchdiffeq import odeint

            pred_z = odeint(self.ODE, z_init, self.Gen_T, rtol=self.rtol,
                        atol=self.atol, method=self.method).permute(1, 0, 2)

        if self.config.type_dec == 'RNN':
            pred_x = self.L_Dec(pred_z, time_dim=len(self.Gen_T))
        else:
            pred_x = self.L_Dec(pred_z)

        return pred_x

    @torch.no_grad()
    def reconstruction(self):

        qz0_mean, qz0_logvar = self.long_encoder()

        eps = torch.randn(qz0_mean.size()).to(self.device)
        z0_long = eps * torch.exp(.5 * qz0_logvar) + qz0_mean

        if self.config.static_data:

            s_samples, qz0_mean_stat, qz0_logvar_stat = self.stat_encoder()
            eps_stat = torch.randn(qz0_mean_stat.size()).to(self.device)
            z0_stat = eps_stat * torch.exp(.5 * qz0_logvar_stat) + qz0_mean_stat

            # Reconstruct Static data
            y_latent = self.S_Enc.y_forward(z0_stat)

            out, _, _, _ = self.S_Dec(
                y_latent, s_samples, self.S_OneHot, self.S_Types, self.S_True_MMask,
                tau=self.tau, batch_norm=self.config.batch_norm_static,
                batch_mean=self.s_batch_mean, batch_var=self.s_batch_var)

            z_init = torch.cat((z0_long, z0_stat), dim=1)  # Extending z0
        else:
            z_init = z0_long
            out = None

        pred_x = self.base_decoder(z_init)
        self.save_gen(pred_x, out, self.Gen_T, self.T, self.W, name='Rec')

    @torch.no_grad()
    def generation_prior(self):

        # s_prob: vector with propability of s during sampling
        z0_long = np.random.normal(size=(self.X.shape[0], self.l_dim_number))
        z0_long = torch.from_numpy(z0_long).float().to(self.device)

        if self.config.static_data:

            s_prob = self.config.s_prob
            sd_stat = self.config.s_dim_static
            if len(s_prob) == 0:
                s_samples = np.random.choice(a=sd_stat, p=[1/sd_stat]*sd_stat,
                    size=self.X.shape[0])
            else:
                assert (len(s_prob) == sd_stat), 'You need to specify a vector of length %d'%(sd_stat)
                s_samples = np.random.choice(a=sd_stat, p=s_prob,
                    size=self.X.shape[0])

            s_samples_dummy = pd.get_dummies(s_samples).values

            for i in range(sd_stat):
                if any (s_samples == i) == False:
                    s_samples_dummy = np.insert(s_samples_dummy, obj=i,
                        values=np.zeros(self.X.shape[0]), axis=1)

            s_samples_dummy =torch.from_numpy(s_samples_dummy).float().to(self.device)

            #Computing the meanpz and logvarpz of the prior distribution
            qz0_mean_stat, qz0_logvar_stat = self.S_Dec(None, s_samples_dummy, None,
                                                    None, None, gen_prior=True)
            qz0_mean_stat = qz0_mean_stat.cpu().numpy()
            qz0_logvar_stat = qz0_logvar_stat.cpu().numpy()

            z0_size = (self.X.shape[0], self.config.z_dim_static)
            z0_stat = np.random.normal(loc=qz0_mean_stat,
                scale=np.exp(0.5*qz0_logvar_stat), size=z0_size)
            z0_stat = torch.from_numpy(z0_stat).float().to(self.device)
        
            y_latent = self.S_Enc.y_forward(z0_stat)

            static_data = self.dataloader.dataset.get_static()
            S_Types = static_data[1]

            out = self.S_Dec(
                y_latent, s_samples_dummy, None, S_Types, None,
                tau=self.tau, batch_norm=False, batch_mean=None,
                batch_var=None, out_prior=True)

            z_init = torch.cat((z0_long, z0_stat), dim=1)  # Extending z0
        else:
            z_init = z0_long
            out=None

        pred_x = self.base_decoder(z_init)
        self.save_gen(pred_x, out, self.Gen_T, self.T, None, name='Gen_Prior')

    @torch.no_grad()
    def generation_posterior(self):

        # sigma_long: Describes the sd of the epsilon of the reparameterization Trick
        #   of the longitudinal data, default = 1
        # sigma_stat: Describes the sd of the epsilon of the reparameterization Trick
        #   of the static data, default = 1
        # N: Describes the number of how often the complete set of patients is used
        #   (N=1 has 354 patients in PPMI)

        qz0_mean, qz0_logvar = self.long_encoder()
        qz0_mean = torch.cat([qz0_mean]*self.config.N_pop, dim=0)
        qz0_logvar = torch.cat([qz0_logvar]*self.config.N_pop, dim=0)

        sigma_long = self.config.sigma_long
        sigma_stat = self.config.sigma_stat

        eps = torch.normal(torch.zeros(qz0_mean.size()), 
            torch.zeros(qz0_mean.size()) + sigma_long).to(self.device)
        z0_long = eps * torch.exp(.5 * qz0_logvar) + qz0_mean

        if self.config.static_data:

            s_samples, qz0_mean_stat, qz0_logvar_stat = self.stat_encoder()
            # Making N representations of the Original data
            qz0_mean_stat = torch.cat([qz0_mean_stat]*self.config.N_pop, dim=0)
            qz0_logvar_stat = torch.cat([qz0_logvar_stat]*self.config.N_pop, dim=0)

            eps_stat = torch.normal(torch.zeros(qz0_mean_stat.size()), 
                torch.zeros(qz0_mean_stat.size()) + sigma_stat).to(self.device)
            z0_stat = eps_stat * torch.exp(.5 * qz0_logvar_stat) + qz0_mean_stat

            # Reconstruct Static data
            y_latent = self.S_Enc.y_forward(z0_stat)

            s_samples = torch.cat([s_samples]*self.config.N_pop, dim=0)
            S_OneHot = torch.cat([self.S_OneHot]*self.config.N_pop, dim=0)
            S_True_MMask = torch.cat([self.S_True_MMask]*self.config.N_pop, dim=0)

            out, _, _, _ = self.S_Dec(
                y_latent, s_samples, S_OneHot, self.S_Types, S_True_MMask,
                tau=self.tau, batch_norm=self.config.batch_norm_static,
                batch_mean=self.s_batch_mean, batch_var=self.s_batch_var)

            z_init = torch.cat((z0_long, z0_stat), dim=1)  # Extending z0
        else:
            z_init = z0_long
            out = None

        pred_x = self.base_decoder(z_init)
        name = 'Gen_Posterior_SL%d_SS%d'%(sigma_long, sigma_stat)
        self.save_gen(pred_x, out, self.Gen_T, self.T, None, name=name)

    def save_gen(self, pred_x, pred_s, gen_t, train_t, w=None, name='Rec'):
        if self.config.from_best:
            name = 'Best_' + name
        name = '%s_%d.pth'%(name, self.config.epoch_init)

        s_path = os.path.join(self.config.save_path_samples, name)

        s_dict = {}
        s_dict['Long_Values'] = pred_x.cpu()
        s_dict['Gen_Time'] = gen_t.cpu()
        s_dict['Train_Time'] = train_t.cpu()
        s_dict['Var_Names'] = self.var_names_save
        s_dict['Var_Names_Static'] = self.var_names_static

        if pred_s is not None:
            pred_s = pred_s.cpu()
        s_dict['Stat_Values'] = pred_s

        if w is not None:
            s_dict['W_train'] = w.cpu()
        
        torch.save(s_dict, s_path)
