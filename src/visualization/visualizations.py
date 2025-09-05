import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, sem, iqr
import torch
import sys
sys.path.append('../')
from data.load_data import load_dataset
import warnings
warnings.filterwarnings('ignore')

def base_parser():
        
    import argparse
    parser = argparse.ArgumentParser()

    # General things
    parser.add_argument('--dataset', type=str,
                        default='PPMI')
    parser.add_argument('--train_dir', type=str,
                        default='/home/valderramanino/MultiNODEs_Clean/data')
    parser.add_argument('--save_path', type=str,
                        default='/home/valderramanino/MultiNODEs_Clean/models')
    parser.add_argument('--exp_name', type=str,
                        default='debug')
    parser.add_argument('--from_best', type=bool,
                        default=False)

    # Specific training parameters just necessary for load data
    parser.add_argument('--n_long_var', type=int,
                        default=25,
                        help='Number of longitudinal variables of the dataset')
    parser.add_argument('--num_epoch', type=int,
                        default=1900)
    parser.add_argument('--t_steps', type=str,
                        default='0,3,6,9,12,18,24,30,36,42,48,54',
                        help='Time steps in months')
    parser.add_argument('--batch_norm_static', type=bool,
                        default=False)
    parser.add_argument('--static_data', type=bool,
                        default=False)
    parser.add_argument('--sigma_long', type=int,
                        default=1) 
    parser.add_argument('--sigma_stat', type=int,
                        default=1)

    config = parser.parse_args()
    return config

def plot_traj_per_patient(org_data, gen_data, epoch, s_path,
                          n_p=3, n_v=5, title='Rec'):

    t_org = org_data.T
    t_gen = gen_data['Gen_Time']

    var_names_long = org_data.var_names_long

    # patients, time, longvar
    x = org_data.X
    pred_x = gen_data['Long_Values']

    # Random n_p patients n_v vars
    # num_patients, _, num_vars = x.shape
    # patients = torch.randperm(num_patients)[:n_p]
    # vars_ = torch.randperm(num_vars)[:n_v]

    # First n_p and n_v vars
    patients = [i for i in range(n_p)]
    vars_ = [i for i in range(n_v)]

    x, pred_x = x[patients, :, :], pred_x[patients, :, :]
    x, pred_x = x[:, :, vars_], pred_x[:, :, vars_]

    for p in range(n_p):
        for v in range(n_v):
            x_ = x[p, :, v]
            pred_x_ = pred_x[p, :, v]

            savename = '%s_Pat%d_LVar%d_Ep%d.png'%(
                title, patients[p], vars_[v], epoch)
            savename = os.path.join(s_path, savename)

            longvar = var_names_long[vars_[v]]
            title_ = '%s Pat %d LVar %s'%(title, patients[p], longvar)

            plt.plot(t_gen, pred_x_, '-g', label='Generated', linewidth=2)
            plt.plot(t_org, x_, 'ok', label='Original', linewidth=2)
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title(title_)
            plt.savefig(savename)
            plt.close()


def vis_mean_traj(org_data, gen_data, epoch, s_path, typ_data='Rec'):

    t_org = org_data.T *54
    t_gen = gen_data['Gen_Time']*54

    pred_x = gen_data['Long_Values']
    _, _, num_vars = pred_x.shape
    var_names_long = org_data.var_names_long

    X, W = org_data.get_XW()
    X, W = X.numpy(), W.numpy()

    X[W==0] = np.nan
    X_mean = torch.from_numpy(np.nanmean(X, axis=0))
    X_std = torch.from_numpy(np.nanstd(X, axis=0))

    W_sum = W.sum(axis = 0)

    for n_var in range(num_vars):

        var_name = var_names_long[n_var]
        save_name = '%s_Ep%d_%s.png'%(typ_data, epoch, var_name)
        save_name = os.path.join(s_path, save_name)
        var_name = '%s  %s'%(typ_data, var_name)

        X_meanvar = X_mean[:, n_var]
        X_stdvar = X_std[:, n_var]

        #indicator of non-missing values
        W_sumvar = W_sum[:, n_var]
        non_missing = W_sumvar > 0
    
        #KI of mean
        #Usind W_sumvar, because of missingness is the n different in samp_trajs
        #for every time step and every variable.
        alpha = 0.05
        KIsampu = X_meanvar - t.ppf(1-alpha/2,W_sumvar)*X_stdvar.numpy()/np.sqrt(W_sumvar) 
        KIsampo = X_meanvar + t.ppf(1-alpha/2,W_sumvar)*X_stdvar.numpy()/np.sqrt(W_sumvar)
    
        pred_x = pred_x.float()

        pred_x_mean = pred_x.mean(dim=0)
        pred_x_std = pred_x.std(dim=0)
    
        #KI of mean
        KIxspu = pred_x_mean - t.ppf(1-alpha/2,pred_x.shape[0])*pred_x_std/np.sqrt(pred_x.shape[0]) 
        KIxspo = pred_x_mean + t.ppf(1-alpha/2,pred_x.shape[0])*pred_x_std/np.sqrt(pred_x.shape[0])
    
        plt.figure()
        plt.plot(t_gen, pred_x_mean[:, n_var], 'orange', label='Generated + 95% CI')
        plt.plot(t_gen, KIxspo[:,n_var], 'orange', linestyle = ':')
        plt.plot(t_gen, KIxspu[:,n_var], 'orange', linestyle = ':')
 
        plt.scatter(t_org[non_missing], X_meanvar[non_missing],
                    label='real data + 95% CI', s = 7, linestyle = 'None')
        plt.errorbar(t_org[non_missing], X_meanvar[non_missing],
                    yerr = (KIsampu[non_missing]-KIsampo[non_missing])/2,
                    linestyle = 'None')
        plt.xlabel('Time in months')
        plt.ylabel('Progression Score')
        plt.legend()
        plt.title(var_name)
        plt.savefig(save_name, dpi=500)
        plt.close()
        print('%s\'s image saved'%(var_name))


def vis_prior_post_mean_traj(org_data, prior_data, post_data, epoch,
                            s_path, typ_data='Prior and Posterior sampling'):

    t_org = org_data.T *54
    t_prior = prior_data['Gen_Time']*54
    t_post = post_data['Gen_Time']*54

    prior_x = prior_data['Long_Values']
    post_x = post_data['Long_Values']
    _, _, num_vars = prior_x.shape
    var_names_long = org_data.var_names_long

    X, W = org_data.get_XW()
    X, W = X.numpy(), W.numpy()

    X[W==0] = np.nan
    X_mean = torch.from_numpy(np.nanmean(X, axis=0))
    X_std = torch.from_numpy(np.nanstd(X, axis=0))

    W_sum = W.sum(axis = 0)

    for n_var in range(num_vars):

        var_name = var_names_long[n_var]
        save_name = '%s_Ep%d_%s.png'%(typ_data, epoch, var_name)
        save_name = os.path.join(s_path, save_name)
        var_name = '%s  %s'%(typ_data, var_name)

        X_meanvar = X_mean[:, n_var]
        X_stdvar = X_std[:, n_var]

        #indicator of non-missing values
        W_sumvar = W_sum[:, n_var]
        non_missing = W_sumvar > 0
    
        #KI of mean
        #Usind W_sumvar, because of missingness is the n different in samp_trajs
        #for every time step and every variable.
        alpha = 0.05
        KIsampu = X_meanvar - t.ppf(1-alpha/2,W_sumvar)*X_stdvar.numpy()/np.sqrt(W_sumvar) 
        KIsampo = X_meanvar + t.ppf(1-alpha/2,W_sumvar)*X_stdvar.numpy()/np.sqrt(W_sumvar)
    
        prior_x = prior_x.float()
        prior_x_mean = prior_x.mean(dim=0)
        prior_x_std = prior_x.std(dim=0)
    
        #KI of mean
        KIxpriorpu = prior_x_mean - t.ppf(1-alpha/2,prior_x.shape[0])*prior_x_std/np.sqrt(prior_x.shape[0]) 
        KIxpriorpo = prior_x_mean + t.ppf(1-alpha/2,prior_x.shape[0])*prior_x_std/np.sqrt(prior_x.shape[0])
    
        plt.figure()
        plt.plot(t_prior, prior_x_mean[:, n_var], 'orange', label='Prior + 95% CI')
        plt.plot(t_prior, KIxpriorpo[:,n_var], 'orange', linestyle = ':')
        plt.plot(t_prior, KIxpriorpu[:,n_var], 'orange', linestyle = ':')

        post_x = post_x.float()
        post_x_mean = post_x.mean(dim=0)
        post_x_std = post_x.std(dim=0)

        #KI of mean
        KIxpostpu = post_x_mean - t.ppf(1-alpha/2,prior_x.shape[0])*post_x_std/np.sqrt(post_x.shape[0]) 
        KIxpostpo = post_x_mean + t.ppf(1-alpha/2,prior_x.shape[0])*post_x_std/np.sqrt(post_x.shape[0])
    
        plt.plot(t_post, post_x_mean[:, n_var], 'r', label='Post + 95% CI')
        plt.plot(t_post, KIxpostpo[:,n_var], 'r', linestyle = ':')
        plt.plot(t_post, KIxpostpu[:,n_var], 'r', linestyle = ':')

        plt.scatter(t_org[non_missing], X_meanvar[non_missing],
                    label='real data + 95% CI', s = 7, linestyle = 'None')
        plt.errorbar(t_org[non_missing], X_meanvar[non_missing],
                    yerr = (KIsampu[non_missing]-KIsampo[non_missing])/2,
                    linestyle = 'None')
        plt.xlabel('Time in months')
        plt.ylabel('Progression Score')
        plt.legend()
        plt.title(var_name)
        plt.savefig(save_name, dpi=500)
        plt.close()
        print('%s\'s image saved'%(var_name))


if __name__ == '__main__':

    config = base_parser()

    # Specific training parameters just necessary for load data
    config.t_steps = np.array([int(i) for i in config.t_steps.split(',')])
    config.t_over = config.t_steps / config.t_steps[-1]
    config.t_visits = len(config.t_steps)

    # Paths to folder with the sxamples
    config.train_dir = os.path.join(config.train_dir, config.dataset)
    config.save_path = os.path.join(config.save_path, config.exp_name)
    config.save_path_samples = os.path.join(config.save_path, 'samples')

    # Training data
    org_dataset = load_dataset(config, only_data=True)
    sigma_long = config.sigma_long
    sigma_stat = config.sigma_stat

    # Generated data
    rec_name = 'Rec_%d.pth'%(config.num_epoch)
    prior_name = 'Gen_Prior_%d.pth'%(config.num_epoch)
    post_name = 'Gen_Posterior_SL%d_SS%d_%d.pth'%(sigma_long,
        sigma_stat, config.num_epoch)

    if config.from_best:
        rec_name = 'Best_' + rec_name
        prior_name = 'Best_' + prior_name
        post_name = 'Best_' + post_name

    rec_path = os.path.join(config.save_path_samples, rec_name)
    prior_path = os.path.join(config.save_path_samples, prior_name)
    post_path = os.path.join(config.save_path_samples, post_name)

    rec_data = torch.load(rec_path)
    prior_data = torch.load(prior_path)
    post_data = torch.load(post_path)

    save_path = os.path.join(config.save_path_samples,
                            'Ep%d'%(config.num_epoch))
    os.makedirs(save_path, exist_ok=True)

    # plot_traj_per_patient(org_dataset, rec_data, config.num_epoch,
    #     config.save_path_samples, title='Rec')
    # # plot_traj_per_patient(org_dataset, prior_data, config.num_epoch,
    # #     config.save_path_samples, title='Prior')
    # plot_traj_per_patient(org_dataset, post_data, config.num_epoch,
    #     config.save_path_samples, title='Post')

    vis_mean_traj(org_dataset, rec_data, config.num_epoch,
                  save_path, typ_data='Reconstruction')
    vis_mean_traj(org_dataset, prior_data, config.num_epoch,
                  save_path, typ_data='Prior sampling')
    vis_mean_traj(org_dataset, post_data, config.num_epoch,
                  save_path, typ_data='Posterior sampling')
    
    vis_prior_post_mean_traj(org_dataset, prior_data, post_data,
                             config.num_epoch, save_path)
