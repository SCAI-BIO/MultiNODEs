def base_parser():
        
    import argparse
    parser = argparse.ArgumentParser()

    # General things
    parser.add_argument('--mode',  type=str,
                        default='train_full',
                        choices=['train_full', 'only_train', 'only_rec',
                                 'only_prior', 'only_posterior'])
    parser.add_argument('--dataset', type=str,
                        default='PPMI')
    parser.add_argument('--GPU', type=str, default='-1',
                        help='Set -1 for CPU running')
    parser.add_argument('--seed', type=int,
                        default=666)
    parser.add_argument('--train_dir', type=str,
                        default='/home/valderramanino/MultiNODEs_Clean/data')
    parser.add_argument('--save_path', type=str,
                        default='/home/valderramanino/MultiNODEs_Clean/models')
    parser.add_argument('--exp_name', type=str,
                        default='debug')

    # Specific training parameters
    parser.add_argument('--n_long_var', type=int,
                        default=25,
                        help='Number of longitudinal variables of the dataset')
    parser.add_argument('--static_data', type=bool,
                        default=False)
    parser.add_argument('--z_dim_static', type=int,
                        default=3,
                        help='Dimension of the zn of the Gaussian Mixture of the static data')
    parser.add_argument('--s_dim_static', type=int,
                        default=6,
                        help='Dimension of the sn of the Gaussian Mixture of the static data')
    parser.add_argument('--solver', type=str,
                        default='Adjoint',
                        choices=['Adjoint', 'Normal', 'Julia'],
                        help='Solver for the ODE system')
    parser.add_argument('--method_solver', type=str,
                        default='dopri5',
                        choices=['dopri5', 'dopri8', 'bosh3',
                                'fehlberg2', 'adaptive_heun'],
                        help='Solver for the ODE system')
    parser.add_argument('--rtol', type=float,
                        default=1e-7,
                        help='rtol option for the odeint solver')
    parser.add_argument('--atol', type=float,
                        default=1e-8,
                        help='atol option for the odeint solver')
    parser.add_argument('--step_size_solver', type=float,
                        default=1e-3,
                        help='h option for the odeint solver')
    parser.add_argument('--t_steps', type=str,
                        default='0,3,6,9,12,18,24,30,36,42,48,54',
                        help='Time steps in months')
    parser.add_argument('--type_enc', type=str,
                        default='RNN',
                        choices=['RNN', 'LSTM'])
    parser.add_argument('--type_dec', type=str,
                        default='orig',
                        choices=['RNN', 'LSTM', 'Orig'])
    parser.add_argument('--drop_dec', type=float,
                        default=0.4951111384939868)
    parser.add_argument('--act_dec', type=str,
                        default='none',
                        choices=['none', 'tanh', 'relu'])
    parser.add_argument('--num_ode_layers', type=int,
                        default=1,
                        choices=[1, 2, 3])
    parser.add_argument('--act_ode', type=str,
                        default='none',
                        choices=['none', 'tanh', 'relu'])
    parser.add_argument('--act_rnn', type=str,
                        default='none',
                        choices=['none', 'relu', 'tanh'])
    parser.add_argument('--batch_norm_static', type=bool,
                        default=False)
    parser.add_argument('--batch_size', type=float,
                        default=0.6400910825235765,
                        help='in percentage of the dataset')
    parser.add_argument('--from_best', type=bool,
                        default=False)
    parser.add_argument('--latent_dim',
                        default=1.8837462054054002,
                        help='in percentage of the long variables')
    parser.add_argument('--lr', type=float,
                        default=0.0015680290621642827)
    parser.add_argument('--num_epochs', type=int,
                        default=1900)
    parser.add_argument('--epoch_init', type=int,
                        default=1)
    parser.add_argument('--nhidden',
                        default=4.584466407303982,
                        help='nhidden factor for multiply the latent dim')
    parser.add_argument('--rnn_nhidden_dec',
                        default=0.6974659448314735,
                        help='in percentage')
    parser.add_argument('--rnn_nhidden_enc',
                        default=0.49866725741363976,
                        help='in percentage')
    parser.add_argument('--scaling_ELBO', type=float,
                        default=0.9675595125571386)
    parser.add_argument('--save_freq', type=int, default=300)
    parser.add_argument('--print_freq', type=int, default=1)

    # Specific simulation parameters
    parser.add_argument('--time_max', type=int,
                        default=1,
                        help='Upper limit of simulated time')
    parser.add_argument('--time_min', type=int,
                        default=0,
                        help='Lower limit of simulated time')
    parser.add_argument('--time_steps', type=int,
                        default=2000,
                        help='Number of steps in the simulated time')
    parser.add_argument('--s_prob',
                        default=[],
                        help='vector with propability of s during prior sampling')
    # These 2 variables describe the level of noise in the 
    # reparametrization trick when using posterior sampling
    parser.add_argument('--sigma_long', type=int,
                        default=1) 
    parser.add_argument('--sigma_stat', type=int,
                        default=1)
    # Describes how often a complete population is generated
    parser.add_argument('--N_pop', type=int,
                        default=1)

    config = parser.parse_args()
    return config