import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

def simulation(N,I0,R0,beta,gamma,timemax,timepoints,vital=False,mu=None,lambd=None, eps=None, simnum=1):
    

    # Total population, N.
    # Initial number of infected and recovered individuals, I0 and R0.
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    # maximal time timemax
    # number of timepoints timepoints
    # noise, variance of epsilon
    # simnum: Number of simulations (different noise)
    
    t = np.linspace(0, timemax, timepoints)

    
    if vital:
        # The SIR model differential equations.
        def deriv(y, t, N, beta, gamma, mu, lambd):
            S, I, R = y
            dSdt = lambd*N - mu * S -beta * S * I / N
            dIdt = beta * S * I / N - gamma * I - mu * I
            dRdt = gamma * I - mu * R
            return dSdt, dIdt, dRdt
    else:
        def deriv(y, t, N, beta, gamma):
            S, I, R = y
            dSdt = -beta * S * I / N
            dIdt = beta * S * I / N - gamma * I
            dRdt = gamma * I
            return dSdt, dIdt, dRdt
    
    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    if (vital):
        ret = odeint(deriv, y0, t, args=(N, beta, gamma, mu, lambd))
    else:
        ret = odeint(deriv, y0, t, args=(N, beta, gamma))
        
    S = []
    I = []
    R = []
    
    for i in (np.arange(0,simnum)):
            
        noise = np.random.normal(0,1,(timepoints,3))
        noise_1 = np.random.normal(0,eps*np.max(ret[:,0]),size=timepoints)
        noise_2 = np.random.normal(0,eps*np.max(ret[:,1]),size=timepoints)
        noise_3 = np.random.normal(0,eps*np.max(ret[:,2]),size=timepoints)
        noise = np.stack([noise_1,noise_2,noise_3],axis=-1)
    
        ret_help = ret+noise
        
        S_help, I_help, R_help = ret_help.T
        S.append(S_help)
        I.append(I_help)
        R.append(R_help)
    
    return S, I, R

# for n in [7000,3000,2100]:
#     for eps in [0.01, 0.05, 0.1]:
        
#         #1000,150,800
#         #eps=2
        
#         N = 1000
#         # Initial number of infected and recovered individuals, I0 and R0.
#         I0, R0 = 1, 0
#         # Everyone else, S0, is susceptible to infection initially.
#         S0 = N - I0 - R0
#         # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
#         beta, gamma = 2, 1 
        
#         timemax = 40
        
#         timepoints = 200
    
#         S,I,R = simulation(N,I0,R0,beta,gamma,timemax,timepoints,eps=eps,simnum=n)
        
#         S = pd.DataFrame(S)
#         I = pd.DataFrame(I)
#         R = pd.DataFrame(R)
    
#         #S_test = np.array(S)
#         #S_test = S[:,:,np.newaxis]
#         #I_test = np.array(I)
#         #I_test = I[:,:,np.newaxis]
#         #R_test = np.array(R)
#         #R_test = R[:,:,np.newaxis]
        
#         datatrain= pd.concat((S,I,R),axis=1)
        
#         head = []
        
#         for i in range(0,timepoints):
#             head.append('S' + str(i))
            
#         for i in range(0,timepoints):
#             head.append('I' + str(i))
            
#         for i in range(0,timepoints):
#             head.append('R' + str(i))
        
#         datatrain.to_csv('/home/pwendland/Dokumente/GitHub/masterarbeit-philipp/NeuralODE_code/Analyse/SIR/SIR-data_correct_n' + str(n) + '_eps' + str(eps) + '.csv',header=head,index=True,na_rep='NaN')
        
N=2100
I0,R0 = 1,0
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = 2, 1 

timemax = 40

timepoints = 200

S,I,R = simulation(N,I0,R0,beta,gamma,timemax,timepoints,eps=1,simnum=1000)

S = pd.DataFrame(S)
I = pd.DataFrame(I)
R = pd.DataFrame(R)

#S_test = np.array(S)
#S_test = S[:,:,np.newaxis]
#I_test = np.array(I)
#I_test = I[:,:,np.newaxis]
#R_test = np.array(R)
#R_test = R[:,:,np.newaxis]

datatrain= pd.concat((S,I,R),axis=1)

head = []

for i in range(0,timepoints):
    head.append('S' + str(i))
    
for i in range(0,timepoints):
    head.append('I' + str(i))
    
for i in range(0,timepoints):
    head.append('R' + str(i))

datatrain.to_csv('/home/pwendland/Dokumente/GitHub/mnode/NeuralODE_code/SIR-example/SIR_n2100_eps05.csv',header=head,index=True,na_rep='NaN')
