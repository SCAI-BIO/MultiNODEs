import numpy as np

def padding(values, timesteps):
    
    timepoints = np.arange(timesteps[-1]+1)
    
    Xd = np.zeros(shape = (values.shape[0], len(timepoints), values.shape[-1]))
    for x in range(len(values)):
      idx = 0
      for t in timepoints:
        if t in timesteps:
            Xd[x,t] = values[x][idx]
            idx += 1
        else:
            mynans = np.zeros(shape=values.shape[-1])
            mynans.fill(np.nan)
            Xd[x,t] = mynans
    
    return np.array(Xd)
    
    
def weighter(values):
    
    weight_matrix = np.where(np.isnan(values)==True, 0.0, 
             np.where(np.isnan(values)==False, 1, values))
    values_matrix = np.where(np.isnan(values)==True, 0.0, values)
    
    return values_matrix, weight_matrix
    