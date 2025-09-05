import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# ==================================================================#
# ==================================================================#
def log_normal_pdf(data, mean, logvar):
    #import ipdb; ipdb.set_trace()
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(data.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (data - mean) ** 2. / torch.exp(logvar))


# ==================================================================#
# ==================================================================#
class RunningAverageMeter(object):
    # Computes and stores the average and current values
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

# ==================================================================#
# ==================================================================#
def define_logs(config):
    import os
    import time
    typ = 'a' if os.path.exists(config.save_path_losses) else 'wt'
    with open(config.save_path_losses, typ) as opt_file:
        now = time.strftime("%c")
        opt_file.write('================ Training Loss (%s) ================\n' % now)
    
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(config).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    opt_name = os.path.join(config.save_path, 'train_opt.txt')
    with open(opt_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


# ==================================================================#
# ==================================================================#
def print_current_losses(epoch, iters, losses, t_epoch, t_comp,
                         log_name, s_excel=True):
    """print current losses on console; also save the losses to the disk
    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        t_comp (float) -- computational time per data point (normalized by batch_size)
        t_data (float) -- data loading time per data point (normalized by batch_size)
    """
    t_epoch = human_format(t_epoch)
    t_comp = human_format(t_comp)
    # 
    if isinstance(t_epoch, float)  and isinstance(t_comp, float):
        message = '[epoch: %d], [iters: %d], [epoch time: %.3f], [total time: %.3f] ' % (epoch, iters, t_epoch, t_comp)
    elif isinstance(t_epoch, float) and isinstance(t_comp, str):
        message = '[epoch: %d], [iters: %d], [epoch time: %.3f], [total time: %s] ' % (epoch, iters, t_epoch, t_comp)
    elif isinstance(t_epoch, str) and isinstance(t_comp, float):
        message = '[epoch: %d], [iters: %d], [epoch time: %s], [total time: %.3f] ' % (epoch, iters, t_epoch, t_comp)
    else:
        message = '[epoch: %d], [iters: %d], [epoch time: %s], [total time: %s] ' % (epoch, iters, t_epoch, t_comp)

    for k, v in losses.items():
        message += '%s=%.6f, ' % (k, v)
    message = message[:-2]
    print(message)  # print the message
    with open(log_name, 'a') as log_file:
        log_file.write('%s\n' % message)  # save the message

    if s_excel:
        # Save the losses in an excel sheet and plot in an image
        excel_name = log_name[:-3] + 'xlsx'
        data = {}

        # It is necessary to do 2 times this for
        # to get the complete vectors for the graphs
        if os.path.exists(excel_name):
            loss_df = pd.read_excel(excel_name, index_col=0)
            for k in loss_df.keys():
                vect = loss_df[k].values.tolist()
                if k == 'Epoch':
                    vect.append(epoch)
                else:
                    vect.append(losses[k])
                data[k] = vect
        else:
            data['Epoch'] = epoch
            for k, v in losses.items():
                data[k] = [v]

        df = pd.DataFrame(data)
        df.to_excel(excel_name)

        if epoch > 1:
            for k in loss_df.keys():
                if k == 'Epoch':
                    time_vect = loss_df[k].values.tolist()
                else:
                    vect = loss_df[k].values.tolist()           
                    plt.plot(time_vect, vect, label=k, linewidth=2)

            img_name = log_name[:-3] + 'png'
            plt.legend()
            plt.title('Losses')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.savefig(img_name)
            plt.close()

def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    if magnitude == 0:
        # return str(num)
        return num
    else:
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
