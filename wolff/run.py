import sys
sys.path.append('.')

import numpy as np
import torch
import matplotlib.pyplot as plt

from monte_carlo import wolff_algorithm
from xy_model import XYModel2DWolff
from logger import Logger
import pickle
import os

def show_spin(xy_model: XYModel2DWolff):
    x, y = np.meshgrid(np.arange(xy_model.grid_size), np.arange(xy_model.grid_size))
    spin = xy_model.get_spin()

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(3, 3)
    ax.quiver(x, y, spin.cos(), spin.sin())
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax

def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

get_M = lambda spin: (spin.cos().sum().square() + spin.sin().sum().square()).sqrt() / spin.size(0) ** 2


# %%
# run and save

def run_exp(L, step_num, T):
    xy_model = XYModel2DWolff(L, )
    logger = wolff_algorithm(xy_model, T, step_num)
    pickle.dump(logger, open(f'./save/xy_model_L={L}_T={T}_step_num={step_num}.pkl', 'wb'))
    return logger

if __name__ == '__main__':
    
    if not os.path.exists(os.path.dirname(__file__) + '/save/'):
        os.makedirs(os.path.dirname(__file__) + '/save/')

    set_seed(0)
    step_num = 1000
    L = 10
    Ts = np.linspace(0.1, 1, 100)
    for i, T in enumerate(Ts):
        print(f'{i} runs out of {Ts.shape[0]}')
        run_exp(L, step_num, T)
