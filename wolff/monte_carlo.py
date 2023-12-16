import tqdm
from logger import Logger
import numpy as np
import torch

from xy_model import XYModel2DWolff

def wolff_algorithm(
        xy_model: XYModel2DWolff, 
        temperature, 
        num_steps, 
        stamp_interval = 1, 
        logger: Logger = None, 
        flip_all = True, 
):
    # Get lattice size
    L = xy_model.get_spin().size(0)

    # Perform Wolff algorithm for num_steps
    if logger is None:
        logger = Logger(
            num_steps, 
            (3, L, L), 
            num_steps // stamp_interval, 
            {'temperature': temperature, 'num_steps': num_steps, 'grid_size': L, 'algorithm': 'wolff'}, 
        )
        step_offset = 0
    else:
        logger = Logger(
            num_steps, 
            (3, L, L), 
            num_steps // stamp_interval, 
            logger = logger # initialize with the previous logger 
        )
        step_offset = logger.states.shape[0] + 1

    
    for step in tqdm.trange(num_steps):

        # randomly choose a Wolff plane        
        nW = 2 * torch.pi * torch.rand((1, )).item()
        # print(nW)
        # update bond according to the Wolff plane
        xy_model.update_bond(nW=nW, beta=1 / temperature)
        
        # select one cluster according to the bond
        xy_model.flip(nW=nW, flip_one_cluster=not flip_all)

        if step % stamp_interval == 0:
            logger.log(step + step_offset, xy_model.get_state().clone().cpu())
    return logger

