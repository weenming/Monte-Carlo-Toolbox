import numpy as np
import torch

from wolff.utils import set_seed

class XYModel2D:
    def __init__(self, grid_size, seed=0, theta_arr=None, device='cpu'):
        '''
        Initialize a 2D XY model
        '''
        # set seed
        set_seed(seed)
        if theta_arr is None:
            self.state = 2 * torch.pi * torch.rand((grid_size, grid_size)).to(device)
        else:
            # check validity of theta_arr
            assert theta_arr.shape == (grid_size, grid_size)
            assert theta_arr.max() <= 2 * torch.pi and theta_arr.min() >= 0
            self.state = theta_arr
        
    
    def get_state(self):
        '''
        Return the current state of the model
        '''
        return self.state
    
    def get_lattice(self):
        return self.get_state()
    

class XYModel2DWolff:
    '''
    XY model with the extra bond activation state

    Attributes: 
        state: (3, grid_size, grid_size). The first dimension stores 
        0: spin
        1: the edge BELOW the current grid
        2: the edge to the RIGHT of the current grid
    '''
    def __init__(self, grid_size, seed=0, theta_arr=None, device='cpu'):
        '''
        Initialize a 2D XY model
        '''
        # set seed
        set_seed(seed)
        if theta_arr is None:
            self.state = 2 * torch.pi * torch.rand((3, grid_size, grid_size)).to(device) # see class doc
        else:
            # check validity of theta_arr
            assert theta_arr.shape == (3, grid_size, grid_size) # see class doc
            assert theta_arr.max() <= 2 * torch.pi and theta_arr.min() >= 0
            self.state = theta_arr
        
    
    def get_state(self):
        '''
        Return the current state of the model
        '''
        return self.state
    
    def get_lattice(self):
        return self.get_state()