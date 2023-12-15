import torch
import numpy as np
from xy_model import XYModel2DWolff, get_randperm_1strow

class Ising2DWolff(XYModel2DWolff):
    def __init__(self, grid_size, theta_arr=None, device='cpu'):
        '''
        Initialize a 2D XY model
        '''
        if theta_arr is None:
            self.state = -1 + 2 * torch.randint(0, 2, (3, grid_size, grid_size)).to(device).to(torch.float64).detach() # see class doc
        else:
            # check validity of theta_arr
            assert theta_arr.shape == (3, grid_size, grid_size) # see class doc
            self.state = theta_arr
        
        self.grid_size = grid_size

        # preallocate space for used matrices
        self._tmp_bond_sampling = torch.zeros_like(self.state[1:])
        self._tmp_already_flipped = torch.zeros_like(self.state[0])
        self._tmp_in_cluster = torch.zeros_like(self.state[0])
        self._tmp_bfs_stack = [] # pretend this is a stack
    

    # Wolff related
    def update_bond(
            self, 
            bond_probability_given_states = None, 
            nW = None, 
            beta = None, 
            J = 1, 
    ):
        '''
        Arguments:
            bond_probability_given_states: callable (s1, s2) -> float \in [0, 1]
                default: Descombes, 2021 (MS thesis)
            nW: The ANGLE of the Wolff plane. Must be specified if using default p
            beta: Temperature parameter. Must be specified if using default p
            J: Bond energy. Default: 1
        '''
        # compose the default sampling probability as in Descombes, 2021 (MS thesis)
        if bond_probability_given_states is None:
            assert nW is not None
            bond_probability_given_states = lambda s1, s2: (1 - torch.exp(-2 * beta * J * s1 * s2)).clamp(0, 1)
        # alias
        p = bond_probability_given_states
        
        # edge below the grid
        self._tmp_bond_sampling[0, :-1, :] = p(self.get_spin()[1:, :], self.get_spin()[:-1, :])
        self._tmp_bond_sampling[0, -1, :] = p(self.get_spin()[-1, :], self.get_spin()[0, :]) # periodic BC
        if not (self._tmp_bond_sampling.min() >= 0) or not self._tmp_bond_sampling.max() <= 1:
            print('badbad!')
        self.get_bond()[0] = torch.bernoulli(self._tmp_bond_sampling[0])

        # edge to the right
        self._tmp_bond_sampling[1, :, :-1] = p(self.get_spin()[:, 1:], self.get_spin()[:, :-1])
        self._tmp_bond_sampling[1, :, -1] = p(self.get_spin()[:, -1], self.get_spin()[:, 0]) # periodic BC
        self.get_bond()[1] = torch.bernoulli(self._tmp_bond_sampling[1])
        
    def _flip_func(self, state, in_cluster, nW):
        return (
            state * (1 - in_cluster) + # not flip spins out of the cluster
            -state * in_cluster # flip the cluster
        )