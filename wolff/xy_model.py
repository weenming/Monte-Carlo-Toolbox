import numpy as np
import torch
from abc import ABC, abstractmethod

get_randperm_1strow = lambda x: x[torch.randperm(x.size(0))[0]]

class BaseXYModel(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_spin(self):
        '''Get the spin lattice in torch array
        '''
        raise NotImplementedError


class XYModel2D:
    def __init__(self, grid_size, theta_arr=None, device='cpu'):
        '''
        Initialize a 2D XY model
        '''
        if theta_arr is None:
            self.state = 2 * torch.pi * torch.rand((grid_size, grid_size)).to(device)
        else:
            # check validity of theta_arr
            assert theta_arr.shape == (grid_size, grid_size)
            assert theta_arr.max() <= 2 * torch.pi and theta_arr.min() >= 0
            self.state = theta_arr
        
    def get_spin(self):
        '''
        Return the current state of the model
        '''
        return self.state
    

class XYModel2DWolff:
    '''
    XY model with the extra bond activation state. 
    Using periodic boundary condition!

    Attributes: 
        state: (3, grid_size, grid_size). The first dimension stores 
        0: spin
        1: the edge BELOW the current grid
        2: the edge to the RIGHT of the current grid
    '''
    def __init__(self, grid_size, theta_arr=None, device='cpu'):
        '''
        Initialize a 2D XY model
        '''
        if theta_arr is None:
            self.state = 2 * torch.pi * torch.rand((3, grid_size, grid_size)).to(device).to(torch.float64).detach() # see class doc
        else:
            # check validity of theta_arr
            assert theta_arr.shape == (3, grid_size, grid_size) # see class doc
            assert theta_arr.max() <= 2 * torch.pi and theta_arr.min() >= 0
            self.state = theta_arr
        
        self.grid_size = grid_size

        # preallocate space for used matrices
        self._tmp_bond_sampling = torch.zeros_like(self.state[1:])
        self._tmp_already_flipped = torch.zeros_like(self.state[0])
        self._tmp_in_cluster = torch.zeros_like(self.state[0])
        self._tmp_bfs_stack = [] # pretend this is a stack
    
    def get_spin(self):
        '''
        Return the current state of the model, in the scalar of theta
        '''
        return self.state[0]
    
    def get_bond(self):
        return self.state[1: , ...]
    
    def get_state(self):
        return self.state
    

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
            bond_probability_given_states = lambda s1, s2: (
                (
                    ((torch.cos(s1 - nW) * torch.cos(s2 - nW)) > 0).to(torch.float64)
                ) * (
                    1 - torch.exp(-2 * beta * J * torch.cos(s1 - nW) * torch.cos(s2 - nW))
                )
            ).clamp(0, 1)
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
        
    def flip(self, nW, flip_one_cluster=True):
        '''
        Constructs clusters based on the calculated bond, and then flips the
        spin w.r.t. the Wolff plane.

        Arguments:
            nW: the ANGLE of the Wolff plane
            flip_one_cluster: if True, in each iteration, only flip one cluster.
        '''

        if flip_one_cluster:
            self.flip_one_cluster(nW)
        else:
            self.flip_all_clusters(nW)
        
    def flip_one_cluster(self, nW, debug=False):
        # randomly sample a starting point (a cluster)
        i, j = torch.randint(0, self.grid_size, (2, ))
        # reset the used flag matrix before recursion
        self._tmp_in_cluster[...] = 0
        self.bond_dfs(i, j)
        
        # flip theta
        self.state[0] = self._flip_func(self.state[0], self._tmp_in_cluster, nW)
        
        if debug:
            assert self._tmp_bfs_stack == []
            return self._tmp_in_cluster
        # reset, although not necessary for now
        self._tmp_in_cluster[...] = 0

    def flip_all_clusters(self, nW):
        # reset the used flag matrix before recursion
        self._tmp_already_flipped[...] = 0

        while (self._tmp_already_flipped != 1).any():
            # select only from unflipped clusters
            i, j = get_randperm_1strow(torch.nonzero(self._tmp_already_flipped == 0))
            
            self._tmp_in_cluster[...] = 0
            self._tmp_in_cluster[i, j] = 1
            self.bond_dfs(i, j)
            # flip theta in this cluster
            if np.random.rand() > 0.5:
                self.state[0] = self._flip_func(self.state[0], self._tmp_in_cluster, nW)
                
            self._tmp_already_flipped[self._tmp_in_cluster.nonzero()[:, 0], self._tmp_in_cluster.nonzero()[:, 1]] = 1
            # reset, although not necessary for now
            self._tmp_in_cluster[...] = 0
            assert self._tmp_bfs_stack == [] # well this must be empty
        
    def _flip_func(self, state, in_cluster, nW):
        return (
            state * (1 - in_cluster) + # not flip spins out of the cluster
            ((2 * nW - state + torch.pi) % (2 * torch.pi)) * in_cluster # flip the cluster
        )
    
    def bond_dfs(self, i, j):
        '''
        Depth first search given initial i and j. Use 
        '''
        # try four different directions
 
        for i_neighbor, j_neighbor in [
            ((i - 1) % self.grid_size, j), 
            ((i + 1) % self.grid_size, j), 
            (i, (j - 1) % self.grid_size), 
            (i, (j + 1) % self.grid_size), 
        ]:
            self._rec_neighbor(i, j, i_neighbor, j_neighbor)
        
        if len(self._tmp_bfs_stack) == 0: # end recursion
            return
        else:
            return self.bond_dfs(*self._tmp_bfs_stack.pop())
       
    def _rec_neighbor(self, i, j, i_n, j_n):
        '''
        Arguments:
            i, j: i and j of the current node
            i_n, j_n: i and j of the neighbor node
        '''
        hor_ver = 1 if i != i_n else 0 # 1 if vertical, 0 if horizontal
        i_bond, j_bond = min(i, i_n), min(j, j_n) # the bond is always stored in the smaller index
        if not self._tmp_in_cluster[i_n, j_n] and self.get_bond()[hor_ver, i_bond, j_bond]:
            self._tmp_in_cluster[i_n, j_n] = 1
            self._tmp_bfs_stack.append((i_n, j_n))
            return self.bond_dfs(i_n, j_n)
        
    # def bad_bond_dfs(self, i, j):
    #     # NOTE: this bad implementation is commented out
    #     # above (i, j)
    #     if not self._tmp_in_cluster[i - 1, j] and self.get_bond()[0, i - 1, j]:
    #         self._tmp_in_cluster[i - 1, j] = True
    #         self._tmp_bfs_stack.append((i, j))
    #         return self.bond_dfs(i - 1, j)
    #     # below (i, j)
    #     elif not self._tmp_in_cluster[i + 1, j] and self.get_bond()[0, i + 1, j]:
    #         self._tmp_in_cluster[i + 1, j] = True
    #         self._tmp_bfs_stack.append((i, j))
    #         return self.bond_dfs(i + 1, j)
    #     # left to (i, j)
    #     elif not self._tmp_in_cluster[i, j - 1] and self.get_bond()[1, i, j - 1]:
    #         self._tmp_in_cluster[i, j - 1] = True
    #         self._tmp_bfs_stack.append((i, j))
    #         return self.bond_dfs(i, j - 1)
    #     # right to (i, j)
    #     elif not self._tmp_in_cluster[i, j + 1] and self.get_bond()[1, i, j + 1]:
    #         self._tmp_in_cluster[i, j + 1] = True
    #         self._tmp_bfs_stack.append((i, j))
    #         return self.bond_dfs(i, j + 1)
    #
    #     if len(self._tmp_bfs_stack) == 0: # end recursion
    #         return
    #     else:
    #         return self.bond_dfs(*self._tmp_bfs_stack.pop())
        