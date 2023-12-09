import torch
import numpy as np
import copy

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

class Logger:
    '''
    Attributes:
        stamp_steps: list of int, SIMULATION step number of logged states
        states: torch.tensor of shape (max_stamp_steps, *state_size), states of the system
        order_quantities: list of float, order parameters
        other_quantities: list of dict, other quantities
    '''
    def __init__(self, sim_step_num, state_size, stamp_num=None, settings={}, logger=None, device='cpu'):
        '''
        logger: initialize the current logger with a previous logger, continue the simulation
        '''
        if stamp_num is None:
            stamp_num = sim_step_num # if not specified, log all states

        if logger is None: 
            self.stamp_steps = []
            self.stamp_num = stamp_num
            self.sim_step_num = sim_step_num
            # initialize states. preallocate memory for the expensive state storage
            self.states = torch.zeros((self.stamp_num, *state_size)).to(device)
            self.other_quantities = []
            
            # save settings
            if settings == {}:
                print('WARNING: Unset settings!')
            self.settings = settings

        else:
            assert type(logger) is Logger
            self.stamp_steps = copy.deepcopy(logger.stamp_steps)
            self.stamp_num = stamp_num + logger.stamp_num
            self.sim_step_num = sim_step_num + logger.sim_step_num
            # copy all states from previous logger
            self.states = torch.zeros((self.stamp_num, *state_size)).to(device)
            # load from previous state
            self.states[:logger.stamp_num] = logger.states.clone()
            self.other_quantities = copy.deepcopy(logger.other_quantities)

            # save settings
            if logger.settings == settings: 
                print('WARNING: Settings are inconsistent between the current setting and the setting used in the previous logger!')
            if settings == {}:
                print('WARNING: Unset settings. Using that from the previous logger!')
            self.settings = logger.settings

        
    def log(self, sim_step, state: torch.Tensor, **kwargs_other_quantities):
        '''
        Add a record of state to the logger
        
        Parameters:
            record: dict of state, (optional) order parameter and other quantities
        '''
        self.stamp_steps.append(sim_step)
        self.other_quantities.append(kwargs_other_quantities)
        self.states[self.get_current_stamp_idx()] = state.clone()

    def get_current_stamp_idx(self):
        '''
        Return the index of the current stamp, based on the length of the list stamp_steps.
        From 0 to max_stamp_steps - 1.
        '''
        return len(self.stamp_steps) - 1
    
    def get_log(self, stamp_idx, ret_other_quantities=False):
        if ret_other_quantities:
            return self.states[stamp_idx], self.other_quantities[stamp_idx]
        else:
            return self.states[stamp_idx]
    