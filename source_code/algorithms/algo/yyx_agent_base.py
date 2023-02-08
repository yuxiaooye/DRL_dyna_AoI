import os
import torch

class YyxAgentBase():
    '''
    motivation: unify all save_nets() in all methods
    '''
    def __init__(self):
        pass

    def save_nets(self, dir_name, episode=0, is_newbest=False):
        if not os.path.exists(dir_name + '/Models'):
            os.mkdir(dir_name + '/Models')
        prefix = 'best' if is_newbest else str(episode)
        # TODO only save actors now
        torch.save(self.actors.state_dict(), dir_name + '/Models/' + prefix + '_actor.pt')
        print('RL saved successfully')