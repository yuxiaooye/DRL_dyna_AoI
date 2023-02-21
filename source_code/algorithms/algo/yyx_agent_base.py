import os
import torch

class YyxAgentBase():
    '''
    motivation: unify all save_nets() in all methods
    '''
    def __init__(self):
        pass

    def save_nets(self, dir_name, iter=0, is_newbest=False):
        if not os.path.exists(dir_name + '/Models'):
            os.mkdir(dir_name + '/Models')
        prefix = 'best' if is_newbest else str(iter)
        torch.save(self.actors.state_dict(), dir_name + '/Models/' + prefix + '_actor.pt')
        if self.input_args.algo.startswith('G2ANet'):
            torch.save(self.g2a_embed_hard_net.state_dict(), dir_name + '/Models/' + prefix + '_g2aAtt.pt')

        print('RL saved successfully')

    def load_nets(self, dir_name, iter=0, best=False):
        prefix = 'best' if best else str(iter)
        self.actors.load_state_dict(torch.load(dir_name + '/Models/' + prefix + '_actor.pt'))
        if self.input_args.algo.startswith('G2ANet'):
            self.g2a_embed_hard_net.load_state_dict(torch.load(dir_name + '/Models/' + prefix + '_g2aAtt.pt'))


