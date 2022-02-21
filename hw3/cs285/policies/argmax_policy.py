import imp
import numpy as np
from cs285.critics.dqn_critic import DQNCritic

class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        self.critic : DQNCritic
        # 注意其维度应该是N*n，其中N为obs的个数，n为action space的维度
        actions = np.argmax(self.critic.qa_values(obs),axis=-1)
        return actions.squeeze()