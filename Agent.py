import torch as T
from torch import nn, tensor, rand, optim, device, cat, save, load, softmax
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, GraphConv, NNConv
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


from environmental_variables import NEURAL_NETWORK, PATH_SIMULATION, SIM_NR


class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, n_agents, agent_idx, chkpt_dir,
                    alpha=0.01, beta=0.01, fc1=64, 
                    fc2=64, fa1=64, fa2=64, gamma=0.95, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        #fa1 = fc1 # tau 0.0001
        #fa2 = fc2 #
        self.agent_name = 'agent_%s' %agent_idx
        self.load_name = self.agent_name
        self.actor = ActorNetwork(alpha, actor_dims, fa1, fa2, n_actions, 
                                  chkpt_dir=chkpt_dir,  name=self.agent_name+'_actor', load_file=self.load_name+'_actor')
        self.critic = CriticNetwork(beta, critic_dims, 
                            fc1, fc2, n_agents, n_actions, 
                            chkpt_dir=chkpt_dir, name=self.agent_name+'_critic', load_file=self.load_name+'_critic')
        self.target_actor = ActorNetwork(alpha, actor_dims, fa1, fa2, n_actions,
                                        chkpt_dir=chkpt_dir, 
                                        name=self.agent_name+'_target_actor', load_file=self.load_name+'_target_actor')
        self.target_critic = CriticNetwork(beta, critic_dims, 
                                            fc1, fc2, n_agents, n_actions,
                                            chkpt_dir=chkpt_dir,
                                            name=self.agent_name+'_target_critic', load_file=self.load_name+'_target_critic')

        self.update_network_parameters(tau=1)
 
    def choose_action(self, observation):
        observation_array = np.array([observation], dtype=np.float32)
        #print("\nobservation array []: ", observation_array)
        state = T.tensor(observation_array, dtype=T.float).to(self.actor.device)

        actions = self.actor.forward(state)
        noise = T.rand(self.n_actions).to(self.actor.device)
        action = actions ##
        #action = actions + noise
        return action.detach().cpu().numpy()[0]

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
                 n_agents, n_actions, name, chkpt_dir, load_file):
        super(CriticNetwork, self).__init__()

        self.file_name = f"{name}.sync"  # os.path.join(chkpt_dir, name)
        self.chkpt_file = f'{PATH_SIMULATION}/agent_files{SIM_NR}/{self.file_name}'
        self.load_file = f'{PATH_SIMULATION}/agent_files{SIM_NR}/{load_file}.sync'


        #change accordingly
        self.fc1 = nn.Linear(input_dims + n_actions, fc1_dims).float()
        #self.fc1 = nn.Linear(input_dims + n_agents * n_actions, fc1_dims)
        #self.fc2 = nn.Linear(fc1_dims, fc2_dims) ## hidden layer 1
        #self.q = nn.Linear(fc1_dims, 1)

        if NEURAL_NETWORK == "duelling_q_network":
            self.q = nn.Linear(fc1_dims, 1)          #       
            self.q_values = nn.Linear(fc1_dims, n_actions)      #advantage
            self.output = nn.Linear(n_actions, 1)
        elif NEURAL_NETWORK == "simple_q_network":
            self.q = nn.Linear(fc1_dims, 1)                     #1 dimention output

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        #change accordingly
        if NEURAL_NETWORK == "duelling_q_network":
            x = F.relu(self.fc1(T.cat([state, action], dim=1)))
            #x = F.relu(self.fc2(x)) ##
            value = self.q(x)   #output 1
            
            q_values = T.softmax(self.q_values(x), dim=1)  
            average = T.mean(q_values, dim = 1, keepdim=True)
            advantage_function = q_values - average

            q = value + advantage_function         #q = value + (q_value - average q values)

            #q = self.output(q)  ##verificar

            q , _= T.max(q, dim=1, keepdim=True)

        elif NEURAL_NETWORK == "simple_q_network":
            x = F.relu(self.fc1(T.cat([state, action], dim=1)))
            #x = F.relu(self.fc2(x)) ##
            q = self.q(x)
            #print("\n simple q network q: ", q)
            #print("\n shape: ", q.shape)
    
        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.load_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions, name, chkpt_dir, load_file):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = f"{name}.sync"  # os.path.join(chkpt_dir, name)
        self.chkpt_file = f'{PATH_SIMULATION}/agent_files{SIM_NR}/{self.chkpt_file}'
        self.load_file = f'{PATH_SIMULATION}/agent_files{SIM_NR}/{load_file}.sync'

        #change accordingly
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        #self.fc2 = nn.Linear(fc1_dims, fc2_dims) ##
        self.pi = nn.Linear(fc1_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        #change accordingly
        
        x = F.relu(self.fc1(state))            #activation function
        #x = F.relu(self.fc2(x)) ##
        #x = self.fc2(x)
        pi = T.softmax(self.pi(x), dim=1)      #hidden layer -> output

        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.load_file))

