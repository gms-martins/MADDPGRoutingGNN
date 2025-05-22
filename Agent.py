import torch as T
from torch import nn, tensor, rand, optim, device, cat, save, load, softmax
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, GraphConv, NNConv
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import functools
from collections import OrderedDict

from environmental_variables import NEURAL_NETWORK, PATH_SIMULATION, SIM_NR, USE_GNN, STATE_SIZE, TOPOLOGY_TYPE, CRITIC_DOMAIN


class GNNProcessor(nn.Module):
    """
    Processador GNN para transformar o estado da rede antes de enviá-lo ao agente.
    Implementa uma Graph Convolutional Network simples.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNProcessor, self).__init__()
        
        # Camadas GCN
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, output_dim)
        
        # Camada extra para garantir dimensionalidade correta
        #self.fc_output = nn.Linear(output_dim, output_dim)
        
        # Dispositivo para processamento
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x, edge_index):
        # Primeira camada convolucional com ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Segunda camada convolucional
        x = self.conv2(x, edge_index)
        
        return x
        
    def process_state(self, state, graph_data=None):
        """
        Processa o estado usando a GNN.
        Se graph_data não for fornecido, cria um grafo simples.
        """
        # Usar o STATE_SIZE global para todas as topologias
        # Isso garante consistência entre todas as topologias (arpanet, internet, service_provider)
        expected_dim = STATE_SIZE  # Usar o STATE_SIZE configurado para cada topologia
        
        # Manter uma verificação de fallback caso não exista STATE_SIZE por algum motivo
        if expected_dim is None or expected_dim <= 0:
            expected_dim = state.shape[1] if hasattr(state, 'shape') and len(state.shape) > 1 else len(state)
        
        if graph_data is None:
            # Caso não tenhamos dados do grafo, criar um grafo simples linha
            # Este é um fallback e idealmente deve ser substituído por dados reais da topologia
            num_nodes = len(state)
            x = T.tensor(state, dtype=T.float).reshape(-1, 1).to(self.device)
            
            # Criar arestas conectando nós adjacentes (grafo linha)
            edge_index = []
            for i in range(num_nodes-1):
                edge_index.extend([[i, i+1], [i+1, i]])  # Bidirecionais
            
            # Garantir que edge_index está na forma correta (2 x n_edges)
            edge_index = T.tensor(edge_index, dtype=T.long).t().contiguous().to(self.device)
        else:
            # Usar os dados do grafo fornecidos
            x = T.tensor(graph_data['x'], dtype=T.float).to(self.device)
            
            # Certificar que o edge_index está no formato correto (2 x n_edges)
            edge_index = T.tensor(graph_data['edge_index'], dtype=T.long)
            if edge_index.shape[0] != 2:
                # Se as dimensões estiverem invertidas, transpor para obter (2 x n_edges)
                edge_index = edge_index.t()
            edge_index = edge_index.contiguous().to(self.device)
        
        # Processar através da GNN
        output = self(x, edge_index)
        
        # Obter um vetor representativo do grafo (média dos embeddings dos nós)
        mean_output = T.mean(output, dim=0)
        
        # Ajustar para a dimensão esperada pelo ator
        final_output = mean_output[:expected_dim] if mean_output.shape[0] > expected_dim else mean_output
        
        # Se a saída for menor que o esperado, adicionar padding
        if final_output.shape[0] < expected_dim:
            padding = T.zeros(expected_dim - final_output.shape[0], device=self.device)
            final_output = T.cat([final_output, padding])
        
        return final_output.detach().cpu().numpy()


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
        
        # Adicionar processador GNN se habilitado
        self.use_gnn = USE_GNN
        if self.use_gnn:
            # Input_dim=1 (característica por nó), hidden_dim=16, output_dim=fa1 (igual à dim da camada fc1 do ator)
            self.gnn_processor = GNNProcessor(input_dim=1, hidden_dim=16, output_dim=fa1)
        else:
            self.gnn_processor = None

        # Actor usa o estado local (Slocal) para escolher uma ação
        # Critic usa o estado central/global (Sglobal) para avaliar a ação
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
 
    def choose_action(self, observation, graph_data=None):
        # Processar com GNN se estiver habilitada
        if self.use_gnn and self.gnn_processor is not None:
            # PASSO 1: Processar o estado com a GNN
            processed_observation = self.gnn_processor.process_state(observation, graph_data)
            
            # PASSO 2: Preparar o array de observação
            observation_array = np.array([processed_observation], dtype=np.float32)
            
            # PASSO 3: Ajustar as dimensões se necessário
            if observation_array.shape[1] != self.actor.fc1.in_features:
                # Determinar se precisamos adicionar padding ou truncar
                if observation_array.shape[1] < self.actor.fc1.in_features:
                    # Caso 1: A saída da GNN é menor que o esperado - adicionar padding
                    padding = np.zeros((1, self.actor.fc1.in_features - observation_array.shape[1]), dtype=np.float32)
                    observation_array = np.concatenate([observation_array, padding], axis=1)
                    print(f"AVISO: Adicionando padding ao estado processado por GNN: {processed_observation.shape} -> {observation_array.shape}")
                else:
                    # Caso 2: A saída da GNN é maior que o esperado - truncar
                    observation_array = observation_array[:, :self.actor.fc1.in_features]
                    print(f"AVISO: Truncando estado processado por GNN: {processed_observation.shape} -> {observation_array.shape}")
        else:
            # PASSO ALTERNATIVO: Usar a observação original se GNN estiver desabilitada
            observation_array = np.array([observation], dtype=np.float32)
          # Converter para tensor e enviar ao dispositivo apropriado
        state = T.tensor(observation_array, dtype=T.float).to(self.actor.device)
        
        actions = self.actor.forward(state)
        return actions.detach().cpu().numpy()[0]

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau # Taxa de atualização lenta (ex: 0.001)

        # Atualização do ator target 
        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        # Atualização do crítico target e do ator target
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

        # Define a camada de entrada (Scentral = BWs + D)
        #change accordingly
        self.fc1 = nn.Linear(input_dims + n_actions, fc1_dims).float()  # Define a primeira camada da rede neural do crítico
        #self.fc1 = nn.Linear(input_dims + n_agents * n_actions, fc1_dims)
        #self.fc2 = nn.Linear(fc1_dims, fc2_dims) ## hidden layer 1
        #self.q = nn.Linear(fc1_dims, 1)  # Define a camada de saída da rede neural do crítico

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
        self.fc1 = nn.Linear(input_dims, fc1_dims) # Define a primeira camada da rede neural do ator
        #self.fc2 = nn.Linear(fc1_dims, fc2_dims) ##
        self.pi = nn.Linear(fc1_dims, n_actions) # Define a camada de saída da rede neural do ator

        self.optimizer = optim.Adam(self.parameters(), lr=alpha) # Define o otimizador da rede neural do ator
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        #change accordingly
        
        x = F.relu(self.fc1(state))            #activation function ReLU
        #x = F.relu(self.fc2(x)) ##
        #x = self.fc2(x)
        pi = T.softmax(self.pi(x), dim=1)      #hidden layer -> output

        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.load_file))

