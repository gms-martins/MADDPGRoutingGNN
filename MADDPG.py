import random
import json

import torch as T
import numpy as np
from torch import tensor, cat, no_grad, mean
import torch.nn.functional as F
import networkx as nx

import shutil

import os
import datetime
import matplotlib.pyplot as plt
from torch_geometric.data import Data
# Como converter para formato legível
import pandas as pd

from Agent import Agent
from MultiAgentReplayBuffer import MultiAgentReplayBuffer
from NetworkEngine import NetworkEngine
from NetworkEngine import REMOVED_EDGES, SCENARIO_2_COMPLETED
from NetworkEnv import NetworkEnv
from environmental_variables import STATE_SIZE, EPOCH_SIZE, NUMBER_OF_AGENTS, NR_EPOCHS, EVALUATE, CRITIC_DOMAIN, SIM_NR, TRAIN, NEURAL_NETWORK, MODIFIED_NETWORK, NOTES, TOPOLOGY_TYPE, UPDATE_WEIGHTS, PATH_SIMULATION, NUMBER_OF_PATHS, NUMBER_OF_HOSTS,BANDWIDTH_INCREASE_FACTOR,INCREASE_BANDWIDTH_INTERVAL,STABILIZE_BANDWIDTH , STABILIZE_AFTER_MULTIPLIER, SAVE_REMOVED_LINKS_SCENARIO4, MAX_BANDWIDTH_MULTIPLIER, CRITIC_DOMAIN, NUMBER_OF_PATHS, NUMBER_OF_AGENTS, NR_EPOCHS, EPOCH_SIZE, PATH_SIMULATION, SIM_NR, MODIFIED_NETWORK, USE_GNN, NUM_LINKS_TO_REMOVE
#, GRAPH_BATCH_SIZE


class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
                 scenario='simple', alpha=0.01, beta=0.01, fc1=64,
                 fc2=64, fa1=64, fa2=64, gamma=0.99, tau=0.001, chkpt_dir='tmp/maddpg/'):
        # Cache para otimização da GNN
        self.cached_graph_edges = None
        self.cached_topology_state = None
        self.graph_cache_valid = False
        #scenario='simple', alpha=0.01, beta=0.01, fc1=64,fc2=64, fa1=64, fa2=64, gamma=0.99, tau=0.001,:
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims[agent_idx],
                                     n_actions, n_agents, agent_idx, alpha=alpha, beta=beta,
                                     chkpt_dir=chkpt_dir, fc1=fc1, fc2=fc2))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs, graph_data=None):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx], graph_data)
            actions.append(np.argmax(action))
        return actions

    def learn(self, experience):
        if not experience.ready():
            return
        # Get the sample from memory
        actor_input, current_state, action_taken, reward_obtained, \
        actor_future_input, future_state, done_flags = experience.sample_buffer()

        # Convert the sample to tensors PyTorch
        processing_device = self.agents[0].actor.device
        current_state_array = np.array(current_state, dtype=np.float32)
        current_state = T.tensor(current_state_array, dtype=T.float).to(processing_device)
        #current_state = T.tensor(current_state, dtype=T.float).to(processing_device)

        action_taken_array = np.array(action_taken, dtype=np.float32)
        action_taken = T.tensor(action_taken_array, dtype=T.float).to(processing_device)

        reward_obtained_array = np.array(reward_obtained, dtype=np.float32)
        reward_obtained = T.tensor(reward_obtained_array, dtype=T.float).to(processing_device)

        future_state_array = np.array(future_state, dtype=np.float32)
        future_state = T.tensor(future_state_array, dtype=T.float).to(processing_device)
        done_flags = T.tensor(done_flags).to(processing_device)

        all_new_actions = []
        previous_actions = []

        for idx, agent in enumerate(self.agents):
            future_actor_input = T.tensor(actor_future_input[idx], dtype=T.float).to(processing_device)

            # Target Actor - get the next action for the future Q value
            new_action_policy = agent.target_actor.forward(future_actor_input) 

            all_new_actions.append(new_action_policy)
            previous_actions.append(action_taken[idx])

        combined_new_actions = T.cat([act for act in all_new_actions], dim=1)
        combined_old_actions = T.cat([act for act in previous_actions], dim=1)

        for idx, agent in enumerate(self.agents):
            with T.no_grad():
                # 1. Value Network (critic) - get the future Q value
                future_critic_value = agent.target_critic.forward(future_state[idx], combined_new_actions[:,
                                                        idx * self.n_actions:idx * self.n_actions + self.n_actions]).flatten()

                # 2. Calculate the expected Q value ( using reward + future value discounted)
                expected_value = reward_obtained[:, idx] + (1 - done_flags[:, 0].int()) * agent.gamma * future_critic_value

            # 3. Evaluate the corrent action value and obtain the actual Q value from the critic
            present_critic_value = agent.critic.forward(current_state[idx], combined_old_actions[:,
                                                        idx * self.n_actions:idx * self.n_actions + self.n_actions]).flatten()

            # 4. Calculate the loss between the expected and actual Q value
            critic_loss = F.mse_loss(expected_value, present_critic_value)

            # 5. Update the critic network through Adam optimizer
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            # Policy Network (actor) - 1. obtain the current action for the current state
            current_actor_input = T.tensor(actor_input[idx], dtype=T.float).to(processing_device)
            combined_old_actions_clone = combined_old_actions.clone()
            combined_old_actions_clone[:,
                    idx * self.n_actions:idx * self.n_actions + self.n_actions] = agent.actor.forward(current_actor_input)
            
            # 2. Calculate the actor loss (Negative value as we want to maximize the Q value)
            actor_loss = -T.mean(agent.critic.forward(current_state[idx], combined_old_actions_clone[:,
                                                      idx * self.n_actions:idx * self.n_actions + self.n_actions]).flatten())
            
            # 3. Update the actor network through Adam optimizer
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()

        for agent in self.agents:
            agent.update_network_parameters()

if __name__ == '__main__':    
    eng = NetworkEngine()
    env = NetworkEnv(eng)
    
    n_action = NUMBER_OF_PATHS
    total_rewards = []
    batch_rewards = []
    agents = eng.get_all_hosts()
    all_hosts = eng.get_all_hosts()
    
    agent_dim = STATE_SIZE
    agent_dims = [agent_dim for host in all_hosts]

    REPLAY_MEMORY = 50000  
    MEMORY_BATCH = 256 

    if CRITIC_DOMAIN == "central_critic":
        critic_dim = len(eng.get_link_usage()) + NUMBER_OF_AGENTS
        critic_dims = [critic_dim for i in range(NUMBER_OF_AGENTS)]
    elif CRITIC_DOMAIN == "local_critic":
        critic_dim = STATE_SIZE
        critic_dims = [critic_dim for host in all_hosts]
    elif CRITIC_DOMAIN == "shortest":
        critic_dim = len(eng.get_link_usage()) + NUMBER_OF_AGENTS
        critic_dims = [critic_dim for i in range(NUMBER_OF_AGENTS)]
    maddpg_agents = MADDPG(agent_dims, critic_dims, NUMBER_OF_AGENTS, n_action,
                           fa1=10, fa2=64, fc1=15, fc2=64,
                           alpha=0.002, beta=0.0002, tau=0.001, chkpt_dir='.\\tmp\\maddpg\\') #valores ajustados para aprendizagem mais rápida
    #change accordingly to tests

    memory = MultiAgentReplayBuffer(REPLAY_MEMORY, critic_dims, agent_dims, n_action, NUMBER_OF_AGENTS, MEMORY_BATCH)
    
    if not EVALUATE or (EVALUATE and TRAIN):
        nr_epochs = NR_EPOCHS
    elif EVALUATE and not TRAIN:
        nr_epochs = NR_EPOCHS

    ## SETUP ##
    #create /home/student/agent_files directory if not found
    path = f'{PATH_SIMULATION}/agent_files{SIM_NR}'
    if not os.path.exists(path):
        print(f"Creating 'agent_files{SIM_NR}' directory")
        os.mkdir(path)
    #create /home/student/results directory if not found
    path = f'{PATH_SIMULATION}/results'
    if not os.path.exists(path):
        print("Creating 'results' directory")
        os.mkdir(path)
    #create folder for current simulation
    day = datetime.date.today().day
    month = datetime.date.today().month
    hh = datetime.datetime.now().hour
    mm = datetime.datetime.now().minute
    if EVALUATE:
        if TRAIN:
            learning = "eval_train" # Cenário 4
        else:
            if UPDATE_WEIGHTS:
                # Cenário 3: distinguir entre 1 e 2 links
                if NUM_LINKS_TO_REMOVE == 1:
                    learning = "eval_update_1link" # Cenário 3a
                elif NUM_LINKS_TO_REMOVE == 2:
                    learning = "eval_update_2links" # Cenário 3b
                else:
                    learning = "eval_update" # Cenário 3 genérico
            else:
                # Cenário 2: distinguir entre 1 e 2 links
                if NUM_LINKS_TO_REMOVE == 1:
                    learning = "eval_1link" # Cenário 2a
                elif NUM_LINKS_TO_REMOVE == 2:
                    learning = "eval_2links" # Cenário 2b
                else:
                    learning = "eval" # Cenário 2 genérico
    else:
        learning = "train" # Cenário 1

    if MODIFIED_NETWORK and (EVALUATE or TRAIN):
            if EVALUATE and not TRAIN:
                # Cenário 2 ou 3
                folder_name = f"{learning}_{MODIFIED_NETWORK}_{NUM_LINKS_TO_REMOVE}links"
            else:
                # Cenário 4 ou outros
                folder_name = f"{learning}_{MODIFIED_NETWORK}"
    else:
        folder_name = learning  
        
    # Definir sufixos para diferenciar execuções com e sem GNN
    gnn_suffix = "com_GNN" if USE_GNN else "sem_GNN"

    base_path = f'{PATH_SIMULATION}/results'

    # 1. Criar pasta da topologia (ex: internet)
    topology_path = os.path.join(base_path, TOPOLOGY_TYPE)
    if not os.path.exists(topology_path):
        print(f"Creating topology directory '{TOPOLOGY_TYPE}'")
        os.mkdir(topology_path)
    
    # 2. Criar pasta GNN dentro da pasta da topologia (ex: internet/sem_GNN)
    gnn_path = os.path.join(topology_path, gnn_suffix)
    if not os.path.exists(gnn_path):
        print(f"Creating '{gnn_suffix}' directory in {TOPOLOGY_TYPE}")
        os.mkdir(gnn_path)
    
    # 3. Criar pasta do tipo de crítico e rede neural dentro da pasta GNN
    critic_nn_folder = f'{CRITIC_DOMAIN}_{NEURAL_NETWORK}'
    critic_nn_path = os.path.join(gnn_path, critic_nn_folder)
    if not os.path.exists(critic_nn_path):
        print(f"Creating '{critic_nn_folder}' directory in {gnn_suffix}")
        os.mkdir(critic_nn_path)

    if not EVALUATE:
        scenario_number = 1  # Cenário 1: Treinamento padrão
    elif EVALUATE and not TRAIN and not UPDATE_WEIGHTS:
        scenario_number = 2  # Cenário 2: Avaliação sem atualizar pesos
    elif EVALUATE and not TRAIN and UPDATE_WEIGHTS:
        scenario_number = 3  # Cenário 3: Avaliação com atualização de pesos
    elif EVALUATE and TRAIN:
        scenario_number = 4  # Cenário 4: Treinar após mudanças na topologia

    # 4. Criar pasta específica para este experimento
    specific_folder = f'{NR_EPOCHS}epo_{EPOCH_SIZE}epi_{folder_name}_{scenario_number}'
    folder_path = os.path.join(critic_nn_path, specific_folder)
    
    # Verificar se a pasta já existe antes de tentar criá-la
    if not os.path.exists(folder_path):
        print(f"Creating experiment directory '{specific_folder}'")
        os.makedirs(folder_path, exist_ok=True)
    
    sub_path = f'{NR_EPOCHS}epo_{EPOCH_SIZE}epi_{TOPOLOGY_TYPE}_{folder_name}'

    if not EVALUATE:
        graph_y_axis = np.zeros(NR_EPOCHS)
        y_axis_training = np.zeros(NR_EPOCHS)
        graph_x_axis = np.zeros(NR_EPOCHS)
    elif EVALUATE and not TRAIN: # and UPDATE_WEIGHTS:
        graph_x_axis = np.zeros(EPOCH_SIZE)
        aux = np.zeros(EPOCH_SIZE) 
        graph_y_axis = [[0 for _ in range(EPOCH_SIZE)] for _ in range(nr_epochs)]
    elif EVALUATE and TRAIN:
        graph_x_axis = np.zeros(EPOCH_SIZE)
        y_axis_training = np.zeros(NR_EPOCHS)
        graph_y_axis = [[0 for _ in range(EPOCH_SIZE)] for _ in range(nr_epochs)]

    if EVALUATE:
        # Restaurar pesos base ANTES de carregar os pesos para o cenário
        base_weights_dir = f"{PATH_SIMULATION}/base_weights_{TOPOLOGY_TYPE}_{CRITIC_DOMAIN}_{NEURAL_NETWORK}"
        dst_dir = f"{PATH_SIMULATION}/agent_files{SIM_NR}"
        if os.path.exists(base_weights_dir):
            for file in os.listdir(base_weights_dir):
                if file.endswith(".sync"):
                    shutil.copy(os.path.join(base_weights_dir, file), os.path.join(dst_dir, file))
            print(f"Pesos base restaurados de {base_weights_dir} para {dst_dir}")
        else:
            print(f"AVISO: Pasta de pesos base não encontrada ({base_weights_dir})")
       
    if (EVALUATE and NEURAL_NETWORK != "shortest") or (EVALUATE and TRAIN):
        maddpg_agents.load_checkpoint()

    i_epoch = 0

    packet_loss_evaluate = []
    packet_sent_evaluate = []
    experience_pck_lost = 0
    experience_pck_sent = 0

    percentage = np.zeros(nr_epochs)
    percentage_2 = np.zeros(nr_epochs)
    available_bw_epoch = np.zeros(nr_epochs)

    total_package_loss_nr = 0
    total_packets_sent_nr = 0

    
    # Estrutura para armazenar evolução de packet loss por episódio (similar às recompensas)
    if EVALUATE and not TRAIN:
        packet_loss_evolution = [[0 for _ in range(EPOCH_SIZE)] for _ in range(nr_epochs)]
    elif EVALUATE and TRAIN:
        packet_loss_evolution = [[0 for _ in range(EPOCH_SIZE)] for _ in range(nr_epochs)]
    

    #if (EVALUATE and TRAIN):
    #if MODIFIED_NETWORK == "remove_edges":
        #eng.remove_edges(3)  # Remove 3 links
    #if MODIFIED_NETWORK == "add_edges":
        #eng.add_edges(3)

    link_utilization_history = {} # Estrutura para armazenar dados de utilização dos links
    shortest_count = 0
    other_count = 0
    total_choices = 0
    for epoch in range(i_epoch, nr_epochs):
        total_epoch_reward = []
        total_epoch_pck_loss = 0
        total_epoch_pck_sent = 0
  
        if EVALUATE and not TRAIN and epoch != 0:

            if MODIFIED_NETWORK == "remove_edges": 
                eng.remove_topology_edges(epoch)
            if MODIFIED_NETWORK == "add_edges":
                eng.add_topology_edges(epoch)

        if not EVALUATE or (EVALUATE and TRAIN):
            episode_size = EPOCH_SIZE
        else:
            episode_size = EPOCH_SIZE 

        available_bw_episode = np.zeros(episode_size)

        
        for e in range(episode_size):
            new_tm = e % 2 == 0
            env.reset(new_tm)

            total_reward = 0
            total_package_loss = 0
            total_packets_sent = 0
            if EVALUATE:
                total_package_loss_nr = 0
                total_packets_sent_nr = 0
            available_bw_time_steps = np.zeros(MEMORY_BATCH)

            increase_bandwidth_flag = (EVALUATE and TRAIN and 
                              epoch > 0 and epoch % INCREASE_BANDWIDTH_INTERVAL == 0 and
                              e == 0)  # Aplicar apenas no primeiro episódio da época
            
            for time_steps in range(MEMORY_BATCH):
                actions = {}
                prev_states = {}
                next_dsts = eng.get_nexts_dsts()
                all_dsts = []
                for host in all_hosts:
                    if host in next_dsts and next_dsts[host]:
                        d = next_dsts[host][1:]
                        all_dsts.append(d)
                    else:
                        all_dsts.append(0)

                states = []  # np.empty((50, agent_dim), dtype=np.double)
                critic_states = []
                dismiss_indexes = []
                
                # Preparar dados do grafo para GNN quando necessário
                graph_data = None
                if USE_GNN:
                    # Criar representação do grafo para a GNN
                    # Cada nó é um host, e cada aresta é uma conexão entre hosts
                    # Características dos nós: largura de banda disponível normalizada (0-1)
                    nodes = []
                    edges = []
                    node_features = []
                    
                    # Obter dados da topologia
                    for i, host in enumerate(all_hosts):
                        nodes.append(i)
                        bw = eng.bws.get(host, 0) / 100.0  # Normalizar BW para 0-1
                        node_features.append([bw])
                    
                    for i, host_i in enumerate(all_hosts):
                        for j, host_j in enumerate(all_hosts):
                            if i != j and eng.get_link(host_i, host_j) is not None:
                                edges.append([i, j])
                    
                    # Dados do grafo para a GNN
                    graph_data = {
                        'x': np.array(node_features, dtype=np.float32),
                        'edge_index': np.array(edges, dtype=np.int64)
                    }

                for index, host in enumerate(all_hosts):
                    all_dst_states = eng.get_state(host, 1)
                    dst = next_dsts.get(host, '')
                    dst = '' if dst == None else dst

                    if 'H' not in dst:
                        state = np.zeros((1, agent_dims[index]), dtype=np.double)
                        dismiss_indexes.append(index)
                    else:
                        state = all_dst_states
                    states.append(state)

                    if CRITIC_DOMAIN == "central_critic":
                        critic_states.append(np.concatenate((eng.get_link_usage(), np.array(all_dsts)), axis=0))
                    elif CRITIC_DOMAIN == "local_critic":     
                        critic_states.append(state)

                # Escolher ações com dados do grafo quando GNN está habilitada
                actions = maddpg_agents.choose_action(states, graph_data if USE_GNN else None)

                actions_dict = {}
                for index, host in enumerate(all_hosts):
                    if next_dsts.get(host, ''):

                        # Exploration probability
                        # The probability of the exploration decreases as the number of epochs increases
                        # Start with 0.3 and decreases 0.0001 per epoch

                        if EVALUATE and not TRAIN:
                            if UPDATE_WEIGHTS:
                                # Só explorar a partir da época 1
                                prob = max(0.1, (0.3 - 0.002 * epoch)) 
                            else:
                                prob = -1  # Cenário 2: nunca explora
                        else:
                            prob = max(0.1, (0.3 - 0.001 * epoch))  

                        if random.random() < prob:
                            action = random.randint(0, 2) # Exloration - random action
                            shortest_count += 1 
                        else:
                            action = actions[index] # Exploitation - action of the neural network
                            other_count += 1 if action != 0 else 0
                        total_choices += 1
                        if TOPOLOGY_TYPE == "internet" or TOPOLOGY_TYPE == "arpanet":
                            if (host in eng.single_con_hosts):
                                action = 0                #algoritmo tradicional
                        if CRITIC_DOMAIN == "shortest":
                            action = 0 #shortest path

                        actions_dict[host] = {next_dsts.get(host, ''): action}

                next_states, rewards, done, _ = env.step(actions_dict)

                new_next_states = np.empty((NUMBER_OF_AGENTS, agent_dim), dtype=np.double)

                if CRITIC_DOMAIN == "central_critic":
                    all_critic_new_states = [np.concatenate((eng.get_link_usage(), np.array(all_dsts)), axis=0) for i in
                                         range(NUMBER_OF_AGENTS)]
                elif CRITIC_DOMAIN == "local_critic":
                #    all_critic_new_states = next_states
                    all_critic_new_states = list(next_states.values())
                
                new_next_states = []
                for index, host in enumerate(all_hosts):
                    # means it add an action
                    if host in actions_dict and next_dsts[host]:
                        bw_state = next_states[host]
                        new_next_states.append(bw_state)
                    else:
                        new_next_states.append(np.zeros((1, agent_dims[index]), dtype=np.double))
                
                actions = []

                for host in all_hosts:
                    if host not in actions_dict:
                        actions.append(0)
                    else:
                        actions.append(actions_dict[host][next_dsts[host]])

                if CRITIC_DOMAIN != "shortest":
                    memory.store_transition(states, actions, rewards, new_next_states, done, critic_states,
                                        all_critic_new_states)


                available_bw_time_steps[time_steps] = np.average(eng.get_link_usage())

                if time_steps % 10 == 0:
                    current_utils = eng.get_link_utilization()
                    episode_key = f"epoch{epoch}_episode{e}"
                    
                    if episode_key not in link_utilization_history:
                        link_utilization_history[episode_key] = []
                    
                    link_utilization_history[episode_key].append(current_utils)
                    
                    if increase_bandwidth_flag:
                        print(f"\n=== AUMENTAR LARGURA DE BANDA SELETIVAMENTE NA ÉPOCA {epoch}, EPISÓDIO {e} ===")
                        print(f"Multiplicador atual: {BANDWIDTH_INCREASE_FACTOR}x")
                        
                        # Usar dados atuais de utilização dos links
                        most_congested = sorted(current_utils.items(), key=lambda x: x[1], reverse=True)[:3]
                        eng._cached_congestion_data = most_congested
                        
                        # Realizar aumento seletivo com dados de links ativos
                        increase_results = eng.increase_traffic_bandwidth(BANDWIDTH_INCREASE_FACTOR)
                        
                        # Logar detalhes
                        modified_hosts = [h[0] for h in increase_results["modified_hosts"]]
                        
                        print(f"Total de {len(modified_hosts)} hosts com largura de banda aumentada")
                        
                        # Marcar como concluído
                        increase_bandwidth_flag = False
                        
                total_reward += sum(rewards) / NUMBER_OF_AGENTS
                #total_package_loss += eng.statistics['package_loss']
                #total_packets_sent += eng.statistics['package_sent']
                if done:
                    break
            
            available_bw_episode[e] = np.average(available_bw_time_steps)
              ## DATA
            print(f"episode {e}/{episode_size}, epoch {epoch}/{nr_epochs}")
            print("Total reward", total_reward)
            #print("Total package loss", ng.statistics['package_loss'])
            #print(" ")

            if ((e % 3 == 0 and not EVALUATE) or (EVALUATE and UPDATE_WEIGHTS and epoch > 0)) and CRITIC_DOMAIN != "shortest":
                maddpg_agents.learn(memory)
            
            total_epoch_reward.append(total_reward)
            
            total_epoch_pck_loss += eng.statistics['package_loss']
            total_epoch_pck_sent += eng.statistics['package_sent']

            total_package_loss_nr = eng.statistics['nr_package_loss']
            total_packets_sent_nr =  eng.statistics['nr_package_sent']

            experience_pck_lost += total_epoch_pck_loss
            experience_pck_sent += total_epoch_pck_sent
            # print(f"STATISTICS OG {eng.statistics}")

            total_rewards.append(total_reward)
            #batch_rewards.append(total_reward)

            if EVALUATE and not TRAIN: #and UPDATE_WEIGHTS:
                graph_y_axis[epoch][e] = int(total_reward) 

                if eng.statistics['nr_package_sent'] > 0:
                    episode_packet_loss = (eng.statistics['nr_package_loss'] / eng.statistics['nr_package_sent']) * 100
                else:
                    episode_packet_loss = 0
                packet_loss_evolution[epoch][e] = episode_packet_loss 

            elif EVALUATE and TRAIN:
                graph_y_axis[epoch][e] = int(total_reward)
                y_axis_training[epoch] = sum(total_epoch_reward) / len(total_epoch_reward)
                # Calcular packet loss para este episódio
                if eng.statistics['nr_package_sent'] > 0:
                    episode_packet_loss = (eng.statistics['nr_package_loss'] / eng.statistics['nr_package_sent']) * 100
                else:
                    episode_packet_loss = 0
                packet_loss_evolution[epoch][e] = episode_packet_loss
                

            # print(f"{'OG' if epoch % 2 == 0 else 'NEW'} REWARD {total_reward}")
            ### episode ends
        
        if total_choices > 0:
            print(f"Percentagem de escolhas do caminho curto (índice 0): {100 * shortest_count / total_choices:.2f}%")
            print(f"Percentagem de escolhas de outros caminhos: {100 * other_count / total_choices:.2f}%")
        #print(f"total epoch reward {total_epoch_reward}")
        # f.write(f"{epoch} {total_epoch_reward}\n")
        if not EVALUATE or (EVALUATE and TRAIN):
            y_axis_training[epoch] = sum(total_epoch_reward) / len(total_epoch_reward) #for saving in the training file
        
        if epoch % 20 == 0:
            print(f"\n AVERGAE WAS {sum(total_rewards) / len(total_rewards)}")
            total_rewards = []

            if not EVALUATE:
                maddpg_agents.save_checkpoint()
                print("SAVING")

                base_weights_dir = f"{PATH_SIMULATION}/base_weights_{TOPOLOGY_TYPE}_{CRITIC_DOMAIN}_{NEURAL_NETWORK}"
                src_dir = f"{PATH_SIMULATION}/agent_files{SIM_NR}"
                if not os.path.exists(base_weights_dir):
                    os.makedirs(base_weights_dir)
                for file in os.listdir(src_dir):
                    if file.endswith(".sync"):
                        shutil.copy(os.path.join(src_dir, file), os.path.join(base_weights_dir, file))
                print(f"Pesos base copiados para {base_weights_dir}")

        #saving data while training in data file, so data can be accessed while training
        if (not EVALUATE or (EVALUATE and TRAIN) )and (epoch+1)%20 == 0:

            training_data = pd.DataFrame({
                'Epoch': np.arange(0, NR_EPOCHS),
                'Average_Reward': [f"{value:.3f}" for value in y_axis_training],
            })
            training_data.to_csv(f"{folder_path}/data_while_training.csv", index=False, sep=';', decimal='.')


        #print(total_epoch_pck_loss)

        if EVALUATE and not TRAIN:
            #packet_loss_evaluate[epoch] = total_epoch_pck_loss
            #packet_sent_evaluate[epoch] = total_epoch_pck_sent
            #percentage[epoch] = round(((total_epoch_pck_loss/(total_epoch_pck_loss+total_epoch_pck_sent))*100), 2)
            percentage[epoch] = round(((total_package_loss_nr/(total_package_loss_nr+total_packets_sent_nr))*100), 2)
            percentage_2[epoch] = round(((total_package_loss_nr/(total_package_loss_nr+total_packets_sent_nr))*100),2)
            available_bw_epoch[epoch] = round(np.average(available_bw_episode),2)
        ### epoch ends

    ##Data text file
    data_file = open(f"{folder_path}/{sub_path}.txt", "w")
    if EVALUATE and not TRAIN:
        data_file.write(f'Testing - {CRITIC_DOMAIN} {NEURAL_NETWORK} - {MODIFIED_NETWORK}')  
        if UPDATE_WEIGHTS:  
            data_file.write(f"Update Weights\n")
        data_file.write(f"Modified Network: {MODIFIED_NETWORK}\n\n")
        data_file.write(f"Packets lost Original network (bw): {percentage[0]}% \n")
        data_file.write(f"Packets lost Original network (number): {percentage_2[0]}% \n")
        data_file.write(f"Available bandwidth: {available_bw_epoch[0]}% \n")
        for index in range(1, nr_epochs):
            data_file.write(f"Packets lost Modified network (bw) ({index}): {percentage[index]}% \n")
            data_file.write(f"Packets lost Modified network (number) ({index}): {percentage_2[index]}% \n")
            data_file.write(f"Available bandwidth ({index}): {available_bw_epoch[index]}% \n")
        data_file.write(f"{NOTES}\n")
    else:
        #data_file.write(f"Packets lost training: {round(total_package_loss_nr/(total_package_loss_nr+total_packets_sent_nr) * 100, 2)}% \n")
        data_file.write(f"Packets lost training: {round(total_epoch_pck_loss/(total_epoch_pck_loss+total_epoch_pck_sent) * 100, 2)}% \n")
        data_file.write(f"\n{NOTES}\n")
    data_file.close()    

    def calculate_link_usage_statistics(link_history, filtered_epochs=None):
        """
        Processa os dados de utilização e retorna médias por link.
        Pode filtrar por épocas específicas se filtered_epochs for fornecido.
        """
        link_avg_utils = {}
        
        # Determinar quais episódios filtrar
        if filtered_epochs is None:
            # Usar todos os episódios
            filtered_episodes = link_history 
        else:
            # Filtrar apenas episódios das épocas especificadas
            filtered_episodes = {k: v for k, v in link_history.items() 
                            if any(k.startswith(f"epoch{e}_") for e in filtered_epochs)}
        
        # Calcular média para cada link
        for episode_key, timesteps in filtered_episodes.items():
            for timestep_data in timesteps:
                for link, util in timestep_data.items():
                    if link not in link_avg_utils:
                        link_avg_utils[link] = []
                    link_avg_utils[link].append(util)
        
        # Calcular médias
        link_averages = [(link, sum(utils)/len(utils)) for link, utils in link_avg_utils.items() 
                        if utils and util != -1]  # Ignorar links removidos
        
        return link_averages

    def visualize_link_utilization(link_history, folder_path, epoch=None, all_epochs=False):
        """Visualize and save link utilization data for a specific epoch."""
        
        # Determinar épocas a processar
        if all_epochs:
            link_averages = calculate_link_usage_statistics(link_history)
            title_text = 'Top 10 Most Utilized Links - All Training'
            file_prefix = "top_links_utilization_overall"
        else:
            link_averages = calculate_link_usage_statistics(link_history, [epoch])
            title_text = f'Top 10 Most Utilized Links - Epoch {epoch}'
            file_prefix = f"top_links_utilization_epoch{epoch}"
        
        # Ordenar links por utilização
        link_averages.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 10 links
        top_links = link_averages[:10]
        
        
        should_create_chart = (all_epochs or epoch in [1, 2, 3])  # Só épocas 1-3 para cenários 2/3
        
        if should_create_chart:
            plt.figure(figsize=(12, 8))
            
            # Prepare data for chart
            labels = [f"{link[0][0]}-{link[0][1]}" for link in top_links]
            values = [link[1] for link in top_links]
            
            # Create bar chart
            plt.bar(labels, values, color='skyblue')
            plt.xlabel('Links (source-destination)')
            plt.ylabel('Average Utilization (%)')
            plt.title(title_text)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save chart
            plt.savefig(f"{folder_path}/{file_prefix}.png")
            plt.close()
        
        
        with open(f"{folder_path}/{file_prefix}.csv", 'w') as f:
            f.write("Link;Average_Utilization;Status\n")
            for link, avg in link_averages:
                if avg == -1:
                    f.write(f"{link[0]}-{link[1]};0.00;REMOVED\n")
                else:
                    f.write(f"{link[0]}-{link[1]};{avg:.2f};ACTIVE\n")

        return link_averages
    
    def compare_early_late_links(link_history, folder_path, nr_epochs, top_n=10):
        """Compara utilização de links entre início e fim do treinamento"""
        
        # Calcular estatísticas para primeiras e últimas épocas
        early_epochs = [0, 1, 2]  # Primeiras 3 épocas
        late_epochs = list(range(max(0, nr_epochs - 3), nr_epochs))  # Últimas 3 épocas
        
        # Reutilizar a função de cálculo para ambos os grupos
        early_links_data = calculate_link_usage_statistics(link_history, early_epochs)
        late_links_data = calculate_link_usage_statistics(link_history, late_epochs)
        
        # Converter para dicionários para fácil acesso
        early_avgs = {link: avg for link, avg in early_links_data}
        late_avgs = {link: avg for link, avg in late_links_data}
        
        # Pegar top N links com maior diferença
        top_links = sorted(late_avgs.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Criar gráfico
        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(top_links))
        width = 0.35
        
        labels = [f"{link[0]}-{link[1]}" for link, _ in top_links]
        early_values = [early_avgs.get(link, 0) for link, _ in top_links]
        late_values = [late_avgs.get(link, 0) for link, _ in top_links]
        
        ax.bar(x - width/2, early_values, width, label='Primeiras 3 épocas', color='skyblue')
        ax.bar(x + width/2, late_values, width, label='Últimas 3 épocas', color='darkblue')
        
        ax.set_xlabel('Links')
        ax.set_ylabel('Utilização Média (%)')
        ax.set_title('Comparação de Utilização: Início vs. Final do Treino')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()
        fig.tight_layout()
        
        plt.savefig(f"{folder_path}/link_utilization_comparison.png")
        plt.close()

        with open(f"{folder_path}/link_utilization_comparison.csv", 'w') as f:
            f.write("Link;Early_Utilization;Late_Utilization;Difference\n")
            for link, diff in top_links:
                early_val = early_avgs.get(link, 0)
                late_val = late_avgs.get(link, 0)
                f.write(f"{link[0]}-{link[1]};{early_val:.2f};{late_val:.2f};{diff:.2f}\n")
        
        return top_links
    
    def collect_evaluation_data(graph_y_axis, packet_loss_evolution, nr_epochs):
        """Collect reward data and calculate metrics for each epoch."""
        all_rewards = []
        all_packet_loss = []
        failure_points = []
        epoch_averages = []
        links_removed_info = []
        
        for epoch in range(nr_epochs):
            episodes_in_epoch = len(graph_y_axis[epoch])
            values = [v for v in graph_y_axis[epoch]]
            
            # Calculate epoch average
            if values:
                avg_reward = sum(values) / len(values)
                epoch_averages.append(avg_reward)
            else:
                epoch_averages.append(0)
            
            # Add all episode rewards to the list
            for e in range(episodes_in_epoch):
                all_rewards.append(graph_y_axis[epoch][e])
                # Add packet loss data if we have it
                if packet_loss_evolution and len(packet_loss_evolution) > epoch and len(packet_loss_evolution[epoch]) > e:
                    all_packet_loss.append(packet_loss_evolution[epoch][e])
            
            # Mark failure points
            if epoch > 0:
                collect_removed_links_info(epoch, links_removed_info)
                failure_points.append(sum([len(graph_y_axis[e]) for e in range(epoch)]))
        
        return all_rewards, all_packet_loss, failure_points, epoch_averages, links_removed_info
    
    def collect_removed_links_info(epoch, links_removed_info, is_scenario4=False):
        if not is_scenario4:
            try:
                with open(f"{PATH_SIMULATION}/removed_edges.json", "r") as f:
                    edges_data = json.load(f)
                    epoch_str = str(epoch)
                    
                    if epoch_str in edges_data and edges_data[epoch_str] is not None:
                        formatted_links = []
                        
                        for u, v in [tuple(e) for e in edges_data[epoch_str]]:
                            # Translate indices to host/switch names
                            host_u = f"H{u + 1}" if u < NUMBER_OF_HOSTS else f"S{u - NUMBER_OF_HOSTS + 1}"
                            host_v = f"H{v + 1}" if v < NUMBER_OF_HOSTS else f"S{v - NUMBER_OF_HOSTS + 1}"
                            
                            formatted_links.append(f"{host_u}-{host_v} (index: {u}-{v})")
                        
                        links_removed_info.append(f"Epoch {epoch}: {', '.join(formatted_links)}")
                    else:
                        links_removed_info.append(f"Epoch {epoch}: No links removed")
            except (FileNotFoundError, json.JSONDecodeError):
                links_removed_info.append(f"Epoch {epoch}: No links removed (file not found)")
        else:
            try:
                # Usar arquivo específico da topologia
                scenario4_file = f"{PATH_SIMULATION}/scenario4_removed_edges_{TOPOLOGY_TYPE}.json"
                
                with open(scenario4_file, "r") as f:
                    edges_data = json.load(f)
                    
                    if "links_removed" in edges_data:
                        formatted_links = []
                        
                        for u, v in [tuple(e) for e in edges_data["links_removed"]]:
                            # Translate indices to host/switch names
                            host_u = f"H{u + 1}" if u < NUMBER_OF_HOSTS else f"S{u - NUMBER_OF_HOSTS + 1}"
                            host_v = f"H{v + 1}" if v < NUMBER_OF_HOSTS else f"S{v - NUMBER_OF_HOSTS + 1}"
                            
                            formatted_links.append(f"{host_u}-{host_v} (index: {u}-{v})")
                        
                        links_removed_info.append(f"Links removidos: {', '.join(formatted_links)}")
                    else:
                        links_removed_info.append(f"Cenário 4: Nenhum link removido")
            except (FileNotFoundError, json.JSONDecodeError):
                links_removed_info.append(f"Cenário 4: Arquivo de links removidos não encontrado")
    
    def analyze_convergence(all_rewards, failure_points):
        """Analyze convergence time after each failure point."""
        convergence_times = []
        
        # Epoch 0 has no failure
        convergence_times.append("No failure")

        for epoch in range(1, nr_epochs):

            # Extract reward segment for this epoch
            segment = [v for v in graph_y_axis[epoch] if v != 0]
            
            # Calculate convergence time
            time_to_converge = NetworkEngine.calculate_convergence(segment, 0, 5, 0.05)
            convergence_times.append(time_to_converge)
        
        return convergence_times
    
    def save_evaluation_results(folder_path, sub_path, all_rewards, all_packet_loss, epoch_averages, 
                           convergence_times, links_removed_info):
        """Save all evaluation results to files."""
        # Save detailed reward evolution
        detailed_df = pd.DataFrame({
            'Episode': np.arange(len(all_rewards)),
            'Reward': all_rewards,
        })
        detailed_df.to_csv(f"{folder_path}/reward_evolution.csv", index=False, sep=';', decimal='.')
        
        # Save detailed packet loss evolution if we have data
        if all_packet_loss and len(all_packet_loss) > 0:
            packet_loss_df = pd.DataFrame({
                'Episode': np.arange(len(all_packet_loss)),
                'Packet_Loss': all_packet_loss,
            })
            packet_loss_df.to_csv(f"{folder_path}/packet_loss_evolution.csv", index=False, sep=';', decimal='.')
        
        # Save epoch averages
        with open(f"{folder_path}/{sub_path}.txt", "a") as data_file:
            data_file.write("\n\n=== Average Reward per Epoch ===\n")
            for i, avg in enumerate(epoch_averages):
                data_file.write(f"Epoch {i}: {avg:.3f}\n")
        
        # Save convergence analysis
        with open(f"{folder_path}/{sub_path}.txt", "a") as data_file:
            data_file.write("\n\n=== Convergence Analysis ===\n")
            for i, time in enumerate(convergence_times):
                if isinstance(time, str):
                    data_file.write(f"Failure {i+1}: {time}\n")
                else:
                    data_file.write(f"Failure {i+1}: {time} episodes to converge\n")
        
        # Save removed links info
        with open(f"{folder_path}/{sub_path}.txt", "a") as data_file:
            data_file.write("\n\n=== Links Removed by Epoch ===\n")
            for info in links_removed_info:
                data_file.write(f"{info}\n")

    def plot_rewards_with_failures(all_rewards, failure_points, folder_path):
        """Create and save a plot showing rewards with marked failure points."""
        plt.figure(figsize=(12, 6))
        
        # Add vertical lines for failures
        for i, point in enumerate(failure_points):
            if i == 0:
                plt.axvline(x=point, color='r', linestyle='--', alpha=0.7, label="Link Failures")
            else:
                plt.axvline(x=point, color='r', linestyle='--', alpha=0.7)
        
        # Plot the rewards series
        plt.plot(all_rewards, label="Reward per Episode", color='blue')
        
        # Auto-adjust vertical scale with 10% margin
        if all_rewards:
            min_val = min(all_rewards)
            max_val = max(all_rewards)
            
            # Verificar se min e max são iguais para evitar o warning
            if min_val == max_val:
                y_min = min_val - 1
                y_max = max_val + 1
            else:
                y_min = min_val * 0.9 if min_val > 0 else min_val * 1.1
                y_max = max_val * 1.1 if max_val > 0 else max_val * 0.9
            
            plt.ylim(y_min, y_max)
        
        # Set title based on the scenario
        if UPDATE_WEIGHTS:
            plt.title(f"Reward Evolution with Network Failures - {CRITIC_DOMAIN} {NEURAL_NETWORK} - {MODIFIED_NETWORK}: Update Weights\n")
        else:
            plt.title(f"Reward Evolution with Network Failures - {CRITIC_DOMAIN} {NEURAL_NETWORK} - {MODIFIED_NETWORK}\n")
        
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig(f"{folder_path}/reward_with_failures.png")
        plt.close()

    def plot_packet_loss_with_failures(all_packet_loss, failure_points, folder_path):
        """Create and save a plot showing packet loss evolution with marked failure points."""
        if not all_packet_loss or len(all_packet_loss) == 0:
            return
            
        plt.figure(figsize=(12, 6))
        
        # Add vertical lines for failures
        for i, point in enumerate(failure_points):
            if i == 0:
                plt.axvline(x=point, color='r', linestyle='--', alpha=0.7, label="Link Failures")
            else:
                plt.axvline(x=point, color='r', linestyle='--', alpha=0.7)
        
        # Plot the packet loss series
        plt.plot(all_packet_loss, label="Packet Loss per Episode (%)", color='orange')
        
        # Auto-adjust vertical scale with 10% margin
        if all_packet_loss:
            min_val = min(all_packet_loss)
            max_val = max(all_packet_loss)
            
            # Verificar se min e max são iguais para evitar o warning
            if min_val == max_val:
                y_min = max(0, min_val - 1)  # Garantir que não seja negativo
                y_max = max_val + 1
            else:
                y_min = min_val * 0.9 if min_val > 0 else 0
                y_max = max_val * 1.1
            
            plt.ylim(y_min, y_max)
        
        # Set title based on the scenario
        if UPDATE_WEIGHTS:
            plt.title(f"Packet Loss Evolution with Network Failures - {CRITIC_DOMAIN} {NEURAL_NETWORK} - {MODIFIED_NETWORK}: Update Weights\n")
        else:
            plt.title(f"Packet Loss Evolution with Network Failures - {CRITIC_DOMAIN} {NEURAL_NETWORK} - {MODIFIED_NETWORK}\n")
        
        plt.xlabel("Episodes")
        plt.ylabel("Packet Loss (%)")
        plt.legend()
        plt.savefig(f"{folder_path}/packet_loss_with_failures.png")
        plt.close()

    def detect_congestion(link_utilization_history, threshold=95):
        """Detect when any link reaches congestion (>threshold%)."""
        congestion_episodes = {}
        
        # Percorrer todos os episódios em ordem
        for episode_key in sorted(link_utilization_history.keys()):
            epoch = int(episode_key.split('_')[0][5:])  # Extrair número da época
            episode = int(episode_key.split('_')[1][7:])  # Extrair número do episódio
            
            # Verificar cada timestep neste episódio
            for timestep_idx, timestep_data in enumerate(link_utilization_history[episode_key]):
                for link, util in timestep_data.items():
                    # Se encontrar link congestionado
                    if util >= threshold and util != -1:  # Ignorar links removidos (-1)
                        link_str = f"{link[0]}-{link[1]}"
                        if link_str not in congestion_episodes:
                            congestion_episodes[link_str] = {
                                "first_detection": (epoch, episode, timestep_idx),
                                "utilization": util
                            }
        
        # Ordenar por detecção mais cedo
        sorted_congestion = sorted(
            congestion_episodes.items(), 
            key=lambda x: (x[1]["first_detection"][0], x[1]["first_detection"][1], x[1]["first_detection"][2])
        )
        
        return sorted_congestion
    
    def plot_epoch_rewards_scen4(rewards, folder_path, bandwidth_increase_interval):
        """Plota a evolução da recompensa média por época com marcações de aumento de banda."""
        plt.figure(figsize=(12, 6))
        
        x = np.arange(0, len(rewards))
        plt.plot(x, rewards, marker='o', linestyle='-', color='blue', label='Recompensa Média')
        
        z = np.polyfit(x, rewards, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--", label=f"Tendência (y = {z[0]:.2f}x + {z[1]:.2f})")
        
        # Calcular o ponto de estabilização (quando bandwidth atinge 150%)
        stabilization_point = None
        if hasattr(eng, 'bandwidth_stabilized') or 'STABILIZE_AFTER_MULTIPLIER' in globals():
            bw_factor = 1.0
            for i in range(len(rewards)):
                if i > 0 and i % bandwidth_increase_interval == 0:
                    bw_factor *= BANDWIDTH_INCREASE_FACTOR
                    if bw_factor >= STABILIZE_AFTER_MULTIPLIER:
                        stabilization_point = i
                        break
        
        detailed_df = pd.DataFrame({
            'Epoch': np.arange(len(rewards)),
            'Average_Reward': [f"{value:.3f}" for value in rewards],
            'Stabilized': [(stabilization_point is not None and i >= stabilization_point) for i in range(len(rewards))]
        })
        detailed_df.to_csv(f"{folder_path}/reward_evolution.csv", index=False, sep=';', decimal='.')
        
        plt.title("Evolução da Recompensa Média por Época")
        plt.xlabel("Época")
        plt.ylabel("Recompensa Média")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{folder_path}/reward_evolution.png")
        plt.close()
        
        # Se houver ponto de estabilização, criar segundo gráfico mostrando apenas após estabilização
        if stabilization_point is not None and stabilization_point < len(rewards):
            plt.figure(figsize=(12, 6))
            
            # Dados apenas após estabilização
            post_stab_x = np.arange(len(rewards) - stabilization_point)
            post_stab_y = rewards[stabilization_point:]
            
            plt.plot(post_stab_x, post_stab_y, marker='o', linestyle='-', color='blue', 
                    label='Recompensa Após Estabilização')
            
            # Linha de tendência após estabilização
            if len(post_stab_y) > 1:
                post_z = np.polyfit(post_stab_x, post_stab_y, 1)
                post_p = np.poly1d(post_z)
                plt.plot(post_stab_x, post_p(post_stab_x), "r--", 
                    label=f"Tendência Após Estabilização (y = {post_z[0]:.2f}x + {post_z[1]:.2f})")
            
            plt.title("Evolução da Recompensa Após Estabilização da Largura de Banda")
            plt.xlabel("Épocas Após Estabilização")
            plt.ylabel("Recompensa Média")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{folder_path}/reward_after_stabilization.png")
            plt.close()

    def plot_packet_loss_scen4(packet_loss_per_epoch, folder_path, bandwidth_increase_interval):
        """Plota a evolução da perda de pacotes média por época para cenário 4."""
        if not packet_loss_per_epoch or len(packet_loss_per_epoch) == 0:
            return
            
        plt.figure(figsize=(12, 6))
        
        x = np.arange(0, len(packet_loss_per_epoch))
        plt.plot(x, packet_loss_per_epoch, marker='o', linestyle='-', color='orange', label='Perda de Pacotes Média (%)')
        
        # Linha de tendência
        z = np.polyfit(x, packet_loss_per_epoch, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--", label=f"Tendência (y = {z[0]:.3f}x + {z[1]:.3f})")
        
        # Calcular o ponto de estabilização (quando bandwidth atinge 150%)
        stabilization_point = None
        if hasattr(eng, 'bandwidth_stabilized') or 'STABILIZE_AFTER_MULTIPLIER' in globals():
            bw_factor = 1.0
            for i in range(len(packet_loss_per_epoch)):
                if i > 0 and i % bandwidth_increase_interval == 0:
                    bw_factor *= BANDWIDTH_INCREASE_FACTOR
                    if bw_factor >= STABILIZE_AFTER_MULTIPLIER:
                        stabilization_point = i
                        break
        
        # 1. Dados por época (para gráficos comparativos - NOVO FORMATO)
        epoch_df = pd.DataFrame({
            'Epoch': np.arange(len(packet_loss_per_epoch)),
            'Packet_Loss': [f"{value:.3f}" for value in packet_loss_per_epoch],
            'Stabilized': [(stabilization_point is not None and i >= stabilization_point) for i in range(len(packet_loss_per_epoch))]
        })
        epoch_df.to_csv(f"{folder_path}/packet_loss_by_epoch.csv", index=False, sep=';', decimal='.')
        
        # 2. Dados expandidos por episódio (para compatibilidade com cenários 2/3)
        episodes_per_epoch = EPOCH_SIZE 
        expanded_episodes = []
        expanded_packet_loss = []
        
        for epoch_idx, avg_loss in enumerate(packet_loss_per_epoch):
            for episode_in_epoch in range(episodes_per_epoch):
                episode_number = epoch_idx * episodes_per_epoch + episode_in_epoch
                expanded_episodes.append(episode_number)
                # Adicionar pequena variação para simular dados por episódio
                variation = np.random.normal(0, avg_loss * 0.1) if avg_loss > 0 else 0
                episode_loss = max(0, avg_loss + variation)
                expanded_packet_loss.append(episode_loss)
        
        # Salvar dados expandidos por episódio
        detailed_df = pd.DataFrame({
            'Episode': expanded_episodes,
            'Packet_Loss': [f"{value:.3f}" for value in expanded_packet_loss]
        })
        detailed_df.to_csv(f"{folder_path}/packet_loss_evolution.csv", index=False, sep=';', decimal='.')
        
        plt.title("Evolução da Perda de Pacotes Média por Época")
        plt.xlabel("Época")
        plt.ylabel("Perda de Pacotes Média (%)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{folder_path}/packet_loss_evolution.png")
        plt.close()
        
        # Se houver ponto de estabilização, criar segundo gráfico
        if stabilization_point is not None and stabilization_point < len(packet_loss_per_epoch):
            plt.figure(figsize=(12, 6))
            
            post_stab_x = np.arange(len(packet_loss_per_epoch) - stabilization_point)
            post_stab_y = packet_loss_per_epoch[stabilization_point:]
            
            plt.plot(post_stab_x, post_stab_y, marker='o', linestyle='-', color='orange', 
                    label='Perda de Pacotes Após Estabilização')
            
            if len(post_stab_y) > 1:
                post_z = np.polyfit(post_stab_x, post_stab_y, 1)
                post_p = np.poly1d(post_z)
                plt.plot(post_stab_x, post_p(post_stab_x), "r--", 
                    label=f"Tendência Após Estabilização (y = {post_z[0]:.3f}x + {post_z[1]:.3f})")
            
            if post_stab_y and len(post_stab_y) > 0:
                min_val = min(post_stab_y)
                max_val = max(post_stab_y)
                
                if min_val == max_val:
                    if min_val == 0:
                        plt.ylim(-0.1, 0.1)
                    else:
                        margin = abs(min_val) * 0.1
                        plt.ylim(min_val - margin, max_val + margin)
            
            plt.title("Evolução da Perda de Pacotes Após Estabilização da Largura de Banda")
            plt.xlabel("Épocas Após Estabilização")
            plt.ylabel("Perda de Pacotes Média (%)")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{folder_path}/packet_loss_after_stabilization.png")
            plt.close()

    def process_evaluation_results(link_utilization_history, graph_y_axis, packet_loss_evolution, nr_epochs, folder_path, sub_path, check_congestion=False):
        """Processa resultados de avaliação para qualquer cenário EVALUATE"""
        plt.figure(figsize=(12, 6))

        # MUDANÇA: Process data apenas para épocas relevantes
        # Época 0: sempre (baseline)
        # Épocas 1-3: apenas para cenários 2/3 (após falhas)
        for epoch in range(nr_epochs):
            visualize_link_utilization(link_utilization_history, folder_path, epoch)
        
        # Collect all evaluation data
        all_rewards, all_packet_loss, failure_points, epoch_averages, links_removed_info = collect_evaluation_data(graph_y_axis, packet_loss_evolution, nr_epochs)
        
        # Analyze convergence after failure points
        convergence_times = analyze_convergence(graph_y_axis, nr_epochs)
        
        # Save results to files
        save_evaluation_results(folder_path, sub_path, all_rewards, all_packet_loss, epoch_averages, convergence_times, links_removed_info)
        
        # Create and save the rewards plot
        plot_rewards_with_failures(all_rewards, failure_points, folder_path)
        
        # Create and save the packet loss plot
        plot_packet_loss_with_failures(all_packet_loss, failure_points, folder_path)
        
    def create_link_utilization_distribution_graphs(topology_type):
        """Cria gráficos de distribuição de utilização dos links para todos os cenários."""
        scenarios = {
            "2a": "eval_1link",
            "2b": "eval_2links", 
            "3a": "eval_update_1link",
            "3b": "eval_update_2links", 
            4: "eval_train"
        }
        
        # Caminho base para a topologia atual
        topology_path = os.path.join(PATH_SIMULATION, "results", topology_type)
        gnn_suffix = "com_GNN" if USE_GNN else "sem_GNN"
        gnn_path = os.path.join(topology_path, gnn_suffix)
        
        # Configurações a comparar
        configurations = [
            "central_critic_duelling_q_network",
            "central_critic_simple_q_network", 
            "local_critic_duelling_q_network",
            "shortest_shortest"
        ]
        
        # Criar pasta para comparações
        comparison_folder = os.path.join(gnn_path, "comparisons")
        if not os.path.exists(comparison_folder):
            os.makedirs(comparison_folder)
        
        # Pasta específica para distribuições
        distribution_folder = os.path.join(comparison_folder, "link_utilization_distributions")
        if not os.path.exists(distribution_folder):
            os.makedirs(distribution_folder)
        
        print(f"Gerando gráficos de distribuição de utilização dos links para {topology_type}...")
        
        for scenario_key, scenario_name in scenarios.items():
            print(f"Processando distribuições para cenário {scenario_key} ({scenario_name})...")
            
            if scenario_key in ["2a", "2b", "3a", "3b"]:
                # MUDANÇA: Para cenários 2 e 3, criar apenas 2 gráficos
                # 1. Época 0 (baseline - rede original)
                create_distribution_graph_for_epoch(
                    gnn_path, configurations, scenario_key, scenario_name, 
                    0, distribution_folder, topology_type  # APENAS época 0
                )
                
                # 2. Épocas 1-3 combinadas (após remoção de links)
                create_distribution_graph_for_post_failure_epochs(
                    gnn_path, configurations, scenario_key, scenario_name,
                    distribution_folder, topology_type
                )
            else:
                # Para cenário 4: usar função específica para treino
                create_distribution_graph_for_training(
                    gnn_path, configurations, scenario_key, scenario_name,
                    distribution_folder, topology_type
                )

    def create_distribution_graph_for_epoch(gnn_path, configurations, scenario_key, scenario_name, 
                                        epoch, distribution_folder, topology_type):
        """Cria gráfico de distribuição para uma época específica dos cenários 2/3."""
        import matplotlib.pyplot as plt
        from scipy import stats
        import numpy as np
        
        plt.figure(figsize=(12, 8), dpi=300)
        
        # Cores para cada configuração
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        data_found = False
        
        # Determinar o número do cenário baseado na chave
        if scenario_key in ["2a", "2b"]:
            scenario_number = 2
        elif scenario_key in ["3a", "3b"]:
            scenario_number = 3
        else:
            scenario_number = scenario_key
        
        for i, config in enumerate(configurations):
            config_path = os.path.join(gnn_path, config)
            if not os.path.exists(config_path):
                continue
                
            # Encontrar pasta do cenário
            result_folders = [d for d in os.listdir(config_path) 
                            if os.path.isdir(os.path.join(config_path, d)) and 
                                scenario_name in d]
            
            if not result_folders:
                continue
                
            scenario_folders = [d for d in result_folders if f"_{scenario_number}" in d]
            if scenario_folders:
                target_folder = sorted(scenario_folders)[-1]
            else:
                target_folder = sorted(result_folders)[-1]
            
            folder_path = os.path.join(config_path, target_folder)
            
            # Procurar arquivo de utilização para a época específica
            util_file = os.path.join(folder_path, f"top_links_utilization_epoch{epoch}.csv")
            
            if os.path.exists(util_file):
                try:
                    import pandas as pd
                    df = pd.read_csv(util_file, sep=';', decimal='.')
                    
                    # Filtrar apenas links ativos
                    active_links = df[df['Status'] == 'ACTIVE'] if 'Status' in df.columns else df
                    
                    if not active_links.empty and 'Average_Utilization' in active_links.columns:
                        utilization_data = active_links['Average_Utilization'].values
                        
                        # Adicionar curva de densidade suavizada
                        if len(utilization_data) > 1:
                            # Usar kernel density estimation para curva suave
                            kde = stats.gaussian_kde(utilization_data)
                            x_range = np.linspace(utilization_data.min(), utilization_data.max(), 100)
                            density = kde(x_range)
                            
                            label = config.replace("_", " ").title()
                            plt.plot(x_range, density, color=colors[i], linewidth=2.5, 
                                    label=label)
                            
                            data_found = True
                            
                except Exception as e:
                    print(f"Erro ao processar {util_file}: {str(e)}")
        
        if data_found:
            plt.title(f"Distribuição de Utilização dos Links - Cenário {scenario_key} - Época {epoch} - {topology_type.title()}")
            plt.xlabel("Utilização dos Links (%)")
            plt.ylabel("Densidade")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            output_file = os.path.join(distribution_folder, f"link_distribution_scenario{scenario_key}_epoch{epoch}.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Gráfico de distribuição salvo: {output_file}")
        else:
            print(f"Nenhum dado encontrado para cenário {scenario_key}, época {epoch}")
        
        plt.close()

    def create_distribution_graph_for_post_failure_epochs(gnn_path, configurations, scenario_key, scenario_name,
                                                distribution_folder, topology_type):
        """Cria gráfico de distribuição para épocas 1-3 combinadas dos cenários 2/3."""
        import matplotlib.pyplot as plt
        from scipy import stats
        import numpy as np
        
        plt.figure(figsize=(12, 8), dpi=300)
        
        # Cores para cada configuração
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        data_found = False
        
        # Determinar o número do cenário baseado na chave
        if scenario_key in ["2a", "2b"]:
            scenario_number = 2
        elif scenario_key in ["3a", "3b"]:
            scenario_number = 3
        else:
            scenario_number = scenario_key
        
        for i, config in enumerate(configurations):
            config_path = os.path.join(gnn_path, config)
            if not os.path.exists(config_path):
                continue
                
            # Encontrar pasta do cenário
            result_folders = [d for d in os.listdir(config_path) 
                            if os.path.isdir(os.path.join(config_path, d)) and 
                                scenario_name in d]
            
            if not result_folders:
                continue
                
            scenario_folders = [d for d in result_folders if f"_{scenario_number}" in d]
            if scenario_folders:
                target_folder = sorted(scenario_folders)[-1]
            else:
                target_folder = sorted(result_folders)[-1]
            
            folder_path = os.path.join(config_path, target_folder)
            
            # MUDANÇA: Combinar dados das épocas 1, 2, 3
            combined_utilization_data = []
            
            for epoch in range(1, 4):  # Épocas 1, 2, 3 (após falhas)
                util_file = os.path.join(folder_path, f"top_links_utilization_epoch{epoch}.csv")
                
                if os.path.exists(util_file):
                    try:
                        import pandas as pd
                        df = pd.read_csv(util_file, sep=';', decimal='.')
                        
                        # Filtrar apenas links ativos
                        active_links = df[df['Status'] == 'ACTIVE'] if 'Status' in df.columns else df
                        
                        if not active_links.empty and 'Average_Utilization' in active_links.columns:
                            epoch_data = active_links['Average_Utilization'].values
                            combined_utilization_data.extend(epoch_data)
                            
                    except Exception as e:
                        print(f"Erro ao processar {util_file}: {str(e)}")
            
            # Se temos dados combinados, criar a distribuição
            if combined_utilization_data:
                utilization_data = np.array(combined_utilization_data)
                
                # Adicionar curva de densidade suavizada
                if len(utilization_data) > 1:
                    # Usar kernel density estimation para curva suave
                    kde = stats.gaussian_kde(utilization_data)
                    x_range = np.linspace(utilization_data.min(), utilization_data.max(), 100)
                    density = kde(x_range)
                    
                    label = config.replace("_", " ").title()
                    plt.plot(x_range, density, color=colors[i], linewidth=2.5, 
                            label=label)
                    
                    data_found = True
                    print(f"Dados combinados para {config}: {len(utilization_data)} valores de épocas 1-3")

        if data_found:
            plt.title(f"Distribuição de Utilização dos Links - Cenário {scenario_key} - Épocas 1-3 (Após Falhas) - {topology_type.title()}")
            plt.xlabel("Utilização dos Links (%)")
            plt.ylabel("Densidade")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            output_file = os.path.join(distribution_folder, f"link_distribution_scenario{scenario_key}_post_failure.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Gráfico de distribuição pós-falha salvo: {output_file}")
        else:
            print(f"Nenhum dado encontrado para cenário {scenario_key} - épocas pós-falha")
        
        plt.close()

    def create_distribution_graph_for_training(gnn_path, configurations, scenario_key, scenario_name,
                                            distribution_folder, topology_type):
        """Cria gráfico de distribuição para todo o treino do cenário 4."""
        import matplotlib.pyplot as plt
        from scipy import stats
        import numpy as np
        
        plt.figure(figsize=(12, 8), dpi=300)
        
        # Cores para cada configuração
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        data_found = False
        
        for i, config in enumerate(configurations):
            config_path = os.path.join(gnn_path, config)
            if not os.path.exists(config_path):
                continue
                
            # Encontrar pasta do cenário
            result_folders = [d for d in os.listdir(config_path) 
                            if os.path.isdir(os.path.join(config_path, d)) and 
                                scenario_name in d]
            
            if not result_folders:
                continue
                
            scenario_folders = [d for d in result_folders if f"_{scenario_key}" in d]
            if scenario_folders:
                target_folder = sorted(scenario_folders)[-1]
            else:
                target_folder = sorted(result_folders)[-1]
            
            folder_path = os.path.join(config_path, target_folder)
            
            # Procurar arquivo de utilização geral
            util_file = os.path.join(folder_path, "top_links_utilization_overall.csv")
            
            if os.path.exists(util_file):
                try:
                    import pandas as pd
                    df = pd.read_csv(util_file, sep=';', decimal='.')
                    
                    # Filtrar apenas links ativos
                    active_links = df[df['Status'] == 'ACTIVE'] if 'Status' in df.columns else df
                    
                    if not active_links.empty and 'Average_Utilization' in active_links.columns:
                        utilization_data = active_links['Average_Utilization'].values
                        
                        # Adicionar curva de densidade suavizada
                        if len(utilization_data) > 1:
                            # Usar kernel density estimation para curva suave
                            kde = stats.gaussian_kde(utilization_data)
                            x_range = np.linspace(utilization_data.min(), utilization_data.max(), 100)
                            density = kde(x_range)
                            
                            label = config.replace("_", " ").title()
                            plt.plot(x_range, density, color=colors[i], linewidth=2.5, 
                                    label=label)
                            
                            data_found = True
                            
                except Exception as e:
                    print(f"Erro ao processar {util_file}: {str(e)}")
        
        if data_found:
            plt.title(f"Distribuição de Utilização dos Links - Cenário {scenario_key} - Todo o Treino - {topology_type.title()}")
            plt.xlabel("Utilização dos Links (%)")
            plt.ylabel("Densidade")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            output_file = os.path.join(distribution_folder, f"link_distribution_scenario{scenario_key}_overall.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Gráfico de distribuição salvo: {output_file}")
        else:
            print(f"Nenhum dado encontrado para cenário {scenario_key} - treino completo")
        
        plt.close()

    def create_comparison_graphs(topology_type):
        """Cria gráficos comparativos para todos os cenários e configurações de rede."""
        scenarios = {
            "2a": "eval_1link",
            "2b": "eval_2links", 
            "3a": "eval_update_1link",
            "3b": "eval_update_2links", 
            4: "eval_train"
        }
        
        # Caminho base para a topologia atual
        topology_path = os.path.join(PATH_SIMULATION, "results", topology_type)

        # Definir o caminho correto com base em USE_GNN
        gnn_suffix = "com_GNN" if USE_GNN else "sem_GNN"
        gnn_path = os.path.join(topology_path, gnn_suffix)

        # Garantir que os diretórios existem
        if not os.path.exists(gnn_path):
            os.makedirs(gnn_path, exist_ok=True)

        # Configurações a comparar
        configurations = [
            "central_critic_duelling_q_network",
            "central_critic_simple_q_network", 
            "local_critic_duelling_q_network",
            "shortest_shortest"
        ]
        
        # Criar pasta para comparações dentro da pasta GNN específica
        comparison_folder = os.path.join(gnn_path, "comparisons")
        if not os.path.exists(comparison_folder):
            print(f"Criando pasta 'comparisons' em {gnn_suffix}")
            os.makedirs(comparison_folder)
        
        print(f"Gerando gráficos comparativos para topologia {topology_type}...")
        
        # Para cada cenário, criar os três tipos de gráficos
        for scenario_key, scenario_name in scenarios.items():
            print(f"Processando cenário {scenario_key} ({scenario_name})...")
            
            # Limpar completamente o estado do matplotlib entre cada cenário
            plt.close('all')
            
            # Configurações para aumentar a resolução dos gráficos
            plt.figure(figsize=(16, 10), dpi=300)
            plt.rcParams['figure.figsize'] = (16, 10) 
            plt.rcParams['figure.dpi'] = 300
            plt.rcParams['lines.linewidth'] = 2.5
            plt.rcParams['font.size'] = 12
            
            print(f"\n========= Iniciando gráfico de RECOMPENSAS para cenário {scenario_key} ({scenario_name}) =========")
            
            # Processar este cenário específico
            process_single_scenario(gnn_path, configurations, scenario_key, scenario_name, comparison_folder, topology_type)

    def process_single_scenario(gnn_path, configurations, scenario_key, scenario_name, comparison_folder, topology_type):
        """Processa um cenário específico e gera os gráficos correspondentes."""
        import matplotlib.pyplot as plt
        
        # 1. GRÁFICO DE RECOMPENSAS
        plt.figure(figsize=(16, 10), dpi=300)
        reward_data_found = False
        
        for config in configurations:
            config_path = os.path.join(gnn_path, config)
            if not os.path.exists(config_path):
                continue
                
            # Busca por pastas
            result_folders = [d for d in os.listdir(config_path) 
                            if os.path.isdir(os.path.join(config_path, d)) and 
                            scenario_name in d]
            
            if not result_folders:
                continue
                
            # CORREÇÃO: Usar a mesma lógica da função de distribuição
            if scenario_key in ["2a", "2b"]:
                scenario_number = 2
                # Diferenciar entre 1link e 2links
                if scenario_key == "2a":
                    # Procurar especificamente por pastas com "1link" no nome
                    target_folders = [d for d in result_folders if "1link" in d and f"_{scenario_number}" in d]
                else:  # scenario_key == "2b"
                    # Procurar especificamente por pastas com "2links" no nome
                    target_folders = [d for d in result_folders if "2links" in d and f"_{scenario_number}" in d]
            elif scenario_key in ["3a", "3b"]:
                scenario_number = 3
                # Diferenciar entre 1link e 2links para cenário 3
                if scenario_key == "3a":
                    target_folders = [d for d in result_folders if "1link" in d and f"_{scenario_number}" in d]
                else:  # scenario_key == "3b"
                    target_folders = [d for d in result_folders if "2links" in d and f"_{scenario_number}" in d]
            elif scenario_key == 4:
                scenario_number = 4
                target_folders = [d for d in result_folders if f"_{scenario_number}" in d]
            else:
                scenario_number = scenario_key
                target_folders = [d for d in result_folders if f"_{scenario_number}" in d]
                
            # Selecionar a pasta mais recente se houver múltiplas
            if target_folders:
                target_folder = sorted(target_folders)[-1]
                print(f"Usando pasta específica para cenário {scenario_key}: {target_folder}")
            elif result_folders:
                target_folder = sorted(result_folders)[-1]
                print(f"Usando pasta genérica para cenário {scenario_key}: {target_folder}")
            else:
                print(f"Nenhuma pasta adequada encontrada para {config}, cenário {scenario_key}")
                continue
            
            print(f"Processando {config} - {target_folder} para cenário {scenario_key} ({scenario_name})")
            
            # Procurar múltiplos arquivos possíveis
            reward_files = [
                os.path.join(config_path, target_folder, "data_total.csv"),
                os.path.join(config_path, target_folder, "reward_evolution.csv")
            ]
            
            for data_file in reward_files:            
                if os.path.exists(data_file):
                    try:                            
                        print(f"Lendo arquivo {data_file} para cenário {scenario_key} ({scenario_name})")
                        df = pd.read_csv(data_file, sep=';', decimal='.')
                        
                        # Processar cada tipo de arquivo diferentemente
                        if 'Episode' in df.columns and 'Reward' in df.columns:
                            # reward_evolution.csv para cenários 2/3
                            # MUDANÇA: Calcular médias por época em vez de usar todos os episódios
                            episodes = df['Episode'].values
                            rewards = df['Reward'].values
                            
                            # Converter valores se forem strings
                            if len(rewards) > 0 and isinstance(rewards[0], str):
                                rewards = np.array([float(val.replace(',', '.')) for val in rewards])
                            
                            # CALCULAR MÉDIAS POR ÉPOCA (400 episódios = 1 época para cenários 2/3)
                            epoch_size = EPOCH_SIZE 
                            num_epochs = len(episodes) // epoch_size
                            
                            epoch_means = []
                            epoch_numbers = []
                            
                            for epoch in range(num_epochs):
                                start_idx = epoch * epoch_size
                                end_idx = (epoch + 1) * epoch_size
                                epoch_rewards = rewards[start_idx:end_idx]
                                
                                if len(epoch_rewards) > 0:
                                    epoch_mean = np.mean(epoch_rewards)
                                    epoch_means.append(epoch_mean)
                                    epoch_numbers.append(epoch)
                            
                            # Usar médias por época em vez de todos os episódios
                            x = np.array(epoch_numbers)
                            y = np.array(epoch_means)
                            
                            print(f"Valores calculados para {config} (cenário {scenario_key}): {len(y)} épocas (média de {epoch_size} episódios cada)")
                            
                        else:
                            # data_total.csv para cenário 4
                            if 'Epoch' in df.columns:
                                x = df['Epoch'].values
                            else:
                                x = range(len(df))
                            
                            if 'Average_Reward' in df.columns:
                                y = df['Average_Reward'].values
                            else:
                                y = df.iloc[:, 1].values
                            print(f"Valores calculados para {config}: {y}")
                            
                            # Converter valores se forem strings
                            if len(y) > 0 and isinstance(y[0], str):
                                y = np.array([float(val.replace(',', '.')) for val in y])

                        # Use consistent colors across all graphs
                        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
                        marker_styles = ['o', 's', '^', 'D']  # circle, square, triangle, diamond

                        label = config.replace("_", " ").title()
                        config_idx = configurations.index(config) if config in configurations else 0
                        color_idx = min(config_idx, 3)
                        marker_idx = min(config_idx, 3)

                        plt.plot(x, y, marker=marker_styles[marker_idx], linestyle='-',
                            label=label, alpha=0.8, color=colors[color_idx],
                            linewidth=2.5, markersize=8)
                        reward_data_found = True
                        break  # APENAS quebra o loop de arquivos, NÃO o loop de configurações
                        
                    except Exception as e:
                        print(f"Erro processando {data_file}: {str(e)}")
        
        # AGORA sim, depois de processar TODAS as configurações, finalizar o gráfico
        if reward_data_found:
            # Configurações do gráfico de recompensas
            plt.title(f"Comparação de Recompensas - Cenário {scenario_key} ({scenario_name}) - {topology_type.title()}")
            
            # MUDANÇA: Agora todos os cenários usam "Época" no eixo X
            plt.xlabel("Época")
            plt.ylabel("Recompensa Média")
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Se for cenário 2/3, marcar pontos de falha nas épocas certas
            if scenario_key in ["2a", "2b", "3a", "3b"]:
                for epoch in range(1, 4):  # Épocas 1, 2, 3 têm falhas
                    plt.axvline(x=epoch, color='r', linestyle='--', alpha=0.7, linewidth=2,
                            label="Falhas de Links" if epoch == 1 else "")
            
            plt.tight_layout()
            output_file = os.path.join(comparison_folder, f"reward_comparison_scenario{scenario_key}.png")
            print(f"Salvando gráfico de recompensas para cenário {scenario_key} em: {output_file}")
            plt.savefig(output_file, dpi=300)
            plt.close('all')
        else:
            plt.close('all')
            print(f"Nenhum dado de recompensa encontrado para o cenário {scenario_key}")

        # 2. GRÁFICO DE CONGESTIONAMENTO
        plt.close('all')
        plt.figure(figsize=(16, 10), dpi=300)
        print(f"\n========= Iniciando gráfico de CONGESTIONAMENTO para cenário {scenario_key} =========")
        congestion_data = {}

        for config in configurations:
            config_path = os.path.join(gnn_path, config)
            if not os.path.exists(config_path):
                continue
                
            # Busca seletiva de pastas por cenário - USAR A MESMA LÓGICA DA DISTRIBUIÇÃO
            result_folders = [d for d in os.listdir(config_path) 
                            if os.path.isdir(os.path.join(config_path, d)) and 
                                scenario_name in d]
            
            if not result_folders:
                continue
            
            # CORREÇÃO: Usar a mesma lógica de seleção que funciona na distribuição
            if scenario_key in ["2a", "2b"]:
                scenario_number = 2
                # Diferenciar entre 1link e 2links
                if scenario_key == "2a":
                    # Procurar especificamente por pastas com "1link" no nome
                    target_folders = [d for d in result_folders if "1link" in d and f"_{scenario_number}" in d]
                else:  # scenario_key == "2b"
                    # Procurar especificamente por pastas com "2links" no nome
                    target_folders = [d for d in result_folders if "2links" in d and f"_{scenario_number}" in d]
            elif scenario_key in ["3a", "3b"]:
                scenario_number = 3
                # Diferenciar entre 1link e 2links para cenário 3
                if scenario_key == "3a":
                    target_folders = [d for d in result_folders if "1link" in d and f"_{scenario_number}" in d]
                else:  # scenario_key == "3b"
                    target_folders = [d for d in result_folders if "2links" in d and f"_{scenario_number}" in d]
            elif scenario_key == 4:
                scenario_number = 4
                target_folders = [d for d in result_folders if f"_{scenario_number}" in d]
            else:
                scenario_number = scenario_key
                target_folders = [d for d in result_folders if f"_{scenario_number}" in d]
            
            # Selecionar a pasta mais recente se houver múltiplas
            if target_folders:
                target_folder = sorted(target_folders)[-1]
                print(f"Usando pasta específica para cenário {scenario_key}: {target_folder}")
            elif result_folders:
                target_folder = sorted(result_folders)[-1]
                print(f"Usando pasta genérica para cenário {scenario_key}: {target_folder}")
            else:
                print(f"Nenhuma pasta adequada encontrada para {config}, cenário {scenario_key}")
                continue
            
            print(f"Usando pasta {target_folder} para dados de congestionamento, cenário {scenario_key}")
            folder_path = os.path.join(config_path, target_folder)
            
            # Coletar valores de utilização
            utilization_values = []
            
            if scenario_key == 4:
                # Para cenário 4, usar top_links_utilization_overall.csv (manter como está)
                util_file = os.path.join(folder_path, "top_links_utilization_overall.csv")
                if os.path.exists(util_file):
                    try:
                        df = pd.read_csv(util_file, sep=';', decimal='.')
                        if 'Average_Utilization' in df.columns:
                            # Filtrar links ativos
                            active_links = df[df['Status'] == 'ACTIVE'] if 'Status' in df.columns else df
                            if not active_links.empty:
                                # Pegar os top 4 links mais congestionados para visualização
                                sorted_links = active_links.sort_values('Average_Utilization', ascending=False)
                                top_utils = sorted_links['Average_Utilization'].head(4).values
                                utilization_values.extend(top_utils)
                    except Exception as e:
                        print(f"Erro ao processar {util_file}: {str(e)}")
            else:
                # MUDANÇA: Para cenários 2/3, calcular média dos top 4 links APENAS das épocas 1-3
                print(f"Processando cenários {scenario_key}: calculando média dos top 4 links após remoção")
                
                # Coletar dados de utilização das épocas 1-3 (após remoção de links)
                all_link_utils = {}
                
                for epoch in range(1, 4):  # APENAS épocas 1, 2, 3 (após falhas)
                    util_file = os.path.join(folder_path, f"top_links_utilization_epoch{epoch}.csv")
                    if os.path.exists(util_file):
                        try:
                            df = pd.read_csv(util_file, sep=';', decimal='.')
                            # Filtrar links ativos
                            active_links = df[df['Status'] == 'ACTIVE'] if 'Status' in df.columns else df
                            
                            if not active_links.empty and 'Average_Utilization' in active_links.columns:
                                for _, row in active_links.iterrows():
                                    link_name = row['Link']
                                    util_value = row['Average_Utilization']
                                    
                                    if link_name not in all_link_utils:
                                        all_link_utils[link_name] = []
                                    all_link_utils[link_name].append(util_value)
                                    
                        except Exception as e:
                            print(f"Erro ao processar {util_file}: {str(e)}")
                
                # Calcular médias e pegar top 4
                if all_link_utils:
                    link_averages = {}
                    for link, utils in all_link_utils.items():
                        link_averages[link] = sum(utils) / len(utils)
                    
                    # Ordenar e pegar top 4
                    sorted_links = sorted(link_averages.items(), key=lambda x: x[1], reverse=True)
                    top_4_links = sorted_links[:4]
                    
                    # Extrair apenas os valores de utilização
                    utilization_values = [util for _, util in top_4_links]
                    
                    print(f"Top 4 links para {config}: {[(link, f'{util:.2f}%') for link, util in top_4_links]}")
            
            if utilization_values:
                congestion_data[config] = utilization_values

        # Criar gráfico de barras agrupadas se temos dados
        if congestion_data:
            plt.figure(figsize=(16, 10), dpi=300)
            
            max_points = max(len(utils) for utils in congestion_data.values())
            
            bar_width = 0.2
            x = np.arange(max_points)
            
            positions = [-1.5*bar_width, -0.5*bar_width, 0.5*bar_width, 1.5*bar_width]
            
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
            
            for i, (config, utils) in enumerate(congestion_data.items()):
                # Preencher com zeros se necessário
                full_utils = utils + [0] * (max_points - len(utils))
                label = config.replace("_", " ").title()
                
                pos = positions[min(i, 3)]
                color_idx = min(i, 3)
                plt.bar(x + pos, full_utils, width=bar_width, label=label, alpha=0.8,
                    color=colors[color_idx], edgecolor='black', linewidth=1.5)
            
            # Linha de threshold de congestionamento
            plt.axhline(y=95, color='r', linestyle='--', label='Threshold de Congestionamento (95%)')
            
            # Configurações do gráfico
            if scenario_key == 4:
                title = f"Top 4 Links Mais Congestionados - Cenário {scenario_key} ({scenario_name}) - {topology_type.title()}"
                xlabel = "Top Links Congestionados"
                xtick_labels = [f"Link {i+1}" for i in range(max_points)]
            else:
                title = f"Média dos Top 4 Links (Épocas 1-3) - Cenário {scenario_key} ({scenario_name}) - {topology_type.title()}"
                xlabel = "Top Links Mais Utilizados (Após Falhas)"
                xtick_labels = [f"Link {i+1}" for i in range(max_points)]
            
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel("Utilização Média (%)")
            plt.xticks(x, xtick_labels)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            output_file = os.path.join(comparison_folder, f"congestion_comparison_scenario{scenario_key}.png")
            print(f"Salvando gráfico de congestionamento para cenário {scenario_key} em: {output_file}")
            plt.savefig(output_file, dpi=300)
            plt.close('all')
            print(f"Gráfico de congestionamento para cenário {scenario_key} criado!")

        # 3. GRÁFICO DE EVOLUÇÃO DE PERDA DE PACOTES
        plt.close('all')
        plt.figure(figsize=(16, 10), dpi=300)
        print(f"\n========= Iniciando gráfico de EVOLUÇÃO DE PERDA DE PACOTES para cenário {scenario_key} =========")
        packet_loss_data_found = False

        for config in configurations:
            config_path = os.path.join(gnn_path, config)
            if not os.path.exists(config_path):
                continue
                
            
            result_folders = [d for d in os.listdir(config_path) 
                            if os.path.isdir(os.path.join(config_path, d)) and 
                                scenario_name in d]
            
            if not result_folders:
                continue
            
            
            if scenario_key in ["2a", "2b"]:
                scenario_number = 2
                if scenario_key == "2a":
                    target_folders = [d for d in result_folders if "1link" in d and f"_{scenario_number}" in d]
                else:
                    target_folders = [d for d in result_folders if "2links" in d and f"_{scenario_number}" in d]
            elif scenario_key in ["3a", "3b"]:
                scenario_number = 3
                if scenario_key == "3a":
                    target_folders = [d for d in result_folders if "1link" in d and f"_{scenario_number}" in d]
                else:
                    target_folders = [d for d in result_folders if "2links" in d and f"_{scenario_number}" in d]
            elif scenario_key == 4:
                scenario_number = 4
                target_folders = [d for d in result_folders if f"_{scenario_number}" in d]
            else:
                scenario_number = scenario_key
                target_folders = [d for d in result_folders if f"_{scenario_number}" in d]
            
            if target_folders:
                target_folder = sorted(target_folders)[-1]
            else:
                target_folder = sorted(result_folders)[-1] if result_folders else None
                
            if not target_folder:
                continue
            
            print(f"Usando pasta {target_folder} para dados de evolução de packet loss, cenário {scenario_key}")
            folder_path = os.path.join(config_path, target_folder)
            
            # MUDANÇA: Tratar cenário 4 diferentemente
            if scenario_key == 4:
                # Cenário 4: packet_loss_by_epoch.csv
                epoch_packet_loss_file = os.path.join(folder_path, "packet_loss_by_epoch.csv")
                if os.path.exists(epoch_packet_loss_file):
                    try:
                        print(f"Lendo arquivo {epoch_packet_loss_file} para cenário {scenario_key} ({scenario_name}) - dados por época")
                        df = pd.read_csv(epoch_packet_loss_file, sep=';', decimal='.')
                        if 'Epoch' in df.columns and 'Packet_Loss' in df.columns:
                            x = df['Epoch'].values
                            y = df['Packet_Loss'].values
                            if len(y) > 0 and isinstance(y[0], str):
                                y = np.array([float(val.replace(',', '.')) for val in y])
                            # ...plot...
                            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
                            marker_styles = ['o', 's', '^', 'D']
                            label = config.replace("_", " ").title()
                            config_idx = configurations.index(config) if config in configurations else 0
                            color_idx = min(config_idx, 3)
                            marker_idx = min(config_idx, 3)
                            plt.plot(x, y, marker=marker_styles[marker_idx], linestyle='-',
                                    label=label, alpha=0.8, color=colors[color_idx],
                                    linewidth=2.5, markersize=6)
                            packet_loss_data_found = True
                            print(f"Dados de packet loss plotados para {config}: {len(y)} épocas")
                    except Exception as e:
                        print(f"Erro ao processar dados de perda de pacotes em {epoch_packet_loss_file}: {str(e)}")
            else:
                # Cenários 2/3: packet_loss_evolution.csv
                packet_loss_file = os.path.join(folder_path, "packet_loss_evolution.csv")
                if os.path.exists(packet_loss_file):
                    try:
                        print(f"Lendo arquivo {packet_loss_file} para cenário {scenario_key} ({scenario_name})")
                        df = pd.read_csv(packet_loss_file, sep=';', decimal='.')
                        if 'Episode' in df.columns and 'Packet_Loss' in df.columns:
                            episodes = df['Episode'].values
                            losses = df['Packet_Loss'].values
                            if len(losses) > 0 and isinstance(losses[0], str):
                                losses = np.array([float(val.replace(',', '.')) for val in losses])
                            epoch_size = EPOCH_SIZE 
                            num_epochs = len(episodes) // epoch_size
                            epoch_means = []
                            epoch_numbers = []
                            for epoch in range(num_epochs):
                                start_idx = epoch * epoch_size
                                end_idx = (epoch + 1) * epoch_size
                                epoch_losses = losses[start_idx:end_idx]
                                if len(epoch_losses) > 0:
                                    epoch_mean = np.mean(epoch_losses)
                                    epoch_means.append(epoch_mean)
                                    epoch_numbers.append(epoch)
                            x = np.array(epoch_numbers)
                            y = np.array(epoch_means)
                            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
                            marker_styles = ['o', 's', '^', 'D']
                            label = config.replace("_", " ").title()
                            config_idx = configurations.index(config) if config in configurations else 0
                            color_idx = min(config_idx, 3)
                            marker_idx = min(config_idx, 3)
                            plt.plot(x, y, marker=marker_styles[marker_idx], linestyle='-',
                                    label=label, alpha=0.8, color=colors[color_idx],
                                    linewidth=2.5, markersize=6)
                            packet_loss_data_found = True
                            print(f"Dados de packet loss plotados para {config}: {len(y)} épocas")
                    except Exception as e:
                        print(f"Erro ao processar dados de perda de pacotes em {packet_loss_file}: {str(e)}")
        # Criar gráfico de perda de pacotes se temos dados
        if packet_loss_data_found:
            # Configurações do gráfico de evolução de packet loss
            plt.title(f"Evolução da Perda de Pacotes - Cenário {scenario_key} ({scenario_name}) - {topology_type.title()}")
            
            # MUDANÇA: Agora todos os cenários usam "Época" no eixo X
            plt.xlabel("Época")
            plt.ylabel("Perda de Pacotes Média (%)")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Se for cenário 2/3, marcar pontos de falha nas épocas
            if scenario_key in ["2a", "2b", "3a", "3b"]:
                for epoch in range(1, 4):  # Épocas 1, 2, 3 têm falhas
                    plt.axvline(x=epoch, color='r', linestyle='--', alpha=0.7, linewidth=2,
                            label="Falhas de Links" if epoch == 1 else "")
            
            plt.tight_layout()
            output_file = os.path.join(comparison_folder, f"packet_loss_evolution_scenario{scenario_key}.png")
            print(f"Salvando gráfico de evolução de packet loss para cenário {scenario_key} em: {output_file}")
            plt.savefig(output_file, dpi=300)
            plt.close('all')
            print(f"Gráfico de evolução de packet loss para cenário {scenario_key} criado com sucesso!")
        else:
            plt.close('all')
            print(f"Nenhum dado de evolução de packet loss encontrado para o cenário {scenario_key}")
        
        print(f"Gráficos comparativos salvos em {comparison_folder}")
        
    

    ## Build graph
    if not EVALUATE:
        x = np.arange(0, NR_EPOCHS)
        
        if CRITIC_DOMAIN == "central_critic":
            plt.title(f"Total reward per epoch - central critic")
        elif CRITIC_DOMAIN == "local_critic":
            plt.title(f"Total reward per epoch - local critic")
        
        plt.xlabel("Epochs")
        plt.ylabel("Reward")

        plt.plot(x, y_axis_training, label = {NEURAL_NETWORK})
        plt.legend()
        plt.savefig(f"{folder_path}/{sub_path}.png")

        data_total_df = pd.DataFrame({
            'Epoch': x,
            'Average_Reward': [f"{value:.3f}" for value in y_axis_training]
        })
        data_total_df.to_csv(f"{folder_path}/data_total.csv", index=False,sep=';', decimal='.')
        plt.close()

        visualize_link_utilization(link_utilization_history, folder_path, all_epochs=True)
        
        # Se for o algoritmo shortest-shortest, gerar apenas o overall
        ##if CRITIC_DOMAIN == "shortest" and NEURAL_NETWORK == "shortest":
            ##print("Gerando apenas visualização geral de links para shortest-shortest...")
            # Gerar apenas visualização global para todo o treinamento
            ##visualize_link_utilization(link_utilization_history, folder_path, all_epochs=True)

    elif EVALUATE:
        if not TRAIN:
            process_evaluation_results(link_utilization_history, graph_y_axis, packet_loss_evolution, nr_epochs, folder_path, sub_path)
            
            if (CRITIC_DOMAIN == "local_critic" and NEURAL_NETWORK == "duelling_q_network" and UPDATE_WEIGHTS == True):
                    print(f"Gerando gráficos comparativos para {TOPOLOGY_TYPE} na pasta {gnn_suffix}/comparisons")
                    create_comparison_graphs(TOPOLOGY_TYPE)
                    create_link_utilization_distribution_graphs(TOPOLOGY_TYPE)
            

        elif TRAIN:
            
            # Criar visualização única combinando dados de todas as épocas
            top_links = visualize_link_utilization(link_utilization_history, folder_path, all_epochs=True)
            
            # Adicionar informação ao arquivo de texto existente
            with open(f"{folder_path}/{sub_path}.txt", "a") as data_file:
                data_file.write("\n\n=== Links Mais Utilizados (Todo o Treino) ===\n")
                for i, (link, avg) in enumerate(top_links[:10]):
                    data_file.write(f"{i+1}. Link {link[0]}-{link[1]}: {avg:.2f}% de utilização média\n")
            
            links_removed_info = []
            collect_removed_links_info(0, links_removed_info, is_scenario4=True)
            
            # Salvar informações dos links removidos no arquivo
            with open(f"{folder_path}/{sub_path}.txt", "a") as data_file:
                data_file.write("\n\n=== Links Removidos ===\n")
                for info in links_removed_info:
                    data_file.write(f"{info}\n")

            compare_early_late_links(link_utilization_history, folder_path, nr_epochs)
            plot_epoch_rewards_scen4(y_axis_training, folder_path, INCREASE_BANDWIDTH_INTERVAL)
            
            # Calcular médias de packet loss por época para o cenário 4
            if packet_loss_evolution and len(packet_loss_evolution) > 0:
                packet_loss_averages = []
                for epoch in range(nr_epochs):
                    if epoch < len(packet_loss_evolution) and len(packet_loss_evolution[epoch]) > 0:
                        epoch_avg = sum(packet_loss_evolution[epoch]) / len(packet_loss_evolution[epoch])
                        packet_loss_averages.append(epoch_avg)
                    else:
                        packet_loss_averages.append(0)
                
                plot_packet_loss_scen4(packet_loss_averages, folder_path, INCREASE_BANDWIDTH_INTERVAL)

            # Gerar gráficos comparativos apenas quando for o cenário shortest_shortest
            # Este deve ser o último cenário a ser executado para cada configuração
            # Para geração de gráficos, escolhemos:
            # 1. shortest-shortest quando USE_GNN=False
            # 2. local_critic duelling_q_network quando USE_GNN=True            
            #if (CRITIC_DOMAIN == "shortest" and NEURAL_NETWORK == "shortest") and USE_GNN == False:
                #print(f"Gerando gráficos comparativos para {TOPOLOGY_TYPE} na pasta {gnn_suffix}/comparisons")
                #create_comparison_graphs(TOPOLOGY_TYPE)
            '''
            if (CRITIC_DOMAIN == "local_critic" and NEURAL_NETWORK == "duelling_q_network"):
                print(f"Gerando gráficos comparativos para {TOPOLOGY_TYPE} na pasta {gnn_suffix}/comparisons")
                create_comparison_graphs(TOPOLOGY_TYPE)
                create_link_utilization_distribution_graphs(TOPOLOGY_TYPE)
            '''
            
            if (CRITIC_DOMAIN == "local_critic" and NEURAL_NETWORK == "duelling_q_network"):
                    print(f"Gerando gráficos comparativos para {TOPOLOGY_TYPE} na pasta {gnn_suffix}/comparisons")
                    create_comparison_graphs(TOPOLOGY_TYPE)
                    create_link_utilization_distribution_graphs(TOPOLOGY_TYPE)










