import random
import json

import torch as T
import numpy as np
from torch import tensor, cat, no_grad, mean
import torch.nn.functional as F
import networkx as nx

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
from environmental_variables import STATE_SIZE, EPOCH_SIZE, NUMBER_OF_AGENTS, NR_EPOCHS, EVALUATE, CRITIC_DOMAIN, SIM_NR, TRAIN, NEURAL_NETWORK, MODIFIED_NETWORK, NOTES, TOPOLOGY_TYPE, UPDATE_WEIGHTS, PATH_SIMULATION, NUMBER_OF_PATHS, NUMBER_OF_HOSTS,BANDWIDTH_INCREASE_FACTOR,INCREASE_BANDWIDTH_INTERVAL,STABILIZE_BANDWIDTH , STABILIZE_AFTER_MULTIPLIER, SAVE_REMOVED_LINKS_SCENARIO4, MAX_BANDWIDTH_MULTIPLIER, CRITIC_DOMAIN, NUMBER_OF_PATHS, NUMBER_OF_AGENTS, NR_EPOCHS, EPOCH_SIZE, PATH_SIMULATION, SIM_NR, MODIFIED_NETWORK, USE_GNN
#, GRAPH_BATCH_SIZE


class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions,
                 scenario='simple', alpha=0.01, beta=0.01, fc1=64,
                 fc2=64, fa1=64, fa2=64, gamma=0.99, tau=0.001, chkpt_dir='tmp/maddpg/'):
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

    REPLAY_MEMORY = 50000 #Aumentado para melhor estabilidade 
    MEMORY_BATCH = 256 #Batch maior para melhor aprendizado

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
        nr_epochs = 4

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
                learning = "eval_update" # Cenários 3
            else:
                learning = "eval" # Cenário 2
    else:
        learning = "train" # Cenário 1

    # Adicionar modificação da rede separadamente
    if MODIFIED_NETWORK and (EVALUATE or TRAIN):
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
        graph_x_axis = np.zeros(EPOCH_SIZE*4)
        aux = np.zeros(EPOCH_SIZE*4) 
        graph_y_axis = [[0 for _ in range(EPOCH_SIZE*4)] for _ in range(nr_epochs)]
    elif EVALUATE and TRAIN:
        graph_x_axis = np.zeros(EPOCH_SIZE)
        y_axis_training = np.zeros(NR_EPOCHS)
        graph_y_axis = [[0 for _ in range(EPOCH_SIZE)] for _ in range(nr_epochs)]
        
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



    if (EVALUATE and TRAIN):
        if MODIFIED_NETWORK == "remove_edges":
            eng.remove_edges(3)  # Remove 3 links
        if MODIFIED_NETWORK == "add_edges":
            eng.add_edges(3)

    link_utilization_history = {} # Estrutura para armazenar dados de utilização dos links

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
            episode_size = EPOCH_SIZE * 4

        available_bw_episode = np.zeros(episode_size)

        # Adicionar código para aumentar largura de banda periodicamente no cenário 4
        if EVALUATE and TRAIN and epoch > 0 and epoch % INCREASE_BANDWIDTH_INTERVAL == 0:
            eng.increase_traffic_bandwidth(BANDWIDTH_INCREASE_FACTOR)
        
        
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
                    
                    # Criar arestas baseadas nas conexões entre hosts
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

                        prob = -1 if (EVALUATE and not TRAIN) else max(0.1, (0.3 - 0.0001 * epoch))

                        # Decision between exploration and exploitation
                        # if it is exploration, choose a random action
                        # if it is exploitation, choose the action of the neural network

                        if random.random() < prob:
                            action = random.randint(0, 2) # Exloration - random action
                        else:
                            action = actions[index] # Exploitation - action of the neural network

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
                    current_utils = eng.get_link_utilization() #contém todos os links da rede
                    episode_key = f"epoch{epoch}_episode{e}"
                    
                    if episode_key not in link_utilization_history:
                        link_utilization_history[episode_key] = []
                    
                    link_utilization_history[episode_key].append(current_utils)

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

            if (e % 3 == 0 and not EVALUATE) or (EVALUATE and UPDATE_WEIGHTS and CRITIC_DOMAIN != "shortest"): # Atualizar a cada episódio em vez de a cada 3
                #old_weights = maddpg_agents.agents[0].actor.fc1.weight.clone().detach()
                maddpg_agents.learn(memory)
                #new_weights = maddpg_agents.agents[0].actor.fc1.weight.clone().detach()
                #weight_diff = T.sum(T.abs(new_weights - old_weights)).item()
                #print(f"Época {epoch}, Episódio {e}: Diferença de pesos = {weight_diff:.6f}")

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
            elif EVALUATE and TRAIN:
                graph_y_axis[epoch][e] = int(total_reward)
                y_axis_training[epoch] = sum(total_epoch_reward) / len(total_epoch_reward)
                

            # print(f"{'OG' if epoch % 2 == 0 else 'NEW'} REWARD {total_reward}")
            ### episode ends
        
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
            percentage[epoch] = round(((total_epoch_pck_loss/(total_epoch_pck_loss+total_epoch_pck_sent))*100), 2)
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
        data_file.write(f"Packets lost training: {round(total_package_loss_nr/(total_package_loss_nr+total_packets_sent_nr) * 100, 2)}% \n")
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
        plt.figure(figsize=(12, 8))
        
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
        
        # Save data to CSV
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
    
    def collect_evaluation_data(graph_y_axis, nr_epochs):
        """Collect reward data and calculate metrics for each epoch."""
        all_rewards = []
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
            
            # Mark failure points
            if epoch > 0:
                collect_removed_links_info(epoch, links_removed_info)
                failure_points.append(sum([len(graph_y_axis[e]) for e in range(epoch)]))
        
        return all_rewards, failure_points, epoch_averages, links_removed_info
        
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
    
    def save_evaluation_results(folder_path, sub_path, all_rewards, epoch_averages, 
                           convergence_times, links_removed_info):
        """Save all evaluation results to files."""
        # Save detailed reward evolution
        detailed_df = pd.DataFrame({
            'Episode': np.arange(len(all_rewards)),
            'Reward': all_rewards,
        })
        detailed_df.to_csv(f"{folder_path}/reward_evolution.csv", index=False, sep=';', decimal='.')
        
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
        y_min = min(all_rewards) * 0.9 if min(all_rewards) > 0 else min(all_rewards) * 1.1
        y_max = max(all_rewards) * 1.1 if max(all_rewards) > 0 else max(all_rewards) * 0.9
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
        
        # Linha de tendência
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
        
        # Salvar dados incluindo informação sobre estabilização
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

    def process_evaluation_results(link_utilization_history, graph_y_axis, nr_epochs, folder_path, sub_path, check_congestion=False):
        """Processa resultados de avaliação para qualquer cenário EVALUATE"""
        plt.figure(figsize=(12, 6))

        # Process data for each epoch
        for epoch in range(nr_epochs):
            visualize_link_utilization(link_utilization_history, folder_path, epoch)
        
        # Collect all evaluation data
        all_rewards, failure_points, epoch_averages, links_removed_info = collect_evaluation_data(graph_y_axis, nr_epochs)
        
        # Analyze convergence after failure points
        convergence_times = analyze_convergence(graph_y_axis, nr_epochs)
        
        # Save results to files
        save_evaluation_results(folder_path, sub_path, all_rewards, epoch_averages, 
                            convergence_times, links_removed_info)
        
        # Create and save the rewards plot
        plot_rewards_with_failures(all_rewards, failure_points, folder_path)
    
    def create_comparison_graphs(topology_type):
        """Cria gráficos comparativos para todos os cenários e configurações de rede."""
        scenarios = {2: "eval", 3: "eval_update", 4: "eval_train"}
        
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
        
        # Para cada cenário, criar os dois tipos de gráficos
        for scenario_type, scenario_name in scenarios.items():
            print(f"Processando cenário {scenario_type} ({scenario_name})...")
            
            # 1. GRÁFICO DE RECOMPENSAS
            plt.figure(figsize=(14, 8))
            reward_data_found = False
            
            for config in configurations:
                config_path = os.path.join(gnn_path, config)
                if not os.path.exists(config_path):
                    continue
                    
                result_folders = [d for d in os.listdir(config_path) 
                                if os.path.isdir(os.path.join(config_path, d)) and 
                                    scenario_name in d]
                
                if not result_folders:
                    continue
                    
                latest_folder = sorted(result_folders)[-1]
                
                # Procurar múltiplos arquivos possíveis
                reward_files = [
                    os.path.join(config_path, latest_folder, "data_total.csv"),
                    os.path.join(config_path, latest_folder, "reward_evolution.csv")
                ]
                
                for data_file in reward_files:            
                    if os.path.exists(data_file):
                        try:
                            df = pd.read_csv(data_file, sep=';', decimal='.')
                            
                            # Processar cada tipo de arquivo diferentemente
                            if 'Episode' in df.columns and 'Reward' in df.columns:
                                # reward_evolution.csv para cenários 2/3
                                episodes_per_epoch = 16  # 4 episódios * 4 (padrão para cenários 2/3)
                                epochs = len(df) // episodes_per_epoch
                                x = np.arange(epochs)
                                
                                # Calcular média de recompensa para cada época
                                y = []
                                for i in range(epochs):
                                    start = i * episodes_per_epoch
                                    end = (i + 1) * episodes_per_epoch
                                    y.append(df['Reward'][start:end].mean())
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
                            
                            # Converter valores se forem strings
                            if len(y) > 0 and isinstance(y[0], str):
                                y = np.array([float(val.replace(',', '.')) for val in y])
                                
                            label = config.replace("_", " ").title()
                            plt.plot(x, y, marker='o', linestyle='-', label=label, alpha=0.8)
                            reward_data_found = True
                            break  # Se encontrou um arquivo, não processa o outro
                        except Exception as e:
                            print(f"Erro processando {data_file}: {str(e)}")
            
            if reward_data_found:
                # Configurações do gráfico de recompensas
                plt.title(f"Comparação de Recompensas - Cenário {scenario_type} ({scenario_name}) - {topology_type.title()}")
                plt.xlabel("Época")
                plt.ylabel("Recompensa Média")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                
                # Se for cenário 2/3, marcar pontos de falha
                if scenario_type in [2, 3]:
                    for i in range(1, len(x)):
                        plt.axvline(x=i, color='r', linestyle='--', alpha=0.3, 
                                label='Falha de Link' if i==1 else "")
                
                # Se for cenário 4, marcar pontos de aumento de banda
                elif scenario_type == 4:
                    # Calcular ponto de estabilização
                    stabilization_point = None
                    if STABILIZE_BANDWIDTH:
                        bw_factor = 1.0
                        for i in range(len(x)):
                            if i > 0 and i % INCREASE_BANDWIDTH_INTERVAL == 0:
                                bw_factor *= BANDWIDTH_INCREASE_FACTOR
                                if bw_factor >= STABILIZE_AFTER_MULTIPLIER:
                                    stabilization_point = i
                                    break
                    
                    for i in range(INCREASE_BANDWIDTH_INTERVAL, len(x), INCREASE_BANDWIDTH_INTERVAL):
                        if stabilization_point is not None and i == stabilization_point:
                            # Ponto de estabilização - linha roxa
                            plt.axvline(x=i, color='purple', linestyle='-', alpha=0.7,
                                    label='Estabilização da Largura de Banda (150%)')
                        elif stabilization_point is None or i < stabilization_point:
                            # Aumento normal de largura de banda - linha verde
                            plt.axvline(x=i, color='g', linestyle='--', alpha=0.3,
                                    label='Aumento de Banda' if i==INCREASE_BANDWIDTH_INTERVAL else "")
                
                plt.tight_layout()
                plt.savefig(os.path.join(comparison_folder, f"reward_comparison_scenario{scenario_type}.png"))
                plt.close()
            else:
                plt.close()
                print(f"Nenhum dado de recompensa encontrado para o cenário {scenario_type}")
                
            # 2. GRÁFICO DE CONGESTIONAMENTO
            congestion_data = {}
            
            for config in configurations:
                config_path = os.path.join(gnn_path, config)
                if not os.path.exists(config_path):
                    continue
                    
                result_folders = [d for d in os.listdir(config_path) 
                                if os.path.isdir(os.path.join(config_path, d)) and 
                                    scenario_name in d]
                
                if not result_folders:
                    continue
                    
                latest_folder = sorted(result_folders)[-1]
                folder_path = os.path.join(config_path, latest_folder)
                
                # Coletar valores de utilização máxima
                max_utilization = []
                
                if scenario_type == 4:
                    # Para cenário 4, usar top_links_utilization_overall.csv
                    util_file = os.path.join(folder_path, "top_links_utilization_overall.csv")
                    if os.path.exists(util_file):
                        try:
                            df = pd.read_csv(util_file, sep=';', decimal='.')
                            # Para cenário 4, não tem Epoch no arquivo, mas tem Average_Utilization
                            if 'Average_Utilization' in df.columns:
                                # Filtrar links ativos
                                active_links = df[df['Status'] == 'ACTIVE'] if 'Status' in df.columns else df
                                if not active_links.empty:
                                    # Pegar os top 4 links mais congestionados para visualização
                                    sorted_links = active_links.sort_values('Average_Utilization', ascending=False)
                                    top_utils = sorted_links['Average_Utilization'].head(4).values
                                    max_utilization.extend(top_utils)
                        except Exception as e:
                            print(f"Erro ao processar {util_file}: {str(e)}")
                else:
                    # Para cenários 2/3, usar top_links_utilization_epochX.csv
                    for epoch in range(4):  # Tipicamente temos 4 épocas
                        util_file = os.path.join(folder_path, f"top_links_utilization_epoch{epoch}.csv")
                        if os.path.exists(util_file):
                            try:
                                df = pd.read_csv(util_file, sep=';', decimal='.')
                                # Filtrar links ativos
                                active_links = df[df['Status'] == 'ACTIVE'] if 'Status' in df.columns else df
                                if not active_links.empty and 'Average_Utilization' in active_links.columns:
                                    max_util = active_links['Average_Utilization'].max()
                                    max_utilization.append(max_util)
                            except Exception as e:
                                print(f"Erro ao processar {util_file}: {str(e)}")
                
                if max_utilization:
                    congestion_data[config] = max_utilization
            
            # Criar gráfico de barras agrupadas se temos dados
            if congestion_data:
                plt.figure(figsize=(14, 8))
                
                # Determinar número máximo de épocas/pontos por configuração
                max_points = max(len(utils) for utils in congestion_data.values())
                
                # Criar barras agrupadas
                bar_width = 0.2
                x = np.arange(max_points)
                
                # Posições para as barras de cada configuração
                positions = [-1.5*bar_width, -0.5*bar_width, 0.5*bar_width, 1.5*bar_width]
                
                for i, (config, utils) in enumerate(congestion_data.items()):
                    # Preencher com zeros se necessário
                    full_utils = utils + [0] * (max_points - len(utils))
                    label = config.replace("_", " ").title()
                    
                    # Usar índice para posicionar as barras, limitando a 4 configurações
                    pos = positions[min(i, 3)]
                    plt.bar(x + pos, full_utils, width=bar_width, label=label, alpha=0.8)
                
                # Linha de threshold de congestionamento
                plt.axhline(y=95, color='r', linestyle='--', label='Threshold de Congestionamento (95%)')
                
                # Configurações do gráfico
                plt.title(f"Utilização Máxima dos Links - Cenário {scenario_type} ({scenario_name}) - {topology_type.title()}")
                plt.xlabel("Época" if scenario_type != 4 else "Top Links Congestionados")
                plt.ylabel("Utilização Máxima (%)")
                
                # Ajustar as labels do eixo X conforme o cenário
                if scenario_type == 4:
                    plt.xticks(x, [f"Link {i+1}" for i in range(max_points)])
                else:
                    plt.xticks(x, [f"Época {i}" for i in range(max_points)])
                    
                plt.legend()
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                plt.savefig(os.path.join(comparison_folder, f"congestion_comparison_scenario{scenario_type}.png"))
                plt.close()
                print(f"Gráfico de congestionamento para cenário {scenario_type} criado!")
        
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
    elif EVALUATE:
        if not TRAIN:
            process_evaluation_results(link_utilization_history, graph_y_axis, nr_epochs, folder_path, sub_path)
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

            # Gerar gráficos comparativos apenas quando for o cenário shortest_shortest
            # Este deve ser o último cenário a ser executado para cada configuração
            # Para geração de gráficos, escolhemos:
            # 1. shortest-shortest quando USE_GNN=False
            # 2. local_critic duelling_q_network quando USE_GNN=True
            if (CRITIC_DOMAIN == "shortest" and NEURAL_NETWORK == "shortest") and USE_GNN == False:
                print(f"Gerando gráficos comparativos para {TOPOLOGY_TYPE} na pasta {gnn_suffix}/comparisons")
                create_comparison_graphs(TOPOLOGY_TYPE)
            elif (CRITIC_DOMAIN == "local_critic" and NEURAL_NETWORK == "duelling_q_network") and USE_GNN == True:
                print(f"Gerando gráficos comparativos para {TOPOLOGY_TYPE} na pasta {gnn_suffix}/comparisons")
                create_comparison_graphs(TOPOLOGY_TYPE)
            



