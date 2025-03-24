import random

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
from NetworkEnv import NetworkEnv
from environmental_variables import STATE_SIZE, EPOCH_SIZE, NUMBER_OF_AGENTS, NR_EPOCHS, EVALUATE, CRITIC_DOMAIN, SIM_NR, TRAIN, NEURAL_NETWORK, MODIFIED_NETWORK, NOTES, TOPOLOGY_TYPE, UPDATE_WEIGHTS, PATH_SIMULATION, NUMBER_OF_PATHS, CHECKPOINT, CHECKPOINT_FILE
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

    def choose_action(self, raw_obs):#, topology):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx]) #, topology)
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

    REPLAY_MEMORY = 1000 #50000
    MEMORY_BATCH = 100 #256

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
                           alpha=0.001, beta=0.0001, tau=0.0001, chkpt_dir='.\\tmp\\maddpg\\')
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
                learning = "eval" # Cenários 2 
            else:
                learning = "eval_update" # Cenário 3
    else:
        learning = "train" # Cenário 1

    # Adicionar modificação da rede separadamente
    if MODIFIED_NETWORK and (EVALUATE or TRAIN):
        folder_name = f"{learning}_{MODIFIED_NETWORK}"
    else:
        folder_name = learning 

      
    folder_path = f'{PATH_SIMULATION}/results/{NR_EPOCHS}epo_{EPOCH_SIZE}epi_{CRITIC_DOMAIN}_{NEURAL_NETWORK}_{TOPOLOGY_TYPE}_{folder_name}_{hh}_{mm}'
    sub_path = f'{NR_EPOCHS}epo_{EPOCH_SIZE}epi_{CRITIC_DOMAIN}_{NEURAL_NETWORK}_{TOPOLOGY_TYPE}_{folder_name}'
    os.mkdir(folder_path)

    if not EVALUATE or (EVALUATE and TRAIN):
        graph_y_axis = np.zeros(NR_EPOCHS)
        y_axis_training = np.zeros(NR_EPOCHS)
        graph_x_axis = np.zeros(NR_EPOCHS)
    elif EVALUATE and not TRAIN: # and UPDATE_WEIGHTS:
        graph_x_axis = np.zeros(EPOCH_SIZE*4)
        aux = np.zeros(EPOCH_SIZE*4) 
        graph_y_axis = [[0 for _ in range(EPOCH_SIZE*4)] for _ in range(nr_epochs)]

    if (EVALUATE and NEURAL_NETWORK != "shortest") or (EVALUATE and TRAIN):
        maddpg_agents.load_checkpoint()

    i_epoch = 0

    if CHECKPOINT:
        maddpg_agents.load_checkpoint()
        check_file = np.loadtxt(CHECKPOINT_FILE, delimiter=',', dtype=float)[1,:]
        for i in np.arange(0, len(check_file)):
            if check_file[i] != 0:
                y_axis_training = check_file[i]
            else:
                i_epoch = i
                break

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
            eng.remove_edges(3)
        if MODIFIED_NETWORK == "add_edges":
            eng.add_edges(3)


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

                    #print("state: ", state)
                    if CRITIC_DOMAIN == "central_critic":
                        critic_states.append(np.concatenate((eng.get_link_usage(), np.array(all_dsts)), axis=0))
                    elif CRITIC_DOMAIN == "local_critic":     
                        critic_states.append(state)

                actions = maddpg_agents.choose_action(states)

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

            if (e % 3 == 0 and not EVALUATE) or (EVALUATE and UPDATE_WEIGHTS) and CRITIC_DOMAIN != "shortest":
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
            #x = np.arange(0, NR_EPOCHS)
            # np.savetxt(f"{folder_path}/data_while_training.csv", (x, y_axis_training), delimiter=',')
            # Salvar em CSV com formato mais legível
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
    data_file.close    

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

        data_df = pd.DataFrame({
            #'Epoch': x,
            #'Reward': graph_y_axis
        })
        data_df.to_csv(f"{folder_path}/data.csv", index=False,sep=';', decimal='.')

        data_total_df = pd.DataFrame({
            'Epoch': x,
            'Average_Reward': [f"{value:.3f}" for value in y_axis_training]
        })
        data_total_df.to_csv(f"{folder_path}/data_total.csv", index=False,sep=';', decimal='.')

        #Fnp.savetxt(f"{folder_path}/data.csv", (graph_x_axis, graph_y_axis), delimiter=',')
        #np.savetxt(f"{folder_path}/data_total.csv", (x, y_axis_training), delimiter=',')
        plt.show()

    