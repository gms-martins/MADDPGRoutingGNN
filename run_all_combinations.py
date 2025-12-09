import subprocess
import os
import time
import sys
import importlib


#script_dir = os.environ.get("PATH_SIMULATION", "c:/Users/Lenovo/Documents/Tese/RRC_DRL_Update/RRC_DRL_Updates")

script_dir = os.environ.get("PATH_SIMULATION", "/workspaces/MADDPGRoutingGNN")
script_path = os.path.join(script_dir, "MADDPG.py")

# Configurações de topologias com seus respectivos parâmetros
topology_configs = [
    #("internet", 25, 25, 11),
    #("arpanet", 33, 33, 6),
    ("service_provider", 65, 65, 4)
]

# Combinações de treino
train_configs = [
    #("shortest", "shortest"),
    #("central_critic", "duelling_q_network"),
    #("central_critic", "simple_q_network"),
    ("local_critic", "duelling_q_network")
]

test_scenarios = [
    (False, False, False),  # Cenário 1: Apenas treino (executado primeiro para gerar modelos)
    #(True, False, False),   # Cenário 2: Apenas avaliar (usa modelos do cenário 1)
    #(True, True, False),    # Cenário 3: Avaliar com update (usa modelos do cenário 1)
    #(True, True, True),     # Cenário 4: Avaliar + treino (usa modelos do cenário 1)
]

# Adicionar timestamp único para pasta
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# Primeiro executa todas as configurações sem GNN (original)
for use_gnn in [False]: # True para usar GNN
    gnn_status = "COM GNN" if use_gnn else "SEM GNN (ORIGINAL)"
    print(f"\n\n{'=' * 60}")
    print(f"INICIANDO EXECUÇÕES {gnn_status}")
    print(f"{'=' * 60}\n")

    for topology, number_of_hosts, number_of_agents, nr_max_links in topology_configs:

        print(f"\n===== Iniciando experimentos com topologia: {topology} =====")
        print(f"Hosts: {number_of_hosts}, Agents: {number_of_agents}, Max Links: {nr_max_links}\n")
        for critic, network in train_configs:
            if use_gnn and critic == "shortest" and network == "shortest":
                print(f"Pulando configuração 'shortest-shortest' com GNN ativada (não recomendado)")
                continue

            for eval_flag, update_flag, train_flag in test_scenarios:
                # Cenários 2 e 3: Avaliação sem treino (com ou sem update)
                if eval_flag and not train_flag:
                    for num_links_to_remove in [1, 2]:
                        if eval_flag and not train_flag and not update_flag: #Cenário 2
                            scenario_name = f"eval_{num_links_to_remove}link{'s' if num_links_to_remove > 1 else ''}"
                        elif eval_flag and not train_flag and update_flag: #Cenário 3
                            scenario_name = f"eval_update_{num_links_to_remove}link{'s' if num_links_to_remove > 1 else ''}"

                        gnn_suffix = "_GNN" if use_gnn else ""
                        timestamp_with_gnn = f"{timestamp}_{critic}_{network}{gnn_suffix}"

                        env_file = os.path.join(script_dir, "environmental_variables.py")
                        with open(env_file, "w") as f:
                            f.write(f'''
NR_ACTIVE_CONNECTIONS = 10
NUMBER_OF_PATHS = 3

NR_EPOCHS = 200
EPOCH_SIZE = 100

NOTES = ""
PATH_SIMULATION = "{script_dir}"
SIM_NR = "{timestamp_with_gnn}"

INCREASE_BANDWIDTH_INTERVAL = 1
BANDWIDTH_INCREASE_FACTOR = 2
MAX_BANDWIDTH_MULTIPLIER = 2
STABILIZE_BANDWIDTH = True
STABILIZE_AFTER_MULTIPLIER = 2
SAVE_REMOVED_LINKS_SCENARIO4 = True

TOPOLOGY_TYPE = "{topology}"; NUMBER_OF_HOSTS = {number_of_hosts}; NUMBER_OF_AGENTS = {number_of_agents}; NR_MAX_LINKS = {nr_max_links}
STATE_SIZE = NR_MAX_LINKS + 1 + NR_ACTIVE_CONNECTIONS * 2 + 1

CRITIC_DOMAIN = "{critic}"
NEURAL_NETWORK = "{network}"

EVALUATE = {eval_flag}
UPDATE_WEIGHTS = {update_flag}
TRAIN = {train_flag}

MODIFIED_NETWORK = "remove_edges"

NUM_LINKS_TO_REMOVE = {num_links_to_remove}

# Controla se a GNN é usada ou não
USE_GNN = {use_gnn}
''')
                        time.sleep(1)
                        print(f"Executando {gnn_status}: {critic} | {network} | EVAL={eval_flag}, UPDATE={update_flag}, TRAIN={train_flag} | NUM_LINKS_TO_REMOVE = {num_links_to_remove} links ")
                        subprocess.run(["python", script_path], cwd=script_dir)
                        time.sleep(2)
                else:
                    # Cenários 1 e 4 (apenas treino ou treino após falha)
                    if not eval_flag: #Cenário 1
                        scenario_name = "train"
                        num_links_to_remove = 0
                    else: #Cenário 4
                        scenario_name = "eval_train"
                        num_links_to_remove = 0

                    gnn_suffix = "_GNN" if use_gnn else ""
                    timestamp_with_gnn = f"{timestamp}_{critic}_{network}{gnn_suffix}"

                    env_file = os.path.join(script_dir, "environmental_variables.py")
                    with open(env_file, "w") as f:
                        f.write(f'''
NR_ACTIVE_CONNECTIONS = 10
NUMBER_OF_PATHS = 3

NR_EPOCHS = 200
EPOCH_SIZE = 100

NOTES = ""
PATH_SIMULATION = "{script_dir}"
SIM_NR = "{timestamp_with_gnn}"

INCREASE_BANDWIDTH_INTERVAL = 1
BANDWIDTH_INCREASE_FACTOR = 2
MAX_BANDWIDTH_MULTIPLIER = 2
STABILIZE_BANDWIDTH = True
STABILIZE_AFTER_MULTIPLIER = 2
SAVE_REMOVED_LINKS_SCENARIO4 = True

TOPOLOGY_TYPE = "{topology}"; NUMBER_OF_HOSTS = {number_of_hosts}; NUMBER_OF_AGENTS = {number_of_agents}; NR_MAX_LINKS = {nr_max_links}
STATE_SIZE = NR_MAX_LINKS + 1 + NR_ACTIVE_CONNECTIONS * 2 + 1

CRITIC_DOMAIN = "{critic}"
NEURAL_NETWORK = "{network}"

EVALUATE = {eval_flag}
UPDATE_WEIGHTS = {update_flag}
TRAIN = {train_flag}

MODIFIED_NETWORK = "remove_edges"

NUM_LINKS_TO_REMOVE = {num_links_to_remove}

# Controla se a GNN é usada ou não
USE_GNN = {use_gnn}
''')
                    time.sleep(1)
                    print(f"Executando {gnn_status}: {critic} | {network} | EVAL={eval_flag}, UPDATE={update_flag}, TRAIN={train_flag} | NUM_LINKS_TO_REMOVE = {num_links_to_remove} links ")
                    subprocess.run(["python", script_path], cwd=script_dir)
                    time.sleep(2)
                
                