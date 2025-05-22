import subprocess
import os
import time
import sys
import importlib

script_dir = os.environ.get("PATH_SIMULATION", "c:/Users/Utilizador/Ambiente de Trabalho/Tese/RRC_DRL_Update/RRC_DRL_Updates")
#script_dir = os.environ.get("PATH_SIMULATION", "/workspaces/RRC_DRL_Updates")
script_path = os.path.join(script_dir, "MADDPG.py")

# Configurações de topologias com seus respectivos parâmetros
topology_configs = [
    ("internet", 25, 25, 11),
    ("arpanet", 33, 33, 6),
    ("service_provider", 65, 65, 4)
]

# Combinações de treino
train_configs = [
    ("central_critic", "duelling_q_network"),
    ("central_critic", "simple_q_network"),
    ("local_critic", "duelling_q_network"),
    ("shortest", "shortest")
]

test_scenarios = [
    (False, False, False),  # Cenário 1: Apenas treino (executado primeiro para gerar modelos)
    (True, False, False),   # Cenário 2: Apenas avaliar (usa modelos do cenário 1)
    (True, True, False),    # Cenário 3: Avaliar com update (usa modelos do cenário 1)
    (True, True, True),     # Cenário 4: Avaliar + treino (usa modelos do cenário 1)
]

# Adicionar timestamp único para pasta
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# Primeiro executa todas as configurações sem GNN (original)
for use_gnn in [False, True]: # True para usar GNN
    gnn_status = "COM GNN" if use_gnn else "SEM GNN (ORIGINAL)"
    print(f"\n\n{'=' * 60}")
    print(f"INICIANDO EXECUÇÕES {gnn_status}")
    print(f"{'=' * 60}\n")

    for topology, number_of_hosts, number_of_agents, nr_max_links in topology_configs:

        print(f"\n===== Iniciando experimentos com topologia: {topology} =====")
        print(f"Hosts: {number_of_hosts}, Agents: {number_of_agents}, Max Links: {nr_max_links}\n")
        for critic, network in train_configs:
            # Pular a combinação "shortest-shortest" quando GNN está ativada
            if use_gnn and critic == "shortest" and network == "shortest":
                print(f"Pulando configuração 'shortest-shortest' com GNN ativada (não recomendado)")
                continue  # Simplesmente pula para a próxima combinação sem executar
            
            # Para todas as outras combinações, executar normalmente
            for eval_flag, update_flag, train_flag in test_scenarios:
                    # Gerar cenário
                    if not eval_flag:
                        scenario_name = "train"
                    elif eval_flag and not train_flag and not update_flag:
                        scenario_name = "eval"
                    elif eval_flag and not train_flag and update_flag:
                        scenario_name = "eval_update"
                    else:
                        scenario_name = "eval_train"
                    
                    # Adicionar sufixo GNN à pasta para os resultados com GNN
                    gnn_suffix = "_GNN" if use_gnn else ""
                    timestamp_with_gnn = f"{timestamp}_{critic}_{network}{gnn_suffix}"                # Gerar código atualizado para as variáveis
                    env_file = os.path.join(script_dir, "environmental_variables.py")
                    with open(env_file, "w") as f:
                        f.write(f'''
NR_ACTIVE_CONNECTIONS = 10
NUMBER_OF_PATHS = 3

NR_EPOCHS = 2
EPOCH_SIZE = 2

NOTES = ""
PATH_SIMULATION = "{script_dir}"
SIM_NR = "{timestamp_with_gnn}"


INCREASE_BANDWIDTH_INTERVAL = 3
BANDWIDTH_INCREASE_FACTOR = 1.4
MAX_BANDWIDTH_MULTIPLIER = 2.0
STABILIZE_BANDWIDTH = True
STABILIZE_AFTER_MULTIPLIER = 1.75  
SAVE_REMOVED_LINKS_SCENARIO4 = True

TOPOLOGY_TYPE = "{topology}"; NUMBER_OF_HOSTS = {number_of_hosts}; NUMBER_OF_AGENTS = {number_of_agents}; NR_MAX_LINKS = {nr_max_links}
STATE_SIZE = NR_MAX_LINKS + 1 + NR_ACTIVE_CONNECTIONS * 2 + 1

CRITIC_DOMAIN = "{critic}"
NEURAL_NETWORK = "{network}"

EVALUATE = {eval_flag}
UPDATE_WEIGHTS = {update_flag}
TRAIN = {train_flag}

MODIFIED_NETWORK = "remove_edges"

# Controla se a GNN é usada ou não
USE_GNN = {use_gnn}
''')
                    
                    # Garantir que o arquivo seja salvo completamente
                    time.sleep(1)
                    
                    # Exibir configuração atual
                    print(f"Executando {gnn_status}: {critic} | {network} | EVAL={eval_flag}, UPDATE={update_flag}, TRAIN={train_flag}")
                    
                    # Usar o caminho completo e executar em um novo ambiente
                    subprocess.run(["python", script_path], cwd=script_dir)
                    
                    # Forçar um atraso entre execuções para garantir que os
                    # arquivos temporários sejam liberados e o cache seja resetado
                    time.sleep(2)
                
                