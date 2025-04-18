import json
import pickle
import random
from itertools import islice

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
import torch as T
import sys
import time
import os

from Link import Link
from NetworkComponent import NetworkComponent
from environmental_variables import EPOCH_SIZE, STATE_SIZE, NR_MAX_LINKS, EVALUATE,UPDATE_WEIGHTS, MODIFIED_NETWORK, NUMBER_OF_PATHS, TOPOLOGY_TYPE, TRAIN , PATH_SIMULATION, MAX_BANDWIDTH_MULTIPLIER,BANDWIDTH_INCREASE_FACTOR,SAVE_REMOVED_LINKS_SCENARIO4, STABILIZE_BANDWIDTH, STABILIZE_AFTER_MULTIPLIER, INCREASE_BANDWIDTH_INTERVAL, NR_ACTIVE_CONNECTIONS, CHECKPOINT, CHECKPOINT_FILE

REMOVED_EDGES = {1: None, 2: None, 3: None}  # Armazenar links removidos por época
SCENARIO_2_COMPLETED = False  # Flag para identificar quando o cenário 2 foi executado

class NetworkEngine:

    def __init__(self):
        self.graph_has_data = False

        # Choose the topology
        if TOPOLOGY_TYPE == "service_provider":
            self.set_service_provider_topology()
            return
        elif TOPOLOGY_TYPE == "arpanet":
            self.set_arpanet_topology()
            return

        #self.graph_topology = pickle.load(open("c:/Users/Utilizador/Ambiente de Trabalho/Tese/RRC_DRL_Update/RRC_DRL_Updates/TopologyFiles/small_network.pickle", 'rb'))
        self.graph_topology=pickle.load(open(f"{PATH_SIMULATION}/TopologyFiles/small_network.pickle", 'rb'))
        self.links = {}
        self.hosts = {}
        self.switchs = {}
        self.components = {}

        # self.graph_topology = nx.random_internet_as_graph(50)

        self.paths = {}
        self.communication_sequences = {'H1': ['H17', 'H22', 'H15', 'H12', 'H25', 'H13', 'H4', 'H9', 'H10', 'H15'],
                                        'H10': ['', 'H1', 'H8', 'H18', 'H6', 'H20', 'H6', 'H19', '', 'H15'],
                                        'H11': ['H24', 'H14', 'H20', 'H5', 'H17', '', 'H2', 'H3', 'H14', 'H13'],
                                        'H12': ['H1', 'H14', 'H6', 'H7', 'H22', 'H21', 'H18', 'H21', '', 'H23'],
                                        'H13': ['H22', 'H5', 'H4', 'H11', 'H25', 'H8', '', '', 'H11', 'H9'],
                                        'H14': ['H3', '', 'H3', '', '', 'H7', 'H12', 'H3', 'H10', 'H24'],
                                        'H15': ['H1', 'H22', 'H3', '', 'H7', 'H22', '', '', 'H6', 'H12'],
                                        'H16': ['H9', 'H4', '', 'H25', 'H14', 'H10', 'H18', 'H10', 'H18', 'H8'],
                                        'H17': ['H1', 'H4', 'H1', 'H3', 'H9', '', 'H8', '', 'H16', 'H8'],
                                        'H18': ['H5', 'H10', 'H11', 'H8', 'H13', 'H7', 'H21', 'H5', 'H24', 'H10'],
                                        'H19': ['H3', '', 'H13', 'H16', 'H13', '', 'H7', 'H18', '', 'H1'],
                                        'H2': ['H14', '', 'H23', 'H16', '', '', 'H6', 'H20', 'H6', 'H13'],
                                        'H20': ['H1', 'H8', 'H6', 'H10', 'H2', '', 'H1', 'H2', 'H22', 'H7'],
                                        'H21': ['H1', '', 'H9', 'H4', '', 'H15', 'H20', '', 'H9', 'H20'],
                                        'H22': ['', '', 'H3', 'H10', 'H3', 'H6', 'H3', 'H12', 'H11', ''],
                                        'H23': ['', 'H9', 'H12', '', 'H15', 'H25', 'H9', 'H17', 'H12', 'H4'],
                                        'H24': ['', 'H6', 'H11', 'H2', 'H5', 'H7', 'H4', 'H12', 'H3', 'H9'],
                                        'H25': ['', '', 'H6', 'H24', 'H13', 'H19', 'H4', '', 'H21', 'H23'],
                                        'H3': ['H21', 'H4', '', 'H24', 'H10', 'H1', 'H20', 'H13', 'H5', 'H1'],
                                        'H4': ['H1', 'H6', 'H22', 'H8', 'H8', 'H18', 'H18', '', 'H22', 'H13'],
                                        'H5': ['H3', 'H9', 'H1', 'H10', '', 'H2', 'H18', 'H15', 'H1', 'H17'],
                                        'H6': ['H1', '', 'H21', 'H9', 'H10', 'H25', '', 'H2', 'H19', 'H18'],
                                        'H7': ['', 'H1', 'H22', '', 'H9', 'H8', 'H9', '', '', 'H10'],
                                        'H8': ['H1', 'H2', 'H6', 'H12', 'H25', 'H9', 'H19', 'H16', 'H18', 'H20'],
                                        'H9': ['H14', '', 'H12', 'H2', 'H17', 'H15', 'H15', 'H20', 'H5', 'H3']}

        # {'H1': ['H33', 'H49', 'H29', 'H2', 'H48', 'H3', 'H4', 'H16', 'H11', 'H12', 'H42','H20', 'H4', 'H48', 'H8', 'H47', 'H37', 'H21', 'H31', 'H30', 'H44', '', 'H50', 'H16', 'H3', 'H12', 'H13', 'H29', 'H28', ''], 'H2': ['H4', 'H4', 'H37', '', 'H44', 'H18', 'H33', 'H50', 'H38', 'H30', 'H6', 'H46', 'H32', '', 'H36', 'H6', 'H7', 'H33', 'H23', 'H3', 'H42', 'H10', 'H25', 'H28', 'H6', 'H16', 'H11', 'H29', 'H31', 'H29'], 'H3': ['H2', 'H16', 'H47', '', 'H11', '', 'H47', 'H50', 'H26', 'H2', 'H21', 'H28', 'H8', 'H25', '', 'H26', 'H36', 'H48', 'H4', 'H11', '', 'H17', 'H30', 'H5', 'H38', 'H6', 'H38', 'H2', 'H1', 'H36'], 'H4': ['H41', 'H31', 'H13', '', 'H18', '', 'H41', 'H5', 'H35', 'H33', 'H37', 'H36', 'H31', 'H42', 'H34', 'H33', 'H16', 'H23', 'H34', 'H45', 'H24', 'H47', 'H17', 'H47', 'H11', '', 'H40', 'H21', 'H19', 'H25'], 'H5': ['H40', 'H21', 'H7', 'H48', 'H10', '', 'H8', 'H40', 'H13', 'H26', 'H10', 'H4', 'H2', '', 'H41', 'H28', 'H36', '', 'H4', 'H28', '', 'H43', 'H37', 'H37', '', 'H24', 'H43', 'H31', 'H45', 'H9'], 'H6': ['H39', 'H29', '', 'H18', 'H43', 'H3', 'H14', 'H41', 'H19', 'H20', 'H28', '', 'H33', 'H13', '', '', '', 'H4', 'H38', 'H32', 'H8', 'H1', '', 'H39', 'H12', 'H2', 'H48', 'H38', '', 'H50'], 'H7': ['H41', 'H42', 'H1', 'H31', 'H18', 'H35', 'H6', 'H28', 'H25', '', 'H37', 'H19', 'H32', 'H31', 'H12', 'H45', 'H19', 'H12', 'H36', 'H48', 'H29', 'H47', 'H3', 'H32', 'H4', 'H29', 'H46', 'H38', 'H33', 'H20'], 'H8': ['H38', '', 'H45', 'H24', 'H25', 'H22', 'H24', 'H32', 'H36', 'H47', 'H39', 'H30', 'H16', 'H1', 'H40', 'H48', 'H11', 'H48', 'H42', '', 'H23', 'H47', 'H10', 'H18', 'H39', '', 'H43', 'H18', 'H20', 'H30'], 'H9': ['H41', 'H38', 'H30', 'H33', 'H20', 'H32', 'H1', 'H8', 'H20', 'H39', 'H46', '', 'H22', 'H44', 'H42', 'H30', 'H33', 'H16', 'H32', 'H35', 'H25', 'H23', 'H28', 'H17', 'H1', 'H26', 'H11', 'H50', 'H17', 'H32'], 'H10': ['H5', '', 'H29', 'H19', 'H26', 'H32', 'H13', 'H47', 'H45', 'H42', 'H31', 'H46', 'H6', 'H48', '', 'H15', 'H21', 'H19', 'H2', 'H27', 'H11', 'H28', 'H19', 'H28', 'H20', 'H50', 'H14', 'H5', '', 'H24'], 'H11': ['H3', 'H5', 'H23', 'H39', 'H14', 'H13', 'H36', 'H40', 'H50', 'H22', 'H16', 'H47', 'H24', 'H16', 'H4', 'H39', 'H19', 'H30', 'H50', 'H30', 'H40', 'H34', 'H1', '', 'H10', '', 'H49', 'H37', 'H50', 'H24'], 'H12': ['H36', 'H36', 'H23', 'H6', 'H33', 'H5', 'H36', 'H25', 'H15', 'H31', 'H30', 'H37', 'H18', 'H28', 'H5', 'H9', 'H21', 'H45', 'H49', 'H27', '', 'H30', 'H1', 'H13', 'H39', 'H14', '', 'H10', 'H3', 'H30'], 'H13': ['H16', 'H8', 'H31', '', 'H6', 'H40', 'H20', 'H49', 'H22', '', 'H47', 'H35', 'H15', 'H25', '', 'H7', 'H28', 'H38', 'H34', 'H40', 'H25', 'H17', 'H34', '', 'H28', 'H8', 'H9', 'H3', 'H6', 'H34'], 'H14': ['H47', '', 'H32', '', 'H8', 'H44', 'H15', 'H48', 'H3', 'H23', 'H18', 'H44', 'H40', 'H17', 'H30', 'H4', 'H38', 'H48', 'H44', 'H28', '', '', 'H9', 'H24', '', 'H24', '', 'H23', 'H21', 'H43'], 'H15': ['H34', '', 'H30', 'H40', 'H11', 'H16', 'H8', 'H2', 'H12', '', 'H29', 'H18', 'H7', 'H1', 'H10', 'H11', 'H9', 'H36', 'H10', 'H23', 'H39', 'H9', 'H7', 'H10', '', 'H2', 'H12', '', '', 'H21'], 'H16': ['H19', 'H43', 'H5', 'H30', 'H9', 'H33', 'H33', 'H27', 'H17', 'H38', 'H19', 'H22', 'H36', 'H5', '', 'H36', 'H17', 'H19', 'H21', 'H43', 'H49', 'H49', 'H4', 'H30', 'H12', 'H28', 'H25', 'H13', 'H39', ''], 'H17': ['H39', 'H31', 'H44', 'H33', 'H25', 'H4', 'H35', 'H28', 'H35', 'H30', 'H46', 'H25', 'H31', '', 'H8', '', 'H13', 'H44', 'H15', 'H19', 'H1', 'H49', 'H35', 'H36', 'H27', 'H14', 'H36', 'H42', 'H29', 'H40'], 'H18': ['H2', 'H20', 'H48', 'H35', 'H15', 'H16', 'H1', 'H14', 'H9', 'H21', 'H7', 'H15', 'H11', 'H49', 'H9', 'H4', 'H10', 'H42', 'H40', 'H48', 'H45', 'H39', 'H32', 'H7', 'H47', 'H3', 'H28', 'H31', 'H35', 'H49'], 'H19': ['H23', 'H11', 'H38', 'H29', 'H21', 'H43', 'H46', 'H43', 'H34', '', '', 'H50', 'H37', 'H7', 'H4', 'H25', '', 'H39', 'H35', 'H35', 'H7', 'H27', 'H6', 'H16', '', 'H4', 'H44', 'H6', '', 'H39'], 'H20': ['H28', 'H19', '', '', 'H35', 'H46', 'H25', 'H32', 'H40', 'H10', 'H44', 'H35', 'H28', 'H33', 'H29', 'H14', 'H22', 'H5', 'H13', 'H4', '', 'H26', 'H30', 'H22', 'H41', 'H46', 'H16', 'H24', 'H21', 'H15'], 'H21': ['H30', 'H14', 'H20', 'H14', 'H12', 'H49', 'H11', '', 'H41', 'H3', 'H41', 'H11', 'H8', 'H24', 'H7', 'H12', 'H30', '', 'H50', 'H14', 'H44', 'H13', 'H24', 'H42', 'H6', 'H22', 'H31', '', 'H48', 'H2'], 'H22': ['H30', 'H49', 'H16', 'H46', '', 'H16', '', 'H13', 'H13', 'H43', 'H26', 'H47', 'H46', 'H6', 'H31', 'H40', 'H49', 'H9', 'H43', 'H29', 'H32', 'H28', 'H16', 'H3', 'H34', 'H21', 'H7', 'H33', 'H50', 'H21'], 'H23': ['H6', 'H12', 'H3', 'H9', 'H38', 'H24', 'H48', 'H11', 'H33', 'H15', 'H45', 'H18', 'H3', 'H24', 'H29', 'H1', 'H41', 'H39', 'H45', 'H43', 'H12', 'H3', 'H31', 'H6', 'H14', 'H32', 'H39', 'H44', 'H27', ''], 'H24': ['H31', 'H37', 'H13', '', 'H13', 'H34', 'H12', 'H45', 'H33', '', 'H5', 'H13', 'H47', 'H41', 'H46', 'H30', 'H26', 'H50', 'H23', 'H1', 'H34', 'H37', '', 'H27', 'H22', 'H22', 'H43', 'H16', 'H2', 'H48'], 'H25': ['H10', 'H23', 'H44', '', '', 'H32', 'H47', 'H21', 'H30', 'H42', 'H9', '', 'H24', 'H43', 'H4', 'H38', 'H48', '', 'H28', 'H16', 'H2', 'H12', 'H38', 'H1', 'H24', 'H1', '', '', 'H27', 'H30'], 'H26': ['H16', 'H49', 'H9', '', 'H28', 'H10', 'H13', 'H49', 'H6', 'H15', 'H16', 'H35', 'H8', 'H19', 'H5', 'H29', 'H7', 'H16', '', 'H12', 'H2', 'H20', 'H12', 'H25', 'H6', 'H22', 'H35', 'H12', 'H19', 'H45'], 'H27': ['H38', '', 'H35', 'H25', 'H2', '', 'H47', '', 'H5', '', 'H49', 'H8', 'H26', 'H4', 'H17', 'H25', 'H28', 'H15', 'H8', 'H19', 'H33', 'H32', 'H22', '', 'H3', 'H12', 'H11', '', 'H45', 'H36'], 'H28': ['H40', '', 'H2', 'H10', 'H40', 'H10', 'H37', 'H1', 'H2', 'H2', 'H11', 'H29', 'H1', 'H7', 'H29', 'H3', 'H43', 'H18', 'H38', 'H27', 'H39', '', 'H7', 'H3', 'H39', 'H33', 'H14', 'H14', 'H37', 'H30'], 'H29': ['H25', 'H41', 'H45', 'H32', 'H17', '', 'H38', '', 'H19', 'H30', 'H26', 'H7', '', '', 'H16', '', 'H1', 'H8', 'H10', 'H8', '', '', 'H13', 'H13', 'H2', 'H40', 'H16', 'H46', 'H33', 'H24'], 'H30': ['H17', '', 'H47', 'H23', 'H6', '', 'H16', 'H33', '', 'H9', 'H22', 'H21', 'H19', 'H9', 'H34', 'H25', '', 'H48', '', 'H21', 'H49', 'H7', 'H50', 'H35', 'H34', 'H46', 'H19', 'H47', 'H37', 'H4'], 'H31': ['H43', 'H17', 'H4', 'H49', 'H37', 'H23', 'H44', '', 'H37', 'H18', 'H21', 'H4', 'H16', 'H44', '', 'H15', 'H2', 'H35', 'H29', 'H33', 'H49', 'H43', 'H38', '', 'H19', 'H13', 'H40', 'H46', 'H28', 'H8'], 'H32': ['', 'H26', 'H27', 'H37', 'H15', 'H26', 'H23', 'H38', 'H19', 'H6', 'H35', 'H13', 'H29', 'H5', 'H18', 'H25', 'H30', 'H16', 'H35', 'H33', 'H1', 'H38', '', 'H25', 'H14', 'H26', 'H16', 'H30', 'H34', 'H5'], 'H33': ['H48', 'H22', 'H42', 'H24', 'H12', 'H1', 'H6', 'H25', '', 'H47', 'H41', '', 'H15', '', 'H21', 'H34', 'H15', 'H16', 'H10', 'H18', 'H5', 'H28', 'H49', 'H43', 'H6', 'H20', 'H28', 'H50', 'H50', 'H50'], 'H34': ['H23', 'H47', 'H4', 'H26', 'H15', 'H29', 'H44', 'H50', 'H48', 'H37', 'H10', 'H10', 'H32', 'H16', 'H35', '', '', 'H21', 'H1', 'H24', '', 'H29', 'H22', 'H39', 'H15', 'H46', 'H29', 'H33', 'H39', 'H23'], 'H35': ['H43', 'H32', 'H43', 'H7', 'H46', 'H22', 'H37', 'H49', 'H24', 'H32', 'H49', 'H17', 'H8', 'H49', 'H46', 'H11', 'H31', '', '', 'H12', 'H26', 'H6', 'H21', 'H32', 'H49', 'H36', 'H36', 'H8', 'H10', 'H17'], 'H36': ['', 'H13', 'H13', 'H3', 'H45', 'H5', 'H15', 'H28', 'H6', 'H40', 'H5', 'H10', 'H3', 'H46', 'H10', 'H8', 'H2', 'H2', 'H20', 'H16', 'H2', 'H25', 'H50', 'H45', 'H9', 'H26', 'H32', 'H11', 'H30', 'H20'], 'H37': ['H33', 'H32', 'H3', 'H38', '', 'H32', 'H3', 'H26', 'H26', 'H10', 'H20', 'H15', 'H13', 'H36', 'H45', 'H18', 'H29', '', 'H48', 'H39', 'H44', 'H46', 'H31', 'H33', 'H22', 'H45', '', 'H8', 'H13', 'H19'], 'H38': ['H7', 'H40', 'H36', 'H9', 'H35', 'H23', 'H24', 'H19', 'H29', '', 'H7', 'H40', 'H9', 'H36', '', '', 'H48', '', 'H29', 'H48', 'H49', 'H8', 'H13', 'H50', 'H26', 'H25', 'H6', 'H25', 'H14', 'H44'], 'H39': ['H9', 'H47', '', '', 'H36', 'H40', '', 'H9', 'H21', 'H28', 'H48', 'H37', 'H32', 'H23', 'H1', 'H47', 'H45', 'H19', 'H14', 'H30', 'H47', 'H15', 'H9', 'H20', 'H41', 'H21', 'H11', 'H30', 'H6', 'H45'], 'H40': ['H28', 'H17', 'H37', 'H42', 'H35', 'H35', 'H28', 'H22', 'H25', 'H20', 'H30', 'H49', 'H6', 'H2', 'H8', 'H3', 'H8', '', 'H17', 'H27', 'H17', 'H1', 'H24', 'H11', 'H19', 'H22', 'H33', 'H1', 'H16', 'H46'], 'H41': ['H11', 'H12', 'H28', 'H27', 'H34', 'H31', 'H43', 'H20', 'H10', 'H19', 'H18', 'H50', 'H40', 'H28', 'H11', 'H3', 'H33', 'H22', 'H45', 'H1', 'H18', 'H46', 'H3', 'H18', 'H23', 'H32', 'H17', '', 'H1', 'H7'], 'H42': ['H39', 'H37', 'H36', 'H15', 'H18', 'H10', 'H22', 'H5', '', 'H33', 'H3', 'H39', 'H2', 'H9', '', 'H3', 'H25', 'H15', 'H20', 'H8', 'H27', 'H7', 'H19', 'H32', 'H6', 'H36', 'H22', '', 'H30', 'H40'], 'H43': ['H33', 'H16', 'H18', '', 'H32', 'H46', 'H19', 'H48', 'H19', 'H48', 'H22', 'H30', 'H34', 'H46', 'H29', 'H34', 'H41', 'H23', 'H38', 'H25', 'H8', 'H26', '', 'H36', 'H9', 'H41', 'H49', 'H33', 'H29', 'H25'], 'H44': ['H47', 'H17', '', 'H4', 'H16', 'H15', 'H32', 'H6', 'H5', 'H12', 'H41', '', 'H26', 'H17', 'H31', 'H29', 'H3', 'H32', 'H24', 'H35', 'H4', 'H19', 'H22', 'H36', 'H32', 'H13', 'H30', 'H42', 'H13', 'H7'], 'H45': ['H37', 'H29', '', 'H33', 'H18', 'H37', 'H18', 'H20', 'H15', 'H40', 'H35', 'H4', '', 'H20', '', '', 'H34', 'H44', 'H7', 'H13', 'H17', 'H17', 'H50', 'H41', 'H26', '', 'H9', 'H18', '', 'H40'], 'H46': ['H2', '', 'H2', 'H28', 'H36', 'H45', 'H38', 'H17', 'H43', 'H10', 'H2', 'H20', 'H50', 'H11', 'H33', '', 'H8', 'H21', 'H37', 'H19', 'H15', 'H24', 'H30', 'H34', 'H19', 'H27', 'H4', 'H21', 'H32', 'H28'], 'H47': ['H12', 'H30', 'H20', 'H12', 'H32', 'H36', 'H46', '', '', '', '', 'H36', 'H33', 'H45', 'H19', 'H31', 'H32', 'H24', 'H5', 'H7', 'H24', 'H40', 'H8', 'H43', '', 'H7', 'H38', 'H50', 'H23', 'H1'], 'H48': ['H32', 'H1', 'H28', 'H6', 'H23', 'H50', 'H43', 'H7', 'H42', 'H30', 'H32', 'H2', '', 'H4', 'H27', '', 'H7', 'H19', 'H43', 'H21', 'H10', 'H47', 'H8', 'H28', 'H16', 'H40', 'H8', 'H44', 'H16', 'H15'], 'H49': ['H41', 'H32', 'H10', 'H47', 'H38', 'H9', 'H25', 'H42', 'H23', 'H9', 'H27', 'H26', 'H32', '', 'H44', 'H37', 'H40', 'H34', 'H32', 'H38', 'H19', 'H45', 'H31', 'H46', 'H17', 'H32', 'H22', 'H42', 'H10', ''], 'H50': ['H2', 'H8', 'H37', 'H32', 'H31', 'H39', 'H48', 'H25', 'H5', 'H18', 'H46', '', 'H25', 'H30', 'H38', 'H16', 'H23', '', 'H14', 'H21', 'H9', 'H8', 'H23', 'H31', 'H15', 'H7', '', 'H20', 'H19', 'H25']}

        self.create_components(self.graph_topology)

        self.bws = {'H1': 29, 'H2': 28, 'H3': 22, 'H4': 28, 'H5': 33, 'H6': 40, 'H7': 34, 'H8': 29, 'H9': 42, 'H10': 21,
                    'H11': 24, 'H12': 42, 'H13': 34, 'H14': 31, 'H15': 22, 'H16': 26, 'H17': 48, 'H18': 49, 'H19': 50,
                    'H20': 36, 'H21': 34, 'H22': 36, 'H23': 33, 'H24': 24, 'H25': 46, 'H26': 38, 'H27': 38, 'H28': 45,
                    'H29': 21, 'H30': 24, 'H31': 32, 'H32': 50, 'H33': 31, 'H34': 32, 'H35': 49, 'H36': 31, 'H37': 34,
                    'H38': 47, 'H39': 49, 'H40': 29, 'H41': 26, 'H42': 37, 'H43': 28, 'H44': 34, 'H45': 34, 'H46': 43,
                    'H47': 41, 'H48': 24, 'H49': 30, 'H50': 33}

        self.calculate_paths()
        self.hosts = self.get_all_hosts()
        self.number_of_hosts = len(self.hosts)
        self.statistics = {'package_loss': 0, 'package_sent': 0, 'nr_package_loss': 0, 'nr_package_sent': 0}
        self.single_con_hosts = [f"H{int(host) + 1}" for host in self.graph_topology if
                                 len(self.graph_topology.edges(host)) == 1]
        self.bws = {host: bw if host not in self.single_con_hosts else bw // 3 for host, bw in self.bws.items()}
        #print(self.bws)
        #print(self.single_con_hosts)

        #nx.draw(self.graph_topology, with_labels=True)
        #plt.show()

        #self.all_tms = json.load(open("all_tms_test.json", mode="r"))
        self.all_tms = json.load(open("c:/Users/Utilizador/Ambiente de Trabalho/Tese/RRC_DRL_Update/RRC_DRL_Updates/TrafficMatrix/tms_internet_train.json", mode="r"))
        self.current_index = 0
        self.current_tm_index = self.current_index % len(self.all_tms)          #EPOCH_SIZE
        #print("\n current rm index: ", self.current_tm_index)

        self.communication_sequences = self.all_tms[self.current_tm_index]
        #print("\n communication seuqnece all_tms[current_tm_index]: ", '\n'.join([f"{key}: {value}" for key, value in self.communication_sequences.items()]))
        #print("\n len communication seuqences: ", len(self.communication_sequences))

        """
        self.simulate_communication("H1", "H10", 2, 20)
        self.simulate_communication("H2", "H7", 0, 20)
        self.get_state("H1", 2)
        """

    def create_components(self, graph: nx.Graph):

        #Create components for each node in the graph
        for node in graph.nodes:
            host = f"H{node + 1}"
            if host not in self.components:
                self.components[host] = NetworkComponent(host, self.communication_sequences.get(host, []))
        #print("\n hosts: ", self.components)

        #Create links for each edge 
        for edge in graph.edges(data=True):
            dst = f"H{edge[1] + 1}"
            origin = f"H{edge[0] + 1}"
            if 'bw' in edge[2]:
                link_bw = edge[2]['bw']
                link = Link(origin, dst, link_bw)
            else:
                link = Link(origin, dst, 100)
            self.components[origin].add_link(link)
            self.components[dst].add_link(link)
            self.links[link.get_id()] = link
        #print("\n edges: ")
        #print("")
        #for key, values in self.links.items():
        #    print(f"{key}, {values.bw_total}")

    # def get_links(self):
    #     for key, values in self.links.items():
    #         print(f"{key}")
    #     key_array = np.array([list(key) for key in self.links.keys()], dtype=np.object)
    #     print("key array: ", key_array)
    #     return key_array

    def build_graph(self):
        for component in self.components.values():
            self.graph_topology.add_node(component.id)

            for neighbor in component.neighbors:
                self.graph_topology.add_edge(component.id, neighbor)

        pp = nx.draw(self.graph_topology, with_labels=True)
        plt.show()

        """
        with open(TOPOLOGY_FILE_NAME, 'r') as topo:
            for line in topo.readlines():
                nodes = line.split()[:2]
                for node in nodes:
                    if not self.graph.has_node(node):
                        self.graph.add_node(node)
                self.graph.add_edge(nodes[0], nodes[1])
        """

    def reset(self, new_tm=False):
        # self.graph_topology = nx.Graph()
        # if True:
        #  self.communication_sequences = generate_traffic_sequence(self)
        self.links = {}
        self.hosts = {}
        self.switchs = {}
        self.components = {}
        self.create_components(self.graph_topology)

        #if not EVALUATE:
        if new_tm:
            self.current_tm_index += 1
            self.communication_sequences = self.all_tms[self.current_tm_index % len(self.all_tms)] #EPOCH_SIZE
            
        # else:
        #     if new_tm:
        #         if TOPOLOGY_TYPE == "internet":
        #             self.communication_sequences = generate_traffic_sequence(self)
        #         if TOPOLOGY_TYPE == "service_provider":
        #             self.communication_sequences = generate_traffic_sequence_service_provider(self)


        #print("new tm: ", self.communication_sequences)
        
        # self.read_topology("topology_arpanet.txt")
        # self.build_graph()
        # self.calculate_paths()
        self.number_of_hosts = len(self.get_all_hosts())
        self.statistics = {'package_loss': 0, 'package_sent': 0, 'nr_package_loss': 0, 'nr_package_sent': 0}

        # if new_tm:
        #  self.communication_sequences = generate_traffic_sequence(self)

    def k_shortest_paths(self, graph, source, target, k):

        #Calculates the k shortest paths between source and target
        try:
            calc = list(islice(nx.shortest_simple_paths(graph, source, target), k))
        except nx.NetworkXNoPath:
            calc = []

        final_paths = []

        for p in calc:
            path = []
            for dst in p:
                path.append(f"H{dst + 1}")
            final_paths.append(path)
        return final_paths

    def is_direct_neighbour(self, origin, destination):
        paths = self.get_paths(origin, destination)

        for path in paths:
            if len(path) == 2:
                return True
        return False

    def calculate_paths(self):

        #Calculates and stores the k shortest paths between all hosts

        all_hosts = [component for component in self.components if "H" in component]
        #print("\n all hosts: ", all_hosts)

        for src in all_hosts:
            graph_src = int(src[1:]) - 1
            all_dsts = [h for h in all_hosts if h != src]
            for dst in all_dsts:
                #print("\n", dst)
                graph_dst = int(dst[1:]) - 1
                self.paths[(src, dst)] = self.k_shortest_paths(self.graph_topology, graph_src, graph_dst, NUMBER_OF_PATHS)
                self.components[src].set_active_path(dst, 0)
        #print("\n paths")#, self.paths)
        #for key, values in self.paths.items():
        #    print(f"{key}: {values}")
        #print("\n components: ", self.components)

    def get_random_dst(self, origin, all_dsts):

        while True:
            #dst = random.choice(all_dsts + ['', '', '', '', ''])
            dst = random.choice(all_dsts)
            if dst != origin:
                return dst

    def simmulate_turn(self):
        hosts = self.get_all_hosts()

        for host in hosts:
            h = self.components[host]

            if not h.is_busy():
                dst = h.get_dst()

                if dst is None or len(dst) < 2:
                    continue

                h.active_dst = dst
                path_id = h.get_active_path(dst)
                self.simulate_communication(host, dst, path_id, self.bws[host], 2)
                #print(f"\n SENDING FROM {host} to {dst}")
            else:
                h.update_communication()
                # means the communication is finished
                if not h.is_busy():
                    src = h.id
                    dst = h.current_dst
                    path_chosen = h.get_active_path(dst)

                    if path_chosen >= len(self.paths[(src, dst)]):
                        path_chosen = 0

                    path = self.paths[(src, dst)][path_chosen]
                    self.update_bw_path(path, -h.active_communication_bw)

    def get_nexts_dsts(self):
        return {host: self.components[host].get_next_dst() for host in self.get_all_hosts() if
                not self.components[host].is_busy()}
        """dict = {}
        for host in self.get_all_hosts():
            if not self.components[host].is_busy():
                destination = self.components[host].get_next_dst()
                if destination != None:
                    host : self.components[host].get_next_dst()
                    dict[host] = destination
        return dict"""

    def get_busy_hosts(self):
        return [host for host in self.get_all_hosts() if self.components[host].is_busy()]

    def simulate_communication(self, src, dst, path_chosen, bw, nr_turns):
        a = path_chosen
        b = len(self.paths[(src, dst)])
        
        if path_chosen >= len(self.paths[(src, dst)]) or path_chosen is None:
            path_chosen = 0

        path = self.paths[(src, dst)][path_chosen]
        self.update_bw_path(path, bw)
        self.components[src].set_communication(nr_turns, bw, dst)

    def update_bw_path(self, path, bw):
        origin = path[0]
        destiny = path[-1]
        initial_bw = bw
        update_bw = bw < 0    #if bw < 0 True

        # Update the bw for each link chosen in the path
        for index in range(len(path) - 1):
            src: NetworkComponent
            dst: NetworkComponent
            src = self.components.get(path[index])
            dst = self.components.get(path[index + 1])

            if src.id != origin:
                # Update or remove active communication
                if bw > 0:
                    src.add_active_communication(origin, destiny)
                else:
                    src.remove_active_communication(origin, destiny)
            else:
                if bw > 0:
                    src.active_dst = destiny
                else:
                    src.active_dst = -1

            link = src.get_link(dst.id)

            # Update the bw for the link
            if update_bw:
                #get communication bw
                bw = -1 * link.get_active_communication(origin, destiny)

            link.update_bw(bw)    #subtracts 'bw' from link

            if not update_bw:
                link.add_active_communication(origin, destiny, bw)

            if link.bw_available < 0 and not update_bw:  
                self.statistics["package_loss"] += -1 * link.bw_available
                self.statistics["nr_package_loss"] += 1
                bw += link.bw_available
                bw = max(1, bw)

        if not update_bw:
            self.statistics["package_sent"] += bw
            self.statistics["nr_package_sent"] += 1
            c = self.components[origin]
            if initial_bw == 0: ##
                c.bw_pct = 0
            else:
                c.bw_pct = (bw / initial_bw)
        else:
            c = self.components[origin]
            c.bw_pct = 0

    def read_topology(self, file):
        with open(file, 'r') as topology:
            for row in topology.readlines():
                src_id, dst_id, bw = row.split()
                if src_id not in self.components:
                    self.components[src_id] = NetworkComponent(src_id, self.communication_sequences.get(src_id, []))

                if dst_id not in self.components:
                    self.components[dst_id] = NetworkComponent(dst_id, self.communication_sequences.get(dst_id, []))
                link = Link(src_id, dst_id, bw)

                self.components[src_id].add_link(link)
                self.components[dst_id].add_link(link)
                self.links[link.get_id()] = link
                """
                if 'H' in src_id:
                    host = NetworkComponent(src_id)4
                    self.hosts[src_id] = host #dst_id.replace("S", "")
                    link = Link(src_id, dst_id, bw)
                    host.add_link(link)

                elif 'S' in src_id:
                    src_id = src_id.replace("S", "")
                    dst_id = dst_id.replace("S", "")
                    self.switchs[]
                    self.links[(src_id, dst_id)] = int(bw)
                    self.links[(dst_id, src_id)] = int(bw)
                """

    def get_all_hosts(self):
        return [c for c in self.components if "H" in c]

    def get_link(self, src, dst):
        if (src, dst) in self.links:
            return self.links.get((src, dst))
        elif (dst, src) in self.links:
            return self.links.get((dst, src))
        return None

    def get_paths(self, src, dst):
        key = (src, dst)
        return self.paths.get(key, [])

    def get_min_bw(self, path, n):
        min_bw = 200000000
        for index, component in enumerate(path[:n]):
            src = path[index]
            dst = path[index + 1]
            link = self.get_link(src, dst)
            available = link.get_bw_available_percentage()
            if available < min_bw:
                min_bw = available
        return min_bw

    def get_state(self, host, n=1):

        # 1. NBWs (free bw for each link)
        # 2. D - Next destination
        # 3. C - Active communications
        # 4. BW - BW to send

        hostC = self.components.get(host)

        hostC: NetworkComponent
        links = []
        #get neighbors
        for neighbor in hostC.neighbors:
            links.append(self.get_link(host, neighbor))

        
        state = np.empty((STATE_SIZE), dtype=object)
        state = np.full((STATE_SIZE), -1)

        # Slocal = (NBWs, D, C, BW_sent)
        # 1. NBWs - NR_MAX_LINKS: bw available for each neighbour node
        link: Link
        #get available bw for the neighbours
        for index, link in enumerate(links):
            state[index] = link.get_bw_available_percentage() / 100

        # 2. D - NR_MAX_LINKS + 1: next destination
        next_dest = hostC.get_next_dst()
        if not next_dest:
            next_dest = -1
        else:
            next_dest = int(next_dest[1:]) / 10

        state[NR_MAX_LINKS] = next_dest #save current communication dest

        # 3. C - NR_MAX_LINKS + 2: active communications
        active_communication = np.array(hostC.get_active_communications()).flatten() #get active communications
        
        for index, active in enumerate(active_communication):
            state[index + NR_MAX_LINKS + 1] = active / 10

        # 4.BW_sent - 1: bw to send
        state[-1] = self.bws.get(host, 0) / 100  #get bw to send

        #print(f"\n state({host}) : ", state)
        return state

    def set_active_path(self, host, dsts):
        h = self.components.get(host, None)

        if h is not None:
            for dst, path in dsts.items():
                h.set_active_path(dst, path)
        else:
            print(f"Host {host} not found.")

    def get_link_usage(self):
        bws = [link.get_bw_available_percentage() for link in self.links.values()]
        #for key, values in self.links.items():
        #   print(f"{key}, {values.bw_total}, {values.get_bw_available_percentage()}")

        # return numpry array
        #print("\n average: ", np.average(bws))
        return np.asarray(bws)
    
    def get_link_utilization(self):
        """Returns prbability of each link being used (100% - bw_available)"""
        link_utils = {}
    
        # Para evitar links bidirecionais
        processed_links = set()
    
        for (src, dst), link in self.links.items():
            # Verificar se já processamos este link (na direção contrária)
            if (dst, src) in processed_links:
                continue
                
            if link.bw_total == 0:
                # Link removido - marcar com valor especial (-1)
                # ou algum outro indicador que prefira
                link_utils[(src, dst)] = -1  # -1 indica link removido
            else:
                # Link normal - calcular utilização normalmente
                utilization = 100 - link.get_bw_available_percentage()
                link_utils[(src, dst)] = utilization
            
            processed_links.add((src, dst))
    
        return link_utils

    def communication_done(self):
        return all([component.is_done() for name, component in self.components.items() if "H" in name])
    
    def setup(self):
        self.links = {}
        self.hosts = {}
        self.switchs = {}
        self.components = {}
        self.paths = {}
        self.create_components(self.graph_topology)
        self.calculate_paths()
        self.hosts = self.get_all_hosts()
        self.number_of_hosts = len(self.hosts)
        self.statistics = {'package_loss': 0, 'package_sent': 0, 'nr_package_loss': 0, 'nr_package_sent': 0}
        self.single_con_hosts = [f"H{int(host) + 1}" for host in self.graph_topology if 
                                 len(self.graph_topology.edges(host)) == 1]   
        #self.bws = {host: bw if host not in self.single_con_hosts else bw // 3 for host, bw in self.bws.items()}

    
    def remove_edges(self, nr_edges):

        edges = []
        change = []
        
        # Verificar se já existe arquivo para a topologia atual
        scenario4_file = f"{PATH_SIMULATION}/scenario4_removed_edges.json"
        current_topology = TOPOLOGY_TYPE  # Topologia atual
        
        # Se o arquivo existir, verificar se é para a mesma topologia
        if os.path.exists(scenario4_file):
            try:
                with open(scenario4_file, 'r') as f:
                    edges_data = json.load(f)
                    
                    # Se o arquivo tem a informação da topologia e é a mesma atual
                    if "topology" in edges_data and edges_data["topology"] == current_topology:
                        print(f"Reutilizando links removidos da topologia {current_topology}")
                        
                        # Recuperar os links já removidos anteriormente
                        if "links_removed" in edges_data:
                            change = [tuple(e) for e in edges_data["links_removed"]]
                            
                            for edge in change:
                                u, v = edge
                                # Converter índices em nomes de hosts
                                host_u = f"H{u + 1}" if u < self.number_of_hosts else f"S{u - self.number_of_hosts + 1}"
                                host_v = f"H{v + 1}" if v < self.number_of_hosts else f"S{v - self.number_of_hosts + 1}"
                                
                                print(f"Link já removido: {host_u}-{host_v} (índice: {u}-{v})")
                                
                                # Remover o link
                                self.graph_topology.remove_edge(*edge)
                                self.graph_topology.add_edge(u, v, bw = 0)
                            
                            # Configurar a rede com as alterações
                            self.setup()
                            return
            except (json.JSONDecodeError, FileNotFoundError):
                print("Arquivo de links removidos inválido ou não encontrado.")
        
        # Se não encontramos links para reutilizar, selecionar novos aleatoriamente
        # Manter formato original de seleção de edges
        for edge in self.graph_topology.edges():
            n1, n2 = edge
            if self.graph_topology.degree(n1) > 1 and self.graph_topology.degree(n2) > 1:
                edges.append(edge)
        
        change = random.sample(edges, min(nr_edges, len(edges)))
        
        for edge in change:
            u, v = edge
            host_u = f"H{u + 1}" if u < self.number_of_hosts else f"S{u - self.number_of_hosts + 1}"
            host_v = f"H{v + 1}" if v < self.number_of_hosts else f"S{v - self.number_of_hosts + 1}"

            print("changing edge: ", edge)
            print(f"Corresponding links: {host_u} <-> {host_v}")

            self.graph_topology.remove_edge(*edge)
            self.graph_topology.add_edge(u, v, bw = 0)
        
        # Salvar os links removidos com informação de topologia
        scenario4_removed_edges = {
            "topology": current_topology,
            "links_removed": [list(edge) for edge in change]
        }
        
        with open(scenario4_file, 'w') as f:
            json.dump(scenario4_removed_edges, f, indent=4)
        
        self.setup()

    
    def add_edges(self, nr_edges):

        # Edge replacement:
        # 1. Finds existing edges where both nodes have degree > 1
        # 2. Finds potential new edges between nodes under max connections limit
        # 3. Removes nr_edges existing edges and adds nr_edges new ones
        # 4. New edges get bandwidth = 100
        # 5. Calls setup() to reinitialize network structure

        edges = []
        change = []
        edges_add = []
        max_connections = NR_MAX_LINKS - nr_edges

        for edge in self.graph_topology.edges():
            n1, n2 = edge
            if self.graph_topology.degree(n1) > 1 and self.graph_topology.degree(n2) > 1:
                edges.append(edge)

        for n1 in self.graph_topology.nodes():
            for n2 in self.graph_topology.nodes():
                if n1 != n2 and not self.graph_topology.has_edge(n1,n2):
                    if self.graph_topology.degree(n1) < max_connections and self.graph_topology.degree(n2) < max_connections:
                        edges_add.append((n1,n2))
        
        change = random.sample(edges, min(nr_edges, len(edges)))
        add = random.sample(edges_add, min(nr_edges, len(edges_add)))

        for edge1, edge2 in zip(change, add):
            self.graph_topology.remove_edge(*edge1)
            u, v = edge2
            print("removing edge: ", edge1, "adding edge: ", edge2, "with node's degree: ", self.graph_topology.degree(u), ", ", self.graph_topology.degree(v))
            self.graph_topology.add_edge(u, v, bw = 100)
        
        self.setup()


    
    def remove_topology_edges(self, mod):  

        # Advanced edge removal with traffic matrix update:
        # 1. Takes modification level (1-3) to determine number of edges
        # 2. Removes edges like remove_edges()
        # 3. Updates traffic matrix based on topology type
        # 4. Reloads traffic sequences from files
        # 5. Resets traffic matrix index  

        global REMOVED_EDGES, SCENARIO_2_COMPLETED

        # Verificar se existe arquivo centralizado para a topologia
        scenario2_file = f"{PATH_SIMULATION}/scenario2_removed_edges_{TOPOLOGY_TYPE}.json"
        current_topology = TOPOLOGY_TYPE

        
        if mod == 1:
            nr_links_changed = 1
        elif mod == 2:
            nr_links_changed = 1 #2
        elif mod == 3:
            nr_links_changed = 1 #3

        edges = []
        change = []

        if mod == 1:
            # Verificar e carregar links compartilhados entre configurações
            if os.path.exists(scenario2_file):
                    with open(scenario2_file, 'r') as f:
                        edges_data = json.load(f)
                        
                        # Verificar se arquivo tem informações válidas
                        if "topology" in edges_data and edges_data["topology"] == current_topology:
                            print(f"Reutilizando links removidos da topologia {current_topology}")
                            
                            # Preencher REMOVED_EDGES com links do arquivo
                            REMOVED_EDGES = {
                                1: [tuple(e) for e in edges_data.get("1", [])],
                                2: [tuple(e) for e in edges_data.get("2", [])],
                                3: [tuple(e) for e in edges_data.get("3", [])]
                            }
                            SCENARIO_2_COMPLETED = True

        # Preencher a lista de edges candidatos
        for n1, n2, data in self.graph_topology.edges(data=True):
            if self.graph_topology.degree(n1) > 1 and self.graph_topology.degree(n2) > 1:
                if 'bw' in data and data['bw'] == 0:
                    continue
                edges.append((n1,n2))

        # Cenário 2: selecionar e salvar aleatoriamente
        
        if EVALUATE and not TRAIN and not UPDATE_WEIGHTS:
            # Se já temos links armazenados, use-os
            if SCENARIO_2_COMPLETED and mod in REMOVED_EDGES and REMOVED_EDGES[mod]:
                change = REMOVED_EDGES[mod]
            else:
                # Seleciona novos links
                change = random.sample(edges, min(nr_links_changed, len(edges)))
                REMOVED_EDGES[mod] = change
                SCENARIO_2_COMPLETED = True
                
                if mod == 3:
                    edges_data = {
                        "topology": current_topology,
                        "1": [[e[0], e[1]] for e in REMOVED_EDGES[1]],
                        "2": [[e[0], e[1]] for e in REMOVED_EDGES[2]],
                        "3": [[e[0], e[1]] for e in REMOVED_EDGES[3]]
                    }
                    with open(scenario2_file, 'w') as f:
                        json.dump(edges_data, f, indent=4)

        # Cenário 3: usar os mesmos links do cenário 2
        elif EVALUATE and not TRAIN and UPDATE_WEIGHTS:
            if SCENARIO_2_COMPLETED:
                if REMOVED_EDGES[mod] is not None:
                    change = REMOVED_EDGES[mod]
                    REMOVED_EDGES[mod] = change
                    save_removed_edges(REMOVED_EDGES, True)

            else:
                print("[ERROR] Scenario 2 not completed. Cannot proceed with Scenario 3.")
                sys.exit(1)

        # Outros cenários: seleção aleatória
        else:
            change = random.sample(edges, min(nr_links_changed, len(edges)))
    
        # Remover os links selecionados
        for edge in change:
            u, v = edge
            # Converter índices em nomes de hosts
            host_u = f"H{u + 1}" if u < self.number_of_hosts else f"S{u - self.number_of_hosts + 1}"
            host_v = f"H{v + 1}" if v < self.number_of_hosts else f"S{v - self.number_of_hosts + 1}"
            
            print("changing edge: ", edge)
            print(f"Corresponding links: {host_u} <-> {host_v}")
            
            self.graph_topology.remove_edge(*edge)
            self.graph_topology.add_edge(u, v, bw = 0)
        
        #print("\n Modified", self.graph_topology)
        #print("\n edges: ", self.graph_topology.edges(data=True))
        self.setup()
        if TOPOLOGY_TYPE == "internet":
            self.all_tms = json.load(open(f"{PATH_SIMULATION}/TrafficMatrix/tms_internet_test.json", mode="r"))
        if TOPOLOGY_TYPE == "arpanet":
            self.all_tms = json.load(open(f"{PATH_SIMULATION}/TrafficMatrix/tms_arpanet_test.json", mode="r"))
        if TOPOLOGY_TYPE == "service_provider":
            self.all_tms = json.load(open(f"{PATH_SIMULATION}/TrafficMatrix/tms_service_provider_test.json", mode="r"))
        self.current_index = 0
        self.current_tm_index = self.current_index % len(self.all_tms)       
        self.communication_sequences = self.all_tms[self.current_tm_index]

        # Salvar links removidos ao final do cenário 2
        if EVALUATE and not TRAIN and not UPDATE_WEIGHTS and mod == 3:
            save_removed_edges(REMOVED_EDGES, True)

    def add_topology_edges(self, mod): 

        # Advanced edge addition with traffic matrix update:
        # 1. Takes modification level (1-3) to determine number of edges
        # 2. Replaces edges like add_edges()
        # 3. Updates traffic matrix based on topology type
        # 4. Reloads traffic sequences from files
        # 5. Resets traffic matrix index

        if mod == 1:
            nr_links_changed = 1
        elif mod == 2:
            nr_links_changed = 2 #3
        elif mod == 3:
            nr_links_changed = 3 #6 

        edges = []
        change = []
        edges_add = []

        max_connections = NR_MAX_LINKS - nr_links_changed

        for edge in self.graph_topology.edges():
            n1, n2 = edge
            if self.graph_topology.degree(n1) > 1 and self.graph_topology.degree(n2) > 1:
                edges.append(edge)

        for n1 in self.graph_topology.nodes():
            for n2 in self.graph_topology.nodes():
                if n1 != n2 and not self.graph_topology.has_edge(n1,n2):
                    if self.graph_topology.degree(n1) < max_connections and self.graph_topology.degree(n2) < max_connections:
                        edges_add.append((n1,n2))
        
        change = random.sample(edges, min(nr_links_changed, len(edges)))
        add = random.sample(edges_add, min(nr_links_changed, len(edges_add)))

        for edge1, edge2 in zip(change, add):
            self.graph_topology.remove_edge(*edge1)
            u, v = edge2
            print("removing edge: ", edge1, "adding edge: ", edge2, "with node's degree: ", self.graph_topology.degree(u), ", ", self.graph_topology.degree(v))
            self.graph_topology.add_edge(u, v, bw = 100)
        
        #print("\n Modified", self.graph_topology)
        #print("\n edges: ", self.graph_topology.edges(data=True))
        self.setup()
        if TOPOLOGY_TYPE == "internet":
            self.all_tms = json.load(open("TrafficMatrix/tms_internet_test.json", mode="r"))
        if TOPOLOGY_TYPE == "arpanet":
            self.all_tms = json.load(open("TrafficMatrix/tms_arpanet_test.json", mode="r"))
        if TOPOLOGY_TYPE == "service_provider":
            self.all_tms = json.load(open("TrafficMatrix/tms_service_provider_test.json", mode="r"))
        self.current_index = 0
        self.current_tm_index = self.current_index % len(self.all_tms)       
        self.communication_sequences = self.all_tms[self.current_tm_index]
        

    def set_arpanet_topology(self):
        self.links = {}
        self.hosts = {}
        self.switchs = {}
        self.components = {}
        self.paths = {}
        self.bws = {}

        self.graph_topology = pickle.load(open(f"{PATH_SIMULATION}/TopologyFiles/topology_arpanet.pickle", "rb"))

        self.communication_sequences = {"H21": [ "H5", "H5", "H33", "H25", "H27", "H27", "H3", "H24", "H11", "H16", "H2", "H10", "H15", "H29", "H3", "H33", "H19", "H14", "H25", "H32", "H11", "H2", "H33", "H24", "H13", "H31", "H3", "H25", "H6", "H14" ], "H1": [ "H15", "H29", "H32", "H5", "H8", "H29", "H32", "H2", "H18", "H20", "H4", "H33", "H18", "H29", "H8", "H15", "H14", "H16", "H31", "H28", "H9", "H22", "H11", "H27", "H32", "H9", "H25", "H24", "H18", "H4" ], "H22": [ "H13", "H32", "H12", "H25", "H12", "H7", "H3", "H25", "H24", "H7", "H19", "H28", "H2", "H8", "H5", "H25", "H10", "H27", "H3", "H32", "H8", "H21", "H25", "H28", "H20", "H17", "H12", "H27", "H4", "H32" ], "H2": [ "H30", "H20", "H14", "H14", "H14", "H5", "H17", "H23", "H21", "H20", "H3", "H22", "H18", "H23", "H28", "H28", "H4", "H26", "H19", "H9", "H27", "H11", "H4", "H28", "H15", "H30", "H14", "H26", "H16", "H7" ], "H23": [ "H12", "H11", "H14", "H26", "H32", "H6", "H27", "H1", "H13", "H16", "H17", "H17", "H30", "H26", "H7", "H9", "H22", "H15", "H28", "H24", "H21", "H29", "H33", "H16", "H19", "H33", "H1", "H8", "H5", "H9" ], "H3": [ "H20", "H19", "H20", "H30", "H22", "H13", "H18", "H24", "H22", "H14", "H23", "H31", "H25", "H27", "H27", "H9", "H12", "H16", "H24", "H11", "H20", "H12", "H31", "H32", "H14", "H18", "H27", "H24", "H21", "H32" ], "H24": [ "H17", "H2", "H18", "H19", "H20", "H13", "H4", "H12", "H33", "H19", "H29", "H21", "H7", "H9", "H29", "H33", "H14", "H32", "H12", "H22", "H20", "H8", "H30", "H29", "H14", "H17", "H17", "H27", "H19", "H19" ], "H4": [ "H16", "H5", "H13", "H9", "H26", "H3", "H32", "H29", "H8", "H18", "H6", "H9", "H30", "H24", "H9", "H7", "H13", "H7", "H29", "H11", "H29", "H8", "H18", "H29", "H1", "H18", "H17", "H19", "H3", "H18" ], "H25": [ "H18", "H15", "H26", "H30", "H18", "H10", "H29", "H1", "H30", "H8", "H15", "H20", "H14", "H5", "H17", "H27", "H14", "H11", "H27", "H26", "H15", "H31", "H3", "H21", "H33", "H22", "H29", "H2", "H22", "H3" ], "H8": [ "H18", "H2", "H13", "H21", "H15", "H14", "H23", "H15", "H5", "H29", "H24", "H5", "H4", "H1", "H25", "H4", "H4", "H9", "H10", "H1", "H4", "H4", "H29", "H23", "H1", "H32", "H4", "H25", "H30", "H14" ], "H26": [ "H14", "H20", "H7", "H33", "H27", "H10", "H32", "H17", "H21", "H6", "H18", "H22", "H13", "H6", "H20", "H20", "H23", "H18", "H33", "H9", "H4", "H14", "H13", "H2", "H29", "H10", "H29", "H30", "H19", "H25" ], "H11": [ "H32", "H2", "H3", "H7", "H25", "H25", "H21", "H32", "H26", "H21", "H3", "H10", "H17", "H29", "H19", "H33", "H30", "H27", "H31", "H26", "H14", "H7", "H9", "H10", "H7", "H17", "H9", "H32", "H16", "H23" ], "H27": [ "H5", "H19", "H5", "H4", "H28", "H26", "H16", "H10", "H5", "H26", "H24", "H7", "H24", "H20", "H13", "H33", "H18", "H23", "H20", "H14", "H19", "H9", "H15", "H3", "H32", "H24", "H31", "H30", "H3", "H1" ], "H12": [ "H22", "H7", "H10", "H27", "H17", "H29", "H14", "H17", "H11", "H6", "H19", "H14", "H23", "H31", "H27", "H1", "H3", "H23", "H3", "H5", "H1", "H30", "H7", "H9", "H28", "H21", "H23", "H2", "H9", "H19" ], "H28": [ "H1", "H2", "H27", "H15", "H16", "H22", "H6", "H3", "H19", "H32", "H2", "H29", "H29", "H16", "H23", "H23", "H5", "H20", "H24", "H20", "H26", "H2", "H5", "H25", "H12", "H8", "H5", "H5", "H33", "H22" ], "H13": [ "H3", "H30", "H31", "H8", "H25", "H20", "H9", "H3", "H30", "H15", "H24", "H8", "H22", "H16", "H28", "H28", "H7", "H7", "H28", "H23", "H3", "H8", "H26", "H24", "H18", "H25", "H11", "H17", "H23", "H3" ], "H29": [ "H7", "H19", "H2", "H24", "H10", "H12", "H32", "H16", "H30", "H9", "H19", "H23", "H18", "H2", "H9", "H10", "H8", "H21", "H21", "H21", "H25", "H27", "H26", "H16", "H33", "H16", "H30", "H13", "H25", "H23" ], "H16": [ "H19", "H11", "H20", "H22", "H32", "H4", "H6", "H21", "H22", "H11", "H21", "H18", "H8", "H33", "H32", "H28", "H2", "H1", "H23", "H25", "H12", "H19", "H25", "H25", "H1", "H33", "H5", "H25", "H26", "H14" ], "H30": [ "H1", "H22", "H22", "H15", "H8", "H26", "H24", "H4", "H31", "H15", "H16", "H12", "H1", "H32", "H23", "H32", "H11", "H16", "H28", "H24", "H26", "H12", "H20", "H4", "H33", "H7", "H25", "H24", "H17", "H10" ], "H17": [ "H3", "H24", "H26", "H29", "H7", "H30", "H10", "H23", "H9", "H20", "H2", "H13", "H6", "H13", "H6", "H27", "H24", "H19", "H30", "H32", "H24", "H1", "H24", "H10", "H33", "H16", "H6", "H12", "H13", "H24" ], "H31": [ "H18", "H18", "H20", "H18", "H24", "H17", "H11", "H8", "H10", "H24", "H29", "H21", "H13", "H24", "H4", "H22", "H10", "H3", "H1", "H27", "H27", "H26", "H14", "H26", "H3", "H26", "H27", "H19", "H12", "H1" ], "H18": [ "H30", "H31", "H5", "H13", "H33", "H3", "H24", "H3", "H24", "H21", "H28", "H28", "H9", "H9", "H1", "H24", "H7", "H5", "H28", "H15", "H12", "H25", "H29", "H3", "H28", "H1", "H6", "H25", "H14", "H27" ], "H32": [ "H25", "H1", "H22", "H5", "H19", "H7", "H30", "H16", "H7", "H24", "H20", "H16", "H6", "H17", "H1", "H19", "H18", "H19", "H19", "H8", "H24", "H24", "H9", "H2", "H18", "H7", "H28", "H21", "H31", "H30" ], "H19": [ "H24", "H23", "H18", "H18", "H14", "H2", "H26", "H18", "H8", "H24", "H9", "H33", "H23", "H24", "H17", "H18", "H32", "H3", "H17", "H16", "H2", "H31", "H13", "H1", "H24", "H22", "H16", "H21", "H24", "H10" ], "H33": [ "H20", "H24", "H5", "H11", "H31", "H24", "H3", "H31", "H2", "H22", "H13", "H5", "H5", "H6", "H6", "H27", "H28", "H17", "H15", "H10", "H12", "H15", "H5", "H22", "H17", "H3", "H13", "H11", "H5", "H5" ], "H20": [ "H23", "H5", "H17", "H29", "H30", "H12", "H6", "H29", "H16", "H9", "H2", "H32", "H26", "H15", "H24", "H15", "H1", "H22", "H15", "H32", "H1", "H4", "H15", "H22", "H17", "H7", "H18", "H15", "H7", "H16" ], "H6": [ "H7", "H8", "H7", "H8", "H13", "H2", "H25", "H16", "H11", "H32", "H15", "H23", "H25", "H22", "H18", "H16", "H30", "H26", "H12", "H17", "H26", "H18", "H8", "H3", "H18", "H22", "H17", "H30", "H9", "H21" ], "H5": [ "H19", "H26", "H30", "H13", "H19", "H27", "H1", "H19", "H3", "H28", "H31", "H32", "H32", "H7", "H23", "H29", "H33", "H3", "H33", "H19", "H13", "H21", "H26", "H3", "H10", "H30", "H19", "H7", "H8", "H27" ], "H7": [ "H6", "H28", "H4", "H20", "H2", "H32", "H24", "H15", "H30", "H11", "H31", "H1", "H15", "H6", "H23", "H15", "H1", "H30", "H3", "H9", "H12", "H21", "H9", "H26", "H20", "H15", "H2", "H5", "H29", "H11" ], "H10": [ "H20", "H25", "H14", "H28", "H28", "H21", "H22", "H7", "H8", "H15", "H2", "H22", "H5", "H27", "H12", "H18", "H8", "H26", "H31", "H27", "H30", "H28", "H1", "H26", "H2", "H16", "H8", "H6", "H24", "H12" ], "H9": [ "H24", "H28", "H23", "H12", "H4", "H20", "H18", "H5", "H30", "H11", "H29", "H13", "H7", "H7", "H13", "H7", "H8", "H2", "H23", "H20", "H13", "H19", "H13", "H7", "H28", "H20", "H29", "H31", "H7", "H26" ], "H14": [ "H8", "H32", "H1", "H9", "H18", "H18", "H20", "H5", "H32", "H12", "H3", "H20", "H5", "H20", "H26", "H29", "H30", "H3", "H10", "H17", "H5", "H13", "H24", "H30", "H20", "H30", "H10", "H18", "H10", "H24" ], "H15": [ "H20", "H7", "H3", "H31", "H7", "H32", "H33", "H33", "H5", "H23", "H12", "H10", "H30", "H25", "H19", "H32", "H8", "H26", "H30", "H33", "H10", "H16", "H16", "H11", "H9", "H19", "H12", "H20", "H31", "H13" ] }
        #self.graph_has_data = True
        
        #nx.draw(self.graph_topology, with_labels=True)
        #plt.show()
        self.create_components(self.graph_topology) #creates nodes and edges
        #self.communication_sequences = generate_traffic_sequence_arpanet(self)
        self.hosts = self.get_all_hosts()
        #for host in self.hosts:
        #    self.bws[host] = random.randint(10, 35) #change values to experiment, copy and put them in self.bws
        #print(self.bws)
        self.bws = {'H21': 38, 'H1': 39, 'H22': 24, 'H2': 11, 'H23': 39, 'H3': 28, 'H24': 42, 'H4': 34, 'H25': 24, 'H8': 13, 'H26': 41, 'H11': 31, 'H27': 24, 'H12': 12, 'H28': 17, 'H13': 15, 'H29': 42, 'H16': 17, 'H30': 38, 'H17': 45, 'H31': 11, 'H18': 43, 'H32': 28, 'H19': 35, 'H33': 32, 'H20': 28, 'H6': 29, 'H5': 34, 'H7': 22, 'H10': 13, 'H9': 35, 'H14': 39, 'H15': 45}
        self.calculate_paths()
        self.number_of_hosts = len(self.hosts)
        self.statistics = {'package_loss': 0, 'package_sent': 0, 'nr_package_loss': 0, 'nr_package_sent': 0}
        self.single_con_hosts = [f"H{int(host) + 1}" for host in self.graph_topology if len(self.graph_topology.edges(host)) == 1]  
        self.bws = {host: bw if host not in self.single_con_hosts else bw // 3 for host, bw in self.bws.items()}
        
        self.all_tms = json.load(open(f"{PATH_SIMULATION}/TopologyFiles/tms_arpanet_train.json", mode="r"))
        self.current_index = 0
        self.current_tm_index = self.current_index % len(self.all_tms)       
        self.communication_sequences = self.all_tms[self.current_tm_index]


    def set_service_provider_topology(self):
        self.links = {}
        self.hosts = {}
        self.switchs = {}
        self.components = {}
        self.paths = {}
        self.bws = {}
        self.communication_sequences = {'H8': ['H31', 'H40', 'H4', 'H1', 'H46', 'H29', 'H3', 'H64', 'H30', 'H56', 'H28', 'H55', 'H14', 'H55', 'H4', 'H11', 'H35', 'H38', 'H22', 'H58', 'H5', '', 'H32', 'H64', 'H14', 'H50', 'H28', 'H59', '', ''], 
                                        'H13': ['H29', 'H65', 'H53', 'H42', 'H42', 'H42', '', 'H21', 'H21', 'H51', 'H32', 'H44', 'H9', 'H28', 'H65', 'H5', 'H44', 'H55', 'H20', 'H64', 'H18', 'H29', 'H33', 'H5', 'H35', 'H23', 'H59', 'H27', 'H37', 'H57'], 
                                        'H22': ['H50', 'H38', 'H64', 'H54', 'H54', 'H29', 'H63', '', 'H46', 'H47', 'H34', 'H47', 'H3', 'H8', 'H5', 'H25', 'H12', '', 'H18', 'H56', 'H6', 'H26', 'H49', 'H12', 'H13', 'H38', 'H9', 'H36', 'H58', 'H60'], 
                                        'H31': ['', 'H32', 'H53', 'H46', 'H63', 'H11', 'H3', 'H18', 'H14', 'H25', 'H63', 'H18', 'H65', 'H40', 'H53', 'H9', 'H44', 'H22', 'H61', 'H30', 'H36', '', 'H3', 'H46', 'H16', 'H19', 'H37', 'H49', 'H9', 'H11'], 
                                        'H39': ['H6', 'H53', 'H42', 'H61', 'H34', 'H3', 'H48', 'H40', 'H28', 'H44', 'H10', 'H14', 'H17', 'H40', 'H48', '', 'H30', 'H62', 'H29', 'H31', 'H46', 'H10', 'H44', 'H31', 'H20', 'H49', 'H36', 'H34', 'H17', 'H18'], 
                                        'H48': ['H44', 'H50', 'H21', 'H12', 'H30', 'H59', 'H6', 'H23', 'H43', 'H65', 'H24', 'H24', 'H59', 'H56', 'H42', 'H43', 'H24', 'H44', 'H39', 'H32', 'H38', 'H40', 'H65', 'H63', 'H23', 'H38', 'H36', 'H1', 'H61', 'H41'], 
                                        'H57': ['H16', 'H61', 'H13', 'H1', 'H27', 'H4', 'H2', 'H44', 'H22', 'H10', 'H61', 'H23', 'H34', 'H32', 'H8', '', 'H13', 'H11', 'H28', 'H39', 'H44', 'H54', 'H23', 'H39', 'H52', 'H38', 'H37', 'H47', 'H50', 'H34'], 
                                        'H12': ['H64', 'H15', 'H2', 'H37', 'H18', 'H37', 'H52', 'H41', 'H65', 'H45', 'H39', 'H42', 'H36', 'H52', 'H41', 'H28', 'H11', 'H64', 'H28', 'H35', 'H40', 'H30', '', 'H6', 'H30', 'H29', 'H42', 'H34', 'H56', 'H37'], 
                                        'H21': ['H51', 'H15', 'H54', 'H7', 'H15', 'H1', 'H50', 'H62', 'H36', 'H2', 'H54', 'H1', 'H41', 'H33', '', 'H58', 'H24', 'H18', 'H5', 'H6', 'H55', 'H56', 'H62', 'H49', 'H2', 'H37', 'H44', 'H2', 'H4', 'H48'], 
                                        'H30': ['H47', 'H26', 'H59', 'H19', 'H36', 'H54', 'H41', 'H17', 'H56', 'H45', 'H16', 'H63', '', 'H11', 'H11', 'H41', 'H49', 'H11', 'H21', 'H63', 'H29', 'H47', 'H44', 'H57', 'H33', 'H42', 'H17', 'H44', 'H51', 'H50'], 
                                        'H38': ['H32', 'H25', 'H33', 'H24', 'H19', 'H58', 'H9', 'H22', 'H65', 'H25', 'H5', 'H5', 'H45', 'H14', 'H56', 'H16', 'H51', 'H29', '', 'H5', 'H53', 'H31', 'H51', 'H35', 'H23', 'H25', 'H2', 'H26', '', 'H61'], 
                                        'H47': ['', 'H29', 'H18', 'H12', '', 'H23', 'H61', 'H3', '', 'H59', 'H7', 'H60', 'H27', 'H9', 'H57', 'H27', 'H53', 'H3', 'H43', 'H1', 'H64', 'H45', 'H57', 'H27', 'H40', 'H10', 'H26', 'H25', 'H30', ''], 
                                        'H56': ['H29', 'H12', '', 'H16', '', 'H23', 'H41', 'H62', 'H2', 'H16', 'H40', 'H14', 'H48', 'H31', 'H49', 'H46', 'H14', '', 'H20', 'H30', 'H15', 'H33', '', 'H53', 'H11', 'H3', 'H34', 'H10', 'H36', 'H41'], 
                                        'H65': ['H4', 'H28', 'H46', 'H50', 'H5', 'H44', 'H46', 'H38', 'H40', 'H29', 'H4', 'H35', 'H12', 'H11', 'H26', 'H52', 'H38', 'H41', 'H34', 'H2', 'H1', 'H18', 'H43', 'H39', 'H8', 'H33', 'H2', 'H10', 'H45', 'H32'], 
                                        'H1': ['H15', 'H24', 'H42', 'H61', 'H45', 'H20', 'H50', 'H5', 'H10', 'H55', 'H51', 'H12', 'H12', 'H38', 'H3', 'H55', 'H58', 'H53', 'H45', 'H7', 'H47', 'H36', 'H30', 'H39', 'H41', 'H40', 'H43', 'H47', 'H55', 'H2'], 
                                        'H2': ['H56', 'H12', 'H63', 'H14', 'H14', 'H52', 'H36', 'H11', 'H14', 'H18', 'H40', 'H65', 'H36', 'H47', 'H19', '', 'H28', 'H4', 'H24', 'H48', 'H9', '', 'H30', 'H34', 'H16', '', 'H22', 'H40', 'H63', 'H7'], 
                                        'H3': ['H16', '', 'H40', 'H65', 'H32', 'H62', 'H64', 'H40', 'H20', 'H48', 'H26', 'H57', 'H4', 'H15', 'H28', 'H8', 'H31', 'H36', 'H16', 'H36', 'H41', 'H34', '', 'H56', 'H4', 'H35', 'H19', 'H1', 'H52', 'H38'], 
                                        'H4': ['H56', 'H19', 'H64', 'H15', 'H10', 'H54', 'H20', 'H36', 'H44', 'H46', 'H47', 'H39', 'H6', 'H55', 'H52', '', 'H15', 'H61', 'H34', 'H64', 'H8', 'H43', 'H27', 'H1', 'H65', 'H2', '', 'H50', '', 'H30'], 
                                        'H5': ['H15', 'H16', 'H34', 'H50', 'H38', 'H55', 'H61', 'H48', 'H8', 'H11', 'H59', 'H40', 'H56', 'H63', 'H41', 'H24', 'H40', 'H12', 'H4', 'H8', 'H42', '', '', 'H22', 'H6', 'H9', 'H11', 'H1', '', 'H33'], 
                                        'H6': ['H17', 'H57', 'H12', 'H48', 'H62', 'H29', 'H49', 'H33', 'H53', 'H26', 'H16', 'H16', 'H37', 'H20', 'H52', 'H45', 'H7', 'H34', 'H45', 'H41', 'H18', 'H46', 'H30', 'H64', '', 'H5', 'H62', 'H1', 'H63', 'H10'],
                                        'H7': ['H33', 'H13', 'H51', '', 'H14', 'H29', 'H62', 'H59', 'H38', 'H16', 'H38', 'H8', 'H35', '', 'H42', 'H11', 'H8', 'H24', 'H39', 'H50', 'H38', 'H23', 'H27', 'H54', '', 'H55', '', 'H50', 'H40', 'H34'], 
                                        'H9': ['H41', 'H45', 'H2', 'H59', 'H29', 'H14', 'H10', '', 'H41', 'H10', 'H13', 'H20', 'H33', '', 'H55', 'H6', 'H28', 'H31', 'H55', 'H49', 'H48', 'H20', 'H58', 'H13', 'H17', 'H64', 'H8', 'H13', 'H19', 'H14'], 
                                        'H10': ['H12', '', 'H57', 'H3', 'H24', 'H51', 'H46', 'H18', 'H52', 'H39', 'H3', 'H47', 'H38', 'H22', 'H61', 'H24', 'H24', 'H43', 'H1', 'H51', 'H62', 'H58', 'H37', 'H3', '', 'H49', 'H16', 'H41', '', 'H53'], 
                                        'H14': ['H36', 'H62', 'H6', 'H18', 'H52', 'H42', 'H59', 'H56', 'H13', 'H37', 'H4', 'H9', 'H37', 'H65', 'H7', 'H37', 'H61', 'H51', 'H36', '', 'H63', 'H2', 'H12', 'H40', 'H54', 'H52', 'H55', 'H32', 'H34', 'H45'], 
                                        'H11': ['H25', 'H29', '', 'H40', 'H13', '', 'H60', 'H8', 'H30', 'H44', 'H40', 'H49', '', 'H9', 'H46', 'H31', '', 'H49', 'H37', 'H55', 'H38', 'H1', 'H49', '', 'H60', 'H31', 'H31', 'H42', 'H10', 'H45'], 
                                        'H20': ['H4', 'H43', 'H8', 'H25', 'H1', 'H33', 'H13', 'H38', 'H63', 'H63', 'H33', 'H24', '', 'H28', 'H36', 'H11', 'H7', 'H35', 'H54', 'H25', 'H26', 'H51', 'H51', 'H32', 'H13', 'H60', 'H34', 'H63', 'H13', 'H26'], 
                                        'H15': ['H46', 'H42', '', 'H16', 'H14', 'H40', 'H46', 'H65', 'H51', 'H31', 'H35', 'H49', 'H24', 'H64', 'H3', 'H26', 'H54', '', 'H63', 'H27', 'H9', 'H58', 'H49', 'H18', 'H9', 'H37', 'H58', 'H41', 'H41', 'H1'],
                                        'H16': ['H10', 'H50', 'H56', 'H1', 'H4', 'H40', 'H18', 'H49', 'H32', 'H23', 'H19', 'H30', 'H9', 'H45', 'H35', 'H4', 'H17', 'H43', 'H44', 'H38', '', '', 'H53', 'H19', 'H58', 'H27', 'H61', '', 'H37', 'H15'], 
                                        'H17': ['H54', 'H35', 'H36', 'H41', 'H61', 'H12', 'H21', 'H46', 'H11', 'H12', 'H52', 'H20', 'H36', '', 'H8', 'H21', 'H64', 'H22', 'H41', 'H54', 'H37', 'H41', 'H37', 'H25', 'H22', 'H48', 'H12', 'H40', 'H40', 'H55'], 
                                        'H18': ['H43', 'H24', 'H34', 'H51', 'H3', 'H60', 'H44', 'H48', 'H20', 'H28', 'H6', 'H61', 'H43', 'H23', 'H46', 'H45', 'H37', 'H30', 'H55', 'H32', 'H41', 'H38', 'H13', 'H20', 'H47', 'H48', 'H13', 'H23', 'H61', 'H42'], 
                                        'H19': ['H28', 'H33', 'H46', 'H64', '', 'H64', 'H9', 'H4', 'H38', 'H27', 'H6', 'H18', '', 'H12', 'H3', 'H48', 'H35', 'H60', 'H53', 'H21', 'H38', 'H2', 'H57', 'H55', 'H65', 'H60', 'H22', 'H8', 'H30', 'H6'], 
                                        'H23': ['H10', 'H45', 'H20', 'H30', 'H63', 'H9', '', 'H12', 'H26', 'H8', 'H6', 'H57', 'H55', 'H39', 'H12', 'H34', 'H36', 'H41', 'H33', '', 'H51', 'H30', 'H48', '', 'H2', 'H39', 'H53', 'H42', 'H43', 'H33'], 
                                        'H24': ['H36', 'H34', 'H48', 'H20', 'H16', 'H37', 'H35', 'H5', 'H40', 'H4', 'H17', 'H63', 'H28', 'H2', 'H26', 'H44', 'H59', 'H22', '', 'H59', 'H30', 'H32', 'H26', 'H32', 'H43', 'H7', '', 'H5', 'H61', 'H41'], 
                                        'H25': ['H54', 'H33', 'H59', 'H40', 'H9', 'H29', 'H5', 'H49', 'H65', 'H56', 'H45', 'H61', 'H57', 'H28', 'H20', 'H42', 'H54', 'H58', 'H38', 'H65', 'H43', 'H51', 'H43', 'H53', 'H38', 'H33', 'H32', 'H46', 'H60', 'H20'],
                                        'H26': ['H44', 'H46', 'H12', 'H48', '', 'H19', 'H34', 'H13', 'H54', 'H35', 'H5', 'H2', 'H7', 'H53', 'H59', 'H36', 'H46', 'H45', 'H17', 'H44', 'H47', 'H40', 'H9', 'H58', 'H5', 'H22', 'H41', 'H20', 'H50', 'H30'], 
                                        'H27': ['H7', 'H46', 'H40', 'H45', 'H50', 'H6', 'H2', 'H28', 'H5', 'H64', 'H53', 'H31', 'H56', 'H65', 'H16', 'H38', 'H56', 'H22', 'H37', 'H21', 'H15', 'H44', 'H15', 'H63', 'H30', 'H37', 'H45', 'H50', 'H29', 'H7'], 'H28': ['H33', 'H41', 'H24', 'H3', 'H34', 'H31', 'H65', 'H3', 'H49', '', 'H43', 'H25', 'H58', '', 'H19', 'H26', 'H13', 'H50', 'H41', 'H3', 'H34', 'H59', 'H16', 'H33', 'H55', 'H43', 'H12', 'H8', 'H8', 'H56'], 'H29': ['H39', 'H35', 'H27', 'H63', 'H45', 'H23', 'H18', 'H52', 'H5', 'H2', 'H53', 'H24', 'H19', 'H24', 'H34', 'H35', 'H4', 'H32', 'H37', 'H35', 'H38', 'H14', 'H2', 'H3', 'H47', 'H52', 'H47', 'H31', 'H43', 'H55'], 'H32': ['H56', '', 'H31', 'H55', 'H65', 'H23', 'H64', 'H53', 'H48', 'H64', '', 'H30', 'H18', 'H15', 'H5', 'H29', 'H9', 'H26', 'H45', 'H17', 'H1', 'H8', 'H29', 'H20', 'H22', 'H63', 'H37', 'H23', 'H20', 'H18'], 'H33': ['H1', 'H47', 'H18', 'H41', 'H58', 'H50', 'H51', 'H37', 'H59', 'H16', 'H53', 'H64', '', 'H2', '', 'H24', 'H55', 'H49', 'H62', '', 'H34', 'H50', '', 'H51', 'H27', 'H17', 'H10', 'H42', 'H22', 'H15'], 'H34': ['H54', 'H60', 'H39', 'H12', 'H32', 'H22', 'H5', 'H42', 'H1', 'H52', 'H62', 'H45', 'H37', '', 'H39', 'H57', 'H16', 'H12', 'H32', 'H62', 'H63', 'H52', 'H28', 'H33', 'H15', 'H58', 'H23', '', 'H43', 'H37'], 'H35': ['H20', 'H39', 'H41', 'H25', 'H18', 'H36', 'H53', 'H5', 'H2', '', 'H15', 'H26', 'H48', 'H14', 'H50', 'H27', 'H39', '', 'H61', 'H12', 'H23', 'H14', 'H34', 'H37', 'H65', 'H25', 'H39', 'H31', 'H36', 'H64'], 'H36': ['H4', 'H56', '', 'H57', 'H59', 'H57', 'H24', 'H15', 'H21', 'H8', 'H35', 'H34', 'H28', 'H65', '', 'H21', 'H2', 'H40', 'H17', 'H38', 'H2', 'H24', 'H47', 'H63', 'H61', 'H61', '', 'H35', 'H42', 'H51'], 'H37': ['H39', 'H26', 'H12', 'H38', 'H33', '', 'H45', 'H12', 'H57', 'H47', 'H3', 'H65', 'H11', 'H29', 'H14', 'H25', 'H49', 'H3', 'H17', '', 'H12', 'H63', 'H21', 'H1', 'H43', 'H48', 'H19', 'H15', 'H59', 'H39'], 'H40': ['', 'H45', 'H63', 'H6', 'H61', 'H22', 'H15', 'H13', 'H25', 'H58', 'H57', 'H4', 'H42', 'H57', 'H26', 'H33', 'H41', 'H30', 'H61', 'H50', 'H18', '', 'H35', 'H48', 'H10', 'H9', 'H57', 'H65', 'H64', 'H21'], 'H41': ['H2', 'H18', 'H35', 'H43', 'H25', 'H30', 'H36', 'H16', 'H5', '', '', 'H47', 'H55', 'H32', '', 'H43', 'H52', 'H30', 'H14', 'H14', '', 'H56', 'H28', 'H54', 'H20', 'H54', 'H47', 'H55', 'H48', 'H5'], 'H42': ['H44', 'H43', 'H24', 'H56', 'H55', 'H53', 'H28', 'H44', '', 'H32', 'H35', 'H17', 'H12', 'H8', 'H14', '', 'H32', 'H32', 'H4', 'H22', 'H32', 'H54', 'H56', 'H5', 'H38', 'H4', 'H36', 'H6', 'H10', 'H53'], 'H43': ['H57', 'H62', 'H31', 'H48', 'H38', 'H26', 'H32', 'H21', 'H18', 'H23', 'H1', 'H23', 'H50', '', 'H20', 'H5', 'H30', 'H23', 'H65', 'H13', 'H7', 'H10', 'H65', 'H41', 'H15', 'H49', 'H3', 'H58', '', 'H4'], 'H44': ['H34', 'H63', 'H11', 'H8', 'H21', 'H55', 'H11', 'H10', 'H54', 'H6', 'H16', 'H50', 'H23', 'H45', 'H7', 'H45', 'H19', 'H47', 'H28', 'H33', '', 'H42', 'H61', 'H26', 'H65', 'H3', 'H15', 'H18', '', 'H30'], 'H45': ['H12', 'H58', 'H63', 'H41', 'H63', 'H33', 'H11', 'H4', 'H63', 'H11', 'H44', 'H29', 'H23', 'H13', 'H24', 'H40', 'H44', 'H16', 'H13', 'H7', 'H20', 'H5', 'H26', 'H2', 'H11', 'H14', 'H31', 'H41', 'H25', 'H56'], 'H46': ['H65', '', 'H30', '', 'H47', 'H33', 'H41', 'H48', 'H24', 'H65', 'H50', 'H5', 'H54', 'H29', 'H27', 'H15', 'H61', 'H19', 'H61', 'H3', 'H64', 'H20', '', 'H63', 'H58', 'H65', 'H27', 'H12', 'H36', 'H56'], 'H49': ['H13', '', 'H4', 'H7', 'H5', 'H63', '', 'H36', 'H21', 'H52', 'H56', 'H10', 'H12', 'H14', 'H39', 'H38', 'H25', '', 'H28', 'H32', 'H57', 'H11', '', 'H55', 'H19', 'H64', 'H61', 'H22', 'H50', 'H9'], 'H50': ['', 'H21', 'H45', 'H32', '', 'H61', 'H25', 'H10', 'H8', 'H42', 'H52', 'H60', 'H44', 'H26', 'H53', 'H47', 'H48', 'H7', 'H54', 'H19', 'H59', 'H6', 'H35', 'H14', 'H24', 'H41', 'H24', 'H40', 'H10', 'H40'], 'H51': ['H38', 'H2', 'H31', 'H8', 'H28', 'H61', 'H26', 'H30', 'H7', '', 'H7', 'H49', 'H42', 'H60', 'H15', 'H65', 'H29', 'H47', 'H57', 'H57', 'H2', 'H29', '', 'H22', 'H15', '', 'H28', 'H2', 'H39', 'H52'], 'H52': ['H6', 'H39', 'H43', 'H28', 'H31', 'H23', 'H20', 'H6', 'H6', 'H2', 'H17', '', 'H23', 'H54', 'H53', 'H48', 'H64', '', 'H11', 'H12', 'H42', 'H5', 'H3', 'H23', 'H39', 'H25', 'H5', 'H18', '', 'H30'], 'H53': ['H1', 'H43', 'H7', 'H41', 'H20', 'H43', 'H43', 'H4', '', 'H63', 'H29', 'H51', 'H25', 'H3', 'H29', 'H21', 'H28', 'H47', 'H58', 'H26', 'H41', 'H8', 'H21', 'H2', 'H31', 'H24', 'H56', 'H58', 'H21', 'H35'], 'H54': ['H25', 'H55', 'H32', 'H25', 'H21', 'H7', 'H45', 'H30', 'H63', 'H64', 'H40', 'H12', 'H36', 'H62', 'H44', 'H13', 'H8', 'H12', 'H6', '', 'H18', 'H36', 'H49', 'H57', 'H58', 'H41', 'H43', 'H50', 'H32', ''], 'H55': ['H63', 'H59', 'H49', 'H51', 'H31', 'H4', 'H49', 'H32', 'H64', 'H47', 'H59', 'H54', 'H10', 'H1', 'H45', 'H62', 'H38', 'H49', 'H38', 'H27', 'H27', 'H54', 'H18', 'H50', 'H15', 'H61', 'H7', 'H58', 'H25', 'H57'], 'H58': ['H1', 'H59', 'H1', 'H2', 'H5', 'H13', 'H53', 'H22', 'H1', 'H4', '', 'H18', 'H31', 'H7', 'H3', 'H6', 'H1', 'H45', 'H63', 'H28', 'H61', 'H60', 'H65', 'H8', 'H8', 'H24', 'H41', 'H44', 'H60', 'H50'], 'H59': ['H38', 'H35', 'H9', '', 'H64', 'H49', '', 'H56', 'H57', 'H12', '', 'H15', 'H11', 'H37', 'H54', 'H47', 'H53', 'H39', '', 'H49', 'H65', '', 'H37', 'H60', '', 'H29', 'H17', 'H49', 'H14', 'H51'], 'H60': ['H5', 'H31', 'H7', 'H49', '', 'H36', 'H62', 'H33', 'H56', 'H51', 'H65', 'H54', 'H45', 'H43', 'H65', 'H55', 'H57', 'H55', 'H62', 'H58', 'H7', 'H8', 'H31', 'H42', 'H5', 'H25', 'H41', 'H4', 'H18', 'H3'], 'H61': ['H31', 'H59', 'H37', 'H9', 'H12', 'H64', 'H54', 'H33', 'H37', 'H64', 'H10', 'H65', 'H49', 'H17', 'H22', 'H12', 'H49', 'H40', 'H11', 'H2', 'H40', 'H20', 'H19', 'H23', 'H49', 'H16', 'H55', 'H42', 'H6', ''], 'H62': ['H15', 'H10', 'H11', 'H10', 'H37', '', 'H52', 'H31', 'H49', 'H7', 'H5', 'H49', 'H22', 'H59', 'H30', 'H31', 'H64', 'H10', 'H13', 'H6', 'H8', 'H11', 'H40', 'H47', 'H64', 'H14', 'H11', 'H28', 'H61', 'H23'], 'H63': ['H28', 'H8', 'H2', 'H6', 'H17', 'H15', 'H51', 'H25', 'H15', 'H48', 'H45', 'H64', 'H60', 'H21', 'H56', 'H2', 'H28', '', 'H16', 'H45', 'H43', 'H40', 'H49', 'H10', 'H23', 'H43', 'H29', 'H26', 'H39', 'H40'], 
                                        'H64': ['H55', 'H20', 'H12', 'H40', 'H29', '', 'H20', 'H28', 'H15', 'H61', '', 'H6', 'H1', 'H2', 'H29', '', 'H28', 'H26', 'H42', 'H1', 'H24', 'H41', '', 'H41', 'H49', 'H7', 'H18', 'H30', 'H25', '']
                                        }
        #self.communication_sequences = generate_traffic_sequence_service_provider(self)

        self.all_tms = {}
        self.graph_has_data = True
        
        self.graph_topology = pickle.load(open(f"{PATH_SIMULATION}/TopologyFiles/service_provider_network.pickle", "rb"))

        #nx.draw(self.graph_topology, with_labels=True)
        #plt.show()
        self.create_components(self.graph_topology) #creates nodes and edges
        self.hosts = self.get_all_hosts()
        #for host in self.hosts:
        #    self.bws[host] = random.randint(20, 100) #change values to experiment, copy and put them in self.bws
        #print(self.bws)
        self.bws = {'H8': 42, 'H13': 29, 'H22': 40, 'H31': 28, 'H39': 35, 'H48': 28, 'H57': 55, 'H12': 35, 'H21': 22, 'H30': 52, 'H38': 24, 'H47': 44, 'H56': 53, 'H65': 60, 'H1': 42, 'H2': 21, 'H3': 75, 'H4': 60, 'H5': 39, 'H6': 23, 'H7': 80, 'H9': 32, 'H10': 53, 'H14': 75, 'H11': 51, 'H20': 63, 'H15': 23, 'H16': 61, 'H17': 78, 'H18': 53, 'H19': 73, 'H23': 28, 'H24': 44, 'H25': 69, 'H26': 26, 'H27': 61, 'H28': 52, 'H29': 29, 'H32': 47, 'H33': 67, 'H34': 25, 'H35': 66, 'H36': 77, 'H37': 39, 'H40': 73, 'H41': 73, 'H42': 21, 'H43': 39, 'H44': 42, 'H45': 41, 'H46': 49, 'H49': 75, 'H50': 55, 'H51': 66, 'H52': 46, 'H53': 24, 'H54': 34, 'H55': 53, 'H58': 50, 'H59': 27, 'H60': 59, 'H61': 69, 'H62': 69, 'H63': 46, 'H64': 53}
        self.calculate_paths()

        #print("\n hosts: ", self.hosts)
        self.number_of_hosts = len(self.hosts)
        self.statistics = {'package_loss': 0, 'package_sent': 0, 'nr_package_loss': 0, 'nr_package_sent': 0}
        #self.single_con_hosts = [f"H{int(host) + 1}" for host in self.graph_topology if len(self.graph_topology.edges(host)) == 1]  
        #print("single con hosts: ", self.single_con_hosts)
        #self.bws = {host: bw if host not in self.single_con_hosts else bw // 3 for host, bw in self.bws.items()}
        
        #generate_traffic_sequence_service_provider(self)

        self.all_tms = json.load(open(f"{PATH_SIMULATION}/TrafficMatrix/tms_service_provider_test.json", mode="r"))
        self.current_index = 0
        self.current_tm_index = self.current_index % len(self.all_tms)       
        self.communication_sequences = self.all_tms[self.current_tm_index] 

    def get_nx_topology(self):
        return self.graph_topology
    
    def calculate_convergence(rewards, failure_point, window, threshold):
        
        # Iniciar do ponto após a falha
        start_idx = failure_point
        
        # Procurar ponto inicial de convergência
        convergence_point = None
        reference_avg = None
        
        for i in range(start_idx + window, len(rewards) - window + 1, window):
            current_window = rewards[i:i+window]
            previous_window = rewards[i-window:i]
            
            current_avg = sum(current_window) / window
            previous_avg = sum(previous_window) / window
            
            if previous_avg != 0:
                variation = abs((current_avg - previous_avg) / previous_avg)
                
                # Detecta o primeiro ponto de convergência
                if convergence_point is None and variation < threshold:
                    convergence_point = i - failure_point
                    reference_avg = previous_avg  # Guarda a média de referência
                    continue
                
                # Se já temos um ponto de convergência, comparar com a referência original
                if convergence_point is not None:
                    # Comparar com a REFERÊNCIA em vez da janela anterior
                    variation_to_reference = abs((current_avg - reference_avg) / reference_avg)
                    if variation_to_reference >= threshold:
                        # Se afastou demais da referência, reset da convergência
                        convergence_point = None
                        reference_avg = None
        
        if convergence_point is not None:
            return str(convergence_point)
        else:
            return "Not converged"
    
    def increase_traffic_bandwidth(self, factor):
        """
        Aumenta a largura de banda de todo o tráfego pelo fator especificado
        
        Args:
            factor: Multiplicador para a largura de banda (ex: 1.2 = aumento de 20%)
        """
        # Guardar larguras de banda originais se ainda não existirem
        if not hasattr(self, 'original_bws'):
            self.original_bws = {host: bw for host, bw in self.bws.items()}
            self.current_bw_multiplier = 1.0
        
        self.current_bw_multiplier = min(self.current_bw_multiplier * factor, MAX_BANDWIDTH_MULTIPLIER)

        # Verificar se atingiu o limite de estabilização
        if STABILIZE_BANDWIDTH and self.current_bw_multiplier >= STABILIZE_AFTER_MULTIPLIER:
            self.bandwidth_stabilized = True
            self.current_bw_multiplier = STABILIZE_AFTER_MULTIPLIER
            print(f"Largura de banda atingiu {self.current_bw_multiplier*100:.0f}% e foi estabilizada")
        else:
            print(f"Largura de banda do tráfego aumentada para {self.current_bw_multiplier*100:.0f}% do valor original")
        
        # Atualizar larguras de banda para todos os hosts
        for host in self.bws:
            original_bw = self.original_bws[host]
            self.bws[host] = int(original_bw * self.current_bw_multiplier)
        
        print(f"Largura de banda do tráfego aumentada para {self.current_bw_multiplier*100:.0f}% do valor original")
        
def generate_traffic_sequence(network=None):
    if not network:
      network = NetworkEngine()

    hosts = network.get_all_hosts()
    bws = {}
    communications = {}
    list_all_communications = []
    generate_traffic_matrix_file = False

    if generate_traffic_matrix_file:
        # Gera e salva novas matrizes de tráfego
        for j in range(500):
            for host in hosts:
                #print("\n host: ", host)
                bws[host] = random.randint(10, 35)
                for i in range(30):
                    dst = network.get_random_dst(host, hosts)
                    #print("\n dst: ", dst)
                    dsts = communications.get(host, [])
                    #print("\n dsts: ", dsts)
                    dsts.append(dst)
                    communications[host] = dsts
            list_all_communications.append(communications)
            communications = {}
        json.dump(list_all_communications, open("TrafficMatrix/tms_internet_test.json", "w"), indent=4)
    else:
        # Usa matrizes existentes carregadas de arquivos JSON
        for host in hosts:
            bws[host] = random.randint(20, 50)
            for i in range(30):
                dst = network.get_random_dst(host, hosts)
                dsts = communications.get(host, [])
                dsts.append(dst)
                communications[host] = dsts
    return  communications

def generate_traffic_sequence_arpanet(network=None):
    if not network:
      network = NetworkEngine()
    #network = pickle.load(open("topology_arpanet.pickle", "rb"))
    hosts = network.get_all_hosts()
    print("\n generating traffic matrix")
    print("\n hosts: ", hosts)
    bws = {}
    communications = {}
    list_all_communications = []
    
    #for j in range(500):
    for host in hosts:
        bws[host] = random.randint(10, 35)
        for i in range(30):
            dst = network.get_random_dst(host, hosts)
            dsts = communications.get(host, [])
            dsts.append(dst)
            communications[host] = dsts
    list_all_communications.append(communications)
    communications = {}
    #json.dump(list_all_communications, open("tms_arpanet_test.json", "w"), indent=4)
    return  communications

def generate_traffic_sequence_service_provider(network=None):
    if not network:
      network = NetworkEngine()
    hosts = network.get_all_hosts()
    bws = {}
    communications = {}
    list_all_communications = []

    nodes_list = ['H8', 'H13', 'H22', 'H31', 'H39', 'H48','H57','H12', 'H30', 'H21', 'H38', 'H47', 'H56', 'H65']
    
    for j in range(1000):
        #random_sample = random.sample(hosts, 35)
        #communications[j] = {}
        #for host in hosts:
        #for host in random_sample:
        for host in nodes_list:
            #print("\n host: ", host)
            bws[host] = random.randint(10, 30)
            for i in range(20):
                dst = network.get_random_dst(host, hosts)
                dsts = communications.get(host, [])
                dsts.append(dst)
                communications[host] = dsts
        list_all_communications.append(communications)
        communications = {}

    #json.dump(list_all_communications, open("tms_service_provider_train.json", "w"), indent=4)
    return  communications

def save_removed_edges(removed_edges, scenario_completed=True):
    # Serializa edges para formato que pode ser salvo em JSON
    edges_data = {str(k): [[e[0], e[1]] for e in v] if v is not None else None 
                  for k, v in removed_edges.items()}
    
    # Salva os edges removidos em arquivo
    with open(f"{PATH_SIMULATION}/removed_edges.json", "w") as f:
        json.dump(edges_data, f)
    
    # Salva flag indicando que cenário 2 foi completado
    with open(f"{PATH_SIMULATION}/scenario_completed.txt", "w") as f:
        f.write(str(scenario_completed))
    
def load_removed_edges():
    try:
        # Carrega edges removidos do arquivo
        with open(f"{PATH_SIMULATION}/removed_edges.json", "r") as f:
            edges_data = json.load(f)
            removed_edges = {int(k): [tuple(e) for e in v] if v is not None else None 
                          for k, v in edges_data.items()}
        
        # Carrega flag de cenário 2 completo
        with open(f"{PATH_SIMULATION}/scenario_completed.txt", "r") as f:
            scenario_completed = f.read().strip() == "True"
            
        return removed_edges, scenario_completed
    except:
        print("[WARNING] Failed to load removed edges or scenario completion status.")
        
