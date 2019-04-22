import networkx as nx
import numpy as np

from model import *

class model_gm (Model):

    def __init__(self):

        self.total_num_sensor = 10
        self.sensor_accuracy = 0.5

        cpt_H1 = [1/2, 1/2]
        cpt_H2 = [1/2, 1/2]
        cpt_O_H1 = update_cpt(self, self.total_num_sensor/2)
        cpt_O_H2 = update_cpt(self, self.total_num_sensor/2)
        raise NotImplementedError

        self.G = nx.DiGraph()
        self.G.add_node('H1', cpt = cpt_H1)
        self.G.add_node('H2', cpt = cpt_H2)
        self.G.add_node('O_H1', cpt = cpt_O_H1)
        self.G.add_node('O_H2', cpt = cpt_O_H2)
        self.G.add_node('R', cpt = cpt_R)
        self.G.add_node('E', cpt = cpt_E)
        self.G.add_edge('H1', 'O_H1')
        self.G.add_edge('H2', 'O_H2')
        self.G.add_edge('O_H1', 'R')
        self.G.add_edge('O_H2', 'R')
        self.G.add_edge('H1', 'E')
        self.G.add_edge('H2', 'E')
        self.G.add_edge('R', 'E')

    def is_valid(self, modification):

        if sum(modification)<=self.total_num_sensor:
            return True
        else:
            return False

    # create a modified model according to the modification
    def create_modified_model(self, modification):

        cpt_O_H1 = update_cpt(self, modification[0])
        cpt_O_H2 = update_cpt(self, modification[1])
        self.G.add_node('O_H1', cpt = cpt_O_H1)
        self.G.add_node('O_H2', cpt = cpt_O_H2)

        return self

    # used to represent cleanup operations for freeing unused resources
    def clean_up(self):

        raise NotImplementedError

    def update_cpt(self, num_sensor):

        p = self.sensor_accuracy

        return [[1-(1-p)**num_sensor, (1-p)**num_sensor],
                [(1-p)**num_sensor, 1-(1-p)**num_sensor]]
