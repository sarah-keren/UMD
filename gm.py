import networkx as nx
import numpy as np

from model import *

class model_gm (Model):

    def __init__(self):

        self.sensors = np.array([5, 5])
        self.sensor_accuracy = 0.5

        cpt_H1 = [1/2, 1/2]
        cpt_H2 = [1/2, 1/2]
        cpt_O_H1 = update_cpt(self, self.sensors[0])
        cpt_O_H2 = update_cpt(self, self.sensors[1])

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

    def new_model(self, modification):

        self.sensors = self.sensors + modification

        cpt_O_H1 = self.hri_model.update_cpt(self, self.sensors[0])
        cpt_O_H2 = self.hri_model.update_cpt(self, self.sensors[1])
        self.hri_model.G.add_node('O_H1', cpt = cpt_O_H1)
        self.hri_model.G.add_node('O_H2', cpt = cpt_O_H2)

    def update_cpt(self, num_sensor):

        p = self.sensor_accuracy

        return [[1-(1-p)**num_sensor, (1-p)**num_sensor],
                [(1-p)**num_sensor, 1-(1-p)**num_sensor]]

    def gm_get_successors(self):

        return [np.array([-1, -1]), np.array([1, -1]), np.array([-1, 1]), np.array([1, 1])]
