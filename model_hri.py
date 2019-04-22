__author__ = 'sarah'

import networkx as nx
import numpy as np

from model import *
from gm import *

UtilityMaximizingDesign / src / gr_poa_kp.py

class model_hri (Model):

    def __init__(self):

        self.hri_model = model_gm()

    # used to represent domain specific invalidity of modifications that is not captured by the design,pddl file
    def is_valid(self, modification):

        if sum(self.hri_model.sensors + modification)<=self.hri_model.total_num_sensor:
            return True
        else:
            return False

    # create a modified model according to the modification
    def create_modified_model(self, modification):

        modified_model = model_hri()
        modified_model.new_model(modification)
        # modification.apply()

        return modified_model

    # used to represent cleanup operations for freeing unused resources
    def clean_up(self):

        raise NotImplementedError

    def get_successors():

        return self.hri_model.gm_get_successors()
