__author__ = 'sarah'

import networkx as nx
import numpy as np

class Model:

    def __init__(self):

        raise NotImplementedError

    # used to represent domain specific invalidity of modifications that is not captured by the design,pddl file
    def is_valid(self, modification):

        return True

    # create a modified model according to the modification
    def create_modified_model(self, modification):

        raise NotImplementedError

    # used to represent cleanup operations for freeing unused resources
    def clean_up(self):

        raise NotImplementedError
