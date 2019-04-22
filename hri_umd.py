__author__ = 'sarah'

import search
import pddl_parser, utils, defs

from umd import *

class UMD_hri (UMD):

    """Environment Design superclass
       supporting COMPLETE
    """

    def __init__(self, initial_model, constraints, design_file_name, design_problem_file_name ):
        self.initial_model = initial_model
        self.constraints = constraints
        self.design_file_name = design_file_name
        self.design_problem_name =  design_problem_file_name

    # we assume nodes that are evlauated are valid - so there is no need here to check for validity
    def evaluate(self, model):
        raise NotImplementedError

    def is_valid(self, node):

        # if there are no constraints - return True
        if self.constraints is None:
            print('No constraints')
            return True
        # check all constraints - if one is violated, return False
        for constraint in self.constraints:
            if not constraint.is_valid(node):
                return False

        # non of the constraints have been violated
        return True


    # the possible modifications are either expressed in a design,pddl file (as transitions) or as implemented modificaitons
    def get_possible_modifications(self, cur_node):

        modifications = []

        possible_modifications = cur_node.state.get_successors()
        for cur_modif in possible_modifications:
            if cur_node.state.is_valid(cur_modif):
                modifications.append(cur_modification)

        return modifications

    # returns the nodes that represent the models that result from applying the possible modifications
    # get all possible modifications for cur_node
    def successors(self, cur_node, cleanup = True):


        # the umd model represented by the node
        cur_model = cur_node.state


        # get the modifications that can be applied to this node
        modification_list = self.get_possible_modifications(cur_node)
        if modification_list is None:
            return None

        # remove the modifications that violate the constraints
        successor_nodes = []
        for modification in modification_list:

            successor_model = modification.apply(cur_model)
            successor_node  = search.DesignNode(successor_model,cur_node, modification, cur_node.path_cost+modification.cost, cur_node.umd_problem)


            valid = True
            # iterate through the constraints to see if the current modification violates them
            for constraint in self.constraints:
                if not constraint.is_valid(successor_node, modification):
                    valid = False
                    break

            # apply the modifications
            if valid:
                ''' add the succesor node specifying:
                    successor_model(the reuslting model)
                    cur_node (ancestor)
                    modificaiton (the appllied modificaiton)
                    node.path_cost+modification.cost (the accumulated design cost)
                '''
                successor_nodes.append(successor_node)

        if cleanup:
            cur_model.clean_up()

        return successor_nodes

    def is_better(self, val_a, val_b):
        if val_a > val_b:
            return True
        else:
            return False