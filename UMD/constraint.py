__author__ = 'sarah'

import  utils
import copy, os
import search


class Constraint:


    def __init__(self):
        #TODO SARAH : complete
        print('Complete')

    def is_valid(self, model):
       return False

class BudgetConstraint():

    '''
    the number of allowed modifications
    '''
    def __init__(self, budget):
        self.budget = budget
    def __repr__(self):
        return "%d"%self.budget


    '''
    check the budget constraints have not been violated
    '''
    def is_valid(self, design_node, modification):

       #accumulated modification cost
       [modification_cost, sequence] = design_node.cost()
       if modification_cost <= self.budget:
           return True       
       return False



