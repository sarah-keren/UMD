__author__ = 'sarah'

import sys
try:
    import  utils, defs
except ModuleNotFoundError:
    print(sys.path)
    raise
import copy, os, time

class Modification:


    def __init__(self, cost=1):
        self.cost = cost
    def apply(self, model):
        raise NotImplementedError
    def __str__(self):
        raise NotImplementedError
    def get_params(self):
        raise NotImplementedError
