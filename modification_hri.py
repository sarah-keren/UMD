__author__ = 'sarah'

import  utils, defs
import copy, os, time

from modification import *

class Modification_hri (Modification):


    def __init__(self, cost=1):

        self.cost = cost

    def apply(self, model, modf):

        raise NotImplementedError # how to get modified_sensor_placement

        mod_model = model.create_modified_model(modf)

        return mod_model

    def __str__(self):

        raise NotImplementedError

    def get_params(self):

        raise NotImplementedError




class InformationShapingModification(Modification):

    '''
    predicate - the information to share with the agent (i.e., add to the init state)
    '''
    def __init__(self, predicate, neg_predicate = None, cost =1 ):
        self.predicate = predicate
        self.neg_predicate = neg_predicate
        self.cost = cost

    '''
    add the predicate to the initial state
    '''
    def apply(self, model, rand_name=False):

        # copy the original model
        #mod_model = copy.deepcopy(model)

        # template files names
        original_template_file = model.template_file_name
        if rand_name:
            mod_template_file_name = 'template_%s.pddl'%int(round(time.time() * 1000))
        else:
            mod_template_file_name = model.template_file_name.replace('.','_mod_%s.'%self.__str__())
        mod_template_file_name = os.path.join(defs.GEN_FOLDER, os.path.basename(mod_template_file_name))

        print('creating a new problem with template file name %s'%mod_template_file_name)

        if self.predicate is not None:
            # create the modified file
            utils.add_predicate_to_init_state_pddl(self.predicate, original_template_file, mod_template_file_name)
        if self.neg_predicate is not None:
            # create the modified file
            utils.add_predicate_to_init_state_pddl('(not %s)'%self.neg_predicate, original_template_file, mod_template_file_name)

        # update the modified model, doing any necessary updates
        mod_model = model.create_modified_model(defs.TEMPLATE_FILE, mod_template_file_name)

        return mod_model

    def __str__(self):
        name = ''
        if self.predicate:
            par_string = self.predicate.replace('(', '')
            par_string = par_string.replace(')', '')
            name += '%s'%par_string
        if self.neg_predicate:
            no_par_string = self.neg_predicate.replace('(','')
            no_par_string = no_par_string.replace(')', '')
            name += '_not_%s'%no_par_string

        name = name.replace(" ", "=")

        #return (''.join(self.predicate))
        return name

    def get_params(self):

        modification_string = self.__str__().split('=')
        params = modification_string[1:]
        return params

    def get_pos_atom(self):

        pred = self.predicate
        #if pred is not None:
        #    pred = pred.replace('(', '')
        #    pred = pred.replace(')', '')
        return pred

    def get_neg_atom(self):

        pred = self.neg_predicate
        #if pred is not None:
        #    pred = pred.replace('(', '')
        #    pred = pred.replace(')', '')
        return pred

# TODO SARAH : Complete implementation
class ActionRemovalModification(Modification):

    '''
    predicate - the information to share with the agent (i.e., add to the init state)
    '''
    def __init__(self, goal_predicates):
        #TODO SARAH : complete
        print('Complete')
    '''
    add the predicate to the initial state
    '''
    def apply(self, model):
        raise NotImplementedError
