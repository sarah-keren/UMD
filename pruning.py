__author__ = 'sarah'


import causal_graph

def get_pruning_func(prune_func_name, umd_problem):

    if 'cg' in prune_func_name.lower():
        # load the causal graphs into the umd class
        #umd_problem.causal_graphs = causal_graph.get_causal_graphs(umd_problem)
        print('before getting Causal Graph')
        umd_problem.causal_graphs = causal_graph.get_causal_graphs_from_compilation(umd_problem)
        print('after getting Causal Graph')

        # send the pruning method below
        return prune_not_in_causal_graph
    else:
        print('prune_func_name %s is not yet defined'%prune_func_name)
        raise NotImplementedError





def prune_currentFocus(successors, node):
    
    # the umd_model passed to this method should have had its value measured, which means that the current paths should have been calculated
    policy_graphs = node.state.policy_graphs
    if policy_graphs is not None:
        for graph in policy_graphs:
            print(graph)
        raise NotImplementedError
    
    
    else:            
        return successors


def prune_not_in_causal_graph(sucessors, node):

    sucessors_after_pruning = []
    for sucessor in sucessors:
        is_relevant = False
        for graph in node.umd_problem.causal_graphs:
            pos_atom = sucessor.action.get_pos_atom()
            neg_atom = sucessor.action.get_neg_atom()
            if graph.is_atom_in_causal_graph(pos_atom):
                is_relevant = True
                break
            if graph.is_atom_in_causal_graph(neg_atom):
                is_relevant = True
                break

        if is_relevant:
            sucessors_after_pruning.append(sucessor)
    return sucessors_after_pruning

