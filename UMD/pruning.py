__author__ = 'sarah'


def get_pruning_func(prune_func_name, umd_problem):
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


