UtilityMaximizingDesign / src / run_grd_design.py

import os
import grd, defs, utils, gr_poa_prp, gr_poa_kp, modification, constraint, pruning
import pddl_parser
import os, shutil, sys
import search, design, heuristic, time

"hri_umd 바꿔라"


def process_input():

    solver_path = sys.argv[1]  # '/mnt/c/Users/sarah/OneDrive/Documents/UtilityMaximizingDesign/solvers/PRP/planner-for-relevant-policies/src'
    # domain_file_name = sys.argv [2]  # '/mnt/c/Users/sarah/OneDrive/Documents/UtilityMaximizingDesign/benchmarks/grd-apo/pond-benchmarks-sarah/wumpus-clg/wumpus-running-example-grd/wumpus05/d.pddl'
    # template_file_name = sys.argv[3]  # '/mnt/c/Users/sarah/OneDrive/Documents/UtilityMaximizingDesign/benchmarks/grd-apo/pond-benchmarks-sarah/wumpus-clg/wumpus-running-example-grd/wumpus05/template.pddl'
    # hyps_file_name = sys.argv[4]  # '/mnt/c/Users/sarah/OneDrive/Documents/UtilityMaximizingDesign/benchmarks/grd-apo/pond-benchmarks-sarah/wumpus-clg/wumpus-running-example-grd/wumpus05/hyps.dat'
    design_file_name = sys.argv[5] # NA means the changes are encoded
    design_problem_file_name = sys.argv[6]  # NA means the changes are encoded
    design_budget = int(sys.argv[7])
    search_method_name = sys.argv[8]
    heuristic_name = sys.argv[9]
    pruning_method_name = sys.argv[10]

    planner_type = defs.get_planner_type(solver_path)

    log_file = open(defs.DESIGN_LOG_NAME, "a")
    log_file.write('\nDesigning domain::%s\ntemplate::%s\nhyps::%s\nbudget::%d\nmethod::%s\nheuristic::%s\npruning::%s\nplanner::%s\n' %
               (domain_file_name, template_file_name, hyps_file_name, design_budget, search_method_name,heuristic_name,pruning_method_name,planner_type))
    log_file.write('\n----------------------\n')
    log_file.flush()


    results_log_file = open('%s'%defs.DESIGN_RESULTS_LOG_NAME, "a")
    results_log_file.write('\n----------------------')
    results_log_file.write('\nDesigning domain::%s\ntemplate::%s\nhyps::%s\nbudget::%d\nmethod::%s\nheuristic::%s\npruning::%s\nplanner::%s\n' %
               (domain_file_name, template_file_name, hyps_file_name, design_budget, search_method_name,heuristic_name,pruning_method_name,planner_type))
    results_log_file.flush()


    if defs.SOLVER_PRP in planner_type:
        # create the gr problem
        gr_problem = gr_poa_prp.GR_POA_PRP(solver_path, domain_file_name, template_file_name, hyps_file_name)
    elif defs.SOLVER_KP in planner_type:
        # create the gr problem
        gr_problem = gr_poa_kp.GR_POA_KP(solver_path, domain_file_name, template_file_name, hyps_file_name)

    else:
        print('planner type '+ planner_type+" not supported")
        raise NotImplementedError

    # constraints
    constraint_list = []
    bud_constraint = constraint.BudgetConstraint(design_budget)
    constraint_list.append(bud_constraint)

    # create GRD problem
    grd_problem = grd.GRD(gr_problem, constraint_list, design_file_name, design_problem_file_name)


    # create frontier and initialize it with the heuristic
    heuristic_func = None
    if defs.NA not in heuristic_name:
        heuristic_func = heuristic.get_heuristic(heuristic_name,grd_problem)
        # zero heuristic
        frontier = search.PriorityQueue(min, heuristic_func)

    # create closed list
    closed_list = search.ClosedListOfSets()
    termination_criteria = design.TerminationCriteriaOptimalValue(0 ,True)
    if defs.NA in pruning_method_name:
        prune_func = None  # pruning.prune_grd_currentFocus
    else:
        prune_func = pruning.get_pruning_func(pruning_method_name,grd_problem)

    return [grd_problem, search_method_name, frontier,prune_func,log_file, results_log_file]



def run_hri_design(grd_problem, search_method_name, frontier, prune_func, log_file, results_log_file):

    if defs.BFD.lower() in search_method_name.lower():
        # create closed list
        closed_list = search.ClosedListOfSets()
        termination_criteria = design.TerminationCriteriaOptimalValue(0, True)


        # time
        start_time = time.clock()

        # perform design
        [best_value, best_node, explored, ex_terminated, results_log] = design.best_first_design(grd_problem, frontier, closed_list, termination_criteria, prune_func, log_file,results_log_file, defs.ITER_LIMIT, defs.DEFAULT_TIME_LIMIT)

        # get exe time
        calc_time = time.clock() - start_time  # , "seconds"


        # log results
        for cost in results_log:
            log_message = 'Best_value_for_cost-%d:'%cost+defs.SEPARATOR+'cur_value::%d'%results_log[cost][0]+defs.SEPARATOR+'cur_node::'+results_log[cost][1].__repr__()+defs.SEPARATOR+'calc_time::%.2f'%results_log[cost][2]+defs.SEPARATOR+'explored::%d'%results_log[cost][3]+'\n'
            results_log_file.write(log_message)

        log_message = 'Total_best_value:%.2f'%best_value + defs.SEPARATOR + ' best_node:'+ best_node.__repr__() + defs.SEPARATOR + 'cost:%.2f'%best_node.cost()[0] + defs.SEPARATOR + 'explored:%s'%explored+ defs.SEPARATOR + 'time:%.6f'%calc_time+ defs.SEPARATOR+ 'ex_terminated:'+ str(ex_terminated) +'\n'
        log_file.write(log_message)
        results_log_file.write(log_message)

        log_file.close()
        results_log_file.close()

    else:
        print('only BFD search method supported for now: existing')
        raise NotImplementedError
