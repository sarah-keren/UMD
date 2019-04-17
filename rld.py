__author__ = 'sarah'

import copy
import os
import random
import sys
import time

sys.path.append(os.path.join(os.getcwd(), 'pymdptoolbox/src/aima'))
sys.path.append(os.path.join(os.getcwd(), 'pymdptoolbox/src/mdptoolbox'))

import umd
import rl_model, constraint
import defs, heuristic, utils, search
import logging
import modification

from optparse import OptionParser
from mdp_ai import MazeMDP


from example import forest, gridworld

class RLD(umd.UMD):

    def __init__(self, initRL, constraints, mdp, design_problem_file_name):
        super().__init__(
            initRL,
            constraints,
            design_file_name=None,
            design_problem_file_name=design_problem_file_name
        )
        self.mdp = mdp

    def evaluate(self, RLmodel):
        RLmodel.allowed_rewards = 'none' # stop counting rewards from modifications
        return RLmodel.get_utility()

    def is_better(self, val_a, val_b):
        # true if a is better than b
        return val_a > val_b

    # get the modifications that are applicable at the current node (disregarding the GRD constraints)
    def get_possible_modifications(self, cur_rl_node):
        rl_model = cur_rl_node.state # should be an instance of RL_model

        if isinstance(rl_model.mdp, MazeMDP):
            empties = rl_model.mdp.maze.find_cells(' ')
            out = set()
            for (m, x, y) in empties:
                out.add(modification.AddCookieModification(m, x, y))
                out.add(modification.AddNailModification(m, x, y))

        return list(out)

def configure_logging(loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)

    root_logger = logging.getLogger('')
    strm_out = logging.StreamHandler(sys.__stdout__)
    strm_out.setFormatter(logging.Formatter('%(message)s'))
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(strm_out)

if __name__ == '__main__':
    usage_msg = "Usage:  %prog [options]"
    parser = OptionParser(usage=usage_msg)

    def usage(msg):
        print("Error: %s\n".format(msg))
        parser.print_help()
        sys.exit()

    parser.add_option("--loglevel",
                      dest="loglevel",
                      default="info",
                      help="Set the logging level: 'debug' or 'info'"
    )

    parser.add_option("--mdp_type",
                      dest="mdp",
                      default="maze",
                      help="Choose the environment, defined as an MDP in \
                        mdptoolbox.example. Options: maze, gridworld"
    )

    parser.add_option("--gamma",
                      dest="gamma",
                      default=1.0,
                      help="Choose discount factor"
    )

    parser.add_option("--budget",
                      dest="budget",
                      default=2,
                      help="Modification budget"
    )

    parser.add_option("--infile",
                      dest="infile",
                      default="maps/04.map",
                      help="Choose initial map")

    parser.add_option("--design",
                      dest="design",
                      default=None,
                      help="Choose map to run RLD on")

    parser.add_option("--allowed_rewards",
                      dest="allowed_rewards",
                      default="none",
                      help="Choose modified environment's rewards to include - all, none, or negative")

    parser.add_option("--busy_wait",
                      dest="busy_wait",
                      default=True,
                      help="When an episode is finished, initiate a self loop?")

    parser.add_option("--outfile",
                      dest="outfile",
                      default=False,
                      help="Where to write best found map?")

    parser.add_option("--time_limit",
                      dest="time_limit",
                      default=3000,
                      help='How long to run design for?')

    parser.add_option("--n_episodes",
                      dest="n_episodes",
                      default=100,
                      help='how Many QLearning episodes to perform?')

    parser.add_option("--seed",
                      dest="seed",
                      default=5,
                      help='set the random seed')

    parser.add_option("--deterministic",
                      dest="deterministic",
                      default=False,
                      help='decide whether transitions are deterministic')

    (options, args) = parser.parse_args()

    random.seed(int(options.seed))

    configure_logging(options.loglevel)

    if options.design:
        infile = 'rld_designs/one_room/{}.map'.format(options.design)
        outfile = 'rld_designs/one_room/search_best_{}.map'.format(options.design)
    else:
        infile = options.infile
        outfile = options.outfile

    if options.mdp == 'maze':
        mdp = MazeMDP(infile=infile, deterministic=options.deterministic, busy_wait=options.busy_wait)
    else:
        raise ValueError('mdp not recognized')

    if options.gamma > 1.0 or options.gamma <= 0.:
        raise ValueError('gamma not in (0,1] range')

    # create the initial RL model
    rl_model = rl_model.RL_model(mdp=mdp, discount=options.gamma,\
        n_episodes=int(options.n_episodes), allowed_rewards=options.allowed_rewards)

    # specify the design constraints (should be non-zero to search)
    constraints = [constraint.BudgetConstraint(int(options.budget))]


    # create rld problem
    rld_problem = RLD(rl_model,
                      constraints,
                      mdp=mdp,
                      design_problem_file_name=None
    )

    # create the frontier according to which nodes are extracted
    frontier = search.PriorityQueue(min, heuristic.zero_heuristic)

    log_file = open('log_file.txt', 'a')
    closed_list = search.ClosedListOfSets()

    # perform design
    import design
    out = \
        design.best_first_design(rld_problem, \
                                 frontier,
                                 closed_list=closed_list,
                                 log_file=log_file,
                                 raise_errs=True,
                                 time_limit=float(options.time_limit)
        )

    [best_value, best_node, explored_count,terminated ] = out[:4]
    print('bv {} bn {} ex {}'.format(best_value, best_node, explored_count))
    print('demonstrating policy ....')
    best_node.state.test_fun(30)
    if terminated:
        print('terminated due to timeout on infile {}'.format(infile))
        best_node.state.mdp.maze.write_maze(outfile)
    else:
        best_node.state.mdp.maze.write_maze(outfile)
