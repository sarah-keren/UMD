import os 

SEPARATOR = ' '

REC_KNOW_STRING  = ';;; Recognition system knowledge'
AGENT_KNOW_STRING = ';;; Agent knowledge'
HYPS_STRING = ';;; HYPS <hyp>'
ADDED_PREDICATES_STRING = ';;; CHANGES'

FAILURE_STRING = 'FAIL'
NA = 'NA'
ITER_LIMIT = 500#1000
DEFAULT_TIME_LIMIT = 1200#1800
MODIFICATIONS_COUNTER = 0
DEFAULT_MAX_HORIZON = 1000
NON_OBS_STATE_STRING = 'NA'

INFINITE = 8888

TEMPLATE_FILE = 'TEMPLATE_FILE'
HYPS_FILE = 'HYPS_FILE'
DOMAIN_FILE = 'DOMAIN_FILE'

NA = 'NA'

BFD = 'bfd'

if 'src' in os.getcwd() or 'script' in os.getcwd():

    GEN_FOLDER = os.path.join(os.path.abspath('../logs/'),'gen')
    RES_FOLDER = os.path.join(os.path.abspath('../logs/'),'log')
else:
    GEN_FOLDER = os.path.join(os.path.abspath('./logs/'), 'gen')
    RES_FOLDER = os.path.join(os.path.abspath('./logs/'), 'log')

DESIGN_LOG_NAME = os.path.join(RES_FOLDER ,"design_log.txt")
DESIGN_RESULTS_LOG_NAME = os.path.join(RES_FOLDER ,"design_log_results.txt")

INFORMATION_SHAPING_STRING = "infoshaping"

def get_modification_counter():
    MODIFICATIONS_COUNTER== MODIFICATIONS_COUNTER + 1
    return MODIFICATIONS_COUNTER


# Frontier Options
FIFO = 'FIFO'

SOLVER_PRP = 'PRP'
SOLVER_KP = 'KP'

def get_solver_type(solver_path):

    if 'prp' in solver_path.lower():
        return SOLVER_PRP
    elif 'replanner' in solver_path.lower():
        return SOLVER_KP


FF = 'ff'
if 'src' in os.getcwd():
    FF_PATH = os.path.abspath(os.path.join('../solvers/FF-v2.3'))
else: #run by script from the main folder
    FF_PATH = os.path.abspath(os.path.join('./solvers/FF-v2.3'))


FD = 'fd'
if 'src' in os.getcwd():
    FD_PATH =  os.path.abspath('../solvers/Fast-Downward')
else: #run by script from the main folder
    FD_PATH = os.path.abspath('./solvers/Fast-Downward')


current_planner = FD

def get_planner_path(planner_type):
    if FF in planner_type:
        return  FF_PATH
    elif FD in planner_type:
        return FD_PATH
    else:
        print('planner_type %s not supported'%planner_type)
        raise NotImplementedError


