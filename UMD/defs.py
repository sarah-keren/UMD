import os 

SEPARATOR = ' '

FAILURE_STRING = 'FAIL'
NA = 'NA'
ITER_LIMIT = 1000
DEFAULT_TIME_LIMIT = 1800
MODIFICATIONS_COUNTER = 0
DEFAULT_MAX_HORIZON = 1000

INFINITE = 8888

BFD = 'bfd'


# Frontier Options
FIFO = 'FIFO'


if 'src' in os.getcwd() or 'script' in os.getcwd():

    GEN_FOLDER = os.path.join(os.path.abspath('../logs/'),'gen')
    RES_FOLDER = os.path.join(os.path.abspath('../logs/'),'log')
else:
    GEN_FOLDER = os.path.join(os.path.abspath('./logs/'), 'gen')
    RES_FOLDER = os.path.join(os.path.abspath('./logs/'), 'log')

DESIGN_LOG_NAME = os.path.join(RES_FOLDER ,"design_log.txt")
DESIGN_RESULTS_LOG_NAME = os.path.join(RES_FOLDER ,"design_log_results.txt")


def get_modification_counter():
    MODIFICATIONS_COUNTER== MODIFICATIONS_COUNTER + 1
    return MODIFICATIONS_COUNTER




