import os
from pesp_instance import PESPInstance
from structures import *

__author__ = 'jetbrains'


def read_all_lines():
    BASE = "Lines/%s"
    all_lines_map = OrderedDict()
    file_names = os.listdir('Lines')
    for fn in file_names:
        f = open(BASE % fn).readlines()
        all_lines_map[fn] = map(lambda x: x.replace('\n', ''), f)

    return all_lines_map


ALL_LINES_DATA = read_all_lines()
GRAPHS = []

# ========================
# SET PARAMETERS
# ========================
T = 10  # time frequency

time_bounds = {"ub_path": 5, "lb_path": 3,  # time between stations
               "ub_station": 1, "lb_station": 0,  # time on station
               "ub_lines": 7, "lb_lines": 3}  # time between station of different lines

pesp_instance = PESPInstance(ALL_LINES_DATA, T, time_bounds)

# =============
# B&B method
# =============
#

# bb_model = pesp_instance.generate_bb_model()
#
# for v in bb_model.getVars():
#     print('%s %g' % (v.varName, v.x))
# print('Objective funtion: \t %g' % bb_model.objVal)

# ===================
# Genetic algorithm
# ===================

gen_model = pesp_instance.generate_genetic_model()


pass
