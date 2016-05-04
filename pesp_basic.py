from file_reader import one_line_dict, read_all_lines

__author__ = 'galya'

#!/usr/bin/python

from gurobipy import *


#
# PESP basic version page 20, Models for Periodic Timetabeling
#

# one_line_dict('/Users/galya/PycharmProjects/PESPtask/Lines/0102111..txt')


try:

    # Create a new model
    m = Model("mip1")

    T = 60
    # Create variables
    t1 = m.addVar(vtype=GRB.INTEGER, name="t1")
    t2 = m.addVar(vtype=GRB.INTEGER, name="t2")
    t3 = m.addVar(vtype=GRB.INTEGER, name="t3")
    t4 = m.addVar(vtype=GRB.INTEGER, name="t4")

    p21 = m.addVar(vtype=GRB.INTEGER, name="p21")
    p23 = m.addVar(vtype=GRB.INTEGER, name="p23")
    p34 = m.addVar(vtype=GRB.INTEGER, name="p34")
    p41 = m.addVar(vtype=GRB.INTEGER, name="p41")

    # Integrate new variables
    m.update()

    # Set objective

    m.setObjective(
        (t1 - t2 + p21 * T - 6) +
        (t3 - t2 + p23 * T - 2) +
        (t4 - t3 + p34 * T - 2) +
        (t1 - t4 + p41 * T - 5),
        GRB.MINIMIZE)

    # Add constraint: x + 2 y + 3 z <= 4
    m.addConstr(t1 - t2 + p21 * T <= 9, "c11")
    m.addConstr(t1 - t2 + p21 * T >= 6, "c12")

    m.addConstr(t3 - t2 + p23 * T <= 6, "c21")
    m.addConstr(t3 - t2 + p23 * T >= 2, "c22")

    m.addConstr(t4 - t3 + p34 * T <= 5, "c31")
    m.addConstr(t4 - t3 + p34 * T >= 2, "c32")

    m.addConstr(t1 - t4 + p41 * T <= 6, "c41")
    m.addConstr(t1 - t4 + p41 * T >= 5, "c42")

    m.addConstr(t1 <= T, "c51")
    m.addConstr(t1 >= 0, "c52")

    m.addConstr(t2 <= T, "c61")
    m.addConstr(t2 >= 0, "c62")

    m.addConstr(t3 <= T, "c71")
    m.addConstr(t3 >= 0, "c72")

    m.addConstr(t4 <= T, "c81")
    m.addConstr(t4 >= 0, "c82")

    m.optimize()

    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))

    print('Obj: %g' % m.objVal)

except GurobiError:
    print('Error reported')