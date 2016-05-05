__author__ = 'jetbrains'

from structures import *
from gurobipy import *


def generate_model(graph):
    """
    :type graph: LineGraph
    """
    T = 30
    vertices = graph.vertices()
    edges = graph.edges()

    m = Model("PESP")
    j = 0

    dict_of_variables = {}
    objective_function = 0
    for edge in edges:
        vertex1 = edge.vertex1
        vertex2 = edge.vertex2
        lin_expr_for_edge = 0
        p = m.addVar(vtype=GRB.INTEGER, name="P_" + str(j))
        m.update()
        for vertex in vertices:
            if vertex in dict_of_variables.keys():
                t = dict_of_variables[vertex]
            else:
                t = m.addVar(vtype=GRB.INTEGER, name=str(vertex))
                dict_of_variables[vertex] = t
                m.update()

            val = 0
            if vertex == vertex1:
                val = -1
            elif vertex == vertex2:
                val = 1

            lin_expr_for_edge += val * t

        constraint1 = m.addConstr(lin_expr_for_edge + p * T <= edge.ub, "C_%s_1" % str(j))
        constraint2 = m.addConstr(lin_expr_for_edge + p * T >= edge.lb, "C_%s_2" % str(j))
        objective_function += lin_expr_for_edge + p * T - edge.lb
        j += 1

    m.setObjective(objective_function, GRB.MINIMIZE)
    m.optimize()

    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))
    print('Obj: %g' % m.objVal)

