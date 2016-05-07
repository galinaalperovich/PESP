__author__ = 'jetbrains'

from structures import *
from gurobipy import *
import itertools


def generate_model(graphs, T, lb_lines, ub_lines):
    """

    :type graphs: list of LineGraph
    """
    m = Model("PESP")
    j = 0
    objective_function = 0
    dict_of_variables = {}
    for graph in graphs:
        vertices = graph.vertices()
        edges = graph.edges()

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

    # between lines we need to add nes constrains
    for pair in itertools.combinations(graphs, r=2):
        graph1 = pair[0]
        graph2 = pair[1]

        j = 1
        for vert1, vert2 in zip(graph1.vertices(), graph2.vertices()):
            if vert1.station_name == vert2.station_name and vert1.activity == "Arrival" and vert2.activity == 'Departure':
                t1 = dict_of_variables[vert1]
                t2 = dict_of_variables[vert2]
                p = m.addVar(vtype=GRB.INTEGER, name="PB_" + str(j))
                m.update()
                constraint1 = m.addConstr(t2 - t1 + p * T <= ub_lines, "CB_%s_1" % str(j))
                constraint2 = m.addConstr(t2 - t1 + p * T >= lb_lines, "C_%s_2" % str(j))
                j += 1

    m.update()
    m.setObjective(objective_function, GRB.MINIMIZE)
    m.optimize()

    return m

