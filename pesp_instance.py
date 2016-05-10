import numpy

__author__ = 'jetbrains'

from structures import *
from gurobipy import *
import itertools


def _get_norm_matrix(matrix):
    """

    Method returens normal numpy matrix instead of Vertex -> Edge matrix
    :param matrix: OrderedDict()
    """

    def add_to(list1, list2, el):
        el0 = el[0]
        if el0 not in list1:
            list1.append(el0)
        el1 = el[1]
        if el1 not in list2:
            list2.append(el1)


    edges_vertexes = list(set(matrix.keys()))  # array of vertexes
    vertexes = []
    edges = []
    map(lambda x: add_to(edges, vertexes, x), edges_vertexes )

    result_matrix = numpy.zeros(shape=(len(edges), len(vertexes)))
    for edge_vertex, val in matrix.iteritems():
        edge = edge_vertex[0]
        vertex = edge_vertex[1]

        result_matrix[edges.index(edge), vertexes.index(vertex)] = val

    return result_matrix


class PESPInstance:
    def __init__(self, all_lines_data, T, time_bounds):
        self.ub_lines = time_bounds['ub_lines']
        self.lb_lines = time_bounds['lb_lines']

        self.ub_path = time_bounds['ub_path']
        self.lb_path = time_bounds['lb_path']

        self.ub_station = time_bounds['ub_station']
        self.lb_station = time_bounds['lb_station']
        self.T = T
        self.graphs = self.generate_graphs(all_lines_data)


    def generate_graphs(self, all_lines_data=None):
        graphs = []

        for line, line_val in all_lines_data.iteritems():
            first = True
            prev_station = None
            next_station = None  # we have ordered files, we will process it as one
            prev_vertex2 = None
            read_first_station = True
            graph = LineGraph()
            line_data = all_lines_data[line]
            for station_name in line_data:
                if first is True:
                    first = False
                    prev_station = station_name

                else:
                    next_station = station_name
                    # generate two vertexes and one edge
                    vertex1 = Vertex(Activity.DEPARTURE, prev_station, line)
                    vertex2 = Vertex(Activity.ARRIVAL, next_station, line)

                    edge = Edge(vertex1, vertex2, self.lb_path, self.ub_path)  # to test only
                    graph.add_vertex(vertex1)
                    graph.add_vertex(vertex2)
                    graph.add_edge(edge)
                    prev_station = next_station

                    if prev_vertex2 is None:
                        prev_vertex2 = vertex2
                        continue
                    else:
                        prev_edge = Edge(prev_vertex2, vertex1, self.lb_station, self.ub_station)
                        graph.add_edge(prev_edge)
                        prev_vertex2 = vertex2

            graphs.append(graph)
        return graphs


    def generate_bb_model(self):
        """
        :type graphs: list of LineGraph
        """
        bb_model = Model("PESP")
        j = 0
        objective_function = 0
        dict_of_variables = {}
        for graph in self.graphs:
            vertices = graph.vertices()
            edges = graph.edges()

            for edge in edges:
                vertex1 = edge.vertex1
                vertex2 = edge.vertex2
                lin_expr_for_edge = 0
                p = bb_model.addVar(vtype=GRB.INTEGER, name="P_" + str(j))
                bb_model.update()
                for vertex in vertices:
                    if vertex in dict_of_variables.keys():
                        t = dict_of_variables[vertex]
                    else:
                        t = bb_model.addVar(vtype=GRB.INTEGER, name=str(vertex))
                        dict_of_variables[vertex] = t
                        bb_model.update()
                        bb_model.addConstr(t <= self.T - 1, "Ct_%s_1" % str(j))
                        bb_model.addConstr(t >= 0, "Ct_%s_2" % str(j))

                    val = 0
                    if vertex == vertex1:
                        val = -1
                    elif vertex == vertex2:
                        val = 1

                    lin_expr_for_edge += val * t

                bb_model.addConstr(lin_expr_for_edge + p * self.T <= edge.ub, "C_%s_1" % str(j))
                bb_model.addConstr(lin_expr_for_edge + p * self.T >= edge.lb, "C_%s_2" % str(j))
                objective_function += lin_expr_for_edge + p * self.T - edge.lb
                j += 1

        # between lines we need to add nes constrains
        for pair in itertools.combinations(self.graphs, r=2):
            graph1 = pair[0]
            graph2 = pair[1]

            j = 1
            for vert1, vert2 in zip(graph1.vertices(), graph2.vertices()):
                if vert1.station_name == vert2.station_name and vert1.activity == "Arrival" and vert2.activity == 'Departure':
                    t1 = dict_of_variables[vert1]
                    t2 = dict_of_variables[vert2]
                    p = bb_model.addVar(vtype=GRB.INTEGER, name="PB_" + str(j))
                    bb_model.update()
                    # >= and <=
                    bb_model.addConstr(t2 - t1 + p * self.T <= self.ub_lines, "CB_%s_1" % str(j))
                    bb_model.addConstr(t2 - t1 + p * self.T >= self.lb_lines, "C_%s_2" % str(j))
                    j += 1

        bb_model.update()
        bb_model.setObjective(objective_function, GRB.MINIMIZE)
        bb_model.optimize()

        return bb_model

    def get_instance_for_gen(self):

        class GenInstance:
            def __init__(self, matrix, vertex_var, p_var, all_edges, edge_lb, edge_ub):
                self.edge_ub = edge_ub
                self.edge_lb = edge_lb
                self.all_edges = all_edges
                self.p_var = p_var
                self.vertex_var = vertex_var
                self.matrix = matrix

        vertex_var = OrderedDict()
        p_var = OrderedDict()
        matrix = {}
        j = 0
        m = 0
        edge_lb = []
        edge_ub = []
        all_edges = []

        for graph in self.graphs:
            vertices = graph.vertices()
            edges = graph.edges()
            for edge in edges:
                vertex1 = edge.vertex1
                vertex2 = edge.vertex2

                for vertex in vertices:
                    if vertex not in vertex_var:
                        t = str(vertex)
                        vertex_var[vertex] = t

                    if vertex == vertex1:
                        matrix[edge, vertex] = -1
                    elif vertex == vertex2:
                        matrix[edge, vertex] = 1

                    m += 1

                edge_lb.append(edge.lb)
                edge_ub.append(edge.ub)
                all_edges.append(edge)
                p_var[edge] = "P_" + str(j)
                j += 1

        for pair in itertools.combinations(self.graphs, r=2):
            graph1 = pair[0]
            graph2 = pair[1]

            j = 1
            for vert1, vert2 in zip(graph1.vertices(), graph2.vertices()):
                if vert1.station_name == vert2.station_name and vert1.activity == "Arrival" and vert2.activity == 'Departure':
                    new_edge = Edge(vert1, vert2, self.lb_lines, self.ub_lines)
                    p_var[new_edge] = "PB_" + str(j)
                    matrix[edge, vert2] = 1
                    matrix[edge, vert1] = -1
                    edge_lb.append(self.lb_lines)
                    edge_ub.append(self.ub_lines)
                    # >= and <=
                    j += 1

        norm_matrix = _get_norm_matrix(matrix)
        return GenInstance(norm_matrix, vertex_var, p_var, all_edges, edge_lb, edge_ub)

        # def generate_genetic_model(self):
        # variables = []
        # j = 0
        # objective_function = 0
        # dict_of_variables = {}
        # for graph in self.graphs:
        # vertices = graph.vertices()
        #         edges = graph.edges()
        #
        #         for edge in edges:
        #             vertex1 = edge.vertex1
        #             vertex2 = edge.vertex2
        #             lin_expr_for_edge = 0
        #             p = bb_model.addVar(vtype=GRB.INTEGER, name="P_" + str(j))
        #             bb_model.update()
        #             for vertex in vertices:
        #                 if vertex in dict_of_variables.keys():
        #                     t = dict_of_variables[vertex]
        #                 else:
        #                     t = bb_model.addVar(vtype=GRB.INTEGER, name=str(vertex))
        #                     dict_of_variables[vertex] = t
        #                     bb_model.update()
        #                     bb_model.addConstr(t <= self.T - 1, "Ct_%s_1" % str(j))
        #                     bb_model.addConstr(t >= 0, "Ct_%s_2" % str(j))
        #
        #                 val = 0
        #                 if vertex == vertex1:
        #                     val = -1
        #                 elif vertex == vertex2:
        #                     val = 1
        #
        #                 lin_expr_for_edge += val * t
        #
        #             bb_model.addConstr(lin_expr_for_edge + p * self.T <= edge.ub, "C_%s_1" % str(j))
        #             bb_model.addConstr(lin_expr_for_edge + p * self.T >= edge.lb, "C_%s_2" % str(j))
        #             objective_function += lin_expr_for_edge + p * self.T - edge.lb
        #             j += 1
        #
        #     # between lines we need to add nes constrains
        #     for pair in itertools.combinations(self.graphs, r=2):
        #         graph1 = pair[0]
        #         graph2 = pair[1]
        #
        #         j = 1
        #         for vert1, vert2 in zip(graph1.vertices(), graph2.vertices()):
        #             if vert1.station_name == vert2.station_name and vert1.activity == "Arrival" and vert2.activity == 'Departure':
        #                 t1 = dict_of_variables[vert1]
        #                 t2 = dict_of_variables[vert2]
        #                 p = bb_model.addVar(vtype=GRB.INTEGER, name="PB_" + str(j))
        #                 bb_model.update()
        #                 # >= and <=
        #                 bb_model.addConstr(t2 - t1 + p * self.T <= self.ub_lines, "CB_%s_1" % str(j))
        #                 bb_model.addConstr(t2 - t1 + p * self.T >= self.lb_lines, "C_%s_2" % str(j))
        #                 j += 1
        #
        #     number_of_variables = 0
        #
        #     pass