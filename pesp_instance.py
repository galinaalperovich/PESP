import random
import numpy

__author__ = 'jetbrains'

from structures import *
from gurobipy import *
import itertools


def add_to(list1, list2, el):
    el0 = el[0]
    if el0 not in list1:
        list1.append(el0)
    el1 = el[1]
    if el1 not in list2:
        list2.append(el1)


def get_norm_matrix(matrix):
    edges_vertexes = list(set(matrix.keys()))  # array of vertexes
    vertexes = []
    edges = []
    map(lambda x: add_to(edges, vertexes, x), edges_vertexes)

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
        self.graphs = self._generate_graphs(all_lines_data)
        self.gen_structure = self._get_instance_for_gen()


    def _generate_graphs(self, all_lines_data=None):
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
                bb_model.addConstr(p <= 2, "CP_%s_1" % str(j))
                bb_model.addConstr(p >= 0, "CP_%s_2" % str(j))

                for vertex in vertices:
                    if vertex in dict_of_variables.keys():
                        t = dict_of_variables[vertex]
                    else:
                        t = bb_model.addVar(vtype=GRB.INTEGER, name=str(vertex))
                        dict_of_variables[vertex] = t
                        bb_model.update()
                        bb_model.addConstr(t <= self.T - 1, "CT_%s_1" % str(j))
                        bb_model.addConstr(t >= 0, "CT_%s_2" % str(j))

                    val = 0
                    if vertex == vertex1:
                        val = -1
                    elif vertex == vertex2:
                        val = 1

                    lin_expr_for_edge += val * t

                bb_model.addConstr(lin_expr_for_edge + p * self.T <= edge.ub, "CE_%s_1" % str(j))
                bb_model.addConstr(lin_expr_for_edge + p * self.T >= edge.lb, "CE_%s_2" % str(j))
                objective_function += lin_expr_for_edge + p * self.T - edge.lb
                j += 1

        # between lines we need to add nes constrains
        for pair in itertools.combinations(self.graphs, r=2):
            graph1 = pair[0]
            graph2 = pair[1]

            j = 1
            for vert1 in graph1.vertices():
                for vert2 in graph2.vertices():
                    if vert1.station_name == vert2.station_name and vert1.activity == "Arrival" and vert2.activity == 'Departure':
                        t1 = dict_of_variables[vert1]
                        t2 = dict_of_variables[vert2]
                        p = bb_model.addVar(vtype=GRB.INTEGER, name="PB_" + str(j))
                        bb_model.update()
                        bb_model.addConstr(p <= 2, "CPE_%s_1" % str(j))
                        bb_model.addConstr(p >= 0, "CPE_%s_2" % str(j))
                        # >= and <=
                        bb_model.addConstr(t2 - t1 + p * self.T <= self.ub_lines, "CB_%s_1" % str(j))
                        bb_model.addConstr(t2 - t1 + p * self.T >= self.lb_lines, "C_%s_2" % str(j))
                        j += 1

        bb_model.update()
        bb_model.setObjective(objective_function, GRB.MINIMIZE)
        bb_model.optimize()

        return bb_model

    def _get_instance_for_gen(self):

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
            for vert1 in graph1.vertices():
                for vert2 in graph2.vertices():
                    if vert1.station_name == vert2.station_name and vert1.activity == "Arrival" and vert2.activity == 'Departure':
                        new_edge = Edge(vert1, vert2, self.lb_lines, self.ub_lines)
                        p_var[new_edge] = "PB_" + str(j)
                        matrix[edge, vert2] = 1
                        matrix[edge, vert1] = -1
                        edge_lb.append(self.lb_lines)
                        edge_ub.append(self.ub_lines)
                        # >= and <=
                        j += 1

        norm_matrix = get_norm_matrix(matrix)
        return GenInstance(norm_matrix, vertex_var, p_var, all_edges, edge_lb, edge_ub)

    def generate_genetic_model(self):
        gen_structure = self.gen_structure
        n = len(self.gen_structure.vertex_var)
        pop_num = 300

        max_num_step = 1000

        # ========================
        # I. Initial population
        # ========================
        parent_population = numpy.zeros(shape=(pop_num, n))
        for i in range(0, pop_num):
            parent_population[i, :] = self._create_random_solution(n)

        num_steps = 0
        while num_steps < max_num_step:

            # ===========================
            # III. Offspring population: uniform crossover of two vectors + mutation
            # ===========================

            offspring_population = numpy.zeros(shape=(pop_num, n))
            i = 0
            while i < pop_num:
                p1 = self._get_parent(parent_population, 5)
                p2 = self._get_parent(parent_population, 5)

                # p1 = parent_population[random.randrange(0, pop_num), :]
                # p2 = parent_population[random.randrange(0, pop_num), :]
                offspring_1, offspring_2 = self._generate_offspring_cross(p1, p2)
                offspring_population[i, :] = offspring_1
                offspring_population[i + 1, :] = offspring_2
                i += 2

            # population = numpy.vstack((parent_population, offspring_population))
            obj_func_value = numpy.apply_along_axis(self._objective_function, axis=1, arr=offspring_population)
            # parent_population_idx = [x for (y, x) in sorted(zip(obj_func_value, range(0, population.shape[0])))]
            # parent_population = population[parent_population_idx[:pop_num], :]
            print min(obj_func_value)

            parent_population = offspring_population
            num_steps += 1

        pass

    def _objective_function(self, v):
        matrix = self.gen_structure.matrix

        return abs(sum(numpy.dot(matrix, v) - self.gen_structure.edge_lb))

    def _create_random_solution(self, n):
        return map(lambda x: random.randrange(0, self.T), range(0, n))


    def _mutation_gene(self, p):
        prob = 0.05
        coin = random.random()
        if coin <= prob:
            return random.randrange(0, self.T)
        else:
            return p

    def _mutation_offspring(self, p):
        prob = 0.05
        offspring = []
        for p_i in p:
            coin = random.random()
            if coin <= prob:
                offspring.append(random.randrange(0, self.T))
            else:
                offspring.append(p_i)
        return offspring


    def _generate_offspring_uniform(self, parent_1, parent_2):
        # Uniform crossover + mutation
        offspring = []
        for p1, p2 in zip(parent_1, parent_2):
            val = random.randrange(0, 2)
            if val == 0:
                offspring.append(self._mutation_gene(p1))
            else:
                offspring.append(self._mutation_gene(p2))
        return offspring


    def _generate_offspring_cross(self, parent_1, parent_2):
        # Crossover + mutation
        offspring = []
        i = random.randrange(0, len(parent_1))
        offspring_1 = self._mutation_offspring(numpy.hstack((parent_1[:i], parent_2[i:])))
        offspring_2 = self._mutation_offspring(numpy.hstack((parent_2[:i], parent_1[i:])))

        return offspring_1, offspring_2

    def _get_parent(self, population, k):
        n = population.shape[0]
        idx = []
        f = []
        for i in range(0, k):
            i1 = random.randrange(0, n)
            idx.append(i1)
            f.append(self._objective_function(population[i1, :]))

        sorted_p = [x for (y, x) in sorted(zip(f, idx))]

        return population[sorted_p[0], :]




