import random
import numpy
import sys
import time

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


def get_norm_matrix(matrix, vertex_var, all_edges):
    result_matrix = numpy.zeros(shape=(len(all_edges), len(vertex_var)))
    for edge_vertex, val in matrix.iteritems():
        edge = edge_vertex[0]
        vertex = edge_vertex[1]

        i_row = [i for i, x in enumerate(all_edges) if x == edge]
        i_col = [i for i, x in enumerate(vertex_var) if x == vertex]

        for i in i_row:
            for j in i_col:
                result_matrix[i, j] = val

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
                        objective_function += t2 - t1 + p * self.T - self.lb_lines
                        j += 1

        bb_model.update()
        bb_model.setObjective(objective_function, GRB.MINIMIZE)
        bb_model.optimize()

        return bb_model

    def _get_instance_for_gen(self):

        class GenInstance:
            def __init__(self, matrix, vertex_var, p_var, all_edges, edge_lb, edge_ub, idx_borders):
                self.p_mutation = 0.1
                self.idx_borders = idx_borders
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
                # p_var[edge] = "P_" + str(j)
                j += 1

        for pair in itertools.combinations(self.graphs, r=2):
            graph1 = pair[0]
            graph2 = pair[1]

            j = 1
            for vert1 in graph1.vertices():
                for vert2 in graph2.vertices():
                    if vert1.station_name == vert2.station_name and vert1.activity == "Arrival" and vert2.activity == 'Departure':
                        print vert1
                        print vert2
                        new_edge = Edge(vert1, vert2, self.lb_lines, self.ub_lines)
                        # p_var[new_edge] = "PB_" + str(j)
                        all_edges.append(new_edge)

                        # bb_model.addConstr(p <= 2, "CPE_%s_1" % str(j))
                        # bb_model.addConstr(p >= 0, "CPE_%s_2" % str(j))

                        matrix[new_edge, vert2] = 1
                        matrix[new_edge, vert1] = -1
                        edge_lb.append(self.lb_lines)
                        edge_ub.append(self.ub_lines)
                        # >= and <=
                        j += 1

        norm_matrix = get_norm_matrix(matrix, vertex_var, all_edges)

        # BORDERS OF LINES
        borders = []
        prev_line = None
        i = 0
        for v in vertex_var:
            line = v.line
            if prev_line is None:
                prev_line = line
                i += 1
                continue
            if line is not prev_line:
                borders.append(i)
                prev_line = line
            i += 1

        # A = (0, len(vertex_var) )
        A = (0, borders[0])
        B = (borders[0], borders[1])
        C = (borders[1], len(vertex_var))

        idx_borders = {1: A, 2: B, 3: C}
        # idx_borders = {1: A}

        return GenInstance(norm_matrix, vertex_var, p_var, all_edges, edge_lb, edge_ub, idx_borders)

    def generate_genetic_model(self):
        start_time = time.time()

        gen_structure = self.gen_structure

        # All variables
        v_len = len(self.gen_structure.vertex_var)
        p_len = len(self.gen_structure.p_var)
        N = v_len + p_len

        pop_num = 100
        num_of_random_el = 6
        max_num_step = 10000

        # ========================
        # I. Initial population for p and v
        # ========================
        parent_population_v = numpy.zeros(shape=(pop_num, v_len))
        parent_population_p = numpy.zeros(shape=(pop_num, p_len))

        for i in range(0, pop_num):
            # parent_population_v[i, :] = self._create_smart_random_solution(v_len, self.T)
            # parent_population_v[i, :] = self._create_random_solution(v_len, self.T)
            parent_population_v[i, :] = self._create_fixed_solution()
            # parent_population_p[i, :] = self._create_random_solution(p_len, 2)

        num_steps = 0
        min_objective = sys.maxint
        while num_steps < max_num_step:

            # ===========================
            # II. Offspring population: uniform crossover of two vectors + mutation
            # ===========================

            offspring_population_v = numpy.zeros(shape=(pop_num, v_len))
            offspring_population_p = numpy.zeros(shape=(pop_num, p_len))

            i = 0
            final_population = None
            # crossover = "uniform"
            # crossover = "cross"
            crossover = "part"
            while i < pop_num:
                parent_population = numpy.hstack((parent_population_v, parent_population_p))
                parent_1 = self._get_parent(parent_population, num_of_random_el)
                parent_2 = self._get_parent(parent_population, num_of_random_el)

                parent_v_1 = parent_1[:v_len]
                parent_v_2 = parent_2[:v_len]

                parent_p_1 = parent_1[v_len:]
                parent_p_2 = parent_2[v_len:]

                if crossover == "cross":
                    offspring_v_1, offspring_v_2 = self._generate_offspring_cross(parent_v_1, parent_v_2, self.T)
                    # offspring_p_1, offspring_p_2 = self._generate_offspring_cross(parent_p_1, parent_p_2, 2)

                    offspring_population_v[i, :] = offspring_v_1
                    offspring_population_v[i + 1, :] = offspring_v_2

                    # offspring_population_p[i, :] = offspring_p_1
                    # offspring_population_p[i + 1, :] = offspring_p_2

                    i += 2
                elif crossover == "uniform":
                    offspring_v_1 = self._generate_offspring_uniform(parent_v_1, parent_v_2, self.T)
                    offspring_p_1 = self._generate_offspring_uniform(parent_p_1, parent_p_2, 2)

                    offspring_population_v[i, :] = offspring_v_1
                    offspring_population_p[i, :] = offspring_p_1
                    i += 1

                elif crossover == "part":
                    offspring_v_1 = self._generate_offspring_uniform_part(parent_v_1, parent_v_2, self.T)
                    # offspring_p_1 = self._generate_offspring_uniform_part(parent_p_1, parent_p_2, 2)

                    offspring_population_v[i, :] = offspring_v_1
                    # offspring_population_p[i, :] = offspring_p_1
                    i += 1

            offspring_population = numpy.hstack((offspring_population_v, offspring_population_p))
            obj_func_value = numpy.apply_along_axis(self._objective_function, axis=1, arr=offspring_population)
            min_objective = min(obj_func_value)
            print min_objective

            if abs(min_objective) < 5:
                final_population = offspring_population
                argmin_obj_value = obj_func_value.argmin()
                break

            parent_population_v = offspring_population_v
            parent_population_p = offspring_population_p
            num_steps += 1

        vp_final = final_population[argmin_obj_value]
        vertex_var = gen_structure.vertex_var
        p_var = gen_structure.p_var

        v_final = vp_final[:v_len]
        p_final = vp_final[v_len:]
        print "Objective function: " + str(min_objective)
        print("--- %s seconds ---" % (time.time() - start_time))
        for vertex, val in zip(vertex_var.values(), v_final.tolist()):
            print vertex + "\t" + str(val)

        print "\n"

        for vertex, val in zip(p_var.values(), p_final.tolist()):
            print vertex + "\t" + str(val)
        pass

    def _objective_function(self, v):
        matrix = self.gen_structure.matrix
        v_len = len(self.gen_structure.vertex_var)
        p_len = len(self.gen_structure.p_var)
        e = len(self.gen_structure.edge_lb)

        matrix2 = numpy.zeros(shape=(e, p_len))
        all_matrix = numpy.hstack((matrix, matrix2))
        p = v[v_len:]

        div = divmod(numpy.dot(all_matrix, v), self.T)
        result = sum(abs(div[1] - self.gen_structure.edge_lb))
        # result = sum(numpy.dot(all_matrix, v) - self.gen_structure.edge_lb + self.T * p)
        # + abs(self.gen_structure.edge_ub - numpy.dot(all_matrix, v) + self.T * p))
        return result

    def _create_random_solution(self, n, ub):
        return map(lambda x: random.randrange(0, ub), range(0, n))


    def _mutation_gene(self, p, ub):
        prob = self.gen_structure.p_mutation
        coin = random.random()
        if coin <= prob:
            return random.randrange(0, ub)
        else:
            return p

    def _mutation_offspring(self, p, ub):
        prob = self.gen_structure.p_mutation
        offspring = []
        for p_i in p:
            coin = random.random()
            if coin <= prob:
                offspring.append(random.randrange(0, ub))
            else:
                offspring.append(p_i)
        return offspring

    def _mutation_fixed(self, p, ub):
        prob = self.gen_structure.p_mutation
        coin = random.random()
        if coin <= prob:
            delta = random.randrange(0, ub)
            p += delta
            p = divmod(p, self.T)[1]


        return p


    def _generate_offspring_uniform(self, parent_1, parent_2, ub_mutation):
        # Uniform crossover + mutation
        offspring = []
        for p1, p2 in zip(parent_1, parent_2):
            val = random.randrange(0, 2)
            if val == 0:
                offspring.append(self._mutation_gene(p1, ub_mutation))
            else:
                offspring.append(self._mutation_gene(p2, ub_mutation))

        return offspring


    def _generate_offspring_uniform_part(self, parent_1, parent_2, ub_mutation):
        # Uniform crossover + mutation
        offspring = []
        vertex_var = self.gen_structure.vertex_var.keys()

        idx = self.gen_structure.idx_borders

        for i, coords in idx.iteritems():
            val = random.randrange(0, 2)
            if val == 0:
                # offspring += self._mutation_offspring(parent_1[coords[0]:coords[1]], ub_mutation)
                offspring += self._mutation_fixed(parent_1[coords[0]:coords[1]], ub_mutation).tolist()
            else:
                # offspring += self._mutation_offspring(parent_2[coords[0]:coords[1]], ub_mutation)
                offspring += self._mutation_fixed(parent_2[coords[0]:coords[1]], ub_mutation).tolist()

        return offspring


    def _generate_offspring_cross(self, parent_1, parent_2, ub):
        # Crossover + mutation
        i = random.randrange(0, len(parent_1))
        offspring_1 = self._mutation_offspring(numpy.hstack((parent_1[:i], parent_2[i:])), ub)
        offspring_2 = self._mutation_offspring(numpy.hstack((parent_2[:i], parent_1[i:])), ub)

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

    def _create_fixed_solution(self):
        v_result = []
        gen_stucture = self.gen_structure
        all_edges = gen_stucture.all_edges
        vertexes = gen_stucture.vertex_var

        t = random.randrange(0, self.T)
        v_result.append(t)
        prev_line = None
        for vertex in vertexes:
            if prev_line is None:
                prev_line = vertex.line
            line = vertex.line

            if line is not prev_line:
                t = random.randrange(0, self.T)
                v_result.append(t)
                prev_line = line
                continue
            vert_idx = vertexes.keys().index(vertex)
            eddges_idx = numpy.nonzero(gen_stucture.matrix[:, vert_idx])
            edge = None
            for edge_ind in eddges_idx[0]:
                edge = all_edges[edge_ind]
                if edge.vertex1 == vertex and edge.vertex1.line == edge.vertex2.line:
                    break

            t = divmod(t + edge.lb, self.T)[1]
            v_result.append(t)
            prev_line = line
        return v_result[:-1]

    def _create_smart_random_solution(self, v_len, T):
        v_var_result = []
        gen_stucture = self.gen_structure
        all_edges = gen_stucture.all_edges
        vertexes = gen_stucture.vertex_var
        idx = gen_stucture.idx_borders

        t = 0
        prev_line = None
        for vertex in vertexes:
            if prev_line is None:
                prev_line = vertex.line
            line = vertex.line

            if line is not prev_line:
                t = 0

            vert_idx = vertexes.keys().index(vertex)
            eddges_idx = numpy.nonzero(gen_stucture.matrix[:, vert_idx])
            edge = None
            for edge_ind in eddges_idx[0]:
                edge = all_edges[edge_ind]
                if edge.vertex1 == vertex:
                    continue

            v_var_result.append(t)
            rand_time = random.choice([edge.lb, edge.ub])
            t = divmod(t + rand_time, self.T)[1]
            prev_line = line

        return v_var_result




