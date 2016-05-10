from collections import OrderedDict

__author__ = 'galya'


class Vertex(object):
    def __init__(self, activity, station_name, line):
        self.line = line
        self.activity = activity
        self.station_name = station_name

    def __hash__(self):
        return hash((self.activity, self.station_name, self.line))

    def __eq__(self, other):
        return (self.activity, self.station_name, self.line) == (
            other.activity, other.station_name, other.line)

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        return "(%s,%s,%s)" % (self.line, self.activity, self.station_name)


class Edge(object):
    def __init__(self, vertex1, vertex2, lb, ub):
        self.ub = ub
        self.lb = lb
        self.vertex2 = vertex2
        self.vertex1 = vertex1

    def __hash__(self):
        return hash((self.ub, self.lb, self.vertex1, self.vertex2))

    def __eq__(self, other):
        return (self.ub, self.lb, self.vertex1, self.vertex2) == (other.ub, self.lb, other.vertex1, other.vertex2)

    def __ne__(self, other):
        return not (self == other)


class LineGraph(object):
    def __init__(self):
        """ initializes a graph object """
        self._graph_dict = OrderedDict()

    def vertices(self):
        """ returns the vertices of a graph """
        return list(self._graph_dict.keys())

    def edges(self):
        """ returns the edges of a graph """
        return self.generate_edges()

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in
            self.__graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.
        :type vertex: Vertex
        """
        if vertex not in self._graph_dict:
            self._graph_dict[vertex] = []

    def add_edge(self, edge):
        """ assumes that edge is of type set, tuple or list;
            between two vertices can be multiple edges!

        :type edge: Edge
        """

        if edge.vertex1 in self._graph_dict:
            self._graph_dict[edge.vertex1].append(edge)
        else:
            self._graph_dict[edge.vertex1] = [edge]

    def generate_edges(self):
        """ A static method generating the edges of the
            graph "graph". Edges are represented as sets
            with one (a loop back to the vertex) or two
            vertices
        """
        edges = []
        for vertex in self._graph_dict:
            for neighbour_edges in self._graph_dict[vertex]:
                if isinstance(neighbour_edges, Edge):
                    neighbour_edges = [neighbour_edges]
                for neighbour_edge in neighbour_edges:
                    assert isinstance(neighbour_edge, Edge)
                    # if neighbour_edge not in edges:
                    edges.append(neighbour_edge)
        return edges

    def __str__(self):
        res = "vertices: "
        for k in self._graph_dict:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.generate_edges():
            res += str(edge) + " "
        return res

    @property
    def graph_dict(self):
        return self._graph_dict


class Station():
    def __init__(self, station_name, station_data):
        self.station_name = station_name
        self.station_data = station_data


class Activity():
    DEPARTURE = "Departure"
    ARRIVAL = "Arrival"