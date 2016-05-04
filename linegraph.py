from collections import OrderedDict

__author__ = 'galya'


class Vertex(object):
    def __init__(self, activity, station_name):
        self.activity = activity
        self.station_name = station_name

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)


class Edge(object):
    def __init__(self, vertex1, vertex2, lb, ub):
        self.ub = ub
        self.lb = lb
        self.vertex2 = vertex2
        self.vertex1 = vertex1

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
                and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)


class LineGraph(object):
    def __init__(self, graph_dict=OrderedDict()):
        """ initializes a graph object """
        self.__graph_dict = graph_dict

    def vertices(self):
        """ returns the vertices of a graph """
        return list(self.__graph_dict.keys())

    def edges(self):
        """ returns the edges of a graph """
        return self.__generate_edges()

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in
            self.__graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.
        :type vertex: Vertex
        """
        if vertex not in self.__graph_dict:
            self.__graph_dict[vertex] = []

    def add_edge(self, edge):
        """ assumes that edge is of type set, tuple or list;
            between two vertices can be multiple edges!

        :type edge: Edge
        """

        if edge.vertex1 in self.__graph_dict:
            self.__graph_dict[edge.vertex1].append(edge)
        else:
            self.__graph_dict[edge.vertex1] = [edge]

    def __generate_edges(self):
        """ A static method generating the edges of the
            graph "graph". Edges are represented as sets
            with one (a loop back to the vertex) or two
            vertices
        """
        edges = []
        for vertex in self.__graph_dict:
            for neighbour_edges in self.__graph_dict[vertex]:
                for neighbour_edge in neighbour_edges:
                    assert isinstance(neighbour_edge, Edge)
                    if neighbour_edge not in edges:
                        edges.append(neighbour_edge)
        return edges

    def __str__(self):
        res = "vertices: "
        for k in self.__graph_dict:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return res