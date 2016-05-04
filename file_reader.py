from collections import OrderedDict
import os
from linegraph import LineGraph, Edge, Vertex

__author__ = 'galya'

BASE = "Lines/%s"


def read_all_lines():
    all_lines_map = OrderedDict()
    line_number = "Line_%s"
    l = 1
    file_names = os.listdir('Lines')
    for fn in file_names:
        all_lines_map[line_number % l] = one_line_dict(fn)
        l += 1

    return all_lines_map


def one_line_dict(filename):
    """
    :rtype : OrderedDict() of OrderedDict
    """
    line_dict = OrderedDict()
    f = open(BASE % filename).readlines()
    for line in f[5:]:
        line_value = line.split('#')
        station_name = line_value[0]

        arrival_time = line_value[1]
        departure_time = line_value[2]

        km_from_start = int((line_value[3]))
        line_dict[station_name] = {'arrival_time': arrival_time,
                                   'departure_time': departure_time,
                                   'km_from_start': km_from_start}

    return line_dict


ALL_LINES_DATA = read_all_lines()
GRAPH = LineGraph()


class Station():
    def __init__(self, station_name, station_data):
        self.station_name = station_name
        self.station_data = station_data


class Activity():
    DEPARTURE = "Departure"
    ARRIVAL = "Arrival"


j = 0
first = True
prev_station = None
next_station = None # we have ordered files, we will process it as one
for line, line_val in ALL_LINES_DATA.iteritems():
    line_data = ALL_LINES_DATA[line]
    for station_name, station_data in line_data.iteritems():
        if first is True:
            first = False
            prev_station = Station(station_name, station_data)
        else:
            next_station = Station(station_name, station_data)
            # generate two vertexes and one edge
            vertex1 = Vertex(Activity.DEPARTURE, prev_station.station_name)
            vertex2 = Vertex(Activity.ARRIVAL, next_station.station_name)

            # consider different cases DD, DA, AD for ub and lb
            edge = Edge(vertex1, vertex2, 2, 10)  # to test only
            GRAPH.add_vertex(vertex1)
            GRAPH.add_vertex(vertex2)
            GRAPH.add_edge(edge)
            prev_station = next_station
    j += 1
    if j == 2:
        break

pass

GRAPH.vertices()






