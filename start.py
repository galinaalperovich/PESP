from line_reader import *
from model_generator import generate_model
from structures import *

__author__ = 'jetbrains'

ALL_LINES_DATA = read_all_lines()
GRAPH = LineGraph()

direction_value = 0
direction = {0: "normal", 1: "reverse"}
first = True
prev_station = None
next_station = None  # we have ordered files, we will process it as one
prev_vertex2 = None
read_first_station = True

for line, line_val in ALL_LINES_DATA.iteritems():
    line_data = ALL_LINES_DATA[line]
    for station_name, station_data in line_data.iteritems():

        # don't read first line of the second file
        if read_first_station is False:
            read_first_station = True
            continue
        else:
            if first is True:
                first = False
                prev_station = Station(station_name, station_data)

            else:
                next_station = Station(station_name, station_data)
                # generate two vertexes and one edge
                vertex1 = Vertex(Activity.DEPARTURE, prev_station.station_name, direction[direction_value])
                vertex2 = Vertex(Activity.ARRIVAL, next_station.station_name, direction[direction_value])

                # consider different cases DD, DA, AD for ub and lb
                edge = Edge(vertex1, vertex2, 5, 10)  # to test only
                GRAPH.add_vertex(vertex1)
                GRAPH.add_vertex(vertex2)
                GRAPH.add_edge(edge)
                prev_station = next_station

                if prev_vertex2 is None:
                    prev_vertex2 = vertex2
                    continue
                else:
                    prev_edge = Edge(prev_vertex2, vertex1, 0, 1)
                    GRAPH.add_edge(prev_edge)
                    prev_vertex2 = vertex2

    direction_value += 1
    read_first_station = False
    if direction_value == 2:
        break

pass

GRAPH.edges()
generate_model(GRAPH)
pass