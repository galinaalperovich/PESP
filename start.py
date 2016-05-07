import os
from model_generator import generate_model
from structures import *

__author__ = 'jetbrains'

def read_all_lines():
    BASE = "Lines/%s"
    all_lines_map = OrderedDict()
    file_names = os.listdir('Lines')
    for fn in file_names:
        f = open(BASE % fn).readlines()
        all_lines_map[fn] = map(lambda x: x.replace('\n', ''), f)

    return all_lines_map

ALL_LINES_DATA = read_all_lines()
GRAPHS = []

# ========================
# SET PARAMETERS
# ========================
# time frequency
T = 10

# time between stations
ub_path = 4
lb_path = 2

#time on station
ub_station = 1
lb_station = 0

#time between station of differnet lines
ub_lines = 3
lb_lines = 7

for line, line_val in ALL_LINES_DATA.iteritems():
    first = True
    prev_station = None
    next_station = None  # we have ordered files, we will process it as one
    prev_vertex2 = None
    read_first_station = True
    GRAPH = LineGraph()
    line_data = ALL_LINES_DATA[line]
    for station_name in line_data:
        if first is True:
            first = False
            prev_station = station_name

        else:
            next_station = station_name
            # generate two vertexes and one edge
            vertex1 = Vertex(Activity.DEPARTURE, prev_station, line)
            vertex2 = Vertex(Activity.ARRIVAL, next_station, line)

            edge = Edge(vertex1, vertex2, lb_path, ub_path)  # to test only
            GRAPH.add_vertex(vertex1)
            GRAPH.add_vertex(vertex2)
            GRAPH.add_edge(edge)
            prev_station = next_station

            if prev_vertex2 is None:
                prev_vertex2 = vertex2
                continue
            else:
                prev_edge = Edge(prev_vertex2, vertex1, lb_station, ub_station)
                GRAPH.add_edge(prev_edge)
                prev_vertex2 = vertex2

    GRAPHS.append(GRAPH)
pass

model = generate_model(GRAPHS, T, lb_lines, ub_lines)

for v in model.getVars():
    print('%s %g' % (v.varName, v.x))
print('Obj: %g' % model.objVal)