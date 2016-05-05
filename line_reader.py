from collections import OrderedDict
import os

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









