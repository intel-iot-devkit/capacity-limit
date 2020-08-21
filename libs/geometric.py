"""
Copyright (C) 2020 Intel Corporation

SPDX-License-Identifier: BSD-3-Clause
"""

from shapely.geometry import LineString, Point
from collections import deque


def get_line(data):
    return LineString(data)


def get_point(data):
    return Point(data)


def get_ax_b(p1, p2):
    run = (p2[0] - p1[0]) * 1.0
    rise = (p2[1] - p1[1]) * 1.0
    try:
        a = rise / run
    except ZeroDivisionError:
        a = 9999999
    b = p1[1] - (a * p1[0])
    return {"a": a, "b": b}


def get_perpendicular_coords(p1, p2):
    l = get_line([p1, p2])
    c = list(l.centroid.coords)[0]
    run = (p2[0] - p1[0])*1.0
    rise = (p2[1] - p1[1])*1.0

    try:
        a = rise/run
    except ZeroDivisionError:
        cp = get_point(c)
        d = cp.distance(get_point(p1))
        return {"a": None, "k": None, "pp1": (c[0] - d, c[1]), "pp2": (c[0] + d, c[1])}

    try:
        k = c[1] - (-1 / a * c[0])
        p1l = get_line([p1, c])
        p2l = get_line([c, p2])
        cp1l = list(p1l.centroid.coords)[0]
        cp2l = list(p2l.centroid.coords)[0]
        py1 = (-1/a * cp1l[0]) + k
        py2 = (-1/a * cp2l[0]) + k
        return {"a": -1/a, "k": k, "pp1": (cp1l[0], py1), "pp2": (cp2l[0], py2)}

    except ZeroDivisionError:
        cp = get_point(c)
        d = cp.distance(get_point(p1))
        return {"a": None, "k": c[0], "pp1": (c[0], c[1]-d), "pp2": (c[0], c[1]+d)}


def get_projection_point(p1, p2, percent=.3):
    l = get_line([p1, p2])
    e = get_ax_b(p1, p2)
    dist = l.length * percent

    if p1[0] < p2[0]:
        x1, y1 = p1[0] - dist, e["a"]*(p1[0] - dist) + e["b"]
        x2, y2 = p2[0] + dist, e["a"]*(p2[0] + dist) + e["b"]
    else:
        x1, y1 = p2[0] - dist, e["a"] * (p2[0] - dist) + e["b"]
        x2, y2 = p1[0] + dist, e["a"] * (p1[0] + dist) + e["b"]
    return {"points": [[x1, y1], [x2, y2]], "line": get_line([[x1, y1], [x2, y2]])}

class InOutCalculator(object):
    def __init__(self, line, first_point=None, max_distance=200):
        self.line = get_line(line)
        self.zero_point = self.line.centroid
        self.perpendicular = self.get_axis_x()
        self.radius = self.get_radius()
        self.max_distance = max_distance
        self.slope = 1

        if first_point is not None:
            self.first_point = get_point(first_point)

    def get_axis_x(self):
        p1, p2 = list(self.line.coords)
        d = get_perpendicular_coords(p1, p2)
        perpendicular = {"p1": Point(d["pp1"][0], d["pp1"][1]), "p2": Point(d["pp2"][0], d["pp2"][1])}
        self.slope = 0 if d["a"] is None else d["a"]
        self.slope = 1 if self.slope >= 0 else -1
        return perpendicular

    def get_radius(self):
        p1, p2 = list(self.line.coords)
        radius = self.zero_point.buffer(self.zero_point.distance(get_point(p1)))
        return radius

    def get_position(self, point):
        in_radius = self.radius.contains(point)
        dist = self.line.centroid.distance(point)
        p1_dist = self.perpendicular["p1"].distance(point)
        p2_dist = self.perpendicular["p2"].distance(point)
        p1, p2 = list(self.line.coords)
        
        self.get_axis_x()
        inverse = False
        if p1[0] > p2[0]:
            inverse = True

        if p1_dist < p2_dist and self.slope > 0 or p1_dist > p2_dist and self.slope < 0:
            dist = dist * -1

        _l = list(self.line.coords)
        _loga = "a" if _l[0][0] < _l[1][0] else "b"
        _logb = "b" if _l[0][0] < _l[1][0] else "a"

        return {"distance": dist, "in_radius": in_radius, "dir":  inverse}

    def evaluate(self, last_point):
        initial_position = self.get_position(self.first_point)
        last_position = self.get_position(get_point(last_point))

        if initial_position["distance"] * last_position["distance"] < 0:
            r = self.get_direction(last_position["distance"])

        else:
            if initial_position["in_radius"] is not last_position["in_radius"]:
                if last_position["in_radius"]:
                    r = self.get_direction(last_position["distance"])
                else:
                    r = self.get_direction(last_position["distance"])
            else:
                r = None
        return r

    def get_direction(self, dist):
        return "N" if dist < 0 else "P"

    def distance(self, point_tuple):
        return self.get_position(get_point(point_tuple))

    def contains(self, point):
        dist = int(self.zero_point.distance(point))
        return dist < self.max_distance
