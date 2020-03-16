import numpy as np
import math
from numpy import ones, vstack
from numpy.linalg import lstsq
from shapely.geometry import LineString

# Returns a CONTINOUS line that goes from one 2D point to another 2D point.
def get_line(x1, y1, x2, y2):
    points = []
    issteep = abs(y2 - y1) > abs(x2 - x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2 - y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points

# # Returns a CONTINOUS line that goes from one 2D point to another 2D point.
# def bresenham_line(x0, y0, x1, y1):
#     steep = abs(y1 - y0) > abs(x1 - x0)
#     if steep:
#         x0, y0 = y0, x0
#         x1, y1 = y1, x1
#     switched = False
#     if x0 > x1:
#         switched = True
#         x0, x1 = x1, x0
#         y0, y1 = y1, y0
#     if y0 < y1:
#         ystep = 1
#     else:
#         ystep = -1
#     deltax = x1 - x0
#     deltay = abs(y1 - y0)
#     error = -deltax / 2
#     y = y0
#     line = []
#     for x in range(x0, x1 + 1):
#         if steep:
#             line.append((y,x))
#         else:
#             line.append((x,y))
#         error = error + deltay
#         if error > 0:
#             y = y + ystep
#             error = error - deltax
#     if switched:
#         line.reverse()
#     return line

# Returns a rotation matrix for 2D space, made out of a angle represented using radians.
def get_2d_rotation_matrix(radians):
    c, s = np.cos(radians), np.sin(radians)
    return np.array(((c, -s), (s, c)))

# def line_intersection(line1, line2):
#     line1, line2 = line2, line1
#     line1 = (np.array(line1[0]), np.array(line1[1]))
#     line2 = (np.array(line2[0]), np.array(line2[1]))
#     line1_direction = line1[1] - line1[0]
#     line1_direction_angle = math.atan2(line1_direction[1], line1_direction[0])
#     R_line1 = get_2d_rotation_matrix(radians=-line1_direction_angle)
#     line2_start_axises_origin, line2_end_axises_origin = line2[0] - line1[0], line2[1] - line1[0]
#
#     line1_start_axises_origin_rotated = np.array([0, 0])
#     line1_end_axises_origin_rotated = np.dot(R_line1, line1_direction)
#     line2_start_axises_origin_rotated = np.dot(R_line1, line2_start_axises_origin)
#     line2_end_axises_origin_rotated = np.dot(R_line1, line2_end_axises_origin)
#
#     line = [line2_start_axises_origin_rotated, line2_end_axises_origin_rotated]
#     x_coords, y_coords = zip(*line)
#     A = vstack([x_coords, ones(len(x_coords))]).T
#     m, c = lstsq(A, y_coords, rcond=None)[0]
#     intersection_point = None
#     if m != 0:
#         x = -c / m  # (-c/m, 0)
#         if 0 <= x < line1_end_axises_origin_rotated[0]:
#             intersection_point = np.array([x, 0])
#             intersection_point = np.dot(R_line1.T, intersection_point)
#             intersection_point = line1[0] + intersection_point
#             intersection_point = [int(intersection_point[0]), int(intersection_point[1])]
#             # print("intersection_point")
#             # print(intersection_point)
#     return intersection_point

# def line_intersection(line, border, side):
#     # line_possible_directions = []
#     # line_direction = np.array(line[1]) - np.array(line[0])
#     # if 0 < line_direction[0] and 0 < line_direction[1]:
#     #     line_possible_directions.extend(["right", "up"])
#     # elif 0 < line_direction[0] and 0 < line_direction[1]:
#     #     line_possible_directions.extend(["", ""])
#     line = LineString(line)
#     border = LineString(border)
#
#     int_pt = line.intersection(border)
#     if hasattr(int_pt, 'x'):
#         point_of_intersection = int_pt.x, int_pt.y
#     else:
#         point_of_intersection = None
#     # print(point_of_intersection)
#     return point_of_intersection
#
#     # # print(LineString(line1).intersection(LineString(line2)))
#     # # print(type(LineString(line1).intersection(LineString(line2))))
#     # # print(LineString(line1).intersection(LineString(line2)).shape)
#     # point = LineString(line1).intersection(LineString(line2)).xy
#     # # print(point)
#     # point0 = point[0][0]
#     # point1 = point[1][0]
#     # # print(point0)
#     # # print(point1)
#     # return (point0, point1)

# Returns the intersection point between 2 lines, line and border are both lines, where each is made out of 2 2D points.
def line_intersection(line, border):
    xdiff = (line[0][0] - line[1][0], border[0][0] - border[1][0])
    ydiff = (line[0][1] - line[1][1], border[0][1] - border[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None

    d = (det(*line), det(*border))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return (x, y)

if __name__ == '__main__':
    # a = line_intersection(line1=[(0, 0), (5.001, 0)], line2=[(0, 5), (10, -5)])
    a = line_intersection(line=([1, 1], [1.1920960396528244, 1.105015255510807]), border=([63, 0], [63, 63]))
    s = 1