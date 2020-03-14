import numpy as np
import itertools
import matplotlib.pyplot as plt
import math
import copy
from functions import torch2np_u8
from lines_utils import line_intersection, get_line, bresenham_line, get_2d_rotation_matrix

def rgbd_gradients_dataset_first_n(dataset, n, random_start=True, **kw):
    if random_start:
        start = np.random.randint(0, len(dataset) - n)
        stop  = start + n
    else:
        start = 0
        stop  = n
    # [(img0, cls0), ..., # (imgN, clsN)]
    first_n = list(itertools.islice(dataset, start, stop))
    # return rgbd_gradients_dataset_plot(first_n, **kw)
    return rgbd_gradients_dataset_plot(first_n, wspace=0.0, hspace=0.0, **kw)

def rgbd_gradients_dataset_plot(samples: list, figsize=(12, 12), wspace=0.1, hspace=0.2, cmap=None):
    fig, axes = plt.subplots(nrows=len(samples), ncols=5, figsize=figsize,
                             gridspec_kw=dict(wspace=wspace, hspace=hspace, left=0, right=1),
                             subplot_kw={'aspect': 1})
    # Plot each tensor
    for i in range(len(samples)):
        rgb   = samples[i]['rgb']
        depth = samples[i]['depth']
        x     = samples[i]['x'].squeeze(0)
        y     = samples[i]['y'].squeeze(0)

        rgb   = torch2np_u8(rgb)
        depth = torch2np_u8(depth)

        # axes[i, 0].imshow(rgb,   cmap=cmap)
        axes[i, 1].imshow(depth, cmap=cmap)
        axes[i, 4].imshow(depth, cmap=cmap)

        X,Y = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
        axes[i, 3].quiver(X, Y, x, y, pivot='tip', units='xy')
        axes[i, 3].set_ylim(axes[i, 3].get_ylim()[::-1])

        # # percentage_of_lengths_to_use = 0.25
        # gardient_lengths = []
        # for j in range(x.shape[0]):
        #     for k in range(x.shape[1]):
        #         length = np.linalg.norm(np.array([x[j][k], y[j][k]]))
        #         if 0.6 < length:
        #             gardient_lengths.append(length)
        #         # gardient_lengths.append(np.linalg.norm(np.array([x[j][k], y[j][k]])))
        # # gardient_lengths.sort()
        # # gardient_lengths = gardient_lengths[-int(percentage_of_lengths_to_use * len(gardient_lengths)):]

        navigation_image = np.zeros(x.shape, dtype=np.float64)
        for j in range(2, navigation_image.shape[0] - 2):
            for k in range(2, navigation_image.shape[1] - 2):
                x_gardient = x[j][k]
                y_gardient = y[j][k]
                # if x_gardient == 0 and y_gardient == 0: # A gardient with 0 length
                #     continue
                # if np.linalg.norm(np.array([x_gardient, y_gardient])) < 0.5:
                # if np.linalg.norm(np.array([x_gardient, y_gardient])) < 0.6:
                if np.linalg.norm(np.array([x_gardient, y_gardient])) < 0.7:
                    continue
                # Checks the 4 lines of the image's frame for the intersection point in one of them.
                gardient_line = ([j, k], [j + y_gardient.item(), k + x_gardient.item()])
                intersection_point = None
                if x_gardient < 0:
                    intersection_point_candidate = line_intersection(line=gardient_line, border=[(0, 0), (0, navigation_image.shape[1] - 1)], side="left")
                    if intersection_point_candidate is not None and \
                            intersection_point_candidate[0] == 0 and \
                            0 <= intersection_point_candidate[1] < navigation_image.shape[0] - 1:
                        intersection_point = intersection_point_candidate
                if intersection_point is not None:
                    (x_intersection, y_intersection) = intersection_point
                else:
                    intersection_point = None
                    if y_gardient < 0:
                        intersection_point_candidate = line_intersection(line=gardient_line, border=([0, 0], [navigation_image.shape[0] - 1, 0]), side="up")
                        if intersection_point_candidate is not None and \
                                intersection_point_candidate[1] == 0 and \
                                0 <= intersection_point_candidate[0] < navigation_image.shape[1] - 1:
                            intersection_point = intersection_point_candidate
                    if intersection_point is not None:
                        (x_intersection, y_intersection) = intersection_point
                    else:
                        intersection_point = None
                        if 0 < y_gardient:
                            intersection_point_candidate = line_intersection(line=gardient_line, border=([0, navigation_image.shape[1] - 1], [navigation_image.shape[0] - 1, navigation_image.shape[1] - 1]), side="down")
                            if intersection_point_candidate is not None and \
                                    intersection_point_candidate[1] == navigation_image.shape[0] - 1 and \
                                    0 <= intersection_point_candidate[0] < navigation_image.shape[1] - 1:
                                intersection_point = intersection_point_candidate
                        if intersection_point is not None:
                            (x_intersection, y_intersection) = intersection_point
                        else:
                            intersection_point = None
                            if 0 < x_gardient:
                                intersection_point_candidate = line_intersection(line=gardient_line, border=([navigation_image.shape[0] - 1, 0], [navigation_image.shape[0] - 1, navigation_image.shape[1] - 1]), side="right")
                                if intersection_point_candidate is not None and \
                                        intersection_point_candidate[0] == navigation_image.shape[0] - 1 and \
                                        0 <= intersection_point_candidate[1] < navigation_image.shape[0] - 1:
                                    intersection_point = intersection_point_candidate
                            if intersection_point is not None:
                                (x_intersection, y_intersection) = intersection_point
                            else:
                                print("x_gardient", "y_gardient")
                                print(x_gardient, y_gardient)
                                print("gardient_line")
                                print(gardient_line)
                                print("WARNING, a line didn't find an intersection line with the image's frame.")
                                # raise Exception("ERROR, No intersection line.")
                # line_points = bresenham_line(x0=k, y0=j,
                #                              x1=int(x_intersection), y1=int(y_intersection))
                # line_points = bresenham_line(x0=k, y0=j,
                #                              x1=int(x_intersection.item()), y1=int(y_intersection.item()))
                line_points = get_line(x1=j, y1=k,
                                       x2=int(x_intersection), y2=int(y_intersection))
                gardient_length = np.linalg.norm(np.array([x_gardient, y_gardient]))
                gardient_radians = math.atan2(y_gardient, x_gardient)
                R_gardient = get_2d_rotation_matrix(radians=-gardient_radians)
                # lines_percentage_of_screen = 0.42
                # lines_percentage_of_screen = 0.05
                lines_percentage_of_screen = 0.2
                screen_average_of_width_and_height = (navigation_image.shape[0] + navigation_image.shape[1]) / 2
                max_line_length = int(lines_percentage_of_screen * screen_average_of_width_and_height)
                current_line_length = 0
                for point in line_points:
                    if current_line_length == max_line_length:
                        break
                    current_line_length += 1
                    if 0 <= point[0] < navigation_image.shape[1] and 0 <= point[1] < navigation_image.shape[0]:
                        x_current_point_gardient = x[point[1]][point[0]]
                        y_current_point_gardient = y[point[1]][point[0]]
                        # navigation_image[point[1]][point[0]] = 1
                        # navigation_image[point[1]][point[0]] = navigation_image[point[1]][point[0]] + (gardient_length) * ((1 - (current_line_length / max_line_length)) ** (1 / 10))
                        # navigation_image[point[1]][point[0]] = navigation_image[point[1]][point[0]] + gardient_length * (1 - (current_line_length / max_line_length))**(1 / 10)
                        # navigation_image[point[1]][point[0]] = navigation_image[point[1]][point[0]] + gardient_length * (1 - current_line_length / max_line_length)
                        # navigation_image[point[1]][point[0]] = navigation_image[point[1]][point[0]] + gardient_length
                        # navigation_image[point[1]][point[0]] = navigation_image[point[1]][point[0]] + 1
                        # navigation_image[point[1]][point[0]] = navigation_image[point[1]][point[0]] + (1 - current_line_length / max_line_length)
                        # navigation_image[point[1]][point[0]] = navigation_image[point[1]][point[0]] + current_line_length / max_line_length
                        # navigation_image[point[1]][point[0]] = navigation_image[point[1]][point[0]] + (gardient_length ** 4) * ((1 - (current_line_length / max_line_length)) ** (1 / 10))
                        # navigation_image[point[1]][point[0]] = max(navigation_image[point[1]][point[0]], gardient_length)
                        navigation_image[point[1]][point[0]] = max(navigation_image[point[1]][point[0]], gardient_length * ((1 - (current_line_length / max_line_length)) ** (1 / 2)))
                        # Decreases the running gardient length, if opposite gardients are on it's way.
                        rotated_point = np.dot(R_gardient, np.array([x_current_point_gardient, y_current_point_gardient]))
                        if rotated_point[0] < 0:
                            gardient_length += rotated_point[0]
                            if gardient_length <= 0:
                                break
        # Keeps only the best x% navigation_values
        navigation_percentage_to_keep = 0.15
        navigation_values = []
        for j in range(navigation_image.shape[0]):
            for k in range(navigation_image.shape[1]):
                navigation_values.append(navigation_image[j][k])
        navigation_values.sort()
        navigation_values = navigation_values[-int(navigation_percentage_to_keep * len(navigation_values)):]
        for j in range(navigation_image.shape[0]):
            for k in range(navigation_image.shape[1]):
                if navigation_image[j][k] not in navigation_values:
                    navigation_image[j][k] = 0

        navigation_image = navigation_image.T
        axes[i, 2].imshow(navigation_image, cmap=cmap)

        # Finds the pixel to navigate through, weighted average
        x_sum = 0
        y_sum = 0
        sum_values = 0
        for j in range(navigation_image.shape[0]):
            for k in range(navigation_image.shape[1]):
                current_value = navigation_image[j][k]
                sum_values += current_value
                x_sum += k * current_value
                y_sum += j * current_value
        navigate_to_x = int(x_sum / sum_values)
        navigate_to_y = int(y_sum / sum_values)
        color = np.array([255, 0 ,0])
        marker_size = 7
        goto_image = copy.deepcopy(rgb)
        for k in range(navigate_to_y - int(marker_size / 2), navigate_to_y + int(marker_size / 2) + 1):
            if 0 <= k < navigation_image.shape[0]:
                goto_image[k][navigate_to_x] = color
        for k in range(navigate_to_x - int(marker_size / 2), navigate_to_x + int(marker_size / 2) + 1):
            if 0 <= k < navigation_image.shape[1]:
                goto_image[navigate_to_y][k] = color
        axes[i, 0].imshow(goto_image, cmap=cmap)

    return fig, axes