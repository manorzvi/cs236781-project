import numpy as np
import torch
import itertools
import matplotlib.pyplot as plt
import math
import copy
from functions import torch2np_u8
from lines_utils import line_intersection, get_line, get_2d_rotation_matrix

# ------------------Constants------------------
# For the maximum lines' lengths, starting from the gardients' roots, going where each gardient points.
lines_percentage_of_screen = 0.2 # 0.15 # 0.05 # 0.42

# To ignore "weak" gardients
ignore_gardients_shorter_than = 0.65 # 0.00001 # 0.6 # 0.5 # 0.7

# After the navigation map is ready (Each pixel's line added it's scores onto it), keep only the strongest pixels (Like max-pooling, keeps only the strongest pixels).
navigation_percentage_to_keep = 0.15 # 0.5

# When a line runs from it's pixel, directed by it's gardient, how to score each of the line's pixel ?
# gardient_length: The line's starting score, which fades out as opposite gardients are meeting it along the line.
# ((1 - (current_line_length / max_line_length)) ** (1 / 2)): Fade out as the line is more far away from it's starting pixel.
def navigation_score_formula(navigation_image, point, gardient_length, current_line_length, max_line_length):
    return max(navigation_image[point[1]][point[0]], gardient_length * ((1 - (current_line_length / max_line_length)) ** (1 / 2)))
# Other options:
#     return 1
#     return navigation_image[point[1]][point[0]] + (gardient_length) * ((1 - (current_line_length / max_line_length)) ** (1 / 10))
#     return navigation_image[point[1]][point[0]] + gardient_length * (1 - (current_line_length / max_line_length))**(1 / 10)
#     return navigation_image[point[1]][point[0]] + gardient_length * (1 - current_line_length / max_line_length)
#     return navigation_image[point[1]][point[0]] + gardient_length
#     return navigation_image[point[1]][point[0]] + 1
#     return navigation_image[point[1]][point[0]] + (1 - current_line_length / max_line_length)
#     return navigation_image[point[1]][point[0]] + current_line_length / max_line_length
#     return navigation_image[point[1]][point[0]] + gardient_length * ((1 - (current_line_length / max_line_length)) ** (1 / 2))
#     return max(navigation_image[point[1]][point[0]], gardient_length * ((1 - (current_line_length / max_line_length)) ** (1 / 2)))
#     return max(navigation_image[point[1]][point[0]], gardient_length)
# ------------------End Constants------------------


# Fettuccini with Corona sauce below :)


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
    fig, axes = plt.subplots(nrows=len(samples), ncols=4, figsize=figsize,
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

        axes[i, 1].imshow(depth, cmap=cmap)
#         axes[i, 4].imshow(depth, cmap=cmap) # NOTHING HERE, just to keep the plottings close, to prevent the annoying bottom scroller.
        
        X,Y = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
        axes[i, 3].quiver(X, Y, x, y, pivot='tip', units='xy')
        axes[i, 3].set_ylim(axes[i, 3].get_ylim()[::-1]) # Transpose, to look like the original RGB and Depth images.
        
        # Attempt to use only the longest gradients
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

        # Creates the navigation map/image by shooting "navigation" rays from each (x, y) pixel, using that pixel's (x_gardient, y_gardient) values.
        # The ray fades away by:
        # Distance from it's origin.
        # Anti-gardient pixels along it's way. (Gardients who point to the opposite direction of the gardient)
        navigation_image = np.zeros(x.shape, dtype=np.float64)
        number_of_frame_pixels_to_ignore = 3 # Because if a image's frame is black, it may cause false lines with nearby white pixels.
        for j in range(number_of_frame_pixels_to_ignore, navigation_image.shape[0] - number_of_frame_pixels_to_ignore):
            for k in range(number_of_frame_pixels_to_ignore, navigation_image.shape[1] - number_of_frame_pixels_to_ignore):
                x_gardient = x[j][k]
                y_gardient = y[j][k]
                if np.linalg.norm(np.array([x_gardient, y_gardient])) < ignore_gardients_shorter_than:
                    continue
                # Checks the 4 border lines of the image's frame for the intersection point between "gardient_line" and one of them.
                gardient_line = ([j, k], [j + y_gardient.item(), k + x_gardient.item()])
                intersection_point = None
                if x_gardient < 0:
                    intersection_point_candidate = line_intersection(line=gardient_line,
                                                                     border=[(0, 0), (navigation_image.shape[0] - 1, 0)]) # Left image border
                    if intersection_point_candidate is not None and \
                            intersection_point_candidate[1] == 0 and \
                            0 <= intersection_point_candidate[0] < navigation_image.shape[0] - 1:
                        intersection_point = intersection_point_candidate
                if intersection_point is not None:
                    (x_intersection, y_intersection) = intersection_point
                else:
                    intersection_point = None
                    if y_gardient < 0:
                        intersection_point_candidate = line_intersection(line=gardient_line,
                                                                         border=([0, 0], [0, navigation_image.shape[1] - 1])) # Upper image border
                        if intersection_point_candidate is not None and \
                                intersection_point_candidate[0] == 0 and \
                                0 <= intersection_point_candidate[1] < navigation_image.shape[1] - 1:
                            intersection_point = intersection_point_candidate
                    if intersection_point is not None:
                        (x_intersection, y_intersection) = intersection_point
                    else:
                        intersection_point = None
                        if 0 < y_gardient:
                            intersection_point_candidate = line_intersection(line=gardient_line,
                                                                             border=([navigation_image.shape[0] - 1, 0], [navigation_image.shape[0] - 1, navigation_image.shape[1] - 1])) # Bottom image border
                            if intersection_point_candidate is not None and \
                                    intersection_point_candidate[0] == navigation_image.shape[0] - 1 and \
                                    0 <= intersection_point_candidate[1] < navigation_image.shape[1] - 1:
                                intersection_point = intersection_point_candidate
                        if intersection_point is not None:
                            (x_intersection, y_intersection) = intersection_point
                        else:
                            intersection_point = None
                            if 0 < x_gardient:
                                intersection_point_candidate = line_intersection(line=gardient_line,
                                                                                 border=([0, navigation_image.shape[1] - 1], [navigation_image.shape[0] - 1, navigation_image.shape[1] - 1])) # Right image border
                                if intersection_point_candidate is not None and \
                                        intersection_point_candidate[1] == navigation_image.shape[1] - 1 and \
                                        0 <= intersection_point_candidate[0] < navigation_image.shape[0] - 1:
                                    intersection_point = intersection_point_candidate
                            if intersection_point is not None:
                                (x_intersection, y_intersection) = intersection_point
                            else:
                                print("x_gardient", "y_gardient")
                                print(x_gardient, y_gardient)
                                print("gardient_line")
                                print(gardient_line)
                                print("WARNING, a line didn't find an intersection line with the image's frame.")
                                continue
                # Finds all 2D points along current-pixel to intersection-with-image-frame-pixel line
                # line_points = bresenham_line(x0=k, y0=j,
                #                              x1=int(x_intersection), y1=int(y_intersection))
                # line_points = bresenham_line(x0=k, y0=j,
                #                              x1=int(x_intersection.item()), y1=int(y_intersection.item()))
                line_points = get_line(x1=j, y1=k,
                                       x2=int(x_intersection), y2=int(y_intersection))
                gardient_length = np.linalg.norm(np.array([x_gardient, y_gardient]))
                gardient_radians = math.atan2(y_gardient, x_gardient)
                R_gardient = get_2d_rotation_matrix(radians=-gardient_radians)
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
                        # Sets current gardient's line's point's (pixel's) "navigation" value.
                        # Longer gardients, and points closer to the original gardient start point cause the value to be higher.
                        # Very important to find the best formula here.
                        navigation_image[point[1]][point[0]] = navigation_score_formula(navigation_image, point, gardient_length, current_line_length, max_line_length)
                        # Decreases the running gardient length, if opposite gardients are on it's way.
                        rotated_point = np.dot(R_gardient, np.array([x_current_point_gardient, y_current_point_gardient]))
                        if rotated_point[0] < 0:
                            gardient_length += rotated_point[0]
                            if gardient_length <= 0:
                                break # Stop using current gardient, as anti-current-gardient gardients destroyed it along it's route.
        # Keeps only the best x% navigation_values, in term of which (x, y) pixels got high "navigation" score.
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

        # Finds the pixel to navigate through, using weighted average over all of the navigation map.
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
        # Plots the Goto marker on the RGB image.
        color = np.array([255, 0 ,0])
        marker_length = int(navigation_image.shape[0] / 10)
        marker_width = int(marker_length / 3)
        goto_image = copy.deepcopy(rgb)
        for j in range(navigate_to_x - int(marker_width / 2), navigate_to_x + int(marker_width / 2) + 1):
            for k in range(navigate_to_y - int(marker_length / 2), navigate_to_y + int(marker_length / 2) + 1):
                if 0 <= k < navigation_image.shape[0] and 0 <= j < navigation_image.shape[1]:
                    goto_image[k][j] = color
        for j in range(navigate_to_y - int(marker_width / 2), navigate_to_y + int(marker_width / 2) + 1):
            for k in range(navigate_to_x - int(marker_length / 2), navigate_to_x + int(marker_length / 2) + 1):
                if 0 <= k < navigation_image.shape[1] and 0 <= j < navigation_image.shape[0]:
                    goto_image[j][k] = color
        axes[i, 0].imshow(goto_image, cmap=cmap)
        
    return fig, axes