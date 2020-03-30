import numpy as np
import itertools
import matplotlib.pyplot as plt
import math

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from functions import torch2np_u8
from lines_utils import line_intersection, get_line, get_2d_rotation_matrix
from models import SpecialFuseNetModel

# ------------------Constants------------------
 # Because if an image's frame is black, it may cause false lines, starting from the frame and though nearby whiter pixels.
number_of_frame_pixels_to_ignore = 5

# For the maximum lines' lengths, starting from the gardients' roots, going where each gardient points.
lines_percentage_of_screen = 0.2 # 0.15 # 0.05 # 0.42

# To ignore "weak" gardients
ignore_gardients_shorter_than = 0.65 # 0.00001 # 0.6 # 0.5 # 0.7

# After the navigation map is ready (Each pixel's line added it's scores onto it), keep only the strongest pixels (Like max-pooling, keeps only the strongest pixels).
navigation_percentage_to_keep = 0.15 # 0.5

def post_epoch_plot(model:SpecialFuseNetModel, device:torch.device, dl_test:DataLoader,
                    figsize=(8, 8), wspace=0.1, hspace=0.2, cmap=None, stop_training_threshold=0.001):

    sample = next(iter(dl_test))

    rgb_minibatch   = sample['rgb']
    depth_minibatch = sample['depth']

    # Ground-Truth Depth Gradients
    x_gt_minibatch = sample['x']
    y_gt_minibatch = sample['y']

    rgb_minibatch   = rgb_minibatch.to(device)
    depth_minibatch = depth_minibatch.to(device)
    x_gt_minibatch  = x_gt_minibatch.to(device)
    y_gt_minibatch  = y_gt_minibatch.to(device)
    xy_gt_minibatch = torch.cat((x_gt_minibatch, y_gt_minibatch), dim=1)

    model.set_requires_grad(requires_grad=False)
    with torch.no_grad():
        xy_minibatch = model(rgb_batch=rgb_minibatch, depth_batch=depth_minibatch)
        loss         = model.loss(ground_truth_grads=xy_gt_minibatch, approximated_grads=xy_minibatch).item()

    x_minibatch = xy_minibatch[:,0,:,:]
    y_minibatch = xy_minibatch[:,1,:,:]
    if len(x_minibatch.shape) == 3:
        x_minibatch = x_minibatch[:,None,:,:]
    if len(y_minibatch.shape) == 3:
        y_minibatch = y_minibatch[:,None,:,:]

    nrows = rgb_minibatch.shape[0]
    ncols = 4

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                             gridspec_kw=dict(wspace=wspace, hspace=hspace, left=0, right=1),
                             subplot_kw={'aspect': 1})
    # fig.suptitle('MiniBatch Loss: {:4f}'.format(loss), fontsize=12)

    # print(f'|rgb_minibatch|={rgb_minibatch.shape}')
    # print(f'|depth_minibatch|={depth_minibatch.shape}')
    # print(f'|x_gt_minibatch|={x_gt_minibatch.shape}')
    # print(f'|y_gt_minibatch|={y_gt_minibatch.shape}')
    # print(f'|x_minibatch|={x_minibatch.shape}')
    # print(f'|y_minibatch|={y_minibatch.shape}')

    if nrows > 1:
        for i in range(nrows):
            rgb   = rgb_minibatch[i,:,:,:]
            depth = depth_minibatch[i,:,:,:]
            x_gt  = x_gt_minibatch[i,:,:,:].squeeze(0)
            y_gt  = y_gt_minibatch[i,:,:,:].squeeze(0)
            x     = x_minibatch[i,:,:,:].squeeze(0)
            y     = y_minibatch[i,:,:,:].squeeze(0)

            # print(f'|rgb|={rgb.shape}')
            # print(f'|depth|={depth.shape}')
            # print(f'|x_gt|={x_gt.shape}')
            # print(f'|y_gt|={y_gt.shape}')
            # print(f'|x|={x.shape}')
            # print(f'|y|={y.shape}')

            rgb   = torch2np_u8(rgb)
            depth = torch2np_u8(depth)

            axes[i,0].imshow(rgb, cmap=cmap)
            axes[i,0].set_title('Ground Truth RGB')

            axes[i,1].imshow(depth, cmap=cmap)
            axes[i,1].set_title('Ground Truth Depth')

            X_gt,Y_gt = np.meshgrid(np.arange(x_gt.shape[1]), np.arange(x_gt.shape[0]))
            axes[i,2].quiver(X_gt, Y_gt, x_gt, y_gt, pivot='tip', units='xy')
            axes[i,2].set_title('Ground Truth Gradients')

            X,Y = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
            axes[i,3].quiver(X, Y, x, y, pivot='tip', units='xy')
            axes[i,3].set_title('Model Output Gradients')
    else:
        rgb   = rgb_minibatch[0,:,:,:]
        depth = depth_minibatch[0,:,:,:]
        x_gt  = x_gt_minibatch[0,:,:,:].squeeze(0)
        y_gt  = y_gt_minibatch[0,:,:,:].squeeze(0)
        x     = x_minibatch[0,:,:,:].squeeze(0)
        y     = y_minibatch[0,:,:,:].squeeze(0)

        rgb   = torch2np_u8(rgb)
        depth = torch2np_u8(depth)

        axes[0].imshow(rgb, cmap=cmap)
        axes[0].set_title('Ground Truth RGB')

        axes[1].imshow(depth, cmap=cmap)
        axes[1].set_title('Ground Truth Depth')

        X_gt, Y_gt = np.meshgrid(np.arange(x_gt.shape[1]), np.arange(x_gt.shape[0]))
        axes[2].quiver(X_gt, Y_gt, x_gt, y_gt, pivot='tip', units='xy')
        axes[2].set_title('Ground Truth Gradients')

        X, Y = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
        axes[3].quiver(X, Y, x, y, pivot='tip', units='xy')
        axes[3].set_title('Model Output Gradients')
    plt.show()

    # if loss <= stop_training_threshold:
    #     return True
    # else:
    #     return False






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
    """
    Plots first n images of a dataset containing tensor images.
    """
    if random_start:
        start = np.random.randint(0, len(dataset) - n)
        stop  = start + n
    else:
        start = 0
        stop  = n

    # [(img0, cls0), ..., # (imgN, clsN)]
    first_n = list(itertools.islice(dataset, start, stop))
    return rgbd_gradients_dataset_plot(first_n, **kw)

# def rgbd_gradients_dataset_plot(samples: list, figsize=(12, 12), wspace=0.1, hspace=0.2, cmap=None):
#     nrows = len(samples)
#     ncols = 3
#
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
#                              gridspec_kw=dict(wspace=wspace, hspace=hspace),
#                              subplot_kw={'aspect': 1})
#     # Plot each tensor
#     for i in range(len(samples)):
#         rgb   = samples[i]['rgb']
#         depth = samples[i]['depth']
#         x     = samples[i]['x'].squeeze(0)
#         y     = samples[i]['y'].squeeze(0)
#
#         rgb   = torch2np_u8(rgb)
#         depth = torch2np_u8(depth)
#
#         axes[i,0].imshow(rgb,   cmap=cmap)
#         axes[i,1].imshow(depth, cmap=cmap)
#
#         X,Y = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
#         U,V = x,y
#         axes[i,2].quiver(X, Y, U, V, pivot='tip', units='xy')
#
#     return fig, axes

def rgbd_gradients_dataset_plot(samples: list, figsize=(12, 12), wspace=0.1, hspace=0.2, cmap=None):
    nrows = len(samples)
    ncols = 4

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
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

        axes[i,0].imshow(rgb, cmap=cmap)
        axes[i,1].imshow(depth, cmap=cmap)

        X,Y = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
        axes[i,2].quiver(X, Y, x, y, pivot='tip', units='xy')


        # Creates the navigation map/image by shooting "navigation" rays from each (x, y) pixel,
        # using that pixel's (x_gradient, y_gradient) values.
        # The ray fades away by:
        # Distance from it's origin.
        # Anti-gradient pixels along it's way. (Gradients who point to the opposite direction of the gradient)
        navigation_image = np.zeros(x.shape, dtype=np.float64)
        for j in range(2, navigation_image.shape[0] - 2):
            for k in range(2, navigation_image.shape[1] - 2):
                x_gradient = x[j][k]
                y_gradient = y[j][k]
                gradient_size = np.linalg.norm(np.array([x_gradient, y_gradient]))
                gradients_threshold = 0.7 # 0.6 # 0.5

                if gradient_size < gradients_threshold:
                    continue

                # Checks the 4 lines of the image's frame for the intersection point between "gradient_line" and one of them.
                gradient_line = ([j,k], [j + y_gradient.item(), k + x_gradient.item()])
                intersection_point = None
                if x_gradient < 0:
                    intersection_point_candidate = line_intersection(line=gradient_line,
                                                                     border=[(0, 0), (0, navigation_image.shape[1] - 1)]) # Left image border
                    if intersection_point_candidate is not None and \
                            intersection_point_candidate[0] == 0 and \
                            0 <= intersection_point_candidate[1] < navigation_image.shape[0] - 1:
                        intersection_point = intersection_point_candidate
                if intersection_point is not None:
                    (x_intersection, y_intersection) = intersection_point
                else:
                    intersection_point = None
                    if y_gradient < 0:
                        intersection_point_candidate = line_intersection(line=gradient_line,
                                                                         border=([0, 0], [navigation_image.shape[0] - 1, 0])) # Upper image border
                        if intersection_point_candidate is not None and \
                                intersection_point_candidate[1] == 0 and \
                                0 <= intersection_point_candidate[0] < navigation_image.shape[1] - 1:
                            intersection_point = intersection_point_candidate
                    if intersection_point is not None:
                        (x_intersection, y_intersection) = intersection_point
                    else:
                        intersection_point = None
                        if 0 < y_gradient:
                            intersection_point_candidate = line_intersection(line=gradient_line,
                                                                             border=([0, navigation_image.shape[1] - 1], [navigation_image.shape[0] - 1, navigation_image.shape[1] - 1])) # Bottom image border
                            if intersection_point_candidate is not None and \
                                    intersection_point_candidate[1] == navigation_image.shape[0] - 1 and \
                                    0 <= intersection_point_candidate[0] < navigation_image.shape[1] - 1:
                                intersection_point = intersection_point_candidate
                        if intersection_point is not None:
                            (x_intersection, y_intersection) = intersection_point
                        else:
                            intersection_point = None
                            if 0 < x_gradient:
                                intersection_point_candidate = line_intersection(line=gradient_line,
                                                                                 border=([navigation_image.shape[0] - 1, 0], [navigation_image.shape[0] - 1, navigation_image.shape[1] - 1])) # Right image border
                                if intersection_point_candidate is not None and \
                                        intersection_point_candidate[0] == navigation_image.shape[0] - 1 and \
                                        0 <= intersection_point_candidate[1] < navigation_image.shape[0] - 1:
                                    intersection_point = intersection_point_candidate
                            if intersection_point is not None:
                                (x_intersection, y_intersection) = intersection_point
                            else:
                                print("x_gradient", "y_gradient")
                                print(x_gradient, y_gradient)
                                print("gradient_line")
                                print(gradient_line)
                                print("WARNING, a line didn't find an intersection line with the image's frame.")
                                # raise Exception("ERROR, No intersection line.")
                # Finds all 2D points along current-pixel to intersection-with-image-frame-pixel line
                # line_points = bresenham_line(x0=k, y0=j,
                #                              x1=int(x_intersection), y1=int(y_intersection))
                # line_points = bresenham_line(x0=k, y0=j,
                #                              x1=int(x_intersection.item()), y1=int(y_intersection.item()))
                line_points = get_line(x1=j, y1=k,
                                       x2=int(x_intersection), y2=int(y_intersection))
                gardient_length = np.linalg.norm(np.array([x_gradient, y_gradient]))
                gardient_radians = math.atan2(y_gradient, x_gradient)
                R_gardient = get_2d_rotation_matrix(radians=-gardient_radians)
                lines_percentage_of_screen = 0.2 # 0.05 # 0.42
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
                                break # Stop using current gardient, as anti-current-gardient gardients destroyed it along it's route.
        # Keeps only the best x% navigation_values, in term of which (x, y) pixels got high "navigation" score.
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

        axes[i,3].imshow(navigation_image, cmap=cmap)

        # Finds the pixel to navigate through, using weighted average over all of the navigation map.
        x_sum      = 0
        y_sum      = 0
        sum_values = 0
        for j in range(navigation_image.shape[0]):
            for k in range(navigation_image.shape[1]):
                current_value = navigation_image[j][k]
                sum_values   += current_value
                x_sum += k * current_value
                y_sum += j * current_value
        navigate_to_x  = int(x_sum / sum_values)
        navigate_to_y  = int(y_sum / sum_values)
        # Plots the Goto marker on the RGB image.
        color = np.array([255, 0 ,0])
        marker_length = int(navigation_image.shape[0] / 10)
        marker_width = int(marker_length / 3)
        # goto_image = copy.deepcopy(rgb)
        for j in range(navigate_to_x - int(marker_width / 2), navigate_to_x + int(marker_width / 2) + 1):
            for k in range(navigate_to_y - int(marker_length / 2), navigate_to_y + int(marker_length / 2) + 1):
                if 0 <= k < navigation_image.shape[0] and 0 <= j < navigation_image.shape[1]:
                    # goto_image[k][j] = color
                    axes[i, 0].scatter(k, j, s=10, c='red', marker='o')
#                     goto_image[k][navigate_to_x] = color
        for j in range(navigate_to_y - int(marker_width / 2), navigate_to_y + int(marker_width / 2) + 1):
            for k in range(navigate_to_x - int(marker_length / 2), navigate_to_x + int(marker_length / 2) + 1):
                if 0 <= k < navigation_image.shape[1] and 0 <= j < navigation_image.shape[0]:
                    # goto_image[j][k] = color
                    axes[i, 0].scatter(j, k, s=10, c='red', marker='o')

    return fig, axes
