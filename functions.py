import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import shutil

def plot_samples(samples_fnames, depth_dir, rgb_dir,
                 rows=2, cols=2, figsize=(8,8), alpha=1):
    depth_abs_fnames = [os.path.join(depth_dir,f) for f in samples_fnames]
    rgb_abs_fnames   = [os.path.join(rgb_dir,f)   for f in samples_fnames]

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)

    for (i, axi), depth_samp, rgb_samp in zip(enumerate(ax.flat), depth_abs_fnames, rgb_abs_fnames):
        img_depth = cv2.imread(depth_samp, 0)
        img_rgb = cv2.imread(rgb_samp)
        img = np.concatenate((img_rgb, np.stack([img_depth for i in range(3)], axis=2)), 1)
        axi.imshow(img, alpha=alpha)
        rowid = i // cols
        colid = i % cols
        axi.set_title("(" + str(rowid) + "," + str(colid) + ") Idx:" + os.path.basename(depth_samp)[:-3])

    plt.tight_layout(True)
    plt.show()

def calc_depth_grads(depth_fnames, depth_dir, ksize=-1, gradient_degree=1, scale=0.1, to_save=False, save_dir=None):
    '''
    calculate gradients of the depth map using Sobel/Scharr Convolution filters.
    :param depth_fnames: list of depth maps file names
    :param depth_dir: depth maps directory
    :param ksize: convolution filter size (-1 for 3x3 Scharr filter)
    :param gradient_degree: degree of the gradient calculated at each direction (default: 1)
    :param scale: scaling factor for original image (default: 0.1)
    :param to_save: to save? (default: False)
    :param save_dir: save the results in that directory (default: None)
    :return: gradient maps for each depth map specified
    '''
    x_save_dir = save_dir + '/x'
    y_save_dir = save_dir + '/y'
    if to_save:
        def delete_if_exists_and_create_folder(dir):
            if os.path.exists(dir):
                shutil.rmtree(dir)
            os.mkdir(dir)
        delete_if_exists_and_create_folder(x_save_dir)
        delete_if_exists_and_create_folder(y_save_dir)
    depth_abs_fnames = [os.path.join(depth_dir,f) for f in depth_fnames]
    ret_val = {}
    for i,fname in enumerate(depth_abs_fnames):
        print(f'{i}) {os.path.basename(fname)} ...',end=' ')
        img = cv2.imread(fname,0)
        img_r = scaling(img,scale)
        grad_x = cv2.Sobel(img_r,cv2.CV_64F,gradient_degree,0,ksize=ksize)
        grad_y = cv2.Sobel(img_r,cv2.CV_64F,0,gradient_degree,ksize=ksize)
        if to_save:
            np.save(os.path.join(x_save_dir, os.path.basename(fname)[:-4]), grad_x)
            np.save(os.path.join(y_save_dir, os.path.basename(fname)[:-4]), grad_y)
        ret_val[os.path.basename(fname)[:-4]] = {'x' : grad_x,
                                                 'y' : grad_y}
        print('done.')
    return ret_val


def scaling(img,scaling_factor):

    width  = int(img.shape[1] * scaling_factor)
    height = int(img.shape[0] * scaling_factor)

    dsize = (width, height)

    # resize image
    return cv2.resize(img, dsize)


def plot_samples_grads(grads, figsize=(8, 8)):
    nrows = int(np.sqrt(len(grads)))
    ncols = int(np.sqrt(len(grads)))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, subplot_kw={'aspect': 1})

    for (i, axi), (k, v) in zip(enumerate(ax.flat), grads.items()):
        X, Y = np.meshgrid(np.arange(v['x'].shape[1]), np.arange(v['x'].shape[0]))
        U = v['x']
        V = v['y']
        axi.quiver(X, Y, U, V, pivot='tip', units='xy')
        rowid = i // ncols
        colid = i % ncols
        axi.set_title("(" + str(rowid) + "," + str(colid) + ") Image: " + k)

    plt.tight_layout(True)
    plt.show()


def calc_pixelwise_diffs(img):
    diff_x = np.zeros_like(img)
    diff_y = np.zeros_like(img)

    directions = ['up', 'down', 'right', 'left']

    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            pix = img[i, j]
            for dir in directions:
                if dir == 'up':
                    if pix > img[i - 1, j]:
                        continue
                    else:
                        diff_y[i, j] += pix - img[i - 1, j]
                if dir == 'down':
                    if pix > img[i + 1, j]:
                        continue
                    else:
                        diff_y[i, j] += img[i + 1, j] - pix
                if dir == 'right':
                    if pix > img[i, j + 1]:
                        continue
                    else:
                        diff_x[i, j] += img[i, j + 1] - pix
                if dir == 'left':
                    if pix > img[i, j - 1]:
                        continue
                    else:
                        diff_x[i, j] += pix - img[i, j - 1]
    return diff_x,diff_y


def calc_diffs(data_fnames,data_dir, scale=0.1, to_save=False, save_dir=None):

    if to_save:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)

    data_abs_fnames = [os.path.join(data_dir, f) for f in data_fnames]
    ret_val = {}

    for i,fname in enumerate(data_abs_fnames):
        img = cv2.imread(fname, 0)
        img = scaling(img, scale)
        diff_x, diff_y = calc_pixelwise_diffs(img)
        ret_val[os.path.basename(fname)[:-4]] = {'x' : diff_x,
                                                 'y' : diff_y}
        if to_save:
            np.save(os.path.join(save_dir, os.path.basename(fname)[:-4]+'.x'),diff_x)
            np.save(os.path.join(save_dir, os.path.basename(fname)[:-4]+'.y'),diff_y)

    return ret_val


def plot_samples_diffs(diffs, figsize=(8, 8)):
    nrows = int(np.sqrt(len(diffs)))
    ncols = int(np.sqrt(len(diffs)))
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, subplot_kw={'aspect': 1})

    for (i, axi), (k, v) in zip(enumerate(ax.flat), diffs.items()):
        X, Y = np.meshgrid(np.arange(v['x'].shape[1]), np.arange(v['x'].shape[0]))
        U = v['x']
        V = v['y']
        axi.quiver(X, Y, U, V, pivot='tip', units='xy')
        rowid = i // ncols
        colid = i % ncols
        axi.set_title("(" + str(rowid) + "," + str(colid) + ") Image: " + k)

    plt.tight_layout(True)
    plt.show()


if __name__ == '__main__':
    depth_dir = 'depth'
    grad_dir = 'depth_gradients'

    pwd = os.getcwd()
    depth_dir = os.path.join(pwd, depth_dir)
    grad_dir = os.path.join(pwd, grad_dir)

    depth_fnames = os.listdir(depth_dir)
    depth_fnames.sort()

    grads = calc_depth_grads(depth_fnames, depth_dir,
                             ksize=5, gradient_degree=1, to_save=True, save_dir=grad_dir)