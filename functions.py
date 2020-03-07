import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import shutil


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def torch2np_u8(x):
    if len(x.shape) == 4: # Squeeze the batch dimension if exist
        assert x.shape[0] == 1, "Non-empy batch dimension (don't know which sample to pick)"
        x = x.squeeze()
    if x.shape[0] == 1: # Squeeze single color channel
        x = x.squeeze()
    else:
        x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x


# NOTE: Following functions are legacy and deprecated. DON'T USE! (manorz, 03/07/20)
#  ---------------------------------------------------------------------------------

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


def scaling(img,scaling_factor):

    width  = int(img.shape[1] * scaling_factor)
    height = int(img.shape[0] * scaling_factor)

    dsize = (width, height)

    # resize image
    return cv2.resize(img, dsize)


def one_one_normalization(data):
    data = 2*((data-np.min(data))/(np.max(data)-np.min(data)))-1
    return data


def alltogether_normalization(grads):
    for i, (k, v) in enumerate(grads.items()):
        if i == 0:
            datax = v['x']
            datay = v['y']
        else:
            datax = np.concatenate((datax, v['x']), axis=0)
            datay = np.concatenate((datay, v['y']), axis=0)

        print(f'-{i}) |datax|={datax.shape}, |datay|={datay.shape}')

    datax = one_one_normalization(datax)
    datay = one_one_normalization(datay)

    datax = np.split(datax, len(grads), axis=0)
    datay = np.split(datay, len(grads), axis=0)

    for i, (dx, dy, v) in enumerate(zip(datax, datay, grads.values())):
        v['x'] = dx
        v['y'] = dy

    return grads


def eachone_normalization(grads):
    for i, (k, v) in enumerate(grads.items()):
        v['x'] = one_one_normalization(v['x'])
        v['y'] = one_one_normalization(v['y'])
    return grads


def calc_depth_grads(depth_fnames, depth_dir, ksize=-1, gradient_degree=1,
                     scale=0.1, to_save=False, save_dir=None,
                     normalization=None):
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

    assert normalization in [None,'alltogether','eachone'], 'if not None, normalization can be one of the follows:\n' \
                                                            '1) alltogether - normalize the whole data-set\n' \
                                                            '2) eachone - normalize each sample independently'

    if to_save:
        x_dir = os.path.join(save_dir,'x')
        y_dir = os.path.join(save_dir,'y')
        os.mkdir(x_dir)
        os.mkdir(y_dir)

    depth_abs_fnames = [os.path.join(depth_dir,f) for f in depth_fnames]
    ret_val = {}
    for i,fname in enumerate(depth_abs_fnames):
        print(f'{i}) {os.path.basename(fname)} ...',end=' ')
        img = cv2.imread(fname,0)
        img_r = scaling(img,scale)
        grad_x = cv2.Sobel(img_r,cv2.CV_64F,gradient_degree,0,ksize=ksize)
        grad_y = cv2.Sobel(img_r,cv2.CV_64F,0,gradient_degree,ksize=ksize)
        # grad_x = one_one_normalization(grad_x)
        # grad_y = one_one_normalization(grad_y)
        if to_save:
            np.save(os.path.join(x_dir, os.path.basename(fname)[:-4]), grad_x)
            np.save(os.path.join(y_dir, os.path.basename(fname)[:-4]), grad_y)
        ret_val[os.path.basename(fname)[:-4]] = {'x' : grad_x,
                                                 'y' : grad_y}
        print('done.')

    if normalization:
        print('Minus One-One Normalization:\n'
              '---------------------------')
        if normalization == 'alltogether':
            print('All-Together Normalization ... ')
            ret_val = alltogether_normalization(ret_val)
            print('Done.')
        if normalization == 'eachone':
            print('Each-One Normalization ... ')
            ret_val = eachone_normalization(ret_val)
            print('Done.')

    return ret_val


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


if __name__ == '__main__':
    data_dir  = 'data/nyuv2'
    depth_dir = 'depth'
    save_dir  = 'data/nyuv2'

    pwd       = os.getcwd()
    data_dir  = os.path.join(pwd,      data_dir)
    depth_dir = os.path.join(data_dir, depth_dir)
    save_dir  = data_dir

    depth_fnames = os.listdir(depth_dir)
    depth_fnames.sort()

    grads = calc_depth_grads(depth_fnames, depth_dir,
                             ksize=5, gradient_degree=1, to_save=True, save_dir=save_dir, normalization='alltogether')