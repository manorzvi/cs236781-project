import os
import numpy as np
from functions import mkdirs
import time
import pickle

class Visualizer():
    def __init__(self, window_size:int=224, name:str='default_name', palette_file='palette.txt',
                 display_id:int=1, use_html:bool=True, display_ncols:int=4,display_server:str='http://localhost',
                 display_port:int=8097,display_env:str='main', ckpts_dir:str='./checkpoints', phase='train'):
        self.window_size = window_size
        self.name        = name
        assert os.path.exists(palette_file)
        self.impalette   = list(np.genfromtxt(palette_file,dtype=np.uint8).reshape(3*256))
        self.saved       = False
        self.display_id  = display_id
        self.use_html    = use_html

        if self.display_id > 0:
            import visdom
            self.ncols   = display_ncols
            self.vis     = visdom.Visdom(server=display_server, port=display_port, env=display_env,
                                         raise_exceptions=True)

        if self.use_html:
            self.web_dir = os.path.join(ckpts_dir, name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print(f'[debug] - create web directory {self.web_dir} ...')
            mkdirs([self.web_dir, self.img_dir])

        self.loss_log_name = os.path.join(ckpts_dir, name, 'loss_log.txt')
        self.val_log_name  = os.path.join(ckpts_dir, name, 'acc_log.txt')
        if phase == "train":
            with open(self.loss_log_name, "w") as loss_log_file:
                now = time.strftime("%c")
                loss_log_file.write(f'=================== Training Loss ({now}) ===================\n')
            with open(self.val_log_name, "w") as val_log_file:
                now = time.strftime("%c")
                val_log_file.write( f'================ Validation Accuracy ({now}) ================\n')
        self.conf_mat_name = os.path.join(ckpts_dir, name, phase + '_conf_mat.pkl')
        with open(self.conf_mat_name, "wb") as conf_mat_file:
            conf_mat = {}
            pickle.dump(conf_mat, conf_mat_file)

    def reset(self):
        self.saved = False

    def throw_visdom_connection_error(self):
        print('\n\nCould not connect to Visdom server (https://github.com/facebookresearch/visdom) for displaying training progress.\nYou can suppress connection to Visdom using the option --display_id -1. To install visdom, run \n$ pip install visdom\n, and start the server by \n$ python -m visdom.server.\n\n')
        exit(1)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result):
        if self.display_id > 0:  # show images in the browser
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    if label == 'mask' or label == 'output':
                        image_numpy = util.tensor2labelim(image,self.impalette)
                    else:
                        image_numpy = util.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.throw_visdom_connection_error()

            else:
                idx = 1
                for label, image in visuals.items():
                    if label == 'mask' or label == 'output':
                        image_numpy = util.tensor2labelim(image,self.impalette)
                    else:
                        image_numpy = util.tensor2im(image)
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image in visuals.items():
                if label == 'mask' or label == 'output':
                    image_numpy = util.tensor2labelim(image,self.impalette)
                else :
                    image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image in visuals.items():
                    if label == 'mask' or label == 'output':
                        image_numpy = util.tensor2labelim(image,self.impalette)
                    else:
                        image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # losses: dictionary of error labels and values
    def plot_current_losses(self, epoch, counter_ratio, opt, losses):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.throw_visdom_connection_error()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, losses, t, t_data):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_current_scores(self, epoch, loss, glob, mean, iou):
        message = '(epoch {0:} test loss: {1:.3f} '.format(epoch, loss)
        message += 'glob acc : {0:.2f}, mean acc : {1:.2f}, IoU : {2:.2f}'.format(glob, mean, iou)

        print(message)
        with open(self.val_log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def save_confusion_matrix(self, conf_mat, epoch):
        conf_mats = pickle.load(open(self.conf_mat_name, "rb"))
        conf_mats["epoch"+str(epoch)] = conf_mat
        conf_mats["epoch"+str(epoch)+"_scores"] = np.asarray(util.getScores(conf_mat))
        pickle.dump(conf_mats, open(self.conf_mat_name, "wb"))