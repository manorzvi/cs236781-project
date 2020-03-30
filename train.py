import time
import os
import sys
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, Any
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from models import SpecialFuseNetModel
from train_results import BatchResult, EpochResult, FitResult
from data_manager import rgbd_gradients_dataset, rgbd_gradients_dataloader
from functions import make_ckpt_fname

class FuseNetTrainer():
    def __init__(self, model:SpecialFuseNetModel, num_epochs:int=1, device:torch.device=None):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param device: torch.device to run training on (CPU or GPU).
        """
        assert isinstance(model,SpecialFuseNetModel), "model type is not supported"
        assert isinstance(device, torch.device),      "Please provide device as torch.device"
        self.model       = model
        self.device      = device
        self.num_epochs  = num_epochs
        self.current_test_run = 0
        model.net.to(self.device)

    def fit(self, dl_train:DataLoader, dl_test:DataLoader, constant_index:int, print_every:int=1,
            checkpoints:str=None, post_epoch_fn=None, early_stopping:int=None, **kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: DataLoader for the training set.
        :param dl_test: DataLoader for the test set.
        :param checkpoints: Whether to save model to file every time the test set accuracy improves.
        Should be a string containing a filename without extension.
        :param early_stopping: Whether to stop training early if there is no
        test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        train_loss, test_loss      = [],[]
        best_loss                  = None
        actual_num_epochs          = 0
        epochs_without_improvement = 0
        checkpoint_filename        = None
        if checkpoints is not None:
            checkpoint_filename = f'{checkpoints}.pt'
            Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename):
                print(f'[I] - Loading checkpoint file {checkpoint_filename}')
                saved_state = torch.load(checkpoint_filename, map_location=self.device)
                epochs_without_improvement = saved_state.get('ewi', epochs_without_improvement)
                self.model.net.load_state_dict(saved_state['model_state'])
        for epoch in range(self.num_epochs):
            save_checkpoint = False
            verbose         = False
            if epoch % print_every == 0 or epoch == self.num_epochs - 1:
                verbose = True
            self._print(f'--- EPOCH {epoch+1}/{self.num_epochs} ---', verbose) # Conditional verbose
            if constant_index < 0:
                train_result = self.train_epoch(dl_train)
                train_loss.append(train_result.losses)
            test_result  = self.test_epoch(dl_test)
            test_loss.append(test_result.losses)
            actual_num_epochs += 1
            if best_loss is None:
                best_loss = np.mean(test_loss[-1])
            else:
                if best_loss > np.mean(test_loss[-1]):
                    best_loss = np.mean(test_loss[-1])
                    epochs_without_improvement = 0
                    save_checkpoint = True
                else: # Count the number of epochs without improvement for early stopping
                    epochs_without_improvement += 1
            if epochs_without_improvement == early_stopping:
                break
            # Save model checkpoint if requested
            if save_checkpoint and checkpoint_filename is not None:
                saved_state = dict(best_loss=best_loss,
                                   ewi=epochs_without_improvement,
                                   model_state=self.model.net.state_dict())
                torch.save(saved_state, checkpoint_filename)
                print(f'[I] - Saved checkpoint {checkpoint_filename} at epoch {epoch+1}')
            if constant_index < 0:
                if post_epoch_fn:
                    # stop_training = post_epoch_fn(model=self.model, device=self.device, dl_test=dl_test)
                    # if stop_training:
                    #     print(f'[I] - Loss threshold achieved. stop training')
                    #     break
                    post_epoch_fn(model=self.model, device=self.device, dl_test=dl_test)
        return FitResult(actual_num_epochs, train_loss, test_loss)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.net.train(True)  # set train mode
        self.model.set_requires_grad(requires_grad=True)
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.net.train(False)  # set evaluation (test) mode
        self.model.set_requires_grad(requires_grad=False)
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data from a DataLoader.
        :return: A BatchResult containing the value of the loss function and
        the number of correctly classified samples in the batch.
        """
        rgb   = batch['rgb']
        depth = batch['depth']
        # Ground-Truth Depth Gradients
        x_gt  = batch['x']
        y_gt  = batch['y']
        rgb   = rgb.to(self.device)
        depth = depth.to(self.device)
        x_gt  = x_gt.to(self.device)
        y_gt  = y_gt.to(self.device)
        xy_gt = torch.cat((x_gt, y_gt), dim=1)
        xy    = self.model(rgb_batch=rgb,depth_batch=depth)
        loss  = self.model.loss(ground_truth_grads=xy_gt, approximated_grads=xy)
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
        return BatchResult(loss.item())

    def test_batch(self, batch, plot_test_batch_every=4000) -> BatchResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        plot_test_batch = False
        if 0 < plot_test_batch_every and self.current_test_run % plot_test_batch_every == 0:
            plot_test_batch = True
        self.current_test_run += 1
        rgb   = batch['rgb']
        depth = batch['depth']
        # Ground-Truth Depth Gradients
        x_gt  = batch['x']
        y_gt  = batch['y']
        rgb   = rgb.to(self.device)
        depth = depth.to(self.device)
        x_gt  = x_gt.to(self.device)
        y_gt  = y_gt.to(self.device)
        xy_gt = torch.cat((x_gt, y_gt), dim=1)
        with torch.no_grad():
            output_xy   = self.model(rgb_batch=rgb, depth_batch=depth)
            if plot_test_batch:
                output_xy = output_xy.cpu()
                x = output_xy[0][0]
                y = output_xy[0][1]
                x = torch.unsqueeze(x, 0)[0]
                y = torch.unsqueeze(y, 0)[0]
                x_gt = x_gt[0][0]
                y_gt = y_gt[0][0]
                i = 0
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 12),
                                 gridspec_kw=dict(wspace=0.1, hspace=0.2, left=0, right=1),
                                 subplot_kw={'aspect': 1})
                X,Y = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
                axes[1].quiver(X, Y, x, y, pivot='tip', units='xy')
                axes[1].set_ylim(axes[1].get_ylim()[::-1]) # Transpose, to look like the original RGB and Depth images.
                X,Y = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
                axes[0].quiver(X, Y, x_gt.cpu(), y_gt.cpu(), pivot='tip', units='xy')
                axes[0].set_ylim(axes[0].get_ylim()[::-1]) # Transpose, to look like the original RGB and Depth images.
                plt.show()
            loss = self.model.loss(ground_truth_grads=xy_gt.cpu(), approximated_grads=output_xy.cpu())
        return BatchResult(loss.item())

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader, forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses      = []
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)
        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size
        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')
        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)
                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()
                losses.append(batch_res.loss)
            avg_loss = sum(losses) / num_batches
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}')
        return EpochResult(losses=losses)

if __name__ == '__main__':
    cwd = os.getcwd()
    print(f'[I] - cwd = {cwd}')
    dataset_dir = os.path.join(cwd, 'data/nyuv2')
    print(f'[I] - dataset_dir = {dataset_dir}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[I] - device = {device}')

    dl_train, dl_test = rgbd_gradients_dataloader(root=dataset_dir, use_transforms=True)

    sample_batch = next(iter(dl_train))
    rgb_size     = tuple(sample_batch['rgb'].shape[1:])
    depth_size   = tuple(sample_batch['depth'].shape[1:])
    grads_size   = tuple(sample_batch['x'].shape[1:])

    fusenetmodel = SpecialFuseNetModel(rgb_size=rgb_size, depth_size=depth_size, grads_size=grads_size, device=device)

    trainer = FuseNetTrainer(model=fusenetmodel, device=device)

    checkpoint_file = make_ckpt_fname()
    print(f'[I] - checkpoint file = {checkpoint_file}')

    if os.path.isfile(f'{checkpoint_file}.pt'):
        print(f'[W] - remove old checkpoint file ({checkpoint_file})')
        try:
            os.remove(f'{checkpoint_file}.pt')
        except:
            print(f'[E] - failed to remove old checkpoint file ({checkpoint_file})')
            exit()

    res = trainer.fit(dl_train, dl_test, early_stopping=20, print_every=10, checkpoints=checkpoint_file)