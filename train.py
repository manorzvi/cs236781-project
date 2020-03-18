import time
import os
import sys
import tqdm
from pathlib import Path
from typing import Callable, Any
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from models import SpecialFuseNetModel
from train_results import BatchResult, EpochResult, FitResult
from data_manager import rgbd_gradients_dataset, rgbd_gradients_dataloader
from hyperparameters import *


class FuseNetTrainer():
    def __init__(self, model:SpecialFuseNetModel, num_epochs:int=NUM_EPOCHS, device:torch.device=None):
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

        model.net.to(self.device)

    def fit(self, dl_train:DataLoader, dl_test:DataLoader, checkpoints:str=None,
            early_stopping:int=None, print_every:int=1, post_epoch_fn=None, **kw) -> FitResult:
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
        train_loss,test_loss       = [],[]
        best_loss                  = None
        actual_num_epochs          = 0
        epochs_without_improvement = 0

        checkpoint_filename        = None
        if checkpoints is not None:
            checkpoint_filename = f'{checkpoints}.pt'
            Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename):
                print(f'[info] - Loading checkpoint file {checkpoint_filename}')
                saved_state = torch.load(checkpoint_filename, map_location=self.device)
                epochs_without_improvement = saved_state.get('ewi', epochs_without_improvement)
                self.model.net.load_state_dict(saved_state['model_state'])

        for epoch in range(self.num_epochs):
            save_checkpoint = False
            verbose         = False

            if epoch % print_every == 0 or epoch == self.num_epochs - 1:
                verbose = True

            self._print(f'--- EPOCH {epoch+1}/{self.num_epochs} ---', verbose) # Conditional verbose

            train_result = self.train_epoch(dl_train)
            test_result  = self.test_epoch(dl_test)
            train_loss.append(train_result.losses)
            test_loss.append(test_result.losses)
            actual_num_epochs += 1

            if best_loss is None:
                best_loss = test_loss[-1]
            else:
                if best_loss > test_loss[-1]: # If the accuracy in the epoch improved (the higher the better)
                    best_loss = test_loss[-1]
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
                print(f'[info] - Saved checkpoint {checkpoint_filename} at epoch {epoch+1}')

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

        return FitResult(actual_num_epochs, train_loss, test_loss)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        # TODO: check if it replace setting model's parameters manually as in model.set_requires_grad():
        #  self.model.set_requires_grad(True) (manorz, 03/14/20)
        self.model.net.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        # TODO: check if it replace setting model's parameters manually as in model.set_requires_grad():
        #  self.model.set_requires_grad(False) (manorz, 03/14/20)
        self.model.net.train(False)  # set evaluation (test) mode
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

    def test_batch(self, batch) -> BatchResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
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

        with torch.no_grad():
            xy   = self.model(rgb_batch=rgb, depth_batch=depth)
            loss = self.model.loss(ground_truth_grads=xy_gt, approximated_grads=xy)

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
        with tqdm.tqdm(desc=pbar_name, total=num_batches,file=pbar_file) as pbar:
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

    image_size       = (224, 224)
    train_test_ratio = 0.9
    batch_size       = 4
    num_workers      = 4
    print(f'[I] - general hyper-parameters:\n'
          f'      -------------------------\n'
          f'    - image_size = {image_size}\n'
          f'    - train_test_ratio = {train_test_ratio}\n'
          f'    - batch_size = {batch_size}\n'
          f'    - num_workers = {num_workers}')

    betas = (0.9, 0.99)
    lr = 0.001
    momentum = 0.9
    weight_decay = 0.0005
    print(f'[I] - optimizer hyper-parameters:\n'
          f'      ---------------------------\n'
          f'    - betas = {betas}\n'
          f'    - learning_rate = {lr}\n'
          f'    - momentum = {momentum}\n'
          f'    - weight_decay = {weight_decay}')

    rgb_tf = T.Compose([
        # Resize to constant spatial dimensions
        T.Resize(image_size),
        # PIL.Image -> torch.Tensor
        T.ToTensor(),
        # Dynamic range [0,1] -> [-1, 1]
        T.Normalize(mean=(.5, .5, .5), std=(.5, .5, .5)),
    ])
    print(f'[I] - rgb transforms = {rgb_tf}')
    depth_tf = T.Compose([
        # Resize to constant spatial dimensions
        T.Resize(image_size),
        # PIL.Image -> torch.Tensor
        T.ToTensor(),
        # Dynamic range [0,1] -> [-1, 1]
        T.Normalize(mean=(.5,), std=(.5,)),
    ])
    print(f'[I] - depth transforms = {depth_tf}')

    dl_train, dl_test = rgbd_gradients_dataloader(root=dataset_dir,
                                                  batch_size=batch_size,
                                                  num_workers=num_workers,
                                                  train_test_ration=train_test_ratio,
                                                  rgb_transforms=rgb_tf, depth_transforms=depth_tf)
    sample_batch = next(iter(dl_test))
    rgb_size     = tuple(sample_batch['rgb'].shape[1:])
    depth_size   = tuple(sample_batch['depth'].shape[1:])
    grads_size   = tuple(sample_batch['x'].shape[1:])

    fusenetmodel = SpecialFuseNetModel(rgb_size=rgb_size, depth_size=depth_size,
                                       grads_size=grads_size, device=device)

    trainer = FuseNetTrainer(model=fusenetmodel, device=device)

    checkpoint_file = 'checkpoints/special_fusenet'
    if os.path.isfile(f'{checkpoint_file}.pt'):
        print(f'[W] - remove old checkpoint file ({checkpoint_file})')
        try:
            os.remove(f'{checkpoint_file}.pt')
        except:
            print(f'[E] - failed to remove old checkpoint file ({checkpoint_file})')
            exit()

    trainer.fit(dl_train, dl_test, early_stopping=20, print_every=10, checkpoints=checkpoint_file)