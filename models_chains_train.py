import time
import os
import sys
import tqdm
import numpy as np
from pathlib import Path
from typing import Callable, Any
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from models_chain import ModelsChain
from train_results import BatchResult, EpochResult, FitResult
from models_chain_data_manager import models_chain_dataset, models_chain_dataloader
from functions import make_ckpt_fname

class ModelsChainTrainer():
    def __init__(self, models_chain:ModelsChain, num_epochs:int=400, device:torch.device=None, seed=42):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param device: torch.device to run training on (CPU or GPU).
        """
        assert isinstance(models_chain, ModelsChain), "model type is not supported"
        assert isinstance(device, torch.device), "Please provide device as torch.device"

        print(f'[I (ModelsChainTrainer)] - model={models_chain}\n'
              f'                         - num_epochs={num_epochs}\n'
              f'                         - device={device}\n',
              f'                         - seed={seed}\n')

        torch.manual_seed(42)

        self.models_chain = models_chain
        self.device       = device
        self.num_epochs   = num_epochs

        models_chain.special_fusenet.to(self.device)

    def fit(self, dl_train:DataLoader, dl_test:DataLoader, checkpoints_densedepth:str=None, 
            checkpoints_special_fusenet:str=None, early_stopping_densedepth:int=None, 
            early_stopping_special_fusenet:int=None, print_every:int=1, post_epoch_fn=None, **kw) -> (FitResult, FitResult):
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: DataLoader for the training set.
        :param dl_test: DataLoader for the test set.
        :param checkpoints_densedepth: Whether to save model to file every time the test set accuracy improves.
        Should be a string containing a filename without extension.
        :param checkpoints_special_fusenet: Whether to save model to file every time the test set accuracy improves.
        Should be a string containing a filename without extension.
        :param early_stopping_densedepth: Whether to stop training early if there is no
        test loss improvement for this number of epochs.
        :param early_stopping_special_fusenet: Whether to stop training early if there is no
        test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        self.early_stopping_densedepth_activated              = False
        self.early_stopping_special_fusenet_activated         = False
        actual_num_epochs                                     = 0
        train_densedepth_loss, test_densedepth_loss           = [], []
        train_special_fusenet_loss, test_special_fusenet_loss = [], []
        best_loss_densedepth                                  = None
        best_loss_special_fusenet                             = None
        epochs_without_improvement_densedepth                 = 0
        epochs_without_improvement_special_fusenet            = 0

        checkpoint_filename_densedepth = None
        if checkpoints_densedepth is not None:
            checkpoint_filename_densedepth = f'{checkpoints_densedepth}.pt'
            Path(os.path.dirname(checkpoint_filename_densedepth)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename_densedepth):
                print(f'[I] - Loading checkpoint file densedepth {checkpoint_filename_densedepth}')
                saved_state                                = torch.load(checkpoint_filename_densedepth,                                                                                                 map_location=self.device)
                epochs_without_improvement_densedepth = saved_state.get('ewi', epochs_without_improvement_densedepth)
                self.models_chain.densedepth.load_state_dict(saved_state['model_state'])
        checkpoint_filename_special_fusenet = None
        if checkpoints_special_fusenet is not None:
            checkpoint_filename_special_fusenet = f'{checkpoints_special_fusenet}.pt'
            Path(os.path.dirname(checkpoint_filename_special_fusenet)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename_special_fusenet):
                print(f'[I] - Loading checkpoint file special_fusenet {checkpoint_filename_special_fusenet}')
                saved_state                                = torch.load(checkpoint_filename_special_fusenet,                                                                                             map_location=self.device)
                epochs_without_improvement_special_fusenet = saved_state.get('ewi', epochs_without_improvement_special_fusenet)
                self.models_chain.special_fusenet.load_state_dict(saved_state['model_state'])

        for epoch in range(self.num_epochs):
            save_checkpoint_densedepth      = False
            save_checkpoint_special_fusenet = False
            verbose                         = False

            if epoch % print_every == 0 or epoch == self.num_epochs - 1:
                verbose = True

            self._print(f'--- EPOCH {epoch+1}/{self.num_epochs} ---', verbose) # Conditional verbose

            (train_densedepth_result, train_special_fusenet_result) = self.train_epoch(dl_train)
            (test_densedepth_result, test_special_fusenet_result)   = self.test_epoch(dl_test)
            
            train_densedepth_loss.append(train_special_fusenet_result.losses)
            train_special_fusenet_loss.append(train_special_fusenet_result.losses)
            test_densedepth_loss.append(test_special_fusenet_result.losses)
            test_special_fusenet_loss.append(test_special_fusenet_result.losses)
            
            actual_num_epochs += 1
                
            if best_loss_densedepth is None:
                best_loss_densedepth = np.mean(test_densedepth_result[-1])
            else:
                if best_loss_densedepth > np.mean(test_densedepth_result[-1]):
                    best_loss_densedepth = np.mean(test_densedepth_result[-1])
                    epochs_without_improvement_densedepth = 0
                    save_checkpoint_densedepth = True
                else: # Count the number of epochs without improvement for early stopping
                    epochs_without_improvement_densedepth += 1
            if best_loss_special_fusenet is None:
                best_loss_special_fusenet = np.mean(test_special_fusenet_result[-1])
            else:
                if best_loss_special_fusenet > np.mean(test_special_fusenet_result[-1]):
                    best_loss_special_fusenet = np.mean(test_special_fusenet_result[-1])
                    epochs_without_improvement_special_fusenet = 0
                    save_checkpoint_special_fusenet = True
                else: # Count the number of epochs without improvement for early stopping
                    epochs_without_improvement_special_fusenet += 1

            # print(f'best_loss_special_fusenet={best_loss_special_fusenet} | epochs_without_improvement_special_fusenet={epochs_without_improvement_special_fusenet}')
            if epochs_without_improvement_densedepth      == early_stopping_densedepth:
                self.early_stopping_densedepth_activated      = True
            if epochs_without_improvement_special_fusenet == early_stopping_special_fusenet:
                self.early_stopping_special_fusenet_activated = True
            if self.early_stopping_densedepth_activated and self.early_stopping_special_fusenet_activated:   
                break

            # Save models checkpoints if requested
            def save_models_checkpoints_if_requested(save_checkpoint, checkpoint_filename, best_loss,
                                                     epochs_without_improvement, model, model_name):
                if save_checkpoint and checkpoint_filename is not None:
                    saved_state = dict(best_loss=best_loss,
                                       ewi=epochs_without_improvement,
                                       model_state=model.state_dict())
                    torch.save(saved_state, checkpoint_filename)
                    print(f'[I] - Saved ' + model_name + f' checkpoint {checkpoint_filename} at epoch {epoch+1}')
                
            save_models_checkpoints_if_requested(save_checkpoint=save_checkpoint_densedepth, 
                                                 checkpoint_filename=checkpoint_filename_densedepth,
                                                 best_loss=best_loss_densedepth,
                                                 epochs_without_improvement=epochs_without_improvement_densedepth,
                                                 model=self.models_chain.densedepth, model_name="DenseDepth")
            save_models_checkpoints_if_requested(save_checkpoint=save_checkpoint_special_fusenet, 
                                                 checkpoint_filename=checkpoint_filename_special_fusenet,
                                                 best_loss=best_loss_special_fusenet,
                                                 epochs_without_improvement=epochs_without_improvement_special_fusenet,
                                                 model=self.models_chain.special_fusenet, model_name="SpecialFuseNet")

            if post_epoch_fn:
                # stop_training = post_epoch_fn(model=self.model, device=self.device, dl_test=dl_test)
                # if stop_training:
                #     print(f'[I] - Loss threshold achieved. stop training')
                #     break
                post_epoch_fn(model=self.models_chain, device=self.device, dl_test=dl_test)

        return (FitResult(actual_num_epochs, train_densedepth_loss, test_densedepth_loss),                                                       FitResult(actual_num_epochs, train_special_fusenet_loss, test_special_fusenet_loss))

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.models_chain.densedepth.train(True)      # set train mode
        self.models_chain.special_fusenet.train(True) # set train mode
        # self.model.set_requires_grad(requires_grad=True)
        # self.model.set_dropout_train(train=True)
        # self.model.set_batchnorms_train(train=True)
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.models_chain.densedepth.train(False)      # set evaluation (test) mode
        self.models_chain.special_fusenet.train(False) # set evaluation (test) mode
        # self.model.net.eval()
        # self.model.set_requires_grad(requires_grad=False)
        # self.model.set_dropout_train(train=False)
        # self.model.set_batchnorms_train(train=False)
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    def train_batch(self, batch) -> (BatchResult, BatchResult):
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data from a DataLoader.
        :return: A BatchResult containing the value of the loss function and
        the number of correctly classified samples in the batch.
        """
        rgb_rgb2d = batch['rgb_rgb2d']
        rgb       = batch['rgb']
        depth     = batch['depth']

        # Ground-Truth Depth Gradients
        x_gt      = batch['x']
        y_gt      = batch['y']

        rgb_rgb2d = rgb_rgb2d.to(self.device)
        rgb       = rgb.to(self.device)
        depth     = depth.to(self.device)
        x_gt      = x_gt.to(self.device)
        y_gt      = y_gt.to(self.device)
        xy_gt     = torch.cat((x_gt, y_gt), dim=1)

        rgb_rgb2d_batch = rgb_rgb2d
        if self.early_stopping_densedepth_activated:
            rgb_rgb2d_batch = None
        rgb_batch       = rgb
        depth_batch     = depth
        if self.early_stopping_special_fusenet_activated:
            rgb_batch       = None
            depth_batch     = None
            
        (depth_pred, xy_pred) = self.models_chain(rgb_rgb2d_batch=rgb_rgb2d_batch, rgb_batch=rgb_batch, depth_batch=depth_batch)

        batchResult_densedepth = None
        if depth_pred is not None:
            loss_densedepth       = self.models_chain.loss_densedepth(ground_truth_depth=depth, approximated_depth=depth_pred)
            self.models_chain.optimizer_densedepth.zero_grad()
            loss_densedepth.backward()
            self.models_chain.optimizer_densedepth.step()
            batchResult_densedepth = BatchResult(loss_densedepth.item())
        batchResult_special_fusenet = None
        if xy_pred is not None:
            loss_special_fusenet  = self.models_chain.loss_special_fusenet(ground_truth_grads=xy_gt, approximated_grads=xy_pred)
            self.models_chain.optimizer_fusenet.zero_grad()
            loss_special_fusenet.backward()
            self.models_chain.optimizer_fusenet.step()
            batchResult_special_fusenet = BatchResult(loss_special_fusenet.item())
            
        return (batchResult_densedepth, batchResult_special_fusenet)

    def test_batch(self, batch) -> BatchResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        rgb_rgb2d = batch['rgb_rgb2d']
        rgb       = batch['rgb']
        depth     = batch['depth']

        # Ground-Truth Depth Gradients
        x_gt      = batch['x']
        y_gt      = batch['y']

        rgb_rgb2d = rgb_rgb2d.to(self.device)
        rgb       = rgb.to(self.device)
        depth     = depth.to(self.device)
        x_gt      = x_gt.to(self.device)
        y_gt      = y_gt.to(self.device)
        xy_gt     = torch.cat((x_gt, y_gt), dim=1)

        with torch.no_grad():
            (depth_pred, xy_pred) = self.models_chain(rgb_rgb2d_batch=rgb_rgb2d, rgb_batch=rgb, depth_batch=depth)
            loss_densedepth       = self.models_chain.loss_densedepth(ground_truth_depth=depth, approximated_depth=depth_pred)
            loss_special_fusenet  = self.models_chain.loss_special_fusenet(ground_truth_grads=xy_gt, approximated_grads=xy_pred)

        return (BatchResult(loss_densedepth.item()), BatchResult(loss_special_fusenet.item()))

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader, forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None) -> (EpochResult, EpochResult):
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses_densedepth      = []
        losses_special_fusenet = []
        num_samples            = len(dl.sampler)
        num_batches            = len(dl.batch_sampler)

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
                (batch_d_res, batch_xy_res) = forward_fn(data)
                batch_d_res_loss = float('NaN')
                if batch_d_res is not None:
                    batch_d_res_loss = batch_d_res.loss
                batch_xy_res_loss = float('NaN')
                if batch_xy_res is not None:
                    batch_xy_res_loss = batch_xy_res.loss

                pbar.set_description(f'{pbar_name} (DD {batch_d_res_loss:.3f}, SFN {batch_xy_res_loss:.3f})')
                pbar.update()

                losses_densedepth.append(batch_d_res_loss)
                losses_special_fusenet.append(batch_xy_res_loss)

            avg_loss_densedepth = sum(losses_densedepth) / num_batches
            avg_loss_special_fusenet = sum(losses_special_fusenet) / num_batches
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Losses: DD {avg_loss_densedepth:.3f}, SFN {avg_loss_special_fusenet:.3f}')

        return (EpochResult(losses=losses_densedepth), EpochResult(losses=losses_special_fusenet))

    
    
    
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