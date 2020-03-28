import os
from pprint import pprint
import torch
import itertools
from shutil import copyfile

from models import SpecialFuseNetModel
from data_manager import rgbd_gradients_dataloader
from train import FuseNetTrainer
from functions import make_ckpt_fname

def grid_search(combinations:list):

    for combintation in combinations:
        image_size              = combintation[0]
        train_test_ratio        = combintation[1]
        batch_size              = combintation[2]
        num_workers             = combintation[3]
        betas                   = combintation[4]
        lr                      = combintation[5]
        momentum                = combintation[6]
        weight_decay            = combintation[7]
        step_size               = combintation[8]
        gamma                   = combintation[9]
        num_epochs              = combintation[10]

        checkpoint_folder         = 'checkpoints/'
        checkpoint_file_name      = make_ckpt_fname(image_size, batch_size, betas, lr, momentum)
        if OVERFITTING_TRAINING:
            checkpoint_file_name += '_overfit'
        checkpoint_file           = os.path.join(cwd, checkpoint_folder, checkpoint_file_name)
        checkpoint_res_file       = os.path.join(cwd, checkpoint_folder, checkpoint_file_name + '_res')
        checkpoint_hp_file        = os.path.join(cwd, checkpoint_folder, checkpoint_file_name + '_hp')

        if os.path.isfile(f'{checkpoint_file}.pt'):
            print(f'[I] - {checkpoint_file}.pt exist ... remove ', end='')
            os.remove(f'{checkpoint_file}.pt')
            print('done.')
        if os.path.isfile(f'{checkpoint_res_file}.pkl'):
            print(f'[I] - {checkpoint_res_file}.pkl exist ... remove ', end='')
            os.remove(f'{checkpoint_res_file}.pkl')
            print('done.')
        if os.path.isfile(f'{checkpoint_hp_file}.py'):
            print(f'[I] - {checkpoint_hp_file}.py exist ... remove ', end='')
            os.remove(f'{checkpoint_hp_file}.py')
            print('done.')

        os.makedirs(os.path.dirname(checkpoint_hp_file)+'.py', exist_ok=True)
        with open(checkpoint_hp_file+'.py', "w") as hpf:
            print(f"IMAGE_SIZE={image_size}",               file=hpf)
            print(f"TRAIN_TEST_RATIO={train_test_ratio}",   file=hpf)
            print(f"BATCH_SIZE={batch_size}",               file=hpf)
            print(f"NUM_WORKERS={num_workers}",             file=hpf)
            print(f"BETAS={betas}",                         file=hpf)
            print(f"LR={lr}",                               file=hpf)
            print(f"MOMENTUM={momentum}",                   file=hpf)
            print(f"WEIGHT_DECAY={weight_decay}",           file=hpf)
            print(f"STEP_SIZE={step_size}",                 file=hpf)
            print(f"GAMMA={gamma}",                         file=hpf)
            print(f"NUM_EPOCHS={num_epochs}",               file=hpf)

        print("[I] - Current Hyper-Parameters:\n"
              "-------------------------------")
        with open(checkpoint_hp_file+'.py', "r") as hpf:
            print(hpf.read())

        dl_train, dl_test = rgbd_gradients_dataloader(root=DATASET_DIR, use_transforms=True,
                                                      batch_size=batch_size, num_workers=num_workers,
                                                      train_test_ratio=train_test_ratio, image_size=image_size,
                                                      overfit_mode=OVERFITTING_TRAINING)
        sample_batch      = next(iter(dl_train))
        rgb_size          = tuple(sample_batch['rgb'].shape[1:])
        depth_size        = tuple(sample_batch['depth'].shape[1:])
        grads_size        = tuple(sample_batch['x'].shape[1:])
        print(f'[I] - RGB SIZE={rgb_size}, DEPTH SIZE={depth_size}, GRADS SIZE={grads_size}')

        if OVERFITTING_TRAINING:
            fusenetmodel = SpecialFuseNetModel(rgb_size=rgb_size, depth_size=depth_size, grads_size=grads_size,
                                               sgd_lr=lr, sgd_momentum=momentum, sgd_wd=weight_decay,
                                               device=device, overfit_mode=OVERFITTING_TRAINING, dropout_p=0)
        else:
            fusenetmodel = SpecialFuseNetModel(rgb_size=rgb_size, depth_size=depth_size, grads_size=grads_size,
                                               sgd_lr=lr, sgd_momentum=momentum, sgd_wd=weight_decay,
                                               device=device, overfit_mode=OVERFITTING_TRAINING)

        trainer = FuseNetTrainer(model=fusenetmodel, device=device, num_epochs=num_epochs)

        res = trainer.fit(dl_train, dl_test, early_stopping=20, print_every=10, checkpoints=checkpoint_file)

        res.save(checkpoint_res_file)

        print("-"*100)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[I] - Using device: {device}')

    cwd                      = os.getcwd()
    hyperparameters_filename = 'hyperparameters'
    overfit_data_dir_path    = 'data/nyuv2_overfit'
    normal_data_dir_path     = 'data/nyuv2'

    OVERFITTING_TRAINING = True
    # OVERFITTING_TRAINING = False
    print(f'[I] - Overfitting Mode: {OVERFITTING_TRAINING}')

    IMAGE_SIZE           = [(64, 64), (224, 224)]
    TRAIN_TEST_RATIO     = [0.8]
    BATCH_SIZE           = [4, 16, 32, 64]
    NUM_WORKERS          = [4]

    BETAS                = [(0.9, 0.99)]
    LR                   = [0.001, 0.01]
    MOMENTUM             = [0.9, 0.99]
    WEIGHT_DECAY         = [0.0005]

    STEP_SIZE            = [1000]
    GAMMA                = [0.1]

    NUM_EPOCHS           = [400]

    if OVERFITTING_TRAINING:
        DATASET_DIR      = os.path.join(cwd, overfit_data_dir_path)
    else:
        DATASET_DIR      = os.path.join(cwd, normal_data_dir_path)
    print(f'[I] - DATASET_DIR: {DATASET_DIR}')

    all_combintations = list(itertools.product(*[IMAGE_SIZE, TRAIN_TEST_RATIO, BATCH_SIZE,
                                                 NUM_WORKERS, BETAS, LR, MOMENTUM,
                                                 WEIGHT_DECAY, STEP_SIZE, GAMMA, NUM_EPOCHS]))
    print(f'[I] - All Hyperparameters Combinations:\n'
          f'---------------------------------------')
    pprint(all_combintations)

    grid_search(all_combintations)