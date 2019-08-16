
import os
import time
import yaml
import torch
import random
from easydict import EasyDict as edict
from torchvision.utils import save_image


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image_batch(im_batch, name_batch, output_root):

    for i in range(im_batch.size(0)):

        im_path = os.path.join(output_root, name_batch[i])

        if not os.path.isdir('/'.join(im_path.split('/')[:-1])):
            os.makedirs('/'.join(im_path.split('/')[:-1]))
        save_image(im_batch[i], im_path)


def create_dir(dir_path, clear_dir_first=True):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    if clear_dir_first:
        command_line = 'rm -rf ' + dir_path + '/*'
        os.system(command_line)


def random_seed(given_seed=None):
    if given_seed is None:
        given_seed = random.randint(1, 10000)
    print("Random Seed: ", given_seed)
    random.seed(given_seed)
    torch.manual_seed(given_seed)
    torch.cuda.manual_seed_all(given_seed)


def flush_print(input_string):
    print(input_string, end='')
    print('\b' * len(input_string), end='', flush=True)


class Clock:
    def __init__(self, start_tick=False):
        self.pre_time = 0
        if start_tick:
            self.tic()

    def tic(self):
        self.pre_time = time.time()

    def toc(self, mark=''):
        cur_time = time.time()
        print(mark + ' ({:.2f} seconds elapsed)'.format(cur_time - self.pre_time))
        self.pre_time = cur_time

    def long_toc(self, mark=''):
        cur_time = time.time()
        hours = int((time.time() - self.pre_time) // 3600)
        minutes = int(((time.time() - self.pre_time) % 3600) // 60)
        seconds = ((time.time() - self.pre_time) % 3600) % 60
        print(mark + ' ({:d} hours {:d} minutes and {:.2f} seconds elapsed)'.format(hours, minutes, seconds))
        self.pre_time = cur_time

