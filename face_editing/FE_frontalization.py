
import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.parallel import data_parallel
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

from lightcnn.public import LightCNN29_v2

# from face_editing.utils import *
# from face_editing.criterion import *
# from face_editing.dataset import MultiPIE
# from face_editing.network.pose_gan import *

from utils import *
from criterion import *
from dataset import MultiPIE
from network.pose_gan import *


from PIL import Image
class Solver:
    def __init__(self, cfg):

        self.cfg = cfg

        cudnn.benchmark = True
        os.environ['CUDA_VISIBLE_DEVICES'] = self.cfg.gpu

        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

        self.out_root = os.path.join(os.path.expanduser(cfg.base_root), 'model_output', get_solver_name(), self.cfg.version)
        self.tsbd_dir = os.path.join(os.path.expanduser(cfg.base_root), 'tsbd_output', get_solver_name(), self.cfg.version)
        self.ckpt_dir = os.path.join(self.out_root, 'checkpoint')

        create_dir(self.out_root, clear_dir_first=False)
        create_dir(self.tsbd_dir, clear_dir_first=(True if self.cfg.mode == 'train' else False))
        create_dir(self.ckpt_dir, clear_dir_first=False)

        if self.cfg.mode == 'train':
            self.writer = SummaryWriter(log_dir=self.tsbd_dir)
        self.timer = Clock()

        self.cur_iter = 0
        self.epoch = 0

        if self.cfg.dataset == 'FR':
            dataset = FR.get_dataset(mode=self.cfg.mode)
        elif self.cfg.dataset == 'Multi-PIE':
            dataset = MultiPIE.get_dataset(mode=self.cfg.mode)
        else:
            raise NotImplementedError

        # data loader
        if self.cfg.mode == 'train':
            # self.data_loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=16, drop_last=True)
            self.data_loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=0, drop_last=True)


        else:
            self.sample_dir = os.path.join(self.out_root, 'output-' + self.cfg.test_model_epoch)
            create_dir(self.sample_dir, clear_dir_first=False)

            # self.data_loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=16)
            self.data_loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=0)


        # loss function
        if self.cfg.gan == 'wgan':
            self.gan_loss = WassersteinAdversarialLoss()
            self.gp = GradientPenalty()
            self.loss = 'loss_adv_real_d, loss_adv_fake_d, loss_gp_d, loss_adv_g, '

        if self.cfg.gan == 'lsgan':
            self.gan_loss = AdversarialLoss()
            self.loss = 'loss_adv_real_d, loss_adv_fake_d, loss_adv_g, '

        if self.cfg.use_l1_loss:
            self.l1_loss = L1Loss()
            self.loss += 'loss_l1_g, '

        if self.cfg.use_perceptual_loss:
            self.p_loss = PerceptualLoss(feature_extractor=LightCNN29_v2())
            self.loss += 'loss_p, '

        self.loss = '[' + self.loss + ']'

        # model
        self.c_dim = 0

        self.g = ResnetGenerator().cuda()
        self.d = NLayerDiscriminator(image_size=self.cfg.image_size, repeat_num=self.cfg.trans_d_num).cuda()

        self.g.train()
        self.d.train()

        # optimizer
        self.optimizer_g = optim.Adam(filter(lambda p: p.requires_grad, self.g.parameters()),
                                      lr=self.cfg.learning_rate, betas=(0.5, 0.999))

        self.optimizer_d = optim.Adam(filter(lambda p: p.requires_grad, self.d.parameters()),
                                      lr=self.cfg.learning_rate, betas=(0.5, 0.999))

        self.scheduler_g = MultiStepLR(self.optimizer_g, milestones=[cfg.step1, cfg.step2], gamma=0.1)
        self.scheduler_d = MultiStepLR(self.optimizer_d, milestones=[cfg.step1, cfg.step2], gamma=0.1)

    def rgbToGray(self, rgbImg):
        R = rgbImg[:, 0, :, :]
        G = rgbImg[:, 1, :, :]
        B = rgbImg[:, 2, :, :]
        gray = 0.299 * R + 0.587 * G + 0.114 * B
        # newImg = rgbImg.clone()
        #
        # newImg[:, 0, :, :] = gray
        # newImg = newImg[:, 0, :, :]

        newImg = gray.view(48, 1, 128, 128)
        return newImg

    def train(self):

        self.load_model_to_train()

        os.system('clear')
        self.timer.tic()
        for epoch in range(self.cfg.training_epoch + 1):
            self.update_lr_rate()
            for iteration, batch in enumerate(self.data_loader, 0):
                self.cur_iter += 1
                self.epoch = epoch

                source_image, target_image, source_image_uv, target_image_uv = self.parse_batch(batch)

                ##### forward #####
                trans_image = data_parallel(self.g, (source_image, None))
                target_pred = data_parallel(self.d, target_image)
                trans_pred = data_parallel(self.d, trans_image.detach())

                ## forward_uv ##
                trans_image_uv = data_parallel(self.g, (source_image_uv, None))
                target_pred_uv = data_parallel(self.d, target_image_uv)
                trans_pred_uv = data_parallel(self.d, trans_image_uv.detach())

                ##### forward #####

                loss_adv_real_d, loss_adv_fake_d = self.gan_loss.loss_d(target_pred, trans_pred)
                loss_adv_d = 0.5 * (loss_adv_real_d + loss_adv_fake_d)
                loss_d = loss_adv_d

                if self.cfg.gan == 'wgan':
                    loss_gp_d = self.gp(target_image, trans_image, self.d)
                    loss_d = loss_d + 10 * loss_gp_d

                self.optimize(self.optimizer_d, loss_d)

                if self.cur_iter % self.cfg.training_g_stride == 0:

                    ##### forward #####
                    trans_image = data_parallel(self.g, (source_image, None))
                    trans_pred = data_parallel(self.d, trans_image)

                    ## forward_uv ##
                    trans_image_uv = data_parallel(self.g, (source_image_uv, None))
                    trans_pred_uv = data_parallel(self.d, trans_image_uv.detach())
                    ##### forward #####

                    loss_adv_g_1 = self.gan_loss.loss_g(trans_pred)
                    loss_adv_g_2 = self.gan_loss.loss_g(trans_pred)
                    loss_g = 0.5 * (loss_adv_g_1 + loss_adv_g_2)
                    # loss_g = loss_adv_g

                    if self.cfg.use_l1_loss:
                        loss_l1_g = self.cfg.l1_loss_weight * self.l1_loss(trans_image, target_image)

                        loss_l1_g_uv = self.cfg.ls_loss_weight * self.l1_loss(trans_image_uv, target_image_uv)

                        loss_g = loss_l1_g + loss_g + loss_l1_g_uv

                    if self.cfg.use_perceptual_loss:

                        new_trans_image = self.rgbToGray(trans_image)
                        new_source_image = self.rgbToGray(source_image)

                        new_trans_image_uv = self.rgbToGray(trans_image_uv)
                        new_source_image_uv = self.rgbToGray(source_image_uv)

                        loss_p = self.p_loss(new_trans_image, new_source_image)
                        loss_p_uv = self.p_loss(new_trans_image_uv, new_source_image_uv)
                        loss_g = loss_p + loss_g + loss_p_uv

                    self.optimize(self.optimizer_g, loss_g)
                    self.log(eval(self.loss), self.loss)

                if self.cur_iter % self.cfg.report_interval == 0:
                    self.record_scalar(eval(self.loss), self.loss)
                    self.record_image([denorm(item) for item in [source_image, trans_image, target_image]])

            self.save_model()
        self.timer.long_toc('finish.')

    def test(self):
        self.load_model_to_test()
        self.g.eval()

        self.timer.tic()
        with torch.no_grad():
            for iteration, batch in enumerate(self.data_loader, 0):

                ##### forward #####
                source_image, target_image, source_image_name = self.parse_batch(batch)
                trans_image = data_parallel(self.g, (source_image, None))
                ##### forward #####

                output_images = denorm(torch.cat([source_image, trans_image, target_image], dim=3))
                # output_images = denorm(trans_image)

                save_image_batch(output_images, source_image_name, self.sample_dir)

        self.timer.long_toc('epoch' + self.cfg.test_model_epoch)

    def update_lr_rate(self):
        self.scheduler_g.step()
        self.scheduler_d.step()

    def load_model_to_train(self):
        if self.cfg.fine_tune:
            print('loading pretrained network...')
            g_path = os.path.join(self.cfg.train_model_path, '-'.join(['epoch', str(self.cfg.train_model_epoch), 'g.pth']))
            self.g.load_state_dict(torch.load(g_path))
            d_path = os.path.join(self.cfg.train_model_path, '-'.join(['epoch', str(self.cfg.train_model_epoch), 'd.pth']))
            self.d.load_state_dict(torch.load(d_path))
            if self.cfg.use_uv_loss:
                uv_path = os.path.join(self.cfg.train_model_path, '-'.join(['epoch', str(self.cfg.train_model_epoch), 'd_uv.pth']))
                self.d_uv.load_state_dict(torch.load(uv_path))
        self.load_model_to_enhance()

    def load_model_to_enhance(self):
        if self.cfg.goal == 'enhance':
            warnings.warn('loading enhance model is depending on the definitions of the models')

            pretrained_dict = torch.load(os.path.expanduser(self.cfg.enhance_g_path))
            model_dict = self.g.state_dict()
            print('loading transfer net ...')
            for k, v in pretrained_dict.items():
                for kk in model_dict.keys():
                    if '.'.join(['transfer_net', k]) == kk:
                        model_dict[kk] = v
                        # print('initializing {} with pretrained {}'.format(kk, k))
            self.g.load_state_dict(model_dict)

            pretrained_dict = torch.load(os.path.expanduser(self.cfg.enhance_d_path))
            model_dict = self.d.state_dict()
            print('loading attribute net ...')
            for k, v in pretrained_dict.items():
                for kk in model_dict.keys():
                    if '.'.join(['base_net', k]) == kk:
                        model_dict[kk] = v
                        # print('initializing {} with pretrained {}'.format(kk, k))
            self.d.load_state_dict(model_dict)

            if self.cfg.use_uv_loss:
                self.d_uv.load_state_dict(torch.load(os.path.expanduser(self.cfg.enhance_d_uv_path)))

    def load_model_to_test(self):
        g_path = os.path.join(self.cfg.test_model_path, '-'.join(['epoch', str(self.cfg.test_model_epoch), 'g.pth']))
        self.g.load_state_dict(torch.load(g_path))

    def save_model(self):
        torch.save(self.g.state_dict(), os.path.join(self.ckpt_dir, '-'.join(['epoch', str(self.epoch), 'g.pth'])))
        torch.save(self.d.state_dict(), os.path.join(self.ckpt_dir, '-'.join(['epoch', str(self.epoch), 'd.pth'])))

    def record_scalar(self, scalar_list, scalar_name_list):
        scalar_name_list = scalar_name_list[1:-1].split(',')

        for idx, item in enumerate(scalar_list):
            self.writer.add_scalar(scalar_name_list[idx].strip(' '), item, self.cur_iter)

    def record_image(self, image_list):
        image_to_show = torch.cat(image_list, dim=2)[: self.cfg.display_number]
        self.writer.add_image('visualization', make_grid(image_to_show, nrow=self.cfg.display_number), self.cur_iter)

    def log(self, scalar_list, scalar_name_list):
        scalar_name_list = (scalar_name_list[1:-1].split(','))

        basic_information = '{} epoch:{:d} iteration:{:d}'.format(get_solver_name() + '_' + self.cfg.version,
                                                                  self.epoch, self.cur_iter)
        loss_line = ''
        for idx, item in enumerate(scalar_list):
            loss_line += (
                        scalar_name_list[idx].strip(' ') + ' '.join([':', '{:.3f}'.format(item.item()).rjust(5)]) + ' ')

        # flush_print(' '.join([basic_information, loss_line]))
        print(' '.join([basic_information, loss_line]))

    def parse_batch(self, batch):

        source_image, target_image, source_image_name, target_image_name, source_image_uv, target_image_uv, source_image_name_uv, target_image_name_uv = batch

        source_image = source_image.cuda()
        target_image = target_image.cuda()

        source_image_uv = source_image_uv.cuda()
        target_image_uv = target_image_uv.cuda()

        if self.cfg.mode == 'train':
            return source_image, target_image, source_image_uv, target_image_uv
        else:
            return source_image, target_image, source_image_uv, target_image_uv, source_image_name, source_image_name_uv

    @staticmethod
    def optimize(optimizer, loss, retain_graph=False):
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        optimizer.step()

    @staticmethod
    def normalize(input_tensor):
        range_v = torch.max(input_tensor) - torch.min(input_tensor)

        if range_v > 0:
            normalised = (input_tensor - torch.min(input_tensor)) / range_v
        else:
            normalised = torch.zeros(input_tensor.size())

        return normalised


def get_solver_name():
    return str(os.path.basename(__file__).split('.')[0])


def get_config(parser):

    args = parser.parse_args()
    args = edict(vars(args))

    print(args.version)

    cfg_file_path = os.path.join('./config', '_'.join([get_solver_name(), args.version + '.yaml']))

    # ./ config / FE_frontalization_MP.yaml
    with open(cfg_file_path, 'r') as stream:
        config = edict(yaml.load(stream))

    config.update(args)
    return config


def main(config):

    solver = Solver(config)

    if config.mode == 'train':
        solver.train()

    elif config.mode == 'test':
        solver.test()

# note:
# 1) the intensity of the images are scaled to [-1, 1]. Thus, dataloader, tensorboard visualization and the output
# of the generator need to be altered.
# 2) cfg.base_root
# 3) set PYTHONNPATH


if __name__ == '__main__':

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    # --version MP
    parser.add_argument('--version', type=str)
    parser.add_argument('--test_model_path', default=None, type=str)
    parser.add_argument('--test_model_epoch', default=None, type=str)
    parser.add_argument('--label_nc', type=int, default=182,
                        help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')

    main(get_config(parser))
