
import math
import warnings

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parallel import data_parallel


def normalize_tensor(input_tensor):
    # input_tensor: (batch_size, embedding_size)
    input_tensor, states = input_tensor
    input_tensor_norm = torch.unsqueeze(torch.norm(input_tensor, dim=1).detach(), dim=1)
    return input_tensor.div(input_tensor_norm)


def weighted_mse_loss(pred, target, weight=None):
    if weight is None:
        return F.mse_loss(pred, target)
    else:
        if len(list(pred.size())) == 4:
            factor = pred.size(0) * pred.size(1) * pred.size(2) * pred.size(3)
        elif len(list(pred.size())) == 3:
            factor = pred.size(0) * pred.size(1) * pred.size(2)
        elif len(list(pred.size())) == 2:
            factor = pred.size(0) * pred.size(1)
        else:
            assert (list(pred.size())) == 1
            factor = pred.size(0)

        return torch.sum(weight * (pred - target) ** 2) / factor


class AdversarialLoss:

    def __init__(self, real_label=1.0, fake_label=0.0, loss_function=weighted_mse_loss):

        # ordinary gan loss: F.binary_cross_entropy_with_logits
        # lsgan loss: F.mse_loss

        self.real_label = real_label
        self.fake_label = fake_label
        self.loss_function = loss_function

    def create_label_tensor(self, pred, real_label):
        # create label tensor that fit the size of the prediction
        if real_label is True:
            return torch.FloatTensor(pred.size()).fill_(self.real_label).cuda()
        else:
            return torch.FloatTensor(pred.size()).fill_(self.fake_label).cuda()

    def loss_d(self, pred_real, pred_fake, label_real=None, label_fake=None, weight_real=None, weight_fake=None):

        real_label = self.create_label_tensor(pred_real, real_label=True) if label_real is None else label_real
        loss_d_real = self.loss_function(
            pred_real, real_label, weight=weight_real)
        
        fake_label = self.create_label_tensor(pred_fake, real_label=False) if label_fake is None else label_fake
        loss_d_fake = self.loss_function(
            pred_fake, fake_label, weight=weight_fake)

        return loss_d_real, loss_d_fake

    def loss_g(self, pred_fake, weight=None):
        loss_g = self.loss_function(pred_fake, self.create_label_tensor(pred_fake, real_label=True), weight=weight)
        return loss_g


class GradientPenalty:
    def __call__(self, x_real, x_fake, discriminator):
        return self.calculate_gradient_penalty(x_real, x_fake, discriminator)

    def calculate_gradient_penalty(self, x_real, x_fake, discriminator):

        warnings.warn('calculating gp loss requires that the discriminator return tuple.')
        warnings.warn('parallel run when calculating gp loss.')

        x_hat = self.calculate_x_hat(x_real, x_fake)
        # pred = discriminator(x_hat)[0]
        pred = data_parallel(discriminator, x_hat)[0]
        return self.gradient_penalty(pred, x_hat)

    @staticmethod
    def gradient_penalty(y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).cuda()
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    @staticmethod
    def calculate_x_hat(x_real, x_fake):
        alpha = torch.rand(x_real.size(0), 1, 1, 1).cuda()
        x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
        return x_hat


class WassersteinAdversarialLoss:

    @staticmethod
    def loss_d(pred_real, pred_fake):
        loss_d_real = - torch.mean(pred_real)
        loss_d_fake = torch.mean(pred_fake)
        return loss_d_real, loss_d_fake

    @staticmethod
    def loss_g(pred_fake):
        loss_g = - torch.mean(pred_fake)
        return loss_g


class AttrLoss:
    def __call__(self, target, logit):
        """Compute binary or softmax cross entropy loss."""
        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)


class RecLoss:
    def __call__(self, real, rec):
        warnings.warn('for RecLoss, ground truth first.')
        return torch.mean(torch.abs(real - rec))


# actually it is l2 loss
class L1Loss:
    def __init__(self):
        self.loss = nn.SmoothL1Loss().cuda()

    def __call__(self, pred, target):
        return self.loss(pred, target)


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


class PerceptualLoss:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, predicted, target):
        embeddings_predicted = normalize_tensor(data_parallel(self.feature_extractor, denorm(F.adaptive_avg_pool2d(predicted, 128))))
        embeddings_target = normalize_tensor(data_parallel(self.feature_extractor, denorm(F.adaptive_avg_pool2d(target, 128))))
        return F.mse_loss(embeddings_predicted, embeddings_target.detach(), reduction='elementwise_mean')


class UVLoss:
    def __call__(self, pred, gt):

        u_pred = self.slice(pred, 0)
        v_pred = self.slice(pred, 1)
        index_pred = self.slice(pred, 2)

        u_gt = self.slice(gt, 0)
        v_gt = self.slice(gt, 1)
        index_gt = torch.min(u_gt, v_gt)
        index_gt[index_gt > 0] = 1

        loss_u = 10 * F.smooth_l1_loss(u_pred[index_gt > 0], u_gt[index_gt > 0])
        loss_v = 10 * F.smooth_l1_loss(v_pred[index_gt > 0], v_gt[index_gt > 0])

        index_pred = index_pred.permute([0, 2, 3, 1]).contiguous()
        index_pred = index_pred.view(index_pred.size(0), -1)

        index_gt = index_gt.permute([0, 2, 3, 1]).contiguous()
        index_gt = index_gt.view(index_gt.size(0), -1)
        loss_index = F.binary_cross_entropy_with_logits(index_pred, index_gt)

        loss_uv = loss_u + loss_v + loss_index
        return loss_uv

    @staticmethod
    def slice(tensor, dim):
        return torch.unsqueeze(tensor[:, dim], dim=1)


class MutualPerceptualLoss:

    def __init__(self, estimator, mode='nce'):
        # mode: Loss mode. NCE : `nce`, or Donsker-Vadadhan : `dv`.
        warnings.warn('MutualPerceptualLoss dose not support batch_size=1')

        self.estimator = estimator
        self.mode = mode

    @staticmethod
    def donsker_varadhan_loss(l, g):
        N, local_units, n_locs = l.size()
        l = l.permute(0, 2, 1)
        l = l.reshape(-1, local_units)

        u = torch.mm(g, l.t())
        u = u.reshape(N, N, n_locs)

        mask = torch.eye(N).cuda()
        n_mask = (1 - mask)[:, :, None]

        E_pos = (u.mean(2) * mask).sum() / mask.sum()

        u -= 100 * (1 - n_mask)
        u_max = torch.max(u)
        E_neg = torch.log((n_mask * torch.exp(u - u_max)).sum() + 1e-6) + u_max - math.log(n_mask.sum())
        loss = E_neg - E_pos
        return loss

    @staticmethod
    def nce_loss(l, g):
        N, local_units, n_locs = l.size()
        l_p = l.permute(0, 2, 1)
        u_p = torch.matmul(l_p, g.unsqueeze(dim=2))

        l_n = l_p.reshape(-1, local_units)
        u_n = torch.mm(g, l_n.t())
        u_n = u_n.reshape(N, N, n_locs)

        mask = torch.eye(N).unsqueeze(dim=2).cuda()
        n_mask = 1 - mask

        u_n = (n_mask * u_n) - (10. * (1 - n_mask))
        u_n = u_n.reshape(N, -1).unsqueeze(dim=1).expand(-1, n_locs, -1)

        pred_lgt = torch.cat([u_p, u_n], dim=2)
        pred_log = F.log_softmax(pred_lgt, dim=2)
        loss = -pred_log[:, :, 0].mean()
        return loss

    def __call__(self, l, g):

        l_enc = normalize_tensor(data_parallel(self.estimator, F.adaptive_avg_pool2d(l, 128)))
        l_enc = torch.unsqueeze(l_enc, dim=2)
        g_enc = normalize_tensor(data_parallel(self.estimator, F.adaptive_avg_pool2d(g, 128)))

        if self.mode == 'nce':
            loss = self.nce_loss(l_enc, g_enc)
        elif self.mode == 'dv':
            loss = self.donsker_varadhan_loss(l_enc, g_enc)
        else:
            raise NotImplementedError(self.mode)

        return loss


if __name__ == '__main__':

    from PIL import Image
    import torchvision.transforms as transforms
    from lightcnn.private import LightCNN29_V4

    def im2tensor(image_path):
        # read the image from the path and use lightcnn v4 to extract the identity representation
        image = Image.open(image_path)
        image_transform = transforms.Compose([transforms.Resize([128, 128]), transforms.ToTensor()])
        image = image_transform(image)
        image = torch.unsqueeze(image, dim=0)
        image = image.cuda()
        return image

    im1 = im2tensor('/home/jie.cao/main/dataset/CelebA/images/088631.jpg')
    im2 = im2tensor('/home/jie.cao/main/dataset/CelebA/images/088632.jpg')
    im3 = im2tensor('/home/jie.cao/main/dataset/CelebA/images/088633.jpg')
    im4 = im2tensor('/home/jie.cao/main/dataset/CelebA/images/088634.jpg')

    a = torch.cat([im4, im3], dim=0)
    b = torch.cat([im4, im3], dim=0)

    extractor = LightCNN29_V4()
    mp_loss = MutualPerceptualLoss(extractor)
    print(mp_loss(a, b))
