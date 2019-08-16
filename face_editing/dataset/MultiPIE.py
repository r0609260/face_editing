
import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


def parse_line(input_line):
    image_name, image_id = input_line.strip().split(' ')
    return image_name, image_id

class MultiPIE(data.Dataset):

    def __init__(self, data_root,uv_data_root, transform, target_transform,uv_texture, target_uv_texture, mode):

        self.set = ['']

        image_path_list = list()
        image_name_list = list()
        target_image_path_list = list()
        target_image_name_list = list()
        image_id_list = list()

        uv_texture_path_list = list()
        uv_texture_name_list = list()
        target_uv_texture_path_list = list()
        target_uv_texture_name_list = list()
        uv_texture_id_list = list()

        if mode == 'train':
            data_list_path = os.path.join(data_root, 'list/train_S.txt')
            uv_data_list_path = os.path.join(data_root, 'list/train_uv.txt')
        elif mode == 'test':
            data_list_path = os.path.join(data_root, 'list/test_S.txt')
            uv_data_list_path = os.path.join(data_root, 'list/test_uv.txt')

        else:
            raise NotImplementedError

        with open(data_list_path) as f:
            for line in f.readlines():
                image_name, image_id = parse_line(line)

                image_name_list.append(image_name)
                target_im_name = self.get_target_im_name(image_name)
                image_path_list.append(os.path.join(data_root, 'data', image_name))
                target_image_name_list.append(target_im_name)
                target_image_path_list.append(os.path.join(data_root, 'data', target_im_name))

                image_id_list.append(image_id)

        with open(uv_data_list_path) as f:
            for line in f.readlines():
                uv_image_name, uv_image_id = parse_line(line)

                uv_texture_name_list.append(uv_image_name)
                target_uv_texture_name = self.get_target_im_name(uv_image_name)
                uv_texture_path_list.append(os.path.join(uv_data_root, 'uv_texture_multipie', uv_image_name))
                target_uv_texture_name_list.append(target_uv_texture_name)
                target_uv_texture_path_list.append(os.path.join(uv_data_root, 'uv_texture_multipie', target_uv_texture_name))

                uv_texture_id_list.append(uv_image_id)

        self.image_name_list = image_name_list
        self.image_path_list = image_path_list
        self.target_image_path_list = target_image_path_list
        self.target_image_name_list = target_image_name_list
        self.image_id_list = image_id_list

        self.transform = transform
        self.target_transform = target_transform



        self.uv_texture_name_list = uv_texture_name_list
        self.uv_texture_path_list = uv_texture_path_list
        self.target_uv_texture_path_list = target_uv_texture_path_list
        self.target_uv_texture_name_list = target_uv_texture_name_list
        self.uv_texture_id_list = uv_texture_id_list

        self.uv_texture = uv_texture
        self.target_uv_texture = target_uv_texture

    @staticmethod
    def return_contents(input_im_name):
        return input_im_name.split('_')

    def get_target_im_name(self, input_im_name):
        contents = self.return_contents(input_im_name)
        contents[3] = '051'
        return '_'.join(contents)

    def __getitem__(self, index):

        input_im_uv = Image.open(self.uv_texture_path_list[index]).convert('RGB')
        input_im_uv = self.transform(input_im_uv)
        input_im_name_uv = self.uv_texture_name_list[index]

        target_im_uv = Image.open(self.target_uv_texture_path_list[index]).convert('RGB')
        target_im_uv = self.target_transform(target_im_uv)
        target_im_name_uv = self.target_uv_texture_name_list[index]

        input_im = Image.open(self.image_path_list[index]).convert('RGB')
        input_im = self.transform(input_im)
        input_im_name = self.image_name_list[index]
        # input_id = self.image_id_list[index]
        target_im = Image.open(self.target_image_path_list[index]).convert('RGB')
        target_im = self.target_transform(target_im)
        target_im_name = self.target_image_name_list[index]

        return input_im, target_im, input_im_name, target_im_name, input_im_uv, target_im_uv, input_im_name_uv, target_im_name_uv

    def __len__(self):
        return len(self.image_name_list)


def get_dataset(data_root='/data1/mandi.luo/Multi-PIE',uv_data_root="/data1/mandi.luo/cjCodeV1", image_size=128, mode='train'):

    data_root = os.path.expanduser(data_root)
    uv_data_root = os.path.expanduser(uv_data_root)

    transform = list()
    # if mode == 'train':
    #     transform.append(transforms.RandomHorizontalFlip())
    transform.append(transforms.Resize(image_size))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform)

    target_transform = list()
    target_transform.append(transforms.Resize(image_size))
    target_transform.append(transforms.ToTensor())
    target_transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    target_transform = transforms.Compose(target_transform)


    uv_texture = list()
    # if mode == 'train':
    #     transform.append(transforms.RandomHorizontalFlip())
    uv_texture.append(transforms.Resize(image_size))
    uv_texture.append(transforms.ToTensor())
    uv_texture.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    uv_texture = transforms.Compose(uv_texture)

    target_uv_texture = list()
    target_uv_texture.append(transforms.Resize(image_size))
    target_uv_texture.append(transforms.ToTensor())
    target_uv_texture.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    target_uv_texture = transforms.Compose(target_uv_texture)

    return MultiPIE(data_root, uv_data_root, transform, target_transform, uv_texture, target_uv_texture, mode)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    # from face_editing.utils import denorm
    from utils import denorm

    data_set = get_dataset()
    # theDataLoader = DataLoader(data_set, batch_size=32, shuffle=False, num_workers=16)
    theDataLoader = DataLoader(data_set, batch_size=32, shuffle=False, num_workers=0)

    for iteration, batch in enumerate(theDataLoader, 0):

        im, tar_im, im_names, tar_im_names, im_uv, tar_im_uv, im_names_uv, tar_im_names_uv = batch
        print(im_names)
        print(tar_im_names)
        print(im_names_uv)
        print(tar_im_names_uv)
        save_image(denorm(im), '../../Downloads/im.png')
        save_image(denorm(tar_im), '../../Downloads/tar_im.png')
        save_image(denorm(im_names_uv), '../../Downloads/im_names_uv.png')
        save_image(denorm(tar_im_names_uv), '../../Downloads/tar_im_names_uv.png')
        exit()

