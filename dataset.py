import os
import os.path as osp
import random

import cv2
import numpy as np
# from cityscapesscripts.helpers.labels import trainId2label
from torch.utils.data import Dataset
from torchvision import transforms

city_mean, city_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(city_mean, city_std)])

colors = [(0, 0, 0), (255, 255, 255)]
palette = []

for color in colors:
    palette += list(color)
# for key in sorted(trainId2label.keys()):
#     if key != -1 and key != 255:
#         palette += list(trainId2label[key].color)


class Railsem(Dataset):
    """
       Cityscapes dataset is employed to load train or val set
       Args:
        root: the Cityscapes dataset path,
         cityscapes
          ├── gtFine
          ├── leftImg8bit
        split: train, val
        crop_size: (512, 1024), only works for 'train' split
        mean: rgb_mean (0.485, 0.456, 0.406)
        std: rgb_mean (0.229, 0.224, 0.225)
        ignore_label: -1
    """

    def __init__(self, root, split='train', crop_size=(960, 512), mean=city_mean, std=city_std, ignore_label=-1):

        self.split = split
        self.crop_size = crop_size
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        self.std = std
        self.ignore_label = ignore_label
        self.files = []

        for root_dir, dirs, files in os.walk(osp.join(root, 'raw_images', split)):
            for file in files:
                img_file = osp.join(root_dir, file)
                label_file = osp.join(root_dir.replace('raw_images', 'raw_masks'), file.replace('.jpg', '.png'))
                self.files.append({'img': img_file, 'label': label_file, 'name': file})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles['img'], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles['label'], cv2.IMREAD_GRAYSCALE)
        
        if label is None or image is None:
            print(datafiles)
        
        # resize
        image = cv2.resize(image, self.crop_size, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, self.crop_size, interpolation = cv2.INTER_NEAREST)
        
        name = datafiles['name']
        
        label[np.where(label == 255)] = 1
        

        # random resize, multiple scale training
        #if self.split == 'train':
        #    f_scale = random.choice([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
        #    image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        #    label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)
        # change to RGB
        image = image[:, :, ::-1]
        # normalization
        image /= 255.0
        image -= self.mean
        image /= self.std

        # random crop
        # if self.split == 'train':
        #     img_h, img_w = label.shape
        #     pad_h = max(self.crop_h - img_h, 0)
        #     pad_w = max(self.crop_w - img_w, 0)
        #     if pad_h > 0 or pad_w > 0:
        #         img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
        #                                      pad_w, cv2.BORDER_CONSTANT,
        #                                      value=(0.0, 0.0, 0.0))
        #         label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
        #                                        pad_w, cv2.BORDER_CONSTANT,
        #                                        value=(self.ignore_label,))
        #     else:
        #         img_pad, label_pad = image, label
        # 
        #     img_h, img_w = label_pad.shape
        #     h_off = random.randint(0, img_h - self.crop_h)
        #     w_off = random.randint(0, img_w - self.crop_w)
        #     image = img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w]
        #     label = label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w]

        # HWC -> CHW
        image = image.transpose((2, 0, 1))
        label = np.asarray(label, np.long)
    
        # # random horizontal flip
        # if self.split == 'train':
        #     flip = np.random.choice(2) * 2 - 1
        #     image = image[:, :, ::flip]
        #     label = label[:, ::flip]
        
#        mask = label
#        mask[np.where(mask == 1)] = 255
#        print("Mask shape: ", mask.shape)
#        cv2.imwrite("image.jpg", image)
#        cv2.imwrite("mask.png", mask)

        return image.copy(), label.copy(), name
