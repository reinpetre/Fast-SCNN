import argparse
import os

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage

from dataset import transform, palette
from model import FastSCNN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict segmentation result from a given image')
    parser.add_argument('--data_path', default='datasets/railsem19-embedding/raw_images', type=str,
                        help='Data path for railsem dataset')
    parser.add_argument('--model_weight', type=str, default='railsem_model.pth', help='Pretrained model weight')
    parser.add_argument('--input_pic', type=str, default='val/rs06800.jpg',
                        help='Path to the input picture')
    # args parse
    args = parser.parse_args()
    data_path, model_weight, input_pic = args.data_path, args.model_weight, args.input_pic
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, data_path)
    image_path = os.path.join(image_path, input_pic)
    
    print(image_path)

    # image = Image.open('{}/{}'.format(data_path, input_pic)).convert('RGB')
    image = Image.open(image_path).convert('RGB')
    image = image.resize((960, 512), Image.NEAREST)
    image_height, image_width = image.height, image.width
    num_width = 2 if 'test' in input_pic else 3
    target = Image.new('RGB', (image_width * num_width, image_height))
    images = [image]

    image = transform(image).unsqueeze(dim=0).cuda()

    # model load
    model = FastSCNN(in_channels=3, num_classes=2)
    model.load_state_dict(torch.load(model_weight, map_location=torch.device('cpu')), strict=False)
    model = model.cuda()
    model.eval()

    # predict and save image
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1)
        pred_image = ToPILImage()(pred.byte().cpu())
        print(np.unique(pred_image))
        pred_image.putpalette(palette)
        if 'test' not in input_pic:
            gt_image = Image.open(image_path.replace('raw_images', 'raw_masks').replace('jpg', 'png'))
            gt_image = gt_image.resize((960, 512), Image.NEAREST )
            images.append(gt_image)
        images.append(pred_image)
        # concat images
        for i in range(len(images)):
            left, top, right, bottom = image_width * i, 0, image_width * (i + 1), image_height
            target.paste(images[i], (left, top, right, bottom))
        target.save(os.path.split(input_pic)[-1].replace('.jpg', '_result.png'))
