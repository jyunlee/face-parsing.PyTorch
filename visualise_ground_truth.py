import os
import torch
import numpy as np

from torchvision import transforms
from PIL import Image
from evaluate import vis_parsing_maps

if __name__ == "__main__":
    
    in_dir_path = '/home/jihyun/workspace/face_parsing/dataset/CelebAMask-HQ/eval-img'
    gt_dir_path = '/home/jihyun/workspace/face_parsing/dataset/CelebAMask-HQ/mask'
    file_names = ['131', '1658', '2041', '3204', '3640', '6930', '7822', '9949', '23766', '27092']

    for file_name in file_names:
        in_file_path = os.path.join(in_dir_path, file_name + '.jpg')
        gt_file_path = os.path.join(gt_dir_path, file_name + '.png')

        in_img = Image.open(in_file_path)
        in_img = in_img.resize((512, 512), Image.BILINEAR)

        gt_img = Image.open(gt_file_path)
        gt_img = gt_img.resize((512, 512), Image.BILINEAR)
        gt_img = np.array(gt_img)
        
        vis_parsing_maps(in_img, gt_img, stride=1, save_im=True, save_path=os.path.join('/home/jihyun/workspace/face_parsing/dataset/CelebAMask-HQ/gt-visualize', file_name+'.jpg'))
