import argparse
import torch
import torch.nn as nn
import cv2

import torchvision.transforms as transforms
from PIL import Image

from flopth import flopth
from ptflops import get_model_complexity_info
from torchsummary import summary
from torchstat import stat
import numpy as np

import config
from networks.HitNet.models import HITNet


class HitNetEstimator:
    def get_config(self):
        parser = argparse.ArgumentParser(description='HITNet')
        parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
        parser.add_argument('--fea_c', type=list, default=[32, 24, 24, 16, 16], help='feature extraction channels')
        # parse arguments, set seeds
        args = parser.parse_args()
        return args
    def __init__(self):
        self.model = HITNet(self.get_config())
        print(self.model)
        self.model = nn.DataParallel(self.model)
        state_dict = torch.load(config.HITNET_MODEL_PATH)
        self.model.load_state_dict(state_dict['model'])
        self.model.to(config.DEVICE)

    def profile(self):
        print("Profiling Architecture : {}".format(config.ARCHITECTURE))
        width = config.PROFILE_IMAGE_WIDTH
        height = config.PROFILE_IMAGE_HEIGHT
        print("image width: {}, height:{}".format(width, height))
        model = HITNet(self.get_config())
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

        width = config.PROFILE_IMAGE_WIDTH
        height = config.PROFILE_IMAGE_HEIGHT
        print("image width: {}, height:{}".format(width, height))

        dummy_inputs = torch.rand(1, 3, width, height)
        print("=====START Profile With FLOPTH========")
        flops, params = flopth(model, inputs=(dummy_inputs,dummy_inputs))
        print("With flopth -> FLOPS: {}, params: {}".format(flops, params))
        print("=====END Profile With FLOPTH========")
    def get_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')
    def preprocess(self, left_image, right_image):
        left_img = self.load_image(left_image)
        right_img = self.load_image(right_image)

        w, h = left_img.size

        # normalize
        processed = self.get_transform()
        left_img = processed(left_img).numpy()
        right_img = processed(right_img).numpy()

        # pad to size 1280x384
        top_pad = 384 - h
        right_pad = 1280 - w
        assert top_pad > 0 and right_pad > 0
        # pad images
        left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                               constant_values=0)
        return torch.from_numpy(left_img).unsqueeze(0), torch.from_numpy(right_img).unsqueeze(0)
        """
        imgL = cv2.imread(left_image, cv2.IMREAD_COLOR)
        imgR = cv2.imread(right_image, cv2.IMREAD_COLOR)

        input_height, input_width = imgL.shape[:2]

        imgL = cv2.resize(imgL, (input_width, input_height))
        imgR = cv2.resize(imgR, (input_width, input_height))

        # Shape (1, 6, None, None)
        imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
        imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

        imgL = torch.from_numpy(imgL).unsqueeze(0)
        imgR = torch.from_numpy(imgR).unsqueeze(0)

        return imgL, imgR
    """

    def estimate(self, left_image, right_image):
        left_img, right_img = self.preprocess(left_image, right_image)
        self.model.eval()
        outputs = self.model(left_img.to(config.DEVICE), right_img.to(config.DEVICE))
        prop_disp_pyramid = outputs['prop_disp_pyramid']
        print("prop_disp_pyramid: {}".format())
        return prop_disp_pyramid

if __name__ == "__main__":
    estimator = HitNetEstimator()
    print("estimator: {}".format(estimator))
