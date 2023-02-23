import torch
import sys
from PIL import Image
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from flopth import flopth
from ptflops import get_model_complexity_info
from torchsummary import summary
from torchstat import stat

sys.path.insert(2, "networks/PASMnet")

from networks.PASMNet.models import PASMnet

import config

class PASMNetEstimator:
    def __init__(self):
        self.model = PASMnet().to(config.DEVICE)
        if config.PASMNET_MODEL_PATH.split('.')[-1] == 'tar':
            ckpt = torch.load(config.PASMNET_MODEL_PATH)['state_dict']
        else:
            ckpt = torch.load(config.PASMNET_MODEL_PATH)
        self.model.load_state_dict(ckpt)
        self.model.eval()

    def profile(self):
        print("Profiling Architecture : {}".format(config.ARCHITECTURE))
        width = config.PROFILE_IMAGE_WIDTH
        height = config.PROFILE_IMAGE_HEIGHT
        print("image width: {}, height:{}".format(width, height))
        model = PASMnet().to("cpu")
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

        width = config.PROFILE_IMAGE_WIDTH
        height = config.PROFILE_IMAGE_HEIGHT
        print("image width: {}, height:{}".format(width, height))

        dummy_inputs = torch.rand(1, 3, width, height)
        print("=====START Profile With FLOPTH========")
        flops, params = flopth(model, inputs=(dummy_inputs,dummy_inputs))
        print("With flopth -> FLOPS: {}, params: {}".format(flops, params))
        print("=====END Profile With FLOPTH========")

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def get_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def pre_process_image(self, left_image, right_image):
        left_img = self.load_image(left_image)
        right_img = self.load_image(right_image)
        w, h = left_img.size

        processed = self.get_transform()
        left_img = processed(left_img).numpy()
        right_img = processed(right_img).numpy()

        # pad to size 1248x384
        top_pad = 384 - h
        right_pad = 1248 - w
        assert top_pad > 0 and right_pad > 0
        # pad images
        left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                               constant_values=0)
        return torch.from_numpy(left_img).unsqueeze(0), torch.from_numpy(right_img).unsqueeze(0), top_pad, right_pad
    def estimate(self, left_image, right_image):
        left_img, right_img, top_pad, right_pad = self.pre_process_image(left_image, right_image)
        self.model.eval()
        print("type of left_img:{}".format(type(left_img)))
        disp_ests = self.model(left_img.to(config.DEVICE), right_img.to(config.DEVICE), max_disp=192)
        print("type of disp_ests:{}".format(type(disp_ests)))
        disp = torch.clamp(disp_ests[:, :, top_pad:, :-right_pad].squeeze().data.cpu(), 0).numpy()
        return disp