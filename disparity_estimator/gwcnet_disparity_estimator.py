import sys
import torch
import numpy as np

import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

from flopth import flopth
from ptflops import get_model_complexity_info
from torchsummary import summary
from torchstat import stat

import config
sys.path.insert(1, "networks/GwcNet")

from networks.GwcNet.models.gwcnet import GwcNet_G, GwcNet_GC

def make_iterative_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper

@make_iterative_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.cpu().numpy()
    else:
        raise NotImplementedError("invalid input type for tensor2numpy")


class GwcNetEstimator:
    def __init__(self):
        self.model = GwcNet_G(192)
        print(self.model)
        self.model = nn.DataParallel(self.model, device_ids=[1])
        #self.model.to(config.DEVICE)
        state_dict = torch.load(config.GWCNET_MODEL_PATH, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict['model'])
        #self.model.load_state_dict(checkpoint)
        self.model.eval()
    def profile(self):
        print("Profiling Architecture : {}".format(config.ARCHITECTURE))
        width = config.PROFILE_IMAGE_WIDTH
        height = config.PROFILE_IMAGE_HEIGHT
        print("image width: {}, height:{}".format(width, height))
        model = GwcNet_G(192)
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
        return torch.from_numpy(left_img).unsqueeze(0), torch.from_numpy(right_img).unsqueeze(0)

    def estimate(self, left_image, right_image):
        left_img, right_img = self.pre_process_image(left_image, right_image)
        self.model.eval()
        print("type of left_img:{}".format(type(left_img)))
        disp_ests = self.model(left_img.to(config.DEVICE1), right_img.to(config.DEVICE1))
        print("type of disp_ests:{}".format(type(disp_ests)))
        disparity_map = tensor2numpy(disp_ests[-1])
        disparity_map = np.squeeze(disparity_map)
        return disparity_map

