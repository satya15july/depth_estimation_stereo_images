import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image

from flopth import flopth
from ptflops import get_model_complexity_info
from torchsummary import summary
from torchstat import stat

import config
from networks.PSMNet.models import stackhourglass, basic

class PSMNetEstimator:
    def __init__(self, type= 'stackhourglass'):
        if type == 'stackhourglass':
            self.model = stackhourglass(192)
        elif type == 'basic':
            self.model = basic(192)
        else:
            print('no model')
        print("config.DEVICE[:5]: {}".format(config.DEVICE[:5]))
        #if config.DEVICE[:5] == "cuda":
        self.model = nn.DataParallel(self.model)
        self.model.to(config.DEVICE)

        state_dict = torch.load(config.PSMNET_MODEL_PATH)
        self.model.load_state_dict(state_dict['state_dict'])

    def profile(self):
        print("Profiling Architecture : {}".format(config.ARCHITECTURE))
        width = config.PROFILE_IMAGE_WIDTH
        height = config.PROFILE_IMAGE_HEIGHT
        print("image width: {}, height:{}".format(width, height))
        model = stackhourglass(192)
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

        width = config.PROFILE_IMAGE_WIDTH
        height = config.PROFILE_IMAGE_HEIGHT
        print("image width: {}, height:{}".format(width, height))

        dummy_inputs = torch.rand(1, 3, width, height)
        print("=====START Profile With FLOPTH========")
        flops, params = flopth(model, inputs=(dummy_inputs,dummy_inputs))
        print("With flopth -> FLOPS: {}, params: {}".format(flops, params))
        print("=====END Profile With FLOPTH========")

    def preprocess(self, left_image, right_image):
        imgL_o = Image.open(left_image).convert('RGB')
        imgR_o = Image.open(right_image).convert('RGB')

        normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                           'std': [0.229, 0.224, 0.225]}
        infer_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(**normal_mean_var)])

        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o)

        # pad to width and hight to 16 times
        if imgL.shape[1] % 16 != 0:
            times = imgL.shape[1]//16
            top_pad = (times+1)*16 -imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16
            right_pad = (times+1)*16-imgL.shape[2]
        else:
            right_pad = 0

        imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

        return imgL, imgR, top_pad, right_pad
    def estimate(self, left_image, right_image):
        imgL, imgR, top_pad, right_pad = self.preprocess(left_image, right_image)
        self.model.eval()
        with torch.no_grad():
            output = self.model(imgL.to(config.DEVICE), imgR.to(config.DEVICE))
        pred_disp = torch.squeeze(output).data.cpu().numpy()

        if top_pad !=0 or right_pad != 0:
            disp_est = pred_disp[top_pad:,:-right_pad]
        else:
            disp_est = pred_disp
        return disp_est