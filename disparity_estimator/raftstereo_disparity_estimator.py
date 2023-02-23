import cv2
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

from flopth import flopth
from ptflops import get_model_complexity_info
from torchsummary import summary
from torchstat import stat

#import sys
#sys.path.append('./deep_learning_archs/RAFT-Stereo')
#from core.raft_stereo import RAFTStereo
#from core.utils.utils import InputPadder
from networks.RAFTStereo.core.raft_stereo import RAFTStereo
from networks.RAFTStereo.core.utils.utils import InputPadder

from PIL import Image
from matplotlib import pyplot as plt

import config


DEBUG_FLAG = False

class RAFTStereoEstimator:
    def __init__(self):
        self.model = torch.nn.DataParallel(RAFTStereo(self.get_internal_args()), device_ids=[0])
        self.model.load_state_dict(torch.load(config.RAFT_STEREO_MODEL_PATH))

    def get_internal_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--valid_iters', type=int, default=32,
                            help='number of flow-field updates during forward pass')

        # Architecture choices
        parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                            help="hidden state and context dimensions")
        parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg",
                            help="correlation volume implementation")
        parser.add_argument('--shared_backbone', action='store_true',
                            help="use a single backbone for the context and feature encoders")
        parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
        parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
        parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
        parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
        parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")

        args_raft_internal = parser.parse_args()
        return args_raft_internal

    def profile(self):
        print("Profiling Architecture : {}".format(config.ARCHITECTURE))
        model = RAFTStereo(self.get_internal_args())
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

        width = config.PROFILE_IMAGE_WIDTH
        height = config.PROFILE_IMAGE_HEIGHT
        print("image width: {}, height:{}".format(width, height))

        dummy_inputs = torch.rand(1, 3, width, height)
        print("=====START Profile With FLOPTH========")
        flops, params = flopth(model, inputs=(dummy_inputs,dummy_inputs))
        print("With flopth -> FLOPS: {}, params: {}".format(flops, params))
        print("=====END Profile With FLOPTH========")

    def load_image(self, imfile):
        img = np.array(Image.open(imfile)).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(config.DEVICE)

    def estimate(self, left_img, right_img):
        #self.model = self.model.module
        self.model.to(config.DEVICE)
        self.model.eval()
        
        output_directory = Path(self.get_internal_args().output_directory)
        output_directory.mkdir(exist_ok=True)
        with torch.no_grad():
            image1 = self.load_image(left_img)
            image2 = self.load_image(right_img)
            
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = self.model(image1, image2, iters=self.get_internal_args().valid_iters, test_mode=True)
            if DEBUG_FLAG:
                file_stem = "output"
                np.save(output_directory / f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
                plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy().squeeze(), cmap='jet')
            disparity_map = -flow_up.cpu().numpy().squeeze()
            
        return disparity_map
