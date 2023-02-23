import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch

from flopth import flopth
from ptflops import get_model_complexity_info
from torchsummary import summary
from torchstat import stat

import sys
#sys.path.append('./deep_learning_archs/BGNet')
#sys.path.insert(0, "deep_learning_archs/BGNet")
#from models.bgnet import BGNet
from networks.BGNet.models.bgnet_plus import BGNet_Plus
#from deep_learning_archs.BGNet.models.bgnet_plus import BGNet_Plus

import config

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}

def get_transform():

    #normalize = __imagenet_stats
    t_list = [
        transforms.ToTensor(),
        #transforms.Normalize(**normalize),
    ]
    return transforms.Compose(t_list)

class BGNetEstimator:
    def __init__(self):
        self.model = BGNet_Plus().to(config.DEVICE)
        print(self.model)
        checkpoint = torch.load(config.BGNET_PLUS_MODEL_PATH, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def profile(self):
        print("Profiling Architecture : {}".format(config.ARCHITECTURE))
        width = config.PROFILE_IMAGE_WIDTH
        height = config.PROFILE_IMAGE_HEIGHT
        print("image width: {}, height:{}".format(width, height))
        model = BGNet_Plus()
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

        width = config.PROFILE_IMAGE_WIDTH
        height = config.PROFILE_IMAGE_HEIGHT
        print("image width: {}, height:{}".format(width, height))

        dummy_inputs = torch.rand(1, 1, width, height)
        print("=====START Profile With FLOPTH========")
        flops, params = flopth(model, inputs=(dummy_inputs,dummy_inputs))
        print("With flopth -> FLOPS: {}, params: {}".format(flops, params))
        print("=====END Profile With FLOPTH========")

    def load_image(self, left_image, right_image):
        left_img = Image.open(left_image).convert('L')
        right_img = Image.open(right_image).convert('L')
        w, h = left_img.size
        h1 = h % 64
        w1 = w % 64
        h1 = h - h1
        w1 = w - w1
        h1 = int(h1)
        w1 = int(w1)
        left_img = left_img.resize((w1, h1), Image.ANTIALIAS)
        right_img = right_img.resize((w1, h1), Image.ANTIALIAS)
        left_img = np.ascontiguousarray(left_img, dtype=np.float32)
        right_img = np.ascontiguousarray(right_img, dtype=np.float32)
        preprocess = get_transform()
        left_img = preprocess(left_img)
        right_img = preprocess(right_img)
        return left_img, right_img
    def estimate(self, left_image, right_image):
        left_img, right_img = self.load_image(left_image, right_image)
        self.model.eval()
        pred, _ = self.model(left_img.unsqueeze(0).to(config.DEVICE), right_img.unsqueeze(0).to(config.DEVICE))
        pred = pred[0].data.cpu().numpy()
        return pred

if __name__ == '__main__':
    estimator = BGNetEstimator()
    print(estimator)