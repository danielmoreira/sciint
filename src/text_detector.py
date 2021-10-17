# -*- coding: utf-8 -*-
import time
import os
import torch
import torch.backends.cudnn as cudnn
import cv2
import numpy as np

from CRAFT import craft_utils
from CRAFT import imgproc
from CRAFT.craft import CRAFT

def copyStateDict(state_dict):
    from collections import OrderedDict
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

dir_w = os.path.join(os.path.dirname(__file__),'CRAFT/weights')

class TextNet():
    def __init__(self, trained_model=dir_w+'/craft_mlt_25k.pth', text_threshold=0.7, link_threshold=0.4, low_text=0.5, 
                 poly=False, cuda=False, refine=False, refiner_model=dir_w+'/craft_refiner_CTW1500.pth'):
        self.canvas_size = 1280
        self.mag_ratio = 1.5
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        # load net
        net = CRAFT()     # initialize

        #print('Loading weights from checkpoint (' + trained_model + ')')
        if cuda:
            net.load_state_dict(copyStateDict(torch.load(trained_model)))
        else:
            net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))

        if cuda:
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False

        self.net = net.eval()
        self.cuda = cuda
        # LinkRefiner
        refine_net = None
        if refine:
            from CRAFT.refinenet import RefineNet
            refine_net = RefineNet()
            print('Loading weights of refiner from checkpoint (' + refiner_model + ')')
            if args.cuda:
                refine_net.load_state_dict(copyStateDict(torch.load(refiner_model)))
                refine_net = refine_net.cuda()
                refine_net = torch.nn.DataParallel(refine_net)
            else:
                refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

            self.refine_net = refine_net.eval()
            self.poly = True
        else:
            self.refine_net = None
            self.poly = poly
        
    
    def __call__(self, image):
        t0 = time.time()
        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 
                                                                              self.canvas_size, 
                                                                              interpolation=cv2.INTER_LINEAR,
                                                                              mag_ratio=self.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = x.unsqueeze(0)                          # [c, h, w] to [b, c, h, w]
        if self.cuda:
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = self.net(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # refine link
        if self.refine_net is not None:
            with torch.no_grad():
                y_refiner = self.refine_net(y, feature)
            score_link = y_refiner[0,:,:,0].cpu().data.numpy()

        t0 = time.time() - t0
        t1 = time.time()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, self.text_threshold, self.link_threshold, self.low_text, self.poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]

        t1 = time.time() - t1
        
        return polys, (score_text, score_link), t0, t1

