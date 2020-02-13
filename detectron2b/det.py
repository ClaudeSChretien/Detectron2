# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"

class detectronLocal:
    def __init__(self):
        test = 1

    def getvar(self, img):


        cfg = get_cfg()

        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75  # set threshold for this model

        cfg.MODEL.WEIGHTS = 'model_final_f10217.pkl'

        cfg.MODEL.DEVICE = 'cpu'

        predictor = DefaultPredictor(cfg)

        predictions = predictor(img)

        field = predictions["instances"].get_fields()

        boxes = field["pred_boxes"]
        tensor1 = boxes[0].tensor

        numpyArray = tensor1.numpy()
        output = tuple(map(tuple, numpyArray))

        output = output[0]
        rec = (output[0], output[1], output[2] - output[0], output[3] - output[1])

        return rec
