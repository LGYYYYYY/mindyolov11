import os
from ultralytics import YOLO
import logging
import torch
import cv2
import numpy as np


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from model_service.pytorch_model_service import PTServingBaseService
except:
    PTServingBaseService = object


class CustomizeService(PTServingBaseService):
    def __init__(self, model_name, model_path):
        '''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            logger.info('Using GPU for inference')
        else:
            logger.info('Using CPU for inference')
        print(f'running on {self.device}')
        self.code_url = os.path.dirname(os.path.abspath(__file__))
        self.model = YOLO(os.path.join(self.code_url, "best.pt"))
        self.model.to(torch.device(self.device))
        self.labels = self.model.module.name if hasattr(self.model, 'module') else self.model.names
        '''

    def _preprocess(self, data):
        '''
        data_list = []
        for _, v in data.items():
            for _, file_content in v.items():
                file_content = file_content.read()
                img = cv2.imdecode(np.frombuffer(file_content, np.uint8), cv2.IMREAD_COLOR)
                data_list.append(img)
        return data_list
        '''

    def _inference(self, data):
        '''
        with torch.no_grad():
            data = self.model(data[0])
        return data
        '''

    def _postprocess(self, data):
        '''
        result_return = dict()
        data = data[0].to(torch.device('cpu'))
        boxes = data.boxes
        if data is not None:
            boxes = data.boxes
            picked_boxes = [[box[1], box[0], box[3], box[2]] for box in boxes.xyxy.tolist()]
            picked_classes = self.convert_labels(boxes.cls)
            picked_score = boxes.conf
            result_return['detection_classes'] = picked_classes
            result_return['detection_boxes'] = picked_boxes
            result_return['detection_scores'] = picked_score.tolist()
        else:
            result_return['detection_classes'] = []
            result_return['detection_boxes'] = []
            result_return['detection_scores'] = []
        return result_return
        '''

    def convert_labels(self, label_list):
        '''
        if isinstance(label_list, np.ndarray):
            label_list = label_list.tolist()
        label_names = [self.labels[int(index)] for index in label_list]
        return label_names
        '''
