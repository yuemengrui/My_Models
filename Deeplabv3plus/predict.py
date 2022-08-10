# *_*coding:utf-8 *_*
# @Author : yuemengrui
from model import Model
from dataset import ImageResize
from torchvision import transforms
import cv2
import numpy as np
import torch


class Predictor(object):

    def __init__(self, checkpoint_path='./checkpoints/model_best_V5.pth'):
        self.device = torch.device('cpu')
        self.model = Model(num_classes=6)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.pre_processing = ImageResize()

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                             ])

        self.labels = ['background', 'others', 'title', 'paragraph', 'table', 'image']
                                                        #     黄色           绿色         橙色         蓝色
        train_id_to_color = [[255, 255, 255], [0, 0, 0], [0, 255, 255], [0, 255, 0], [0, 97, 255], [255,0,0]]
        # train_id_to_color.append([0, 0, 0])
        self.train_id_to_color = np.array(train_id_to_color)

    def predict(self, img_path):
        ori_img = cv2.imread(img_path)
        h, w = ori_img.shape[:2]
        print(h, w)
        resize_img, _ = self.pre_processing(ori_img)
        img = self.transform(resize_img)
        img_tensor = img.to(self.device)
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img_tensor)

        pred = pred.max(1)[1].cpu().numpy()[0]

        colorized_preds = self.train_id_to_color[pred].astype('uint8')
        colorized_preds = np.array(cv2.resize(colorized_preds, (w, h)))
        print(colorized_preds.shape)

        cv2.imshow('xx', colorized_preds)
        cv2.waitKey(0)

        # ori_img.paste(colorized_preds, (0, 0), colorized_preds)
        #
        # ori_img.show()

        out = cv2.addWeighted(ori_img, alpha=0.8, src2=colorized_preds, beta=0.2, gamma=0)

        cv2.imshow('out', out)
        cv2.waitKey(0)
        # gray = pred.astype(np.uint8)
        # cv2.imshow('xx', gray * (255 / 6))
        # cv2.waitKey(0)


if __name__ == '__main__':
    predictor = Predictor()

    img_path = '/Users/yuemengrui/Data/IDP/test_images/2142.jpg'

    predictor.predict(img_path)
