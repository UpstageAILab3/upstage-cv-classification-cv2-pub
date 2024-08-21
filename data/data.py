from .augmentation import Augmentaion 
import pandas as pd
import numpy as np
from PIL import Image
import os

class Data():
    def __init__(self):
        self.aug = Augmentaion()

        self.PRE_PATH = '../source/data/'
        
        self.TRAIN_CSV_PATH = self.PRE_PATH + 'train.csv'
        self.TRAIN_IMG_PATH = self.PRE_PATH + 'train'
        self.TRAIN_DF = pd.read_csv(self.TRAIN_CSV_PATH)
        
        self.TRAIN_AUG_CSV_PATH = self.PRE_PATH + 'train_aug.csv'
        self.TRAIN_AUG_IMG_PATH = self.PRE_PATH + 'train_aug'

    def getOriginImage(self):
        """
        OUTPUT:
            id_list : List[int ...]
            img_list : List[np.array(image) ...]
            label_list : List[int ...]

        대회에서 주어지는 train 데이터 셋을 가져오는 함수
        """
        id_list = []
        img_list = []
        label_list = []

        for id, target in self.TRAIN_DF.itertuples(index = False):
            img_path = os.path.join(self.TRAIN_IMG_PATH, id)
            img_np = np.array(Image.open(img_path))

            id_list.append(id)
            img_list.append(img_np)
            label_list.append(target)

        return id_list, img_list, label_list
    
    def createAugImage(self, ):
        rotate_ids, rotate_imgs, rotate_labels = self.aug


    def saveImage(self, path, id, )
