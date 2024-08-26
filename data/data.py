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

        self.TRAIN_IMG_ID, self.TRAIN_IMG, self.TRAIN_TARGET = self.getOriginImage()

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
    
    def createAugImage(self):
        '''
            patch(한번 또는 두번) + rotate + flip + noise
            patch(한번 또는 두번) + rotate 
        '''

        aug_ids = []
        aug_imgs = []
        aug_labels = []

        comb1 = ["patch", "rotate", "flip", "noise"]
        comb2 = ["patch", "rotate"]

        comb1_aug_ids, comb1_aug_imgs, comb1_aug_labels = self.aug.getCombinationAug(self.TRAIN_IMG_ID, self.TRAIN_IMG, self.TRAIN_TARGET, comb1)
        aug_ids.append(comb1_aug_ids)
        aug_imgs.append(comb1_aug_imgs)
        aug_labels.append(comb1_aug_labels)

        comb2_aug_ids, comb2_aug_imgs, comb2_aug_labels = self.aug.getCombinationAug(self.TRAIN_IMG_ID, self.TRAIN_IMG, self.TRAIN_TARGET, comb2)
        aug_ids.append(comb2_aug_ids)
        aug_imgs.append(comb2_aug_imgs)
        aug_labels.append(comb2_aug_labels)

        return aug_ids, aug_imgs, aug_labels


    def saveImage(self, path, id, ):
