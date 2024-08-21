'''
Roate / Flip / Shift / Noise
를 적용하는 클래스
'''

import albumentations as A
import numpy as np
import random

class Augmentaion():
    def getRotateImg(self, ids, images, labels):
        """
        INPUT:
            ids : List[string ...] # ID List
            images : List[numpy.array(image) ...] # Image List
            labels : List[int ...]

        OUTPUT:
            id_list : List[string ...] # ID List
            img_list : List[numpy.array(image) ...] # Image List
            label_list : List[int ...]

        np.array() 형태의 이미지 리스트를 입력하면 
        (ID LIST + 30도 단위씩 회전한 np.array() 형태의 IMG LIST + 각 이미지 label List) 반환
        => 입력의 11배 증강
        """
        rotate_funcs = {
            'rotate_030_' : A.Compose([A.Rotate(limit=(30, 30), p=1)]),
            'rotate_060_' : A.Compose([A.Rotate(limit=(60, 60), p=1)]),
            'rotate_090_' : A.Compose([A.Rotate(limit=(90, 90), p=1)]),
            'rotate_120_' : A.Compose([A.Rotate(limit=(120, 120), p=1)]),
            'rotate_150_' : A.Compose([A.Rotate(limit=(150, 150), p=1)]),
            'rotate_180_' : A.Compose([A.Rotate(limit=(180, 180), p=1)]),
            'rotate_210_' : A.Compose([A.Rotate(limit=(210, 210), p=1)]),
            'rotate_240_' : A.Compose([A.Rotate(limit=(240, 240), p=1)]),
            'rotate_270_' : A.Compose([A.Rotate(limit=(270, 270), p=1)]),
            'rotate_300_' : A.Compose([A.Rotate(limit=(300, 300), p=1)]),
            'rotate_330_' : A.Compose([A.Rotate(limit=(330, 330), p=1)]),
        }

        id_list = []
        img_list = []
        label_list = []

        for id, img, label in zip(ids, images, labels):
            for prefix, aug_func in rotate_funcs:
                new_id = prefix + id
                aug_img = aug_func(image = img)['image']

                id_list.append(new_id)
                img_list.append(aug_img)
                label_list.append(label)

        return id_list, img_list, label_list
    
    def getFlipImg(self, ids, images, labels):
        """
        INPUT:
            ids : List[string ...] # ID List
            images : List[numpy.array(image) ...] # Image List
            labels : List[int ...] # Label List

        OUTPUT:
            id_list : List[string ...] # ID List
            img_list : List[numpy.array(image) ...] # Image List
            label_list : List[int ...] # Label List

        (ID LIST + flip 한 이미지 IMG LIST + label list) 반환
        => 입력의 1배 증강
        """
        rotate_funcs = {
            'flip_' : A.Comopose([A.HorizontalFlip(p=1)])
        }

        id_list = []
        img_list = []
        label_list = []

        for id, img, label in zip(ids, images, labels):
            for prefix, aug_func in rotate_funcs:
                new_id = prefix + id
                aug_img = aug_func(image = img)['image']

                id_list.append(new_id)
                img_list.append(aug_img)
                label_list.append(label)

        return id_list, img_list, label_list
    

    def getNoiseImg(self, ids, images, labels):
        """
        INPUT:
            ids : List[string ...] # ID List
            images : List[numpy.array(image) ...] # Image List
            labels : List[int ...] # Label List

        OUTPUT:
            id_list : List[string ...] # ID List
            img_list : List[numpy.array(image) ...] # Image List
            label_list : List[int ...] # Label List

        (ID LIST + 500, 2000 넣은 이미지 IMG LIST + label list) 반환 
        => 입력의 2배 증강
        """
        rotate_funcs = {
            'noise_500_' : A.Comopose([A.GaussNoise(var_limit=(500, 500), mean=0, p=1)]),
            'noise_2000_' : A.Comopose([A.GaussNoise(var_limit=(2000, 2000), mean=0, p=1)]),
        }

        id_list = []
        img_list = []
        label_list = []

        for id, img, label in zip(ids, images, labels):
            for prefix, aug_func in rotate_funcs:
                new_id = prefix + id
                aug_img = aug_func(image = img)['image']

                id_list.append(new_id)
                img_list.append(aug_img)
                label_list.append(label)

        return id_list, img_list, label_list
    
    def getPatchImg(self, ids, images, labels):
        '''
        INPUT:
            ids : List[string ...] # ID List
            images : List[np.array(image) ...] # Image List
            labels : List[int ...] # Label List

        OUTPUT:
            id_list : List[string ...] # Aug Data ID List
            img_list : List[np.array(image) ...] # Aug Image List
            label_list : List[int ...] # Aug Label List
        '''

        id_list = []
        img_list = []
        label_list = []

        for id, img, label in zip(ids, images, labels):
            # 랜덤하게 patch 할 이미지 뽑음
            random_img = random.choice(images) 
            while random_img == img:
                random_img = random.choice(images) 

            # patch로 데이터 증강
            patch_img = self.patch(img, random_img)
            
            new_id = 'patch_' + id

            # 결과 저장
            id_list.append(new_id)
            img_list.append(patch_img)
            label_list.append(label)

        return id_list, img_list, label_list
    
    def patch(self, img1, img2):
        '''
        INPUT;
            img1 : np.array(image)
            img2 : np.array(image)
        OUTPUT:
            img : np.array(image)

        이미지1, 2 를 주면 2의 32 x 32 부분을 랜덤하게 크롭해서 이미지 1에 붙여 만든 이미지를 반환
        '''
        # ph x pw 크기 만큼 patch 진행
        h, w, _ = img1.shape
        ph, pw = min(32, h), min(32, w)
        
        # 랜덤한 위치
        x, y = np.random.randint(0, w - pw + 1), np.random.randint(0, h - ph + 1)
        
        patch_img = np.copy(img1)
        patch_img[y : y + ph, x : x + pw] = img2[y:y + ph, x:x + pw]
        return patch_img