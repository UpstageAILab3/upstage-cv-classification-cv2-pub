[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/FVjNDCrt)
# Title (Please modify the title)
## Team

![](https://github.com/UpstageAILab3/upstage-cv-classification-cv2/blob/kimkihong/ppt_jpg/3.jpg?raw=true)


## 1. Competiton Info

### Overview
<img width="860" alt="image" src="https://github.com/user-attachments/assets/94fe20c5-e769-40c8-b6c7-9d63ccdeda6d">

이미지 분류 대회
1570장의 학습 이미지를 통해 3140장의 평가 이미지를 예측.
총 17가지의 클래스로 분류하면되고 평가지표는 f1 macro 사용.

### Timeline

 - start : 7월 30일 (화) 10:00
- final submission deadline : 8월 11일 (일) 19:00

## 2. Components

### Directory

- _Insert your directory structure_

e.g.
```
├── code
│   ├── jupyter_notebooks
│   │   └── model_train.ipynb
│   └── train.py
├── docs
│   ├── pdf
│   │   └── (Template) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디 .pptx
│   └── paper
└── input
    └── data
        ├── eval
        └── train
```

## 3. Data descrption

### Dataset overview
<img width="566" alt="image" src="https://github.com/user-attachments/assets/5d171b45-f349-41df-80ac-a3452822a2f0">

train dataset : 1570장
test dataset : 3140장

train dataset은 대부분 노이즈가 없고 깔끔하게 있는 반면 test dataset은 노이즈가 많고 대부분의 데이터가 오염이 되어있음.

### EDA
<img width="639" alt="image" src="https://github.com/user-attachments/assets/cd451af8-099d-492d-a8b9-7402c841af8b">

이미지 사이즈 분포

이미지 사이즈 분포를 보면 1000을 넘는 해상도를 가지진 않아서 모델에 알맞는 해상도인 244 혹은 380으로 resizing 가능.
유독 작거나 유독 커다란 데이터가 그렇게 많지 않고 해상도 비율이 이상한 데이터는 없어서 데이터 drop 없이 진행.

<img width="656" alt="image" src="https://github.com/user-attachments/assets/39392645-a559-48dc-91d0-a896523b5674">

클래스별 데이터 수

1, 14번 클래스가 다른 클래스에 비해 적은 수의 데이터를 가지고 있지만 샘플링을 하기보다는 데이터 증강을 통해 충분한 데이터 량을 확보.

<img width="659" alt="image" src="https://github.com/user-attachments/assets/ae23664f-1782-4a6d-898d-337343091e09">

몇 가지 잘못 라벨링 되어있는 데이터 수정.

### Data Processing

<img width="648" alt="image" src="https://github.com/user-attachments/assets/cd94e860-7f56-4c24-b3ea-613cfdc399a3">

<img width="288" alt="image" src="https://github.com/user-attachments/assets/ba171d98-f0ef-4f95-ae49-d582737b292f"><img width="164" alt="image" src="https://github.com/user-attachments/assets/9a912a0c-c880-4752-91d9-e2822b216dfc">

데이터 증강을 rotate, flip, shift, noise, patch 를 사용함.
1570 -> 116180개 로 train dataset 증강됨.

## 4. Modeling

### Model descrition

<img width="651" alt="image" src="https://github.com/user-attachments/assets/f36f82f9-1452-48b2-ab86-e14b4f421134">

CNN 기반 모델 : resnet50, cafomer_s18, efficientnet_b0, efficientnet_b4, ConvNext

Transformer 기반 모델 : vit_laion_2b, swin_t, swin_b, swin_b

위 모델들을 사용해서 성능 비교 실험을 했을때 transformer 기반의 모델들은 학습속도도 현저히 떨어지고 성능도 좋지 않음.
반면 CNN 기반 모델에서 efficientnet_b4 > efficientnet_b0 > resnet50 순으로 성능이 좋다고 판단됨.

base model로 efficientnet_b4를 사용하고 다른 모델들은 모델 앙상블 할때 사용.

## 5. Result

### Leader Board

<img width="403" alt="image" src="https://github.com/user-attachments/assets/8c9edf3d-a2e4-4692-abbf-23ed9e69edde">


### Presentation

[cv competition presentation.pdf](https://github.com/user-attachments/files/16622219/cv.competition.presentation.pdf)

