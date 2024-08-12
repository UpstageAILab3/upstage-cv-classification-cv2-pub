#############################################################
## 📜 문서 타입 분류 대회
## kimkihong / helpotcreator@gmail.com / Upstage AI Lab 3기
## 2024.07.30.화 10:00 ~ 2024.08.11.일 19:00
## 파일 설명:
## - 학습, 평가, 앙상블
#############################################################

import logging
import timm
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('timm')
logger.setLevel(logging.WARNING)
import os
import shutil
import random
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
import albumentations as A
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 상수 설정
PRE_PATH = '/kkh/'
TRAIN_KR_IMAGE_PATH = PRE_PATH + 'data/train_kr'
TRAIN_KR_AUG_IMAGE_PATH = PRE_PATH + 'data/train_kr_aug_4'
VALID_SUNHO_IMAGE_PATH = PRE_PATH + 'data/valid_sunho'
TEST_IMAGE_PATH = PRE_PATH + 'data/test'
META_KR_CSV_PATH = PRE_PATH + 'data/meta_kr.csv'
TRAIN_KR_CSV_PATH = PRE_PATH + 'data/train_kr.csv'
TRAIN_KR_AUG_CSV_PATH = PRE_PATH + 'data/train_kr_aug_4.csv'
VALID_SUNHO_CSV_PATH = PRE_PATH + 'data/valid_sunho.csv'
TEST_CSV_PATH = PRE_PATH + 'data/sample_submission.csv'
MODEL_SAVE_PATH = PRE_PATH + 'test/'

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_NAMES = ['efficientnet_b4', 'resnet50', 'efficientnet_b0']
NUM_EPOCHS = 50
EACH_MODEL_SAVE = True

PRETRAINED_SIZE = 380
PRETRAINED_MEANS = [0.485, 0.456, 0.406]
PRETRAINED_STDS = [0.229, 0.224, 0.225]

LR = 5e-4
BATCH_SIZE = 32
# DROPOUT_RATIO = 0.2
PATIENCE = 5
NUM_WORKERS = os.cpu_count()  # 모든 CPU 코어 사용
NUM_CLASSES = 17

# 폴더 생성 및 백업
def prepare_folders():
    if os.path.exists(MODEL_SAVE_PATH):
        backup_path = PRE_PATH + 'test_backup/'
        shutil.move(MODEL_SAVE_PATH, backup_path)
        print(f"Existing '{MODEL_SAVE_PATH}' directory moved to '{backup_path}'.")

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Load Data
def load_data():
    meta_kr_df = pd.read_csv(META_KR_CSV_PATH)
    train_kr_df = pd.read_csv(TRAIN_KR_CSV_PATH)
    train_kr_aug_df = pd.read_csv(TRAIN_KR_AUG_CSV_PATH)
    valid_sunho_df = pd.read_csv(VALID_SUNHO_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)
    return meta_kr_df, train_kr_df, train_kr_aug_df, valid_sunho_df, test_df

# Dataset class 정의
class ImageDataset(Dataset):
    def __init__(self, df, image_path, transform=None, oversample=False):
        self.df = df
        self.image_path = image_path
        self.transform = transform
        self.oversample = oversample

        if self.oversample:
            class_counts = np.bincount(self.df['target']) # 클래스별로 이미지 데이터 개수를 조회한다.
            max_class_count = max(class_counts) # 최고로 많은 데이터 양을 선정한다.

            # 오버샘플링할 비율 = 최고로 많은 데이터 개수 // 각 클래스별 데이터 개수
            oversample_factors = [max_class_count // count for count in class_counts]

            oversampled_data = []
            for cls, factor in enumerate(oversample_factors):
                cls_data = self.df[self.df['target'] == cls]
                oversampled_data.append(cls_data.sample(factor * len(cls_data), replace=True))

            self.df = pd.concat(oversampled_data).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df.iloc[idx]
        img = np.array(Image.open(os.path.join(self.image_path, name)).convert("RGB"))

        # 변환 적용
        if self.transform:
            try:
                img = self.transform(image=img)['image']
            except Exception as e:
                print(f"Error in transforming image {name}: {e}")
                img = img  # 예외가 발생한 경우 원본 이미지 사용

        return img, target

# 이미지 변환 설정
def get_transforms():
    train_transform = A.Compose([
        A.Resize(PRETRAINED_SIZE, PRETRAINED_SIZE),
        A.Normalize(mean=PRETRAINED_MEANS, std=PRETRAINED_STDS),
        ToTensorV2()
    ])

    valid_transform = A.Compose([
        A.Resize(PRETRAINED_SIZE, PRETRAINED_SIZE),
        A.Normalize(mean=PRETRAINED_MEANS, std=PRETRAINED_STDS),
        ToTensorV2()
    ])
    
    return train_transform, valid_transform

# 학습
def training(model, dataloader, criterion, optimizer, device, epoch, num_epochs, model_name):
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []

    pbar = tqdm(dataloader, desc=f"{model_name} - Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        preds = model(images)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(labels.detach().cpu().numpy())

        pbar.set_description(f"{model_name} - Epoch [{epoch+1}/{num_epochs}] - Train Loss: {loss.item():.4f}")
        
    train_loss /= len(dataloader)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')
    return model, train_loss, train_acc, train_f1, preds_list

# 평가
def evaluation(model, dataloader, criterion, device, epoch, num_epochs, model_name):
    model.eval()
    valid_loss = 0.0
    preds_list = []
    targets_list = []

    with torch.no_grad():
        tbar = tqdm(dataloader, desc=f"{model_name} - Evaluation")
        for images, labels in tbar:
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            loss = criterion(preds, labels)

            valid_loss += loss.item()
            preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
            targets_list.extend(labels.detach().cpu().numpy())
            tbar.set_description(f"{model_name} - Epoch [{epoch+1}/{num_epochs}] - Valid Loss: {loss.item():.4f}")

    valid_loss /= len(dataloader)
    valid_acc = accuracy_score(targets_list, preds_list)
    valid_f1 = f1_score(targets_list, preds_list, average='macro')
    print(f"{model_name} - Epoch [{epoch+1}/{num_epochs}] - Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}, Valid F1: {valid_f1:.4f}")

    return valid_loss, valid_acc, valid_f1, preds_list

# 학습 및 평가
def training_loop(model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, patience, model_name):
    best_valid_f1 = -1
    early_stop_counter = 0
    best_model = None
    best_preds = []

    for epoch in range(num_epochs):
        model, train_loss, train_acc, train_f1, _ = training(model, train_dataloader, criterion, optimizer, device, epoch, num_epochs, model_name)
        valid_loss, valid_acc, valid_f1, valid_preds = evaluation(model, valid_dataloader, criterion, device, epoch, num_epochs, model_name)

        # 모델 저장
        if EACH_MODEL_SAVE and epoch >= 2:
            model_save_path = os.path.join(MODEL_SAVE_PATH, f"{model_name}_Ep{epoch+1}_L_{valid_loss:.4f}_A_{valid_acc:.4f}_F1_{valid_f1:.4f}.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f"{model_name} - Saved Model to {model_save_path}")

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_model = model
            best_preds = valid_preds
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"{model_name} - Early stopping triggered for {model_name} at epoch {epoch+1}")
            break

    return best_model, best_valid_f1, best_preds

# 앙상블 voting
def majority_voting_from_preds(preds_list):
    preds_array = np.array(preds_list)
    majority_preds = []

    for i in range(len(preds_array[0])):  # preds_array의 열 수를 기준으로 예측
        preds = preds_array[:, i]
        unique, counts = np.unique(preds, return_counts=True)
        print(f"{i}번째 이미지 - 모델들이 예측한 클래스: {preds} ==>> 앙상블 voting 선택: ", end="")

        chosen_pred = unique[np.argmax(counts)]
        if len(set(counts)) == 1:  # 유니크한 값의 개수가 모두 동일한 경우
            chosen_pred = preds[0]  # 첫 번째, 모델의 결과를 선택한다.
        majority_preds.append(chosen_pred)
    
    return majority_preds

# 앙상블 평가
def evaluation_ensemble(dataloader, criterion, device, preds, model_name):
    targets_list = []

    with torch.no_grad():
        tbar = tqdm(dataloader, desc=f"{model_name} - Evaluation")
        for images, labels in tbar:
            labels = labels.to(device)
            targets_list.extend(labels.cpu().numpy())
            tbar.set_description(f"{model_name} - Valid Accuracy Calculation")
    valid_acc = accuracy_score(targets_list, preds)
    valid_f1 = f1_score(targets_list, preds, average='macro')
    print(f"{model_name} - Valid Acc: {valid_acc:.4f}, Valid F1: {valid_f1:.4f}")
    return valid_acc, valid_f1

# 메인 함수
def run_training_pipeline():
    prepare_folders()

    meta_kr_df, train_kr_df, train_kr_aug_df, valid_sunho_df, test_df = load_data()
    train_transform, valid_transform = get_transforms()

    train_dataset = ImageDataset(train_kr_aug_df, TRAIN_KR_AUG_IMAGE_PATH, transform=train_transform, oversample=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    valid_dataset = ImageDataset(valid_sunho_df, VALID_SUNHO_IMAGE_PATH, transform=valid_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model_predictions = []
    valid_preds_list = []  # 각 모델의 예측 결과를 저장하기 위한 리스트
    for model_name in MODEL_NAMES:
        model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES, in_chans=3)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=LR)

        best_model, valid_max_f1, valid_preds = training_loop(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=NUM_EPOCHS, patience=PATIENCE, model_name=model_name)
        model_predictions.append(best_model)
        valid_preds_list.append(valid_preds)

    # 앙상블 모델의 예측을 기반으로 평가
    valid_preds_ensemble = majority_voting_from_preds(valid_preds_list)
    ensemble_valid_acc, ensemble_valid_f1 = evaluation_ensemble(valid_loader, criterion, device, valid_preds_ensemble, model_name="ensemble")
    print('=========================================================================================================')
    print('END!!!!!!')

if __name__ == "__main__":
    run_training_pipeline()
