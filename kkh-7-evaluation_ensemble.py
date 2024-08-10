import logging
import timm
import os
import random
import torch
import pandas as pd
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import platform

os_name = platform.system()
if os_name == 'Windows':
    PRE_PATH = ''
elif os_name == 'Linux':
    PRE_PATH = '/kkh/'
elif os_name == 'Darwin': # 맥
    PRE_PATH = '/kkh/'
    
VALID_IMAGE_PATH = PRE_PATH + 'data/valid_sunho'
VALID_CSV_PATH = PRE_PATH + 'data/valid_sunho.csv'
TEST_IMAGE_PATH = PRE_PATH + 'data/test'
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

NUM_CLASSES = 17
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

MODEL_FILES = [
    PRE_PATH + 'data/pt/dongjae/' + 'eff-b4_0.4954_0.9248_epoch7.pt',
    PRE_PATH + 'data/pt/dongjae/' + 'eff-b4_0.5089_0.9283_epoch2.pt',
    PRE_PATH + 'data/pt/dongjae/' + 'eff-b4_0.4055_0.9239_epoch0.pt',
]

def create_directory_with_backup(path):
    try:
        if os.path.exists(path):
            backup_path = path + '_backup'
            os.rename(path, backup_path)
            print(f"Existing folder renamed to: {backup_path}")
        os.makedirs(path)
        print(f"Folder created: {path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Dataset class 정의
class ImageDataset(Dataset):
    def __init__(self, df, image_path, transform=None):
        self.df = df
        self.image_path = image_path
        self.transform = transform

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
    valid_transform = A.Compose([
        A.Resize(380, 380),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return valid_transform

# 앙상블 voting
def majority_voting_from_preds(preds_list):
    preds_array = np.array(preds_list)
    majority_preds = []
    for i in range(len(preds_array[0])):  # preds_array의 열 수를 기준으로 예측
        preds = preds_array[:, i]
        unique, counts = np.unique(preds, return_counts=True)
        chosen_pred = unique[np.argmax(counts)]
        if len(set(counts)) == 1:  # 유니크한 값의 개수가 모두 동일한 경우
            chosen_pred = preds[0]  # 첫 번째 모델의 결과를 선택한다.
        majority_preds.append(chosen_pred)
    return majority_preds

# 평가 수행
def evaluate(models, loader):
    all_preds = []
    all_targets = []
    for model in models:
        model.eval()
        preds_list = []
        targets_list = []
        with torch.no_grad():
            tbar = tqdm(loader, desc="Evaluation")
            for images, targets in tbar:
                images = images.to(device)
                preds = model(images)
                preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
                targets_list.extend(targets.detach().cpu().numpy())
        all_preds.append(preds_list)
        all_targets.append(targets_list)

    ensemble_preds = majority_voting_from_preds(all_preds)
    return ensemble_preds, all_targets[0]

# 최종 테스트 수행
def run_final_test(models, test_loader, test_df):
    all_preds = []
    for model in models:
        model.eval()
        preds_list = []
        with torch.no_grad():
            tbar = tqdm(test_loader, desc="Final Test")
            for images, _ in tbar:
                images = images.to(device)
                preds = model(images)
                preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        all_preds.append(preds_list)
    
    # 앙상블 예측
    ensemble_preds = majority_voting_from_preds(all_preds)
    
    # 결과 저장
    test_df['target'] = ensemble_preds
    test_df.to_csv(os.path.join(MODEL_SAVE_PATH, 'ensemble_test.csv'), index=False)
    print(f"Final test predictions saved to {os.path.join(MODEL_SAVE_PATH, 'ensemble_test.csv')}")

# 메인 함수
def run_final_test_pipeline():
    test_df = pd.read_csv(TEST_CSV_PATH)
    valid_df = pd.read_csv(VALID_CSV_PATH)
    valid_transform = get_transforms()

    test_dataset = ImageDataset(test_df, TEST_IMAGE_PATH, transform=valid_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    valid_dataset = ImageDataset(valid_df, VALID_IMAGE_PATH, transform=valid_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    models = []
    for model_file in MODEL_FILES:
        if 'efficientnet_b4' in model_file:
            model_name = 'tf_efficientnet_b4'
        elif 'efficientnet_b0' in model_file:
            model_name = 'efficientnet_b0'
        elif 'resnet50' in model_file:
            model_name = 'resnet50'
        else:
            raise ValueError(f"Unknown model file: {model_file}")
        
        model = timm.create_model(model_name, pretrained=False, num_classes=NUM_CLASSES, in_chans=3)

        # map_location 인자를 사용하여 모델을 CPU에서 로드
        model.load_state_dict(torch.load(model_file, map_location=device))
        model = model.to(device)
        models.append(model)

    # Validation 평가 수행
    val_preds, val_targets = evaluate(models, valid_loader)
    valid_df['predictions'] = val_preds
    valid_df.to_csv(os.path.join(MODEL_SAVE_PATH, 'ensemble_valid.csv'), index=False)
    print(f"Validation predictions saved to {os.path.join(MODEL_SAVE_PATH, 'ensemble_valid.csv')}")

    # F1 점수 계산 및 저장
    f1 = f1_score(val_targets, val_preds, average='macro')
    f1_score_path = os.path.join(MODEL_SAVE_PATH, f'ensemble_valid_f1_{f1:.4f}.txt')
    with open(f1_score_path, 'w') as f:
        f.write(f'Validation F1 Score: {f1:.4f}\n')
    print(f"Validation F1 score saved to {f1_score_path}")

    # 최종 테스트 수행
    run_final_test(models, test_loader, test_df)

if __name__ == "__main__":
    create_directory_with_backup(MODEL_SAVE_PATH)
    run_final_test_pipeline()
