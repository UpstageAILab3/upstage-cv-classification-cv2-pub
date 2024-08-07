import logging
import timm
import os
import torch
import pandas as pd
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# 상수 설정
PRE_PATH = '/kkh/'
TEST_IMAGE_PATH = PRE_PATH + 'data/test'
TEST_CSV_PATH = PRE_PATH + 'data/sample_submission.csv'
MODEL_SAVE_PATH = PRE_PATH + 'test/'
MODEL_FILE = MODEL_SAVE_PATH + 'efficientnet_b4_Ep8_L_0.7092_A_0.9020_F1_0.8999.pt'

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_CLASSES = 17
PRETRAINED_SIZE = 380
PRETRAINED_MEANS = [0.485, 0.456, 0.406]
PRETRAINED_STDS = [0.229, 0.224, 0.225]
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()  # 모든 CPU 코어 사용

# Dataset class 정의
class ImageDataset(Dataset):
    def __init__(self, df, image_path, transform=None):
        self.df = df
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name = self.df.iloc[idx]['ID']
        img = np.array(Image.open(os.path.join(self.image_path, name)).convert("RGB"))

        # 변환 적용
        if self.transform:
            try:
                img = self.transform(image=img)['image']
            except Exception as e:
                print(f"Error in transforming image {name}: {e}")
                img = img  # 예외가 발생한 경우 원본 이미지 사용

        return img, 0  # 라벨은 테스트셋이므로 의미 없음

# 이미지 변환 설정
def get_transforms():
    valid_transform = A.Compose([
        A.Resize(PRETRAINED_SIZE, PRETRAINED_SIZE),
        A.Normalize(mean=PRETRAINED_MEANS, std=PRETRAINED_STDS),
        ToTensorV2()
    ])
    
    return valid_transform

# 최종 테스트 수행
def run_final_test(model, test_loader, test_df):
    model.eval()
    preds_list = []

    with torch.no_grad():
        tbar = tqdm(test_loader, desc="Final Test")
        for images, _ in tbar:
            images = images.to(device)

            preds = model(images)
            preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
    
    # 결과 저장
    test_df['target'] = preds_list
    test_df.to_csv(os.path.join(MODEL_SAVE_PATH, 'test_predictions_temp2.csv'), index=False)
    print(f"Final test predictions saved to {os.path.join(MODEL_SAVE_PATH, 'test_predictions.csv_temp2')}")

# 메인 함수
def run_final_test_pipeline():
    test_df = pd.read_csv(TEST_CSV_PATH)
    valid_transform = get_transforms()

    test_dataset = ImageDataset(test_df, TEST_IMAGE_PATH, transform=valid_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=NUM_CLASSES, in_chans=3)
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, MODEL_FILE)))
    model = model.to(device)

    run_final_test(model, test_loader, test_df)

    print('Final Test Completed!')

if __name__ == "__main__":
    run_final_test_pipeline()
