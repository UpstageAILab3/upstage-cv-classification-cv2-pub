#############################################################
## 📜 문서 타입 분류 대회
## kimkihong / helpotcreator@gmail.com / Upstage AI Lab 3기
## 2024.07.30.화 10:00 ~ 2024.08.11.일 19:00
## 파일 설명:
## - 원본 이미지의 상단 20%를 crop하여 제목 부분만 추출한다.
#############################################################

import os
import pandas as pd
from PIL import Image

# 경로 설정
folder_path = "data/train_kr_title"
csv_file = "data/train_kr_title.csv"

# 증강할 target 리스트
augment_targets = [14, 11, 13, 3, 15, 4, 10, 6, 12, 7]

# CSV 파일 로드
df = pd.read_csv(csv_file)

# 증강된 데이터 저장을 위한 리스트
new_rows = []

# 이미지 증강 작업
for _, row in df.iterrows():
    if row['target'] in augment_targets:
        image_path = os.path.join(folder_path, row['ID'])
        if os.path.exists(image_path):
            img = Image.open(image_path)
            width, height = img.size

            # 상단 20%만 남기고 아래는 삭제
            cropped_img = img.crop((0, 0, width, int(height * 0.2)))

            # 새로운 파일명 설정 및 저장
            new_filename = f"title_{row['ID']}"
            new_image_path = os.path.join(folder_path, new_filename)
            cropped_img.save(new_image_path)

            # 새로운 row 추가
            new_rows.append({"ID": new_filename, "target": row['target']})

# 기존 CSV에 새로운 행 추가
new_df = pd.DataFrame(new_rows)
df = pd.concat([df, new_df], ignore_index=True)

# 변경된 CSV 파일 저장
df.to_csv(csv_file, index=False)

# 이미지 파일 개수 및 CSV 파일의 레코드 수 출력
image_files_count = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
csv_records_count = len(df)

print(f"Number of images in folder: {image_files_count}")
print(f"Number of records in CSV: {csv_records_count}")
