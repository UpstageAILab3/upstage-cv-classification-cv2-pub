#############################################################
## 📜 문서 타입 분류 대회
## kimkihong / helpotcreator@gmail.com / Upstage AI Lab 3기
## 2024.07.30.화 10:00 ~ 2024.08.11.일 19:00
## 파일 설명:
## - train_kr 폴더 내부의 이미지들을 클래스별로 나누어 폴더링하고, 그 모든 내용을 담은, train_kr_class 폴더를 생성함
#############################################################

import os
import shutil

# 경로 설정
images_folder_path = 'data/train_kr'
destination_folder_path = 'data/train_kr_class'

# 이미지 파일을 클래스명으로 된 폴더로 복사
for filename in os.listdir(images_folder_path):
    if filename.lower().endswith('.jpg'):
        # 파일명에서 클래스명 추출
        class_name = filename.split('_')[1].replace('.jpg', '')
        
        # 클래스명 폴더 경로 생성
        class_folder_path = os.path.join(destination_folder_path, class_name)
        os.makedirs(class_folder_path, exist_ok=True)
        
        # 이미지 파일을 클래스명 폴더로 복사
        source_file_path = os.path.join(images_folder_path, filename)
        destination_file_path = os.path.join(class_folder_path, filename)
        shutil.copy2(source_file_path, destination_file_path)

print("이미지 파일을 클래스별로 분류하여 복사했습니다.")
