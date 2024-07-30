import csv
import os

# 경로 설정
train_kr_csv_path = 'data\\train_kr.csv'
meta_kr_csv_path = 'data\\meta_kr.csv'
train_kr_folder_path = 'data\\train_kr'

# meta_kr.csv 파일을 읽어 target과 class_name_ko 매핑
target_to_ko = {}
with open(meta_kr_csv_path, mode='r', encoding='utf-8') as meta_file:
    reader = csv.DictReader(meta_file)
    for row in reader:
        target_to_ko[row['target']] = row['class_name_ko']

# train_kr.csv 파일을 읽어 새로운 파일명 생성
updated_rows = []
with open(train_kr_csv_path, mode='r', encoding='utf-8') as train_file:
    reader = csv.DictReader(train_file)
    for row in reader:
        original_file_name = row['ID']
        target = row['target']
        new_file_name = original_file_name.replace('.jpg', f'_{target_to_ko[target]}.jpg')
        updated_rows.append({'ID': new_file_name, 'target': target})
        
        # 실제 파일명 변경
        original_file_path = os.path.join(train_kr_folder_path, original_file_name)
        new_file_path = os.path.join(train_kr_folder_path, new_file_name)
        os.rename(original_file_path, new_file_path)

# 업데이트된 파일명으로 train_kr.csv 파일 저장
with open(train_kr_csv_path, mode='w', newline='', encoding='utf-8') as train_file:
    fieldnames = ['ID', 'target']
    writer = csv.DictWriter(train_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(updated_rows)

print("파일 이름 변경 및 CSV 업데이트가 완료되었습니다.")
