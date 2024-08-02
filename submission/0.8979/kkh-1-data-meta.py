#############################################################
## 📜 문서 타입 분류 대회
## kimkihong / helpotcreator@gmail.com / Upstage AI Lab 3기
## 2024.07.30.화 10:00 ~ 2024.08.11.일 19:00
## 파일 설명:
## - meta.csv 파일 내용에 한국어 내용을 추가하여 meta_kr.csv 파일로 만들어 줌
#############################################################

import pandas as pd

def translate_meta(input_file: str, output_file: str):
    translations = {
        'account_number': '계좌번호',
        'application_for_payment_of_pregnancy_medical_expenses': '건강보험 임신출산 진료비 지급 신청서',
        'car_dashboard': '자동차 계기판',
        'confirmation_of_admission_and_discharge': '입퇴원 확인서',
        'diagnosis': '진단서',
        'driver_lisence': '운전면허증',
        'medical_bill_receipts': '진료비영수증',
        'medical_outpatient_certificate': '통원진료 확인서',
        'national_id_card': '주민등록증',
        'passport': '여권',
        'payment_confirmation': '진료비 납입 확인서',
        'pharmaceutical_receipt': '약제비 영수증',
        'prescription': '처방전',
        'resume': '이력서',
        'statement_of_opinion': '소견서',
        'vehicle_registration_certificate': '자동차 등록증',
        'vehicle_registration_plate': '자동차 번호판'
    }

    df = pd.read_csv(input_file)
    df['class_name_ko'] = df['class_name'].map(translations)
    df['class_name'] = df['class_name'].replace('driver_lisence', 'driver_license')
    df.to_csv(output_file, index=False, encoding='utf-8')

translate_meta('data/meta.csv', 'data/meta_kr.csv')