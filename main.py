import streamlit as st
from PIL import Image
import pandas as pd
import torch
import numpy as np
from torchvision import transforms
from importlib import import_module
import os

#모델 불러오기
def load_model(saved_model, device): 
    model_class = "EfficientNet_b2"

    #동적으로 model.py를 import하고 원하는 class를 가져옴
    model_cls = getattr(import_module("model"), model_class) 
    model = model_cls(18)

    # 모델 가중치를 로드한다.
    model_path = os.path.join(saved_model, model_class+"_best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

# 이미지를 모델 입력에 맞게 전처리하는 함수
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.548, 0.504, 0.479], std=[0.237, 0.247, 0.246]),
    ])
    img = preprocess(image)
    img = img.unsqueeze(0)  # 배치 차원 추가
    return img

mask_mapping = {
    0: '올바른 마스크 착용입니다😊',
    1: '잘못된 마스크 착용입니다😢',
    2: '마스크를 착용하지 않았습니다😶'
}

gender_mapping = {
    0: '남성',
    1: '여성'
}

age_mapping = {
    0: '10대~20대',
    1: '30대~50대',
    2: '60대 이상'
}

#모델 불러오기
# CUDA를 사용할 수 있는지 확인
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = load_model("./model", device).to(device)
model.eval()

# Streamlit 앱 구성
st.title("😷Mask Classification")
st.sidebar.header("원하는 사진을 업로드 해주세요")
st.sidebar.write("사람의 정면 모습이 사진을 통해 **나이**와 **성별**, **마스크 착용 여부**를 알려드립니다.")
st.sidebar.markdown("")

# 파일 업로드 기능
uploaded_file = st.sidebar.file_uploader("파일 업로드", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error("이미지를 열 수 없습니다. 올바른 이미지 파일을 업로드하세요")
    else:
        # 업로드된 이미지 표시
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 이미지", use_column_width=False, width=250)

        # 모델 예측
        with st.spinner("이미지 분류 중..."): #연산이 진행되는 도중 메세지 출력
            img_array = preprocess_image(image)
            with torch.no_grad():
                predictions = model(img_array)
            prediction = torch.argmax(predictions).item()

            # class별 분류
            mask_label = (prediction // 6) % 3
            gender_label = (prediction // 3) % 2
            age_label = prediction % 3

            mask = mask_mapping.get(mask_label, "Unknown")
            gender = gender_mapping.get(gender_label, "Unknown")
            age = age_mapping.get(age_label,"Unknown")

            st.balloons() #특수효과
        
        # 결과 출력
        st.markdown("### 분류 결과", unsafe_allow_html=True)
        st.write(f"{age} {gender}입니다. {mask}")