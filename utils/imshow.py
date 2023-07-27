import numpy as np
import cv2
import os
import torch
from PIL import Image

def save_fourier_transform(image_path, save_path):
    # 이미지를 흑백으로 읽어옵니다. (RGB 이미지인 경우 색상 채널을 평균하여 흑백으로 변환합니다.)
    print(os.path.isfile(image_path))
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 이미지를 torch.Tensor로 변환합니다.
    x = torch.tensor(image, dtype=torch.float32)

    # 입력 데이터의 크기를 2D 행렬로 조정합니다.
    x = x.unsqueeze(0).unsqueeze(0)  # 이 부분 추가

    # 푸리에 변환을 수행합니다.
    x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')  # dim 변경

    # 변환 결과의 절댓값을 취하여 주파수 성분의 크기를 얻습니다.
    magnitude_spectrum = torch.abs(x)

    # 변환 결과를 파일로 저장합니다.
    magnitude_spectrum_np = magnitude_spectrum.numpy()
    print(magnitude_spectrum_np.shape)
    magnitude_spectrum_np = np.squeeze(magnitude_spectrum_np)  # 불필요한 차원을 제거
    magnitude_spectrum_np = np.transpose(magnitude_spectrum_np, (1, 0))  # shape 변경
    cv2.imwrite(save_path, np.log(1 + magnitude_spectrum_np))

# 이미지 파일 경로와 저장할 파일 경로를 지정하여 함수를 호출합니다.
image_path = 'example.png'
save_path = 'fft.jpg'
save_fourier_transform(image_path, save_path)
