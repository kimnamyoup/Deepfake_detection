import os
# KMP_DUPLICATE_LIB_OK 환경 변수 설정 (특정 환경에서 발생할 수 있는 오류 방지)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import time


# 디바이스 설정: GPU 사용 가능 시 GPU, 아니면 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 크기 및 전처리 정의
IMG_SIZE = 128 # 적절한 이미지 크기로 조절하세요.
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet 통계치 (RGB)
                         std=[0.229, 0.224, 0.225])
])

# 데이터셋의 평균(mean)과 표준편차(std) 값을 텐서로 정의
# imshow를 위해 이미지 데이터를 역정규화할 때 사용됩니다.
mean_vals = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std_vals = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# 이미지 텐서를 역정규화하는 함수
def denormalize(tensor_image):
    """
    정규화된 텐서 이미지를 [0, 1] 범위로 역변환합니다.
    입력 텐서는 (C, H, W) 형태여야 합니다.
    """
    # 텐서와 동일한 디바이스(CPU/GPU)에 mean_vals와 std_vals가 있는지 확인
    tensor_image = tensor_image * std_vals.to(tensor_image.device) + mean_vals.to(tensor_image.device)
    # 역변환 후에도 부동 소수점 오차로 인해 [0, 1] 범위를 약간 벗어날 수 있으므로 클램핑합니다.
    tensor_image = torch.clamp(tensor_image, 0.0, 1.0)
    return tensor_image

# 데이터셋 경로 설정
# D:\PythonProject\GAN\MVTec\mvtec_anomaly_detection\bottle 경로를
# 실제 train 및 test 폴더의 상위 폴더 경로에 맞춰주세요.
BASE_DATA_PATH = "D:/PythonProject/GAN/MVTec/mvtec_anomaly_detection/bottle"
# BASE_DATA_PATH = "D:\PythonProject\GAN\facetest\aaa"

# ImageFolder를 사용하여 데이터셋 로드
# ImageFolder는 root 경로 아래의 각 서브폴더를 클래스로 인식합니다.
# 예: BASE_DATA_PATH/train/good, BASE_DATA_PATH/test/good, BASE_DATA_PATH/test/broken_bottle 등
train_dataset = ImageFolder(root=os.path.join(BASE_DATA_PATH, "train"), transform=transform)
test_dataset  = ImageFolder(root=os.path.join(BASE_DATA_PATH, "test"), transform=transform)

print(f"Number of training images: {len(train_dataset)}")
print(f"Number of test images: {len(test_dataset)}")

# 학습용 DataLoader 설정 (배치 128, 셔플)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# 테스트용 DataLoader 설정 (배치 128, 셔플 없음)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Convolutional Autoencoder 클래스 정의
class ConvAutoencoder(nn.Module):
    # 생성자: 인코더와 디코더 블록을 정의
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # 인코더: Conv→ReLU→MaxPool로 특성 맵 축소
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # 입력 3채널 (RGB)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),  # 병이미지라 음수값(어두운면)이 강해서 LeakyReLU 값을 0.01보다 높게 0.2로 설정후 낮춰가는 방향으로 할 예정 -> 0.1로 변경예정
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 채널 불일치 수정 (32 -> 64)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2)
        )
        # 디코더: Upsample→Conv→ReLU/Sigmoid로 복원
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), # 인코더의 최종 채널에 맞춰 수정 (64 -> 32)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 3, kernel_size=3, padding=1), # 출력 3채널 (RGB)
            nn.Tanh()                     # 출력 범위 [0,1]
        )

    # 순전파 정의: 입력→인코더→디코더→출력
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 모델 인스턴스 생성 및 디바이스에 할당
model = ConvAutoencoder()

# 여러 개의 GPU를 사용할 수 있는지 확인하고, 가능하다면 DataParallel로 모델을 래핑합니다.
if torch.cuda.device_count() > 1:
  print(f"Let's use {torch.cuda.device_count()} GPUs!")
  model = nn.DataParallel(model)

model.to(device)

# 옵티마이저(Adam) 및 손실 함수(MSE) 정의
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

'''
# <모델 인스턴스 생성 및 옵티마이저/손실 함수 정의 바로 아래에 추가>

# 첫 번째 배치 가져와서 확인
dataiter = iter(train_loader)
first_batch_imgs, _ = next(dataiter)

print(f"Shape of first batch images: {first_batch_imgs.shape}") # (batch_size, C, H, W) 확인

# 첫 번째 이미지 하나만 선택해서 시각화
sample_img_tensor = first_batch_imgs[0]

# 디바이스에 있을 수 있으므로 CPU로 이동
sample_img_tensor = sample_img_tensor.cpu()

# 역정규화
denorm_img = denormalize(sample_img_tensor)

# (C, H, W) -> (H, W, C)로 변경하여 imshow에 맞춤
plt.imshow(denorm_img.permute(1, 2, 0))
plt.title("Sample Image after Denormalization")
plt.axis('off')
plt.show()

# <이후 train(model, train_loader, epochs=100) 코드로 이어서 학습>
'''
# 학습 함수 정의
def train(model, loader, epochs=5): # 에포크 수 기본값 5로 유지 (사용자 코드는 10으로 변경됨)
    breakPoint = 0
    updateCheck = 0
    # 학습 모드로 전환
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        for imgs, _ in loader:
            # 입력 이미지 디바이스로 이동
            imgs = imgs.to(device)
            # 기울기 초기화
            optimizer.zero_grad()
            # 재구성 이미지 계산
            outputs = model(imgs)
            # 재구성 손실 계산
            loss = criterion(outputs, imgs)
            # 역전파 수행
            loss.backward()
            # 파라미터 업데이트
            optimizer.step()
            # 배치 손실 누적
            running_loss += loss.item() * imgs.size(0)
        # 에포크별 평균 손실 출력
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader.dataset):.6f}")
        update_loss=running_loss / len(loader.dataset)

        if updateCheck == 0:
            breakPoint=update_loss
            updateCheck += 1

        if update_loss < breakPoint:
            breakPoint=update_loss
            print(f"update!! update Values [{breakPoint}]")
            updateCheck+=1
            print(f"Update!! Best Loss: [{breakPoint:.6f}] - Model Saved!")
            # 모델의 state_dict를 저장합니다.
            # DataParallel을 사용하는 경우, model.module.state_dict()를 저장해야 합니다.
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), "best_autoencoder_model.pth")
            else:
                torch.save(model.state_dict(), "best_autoencoder_model.pth")
        else:
            print("Didn't Update")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"코드 실행 시간: {elapsed_time:.4f} 초")


# 모델 학습 실행 (사용자가 요청한 10 에포크로 설정)
train(model, train_loader, epochs=1000)

# 평가 모드로 전환
model.eval()
# 테스트 전체 이미지와 재구성, 재구성 오차를 저장할 리스트 초기화
all_imgs, all_recons, all_errors = [], [], []

# 그레이디언트 비활성화
with torch.no_grad():
    for imgs, _ in test_loader:
        # 이미지 디바이스로 이동
        imgs = imgs.to(device)
        # 재구성 수행
        recons = model(imgs)
        # 배치별 픽셀 단위 MSE 계산
        batch_errors = torch.mean((recons - imgs) ** 2, dim=[1,2,3])
        # CPU로 옮겨 NumPy 배열로 변환
        all_imgs.append(imgs.cpu())
        all_recons.append(recons.cpu())
        all_errors.append(batch_errors.cpu())

# 리스트를 하나의 텐서/배열로 결합
all_imgs    = torch.cat(all_imgs)
all_recons  = torch.cat(all_recons)
all_errors  = torch.cat(all_errors).numpy()

thresh_95 = np.percentile(all_errors, 95)
threshold= np.percentile(all_errors, 95)
print("95th percentile threshold:", thresh_95)

# 임계값: 평균 + 3*표준편차로 설정
# threshold = all_errors.mean() + 3 * all_errors.std()
print(f"Anomaly Threshold: {threshold:.6f}")

# 이상치 인덱스와 정상치 인덱스 추출
anomaly_idxs = np.where(all_errors > threshold)[0]
normal_idxs  = np.where(all_errors <= threshold)[0]

# 현재 감지된 이상치 및 정상치 개수와 오류 통계 출력
print(f"Number of anomalies found: {len(anomaly_idxs)}")
print(f"Number of normal samples found: {len(normal_idxs)}")
print("---------------------------------------------------")
print(f"Min error: {all_errors.min():.6f}")
print(f"Max error: {all_errors.max():.6f}")
print(f"Mean error: {all_errors.mean():.6f}")
print(f"Std error: {all_errors.std():.6f}")
print(f"Calculated Threshold: {threshold:.6f}")

# 이상치와 정상치 각각 샘플 랜덤 선택
np.random.seed(42)

# 선택할 이상치 샘플의 개수를 조정 (최대 5개, 하지만 감지된 개수 내에서)
num_anom_to_select = min(len(anomaly_idxs), 5)
if num_anom_to_select > 0:
    sel_anom = np.random.choice(anomaly_idxs, num_anom_to_select, replace=False)
else:
    sel_anom = np.array([]) # 이상치가 없으면 빈 NumPy 배열로 설정
    print("시각화할 이상치가 감지되지 않았습니다.")

# 정상 샘플도 5개를 뽑되, 정상 샘플 수가 5개 미만일 경우를 대비
num_norm_to_select = min(len(normal_idxs), 5)
sel_norm = np.random.choice(normal_idxs, num_norm_to_select, replace=False)

# 시각화: 원본, 재구성, 에러 맵(heatmap)
fig, axes = plt.subplots(3, 5, figsize=(15, 9)) # 시각화 크기 조정

for i in range(5):
    idx_to_plot = None
    title_prefix = ""

    # 이상치 샘플이 있다면 먼저 이상치를 표시
    if i < len(sel_anom):
        idx_to_plot = sel_anom[i]
        title_prefix = "Anomaly"
    # 이상치 샘플이 부족하면 정상 샘플로 채움
    elif i < len(sel_norm):
        idx_to_plot = sel_norm[i]
        title_prefix = "Normal"
    else:
        # 더 이상 그릴 샘플이 없으면 루프 종료
        break

    # 샘플이 있는 경우에만 그리기
    if idx_to_plot is not None:
        # 원본 이미지 시각화
        # all_imgs[idx_to_plot]는 (C, H, W) 형태이므로, imshow에 맞게 (H, W, C)로 변경하고 denormalize 적용
        axes[0, i].imshow(denormalize(all_imgs[idx_to_plot]).permute(1, 2, 0))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title(f"{title_prefix} Orig")
        else:
            axes[0, i].set_title(f"Orig {i+1}")

        # 재구성 이미지 시각화
        # all_recons[idx_to_plot]도 (C, H, W) 형태이므로, imshow에 맞게 (H, W, C)로 변경하고 denormalize 적용
        axes[1, i].imshow(denormalize(all_recons[idx_to_plot]).permute(1, 2, 0))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title(f"{title_prefix} Recon")
        else:
            axes[1, i].set_title(f"Recon {i+1}")

        # 에러 히트맵 시각화 (에러 맵은 채널 평균을 내어 (H, W) 형태이므로 그대로 사용 가능)
        # 에러 맵은 재구성 오류의 시각화이므로 denormalize를 적용하지 않습니다.
        err_map = torch.mean((all_recons[idx_to_plot] - all_imgs[idx_to_plot])**2, dim=0)
        axes[2, i].imshow(err_map, cmap='hot')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title(f"{title_prefix} Error Map")
        else:
            axes[2, i].set_title(f"Error Map {i+1}")
    else:
        # 샘플이 없는 칸은 비워두기
        for row in range(3):
            axes[row, i].axis('off')

plt.tight_layout()
plt.show()