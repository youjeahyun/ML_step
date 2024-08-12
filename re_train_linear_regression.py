import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import mlflow.pytorch

# 모델 정의 (저장할 때와 동일한 구조로 정의해야 함)
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

# 모델 인스턴스 생성 및 로드
model = LinearRegressionModel()
model_path = 'linear_regression_model.pth'
model.load_state_dict(torch.load(model_path))

# 모델을 GPU로 이동 (가능한 경우)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 새로운 데이터 로드 및 전처리
new_data = pd.read_csv('optimized_floor_data.csv')

# 데이터 확인
print(pd.isna(new_data).sum())  # NaN 값의 개수 확인
print(np.isinf(new_data.values).sum())  # 무한값의 개수 확인

# 데이터 전처리
X_new = new_data[['Temperature', 'Power_Consumption']].values
y_new = new_data['Optimal_Floor'].values

# PyTorch 텐서로 변환
X_new_tensor = torch.tensor(X_new, dtype=torch.float32)
y_new_tensor = torch.tensor(y_new, dtype=torch.float32).view(-1, 1)

# 데이터 정규화
def normalize_tensor(tensor):
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0)
    return (tensor - mean) / std

X_new_normalized = normalize_tensor(X_new_tensor)

# 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 재학습 시작
num_epochs = 30000
for epoch in range(num_epochs):
    model.train()
    X_new_gpu = X_new_normalized.to(device)
    y_new_gpu = y_new_tensor.to(device)
    outputs = model(X_new_gpu)
    loss = criterion(outputs, y_new_gpu)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 최종 모델 평가 (옵션)
model.eval()
with torch.no_grad():
    test_outputs = model(X_new_gpu)
    test_loss = criterion(test_outputs, y_new_gpu)
    print(f'Test Loss: {test_loss.item():.4f}')

# 모델을 로컬에 저장 (옵션)
model_path_new = 'linear_regression_model_retrained.pth'
torch.save(model.state_dict(), model_path_new)
print(f"모델이 로컬에 저장되었습니다: {model_path_new}")
