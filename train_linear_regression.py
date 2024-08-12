import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# MLflow 설정
mlflow.set_tracking_uri('http://192.168.0.46:5000')  # MLflow 서버 URI
mlflow.set_experiment('linear_regression_experiment')  # 실험 이름

# 데이터 로드 및 전처리
data = pd.read_csv('optimized_floor_data.csv')

# 데이터 확인
print(pd.isna(data).sum())  # NaN 값의 개수 확인
print(np.isinf(data.values).sum())  # 무한값의 개수 확인

# 특성과 타겟 변수 분리
X = data[['Temperature', 'Power_Consumption']].values
y = data['Optimal_Floor'].values

# 데이터 정규화
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# PyTorch 텐서로 변환
X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # (N, 1) 형태로 변환

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# 데이터 로더 설정
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# CUDA 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 모델 정의
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)  # 2개의 입력(feature)과 1개의 출력(target)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel().to(device)  # 모델을 GPU로 이동

# 손실 함수 및 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 옵티마이저

# MLflow에 실험 및 메트릭 기록
with mlflow.start_run() as run:
    num_epochs = 2700  # 적절한 에포크 수로 설정
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass 및 옵티마이저 스텝
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            # MLflow에 메트릭 기록
            mlflow.log_metric('loss', loss.item(), step=epoch + 1)

    # 최종 모델 평가
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

        avg_loss = total_loss / len(test_loader)
        print(f'Test Loss: {avg_loss:.4f}')
        # MLflow에 메트릭 기록
        mlflow.log_metric('test_loss', avg_loss)

    # 모델을 로컬에 저장
    model_path = 'linear_regression_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"모델이 로컬에 저장되었습니다: {model_path}")

    # 모델 저장
    mlflow.log_artifact(model_path)  # MLflow에 모델 파일 추가
    mlflow.pytorch.log_model(model, 'model')

    # MLflow에 파라미터 기록 (옵션)
    mlflow.log_param('learning_rate', 0.01)
    mlflow.log_param('num_epochs', num_epochs)

    # MLflow에서 모델 정보 확인
    print("MLflow Run ID:", run.info.run_id)
