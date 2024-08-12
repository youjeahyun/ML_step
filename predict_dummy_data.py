import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# 모델 정의
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1)  # 2개의 입력(feature)과 1개의 출력(target)

    def forward(self, x):
        return self.linear(x)


# 데이터 전처리 및 예측
def predict(model, data_point, scaler, device):
    model.eval()
    with torch.no_grad():
        # 데이터 전처리
        data_point = np.array(data_point).reshape(1, -1)  # (1, 2) 형태로 변환
        data_point_normalized = scaler.transform(data_point)  # 데이터 정규화
        data_tensor = torch.tensor(data_point_normalized, dtype=torch.float32).to(device)

        # 예측
        output = model(data_tensor)
        return output.item()


# 데이터 로드 및 전처리
data = pd.read_csv('optimized_floor_data.csv')

# 특성과 타겟 변수 분리
X = data[['Temperature', 'Power_Consumption']].values
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LinearRegressionModel().to(device)
model.load_state_dict(torch.load('linear_regression_model.pth', map_location=device))
model.to(device)

# 예측할 데이터 포인트
data_point = (280, 276)
predicted_floor = predict(model, data_point, scaler, device)
print(f"예측된 floor 값: {predicted_floor:.1f}")
