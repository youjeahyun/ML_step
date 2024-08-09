import numpy as np
import pandas as pd

# 데이터 생성
np.random.seed(42)  # 난수 생성의 시드를 고정하여 재현 가능한 결과를 생성합니다.

# 100개의 임의의 온도(0~100)와 전력 소비(0~500) 생성
Temperature = np.random.uniform(0, 100, 100)  # 0부터 100까지의 균일 분포에서 100개의 임의의 온도 값을 생성합니다.
Power_Consumption = np.random.uniform(0, 500, 100)  # 0부터 500까지의 균일 분포에서 100개의 임의의 전력 소비 값을 생성합니다.

# 최적화된 층수 계산
Optimal_Floor = np.round(Temperature / 10 + Power_Consumption / 50)  # 온도를 10으로 나누고, 전력 소비를 50으로 나눈 값을 더한 후 반올림하여 최적화된 층수를 계산합니다.

# 1층부터 100층 사이의 값으로 제한
Optimal_Floor = np.clip(Optimal_Floor, 1, 100)  # 계산된 최적화된 층수 값을 1부터 100 사이로 제한합니다. (1보다 작거나 100보다 큰 값은 각각 1 또는 100으로 설정됩니다.)

# 데이터프레임으로 변환
data = pd.DataFrame({
    'Temperature': Temperature,  # 온도 데이터
    'Power_Consumption': Power_Consumption,  # 전력 소비 데이터
    'Optimal_Floor': Optimal_Floor  # 최적화된 층수 데이터
})

# CSV 파일로 저장
data.to_csv('optimized_floor_data.csv', index=False)  # 데이터프레임을 'optimized_floor_data.csv'라는 이름의 CSV 파일로 저장합니다. index=False는 행 인덱스를 파일에 포함시키지 않도록 합니다.

print("CSV 파일이 생성되었습니다: 'optimized_floor_data.csv'")  # 파일 생성 완료 메시지를 출력합니다.
