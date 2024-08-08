import numpy as np
from PIL import Image
import os

def load_raw_image(file_path, width, height, num_images, dtype=np.uint16, endian='little'):
    """
    RAW 이미지 파일을 읽어와서 numpy 배열로 변환합니다.

    Parameters:
        file_path (str): RAW 이미지 파일 경로.
        width (int): 이미지의 너비.
        height (int): 이미지의 높이.
        num_images (int): 이미지의 개수.
        dtype (np.dtype): 이미지 데이터 타입 (기본값: np.uint16).
        endian (str): 바이트 순서 (기본값: 'little').

    Returns:
        numpy.ndarray: RAW 이미지 데이터를 담고 있는 3D numpy 배열.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # RAW 파일에서 바이트 데이터를 읽습니다.
    with open(file_path, 'rb') as f:
        data = f.read()

    # 데이터 타입에 따른 픽셀당 바이트 수를 계산합니다.
    bytes_per_pixel = np.dtype(dtype).itemsize  # 예: 2 바이트 (16비트 unsigned)

    # 전체 픽셀 수를 계산합니다.
    total_pixels = width * height * num_images

    # 예상 데이터 크기를 계산합니다.
    expected_size = total_pixels * bytes_per_pixel

    # 실제 데이터 크기를 확인하고, 필요시 데이터를 잘라냅니다.
    actual_size = len(data)
    if actual_size < expected_size:
        print(f"Warning: File is smaller than expected. Expected: {expected_size}, Got: {actual_size}")
    elif actual_size > expected_size:
        print(f"Warning: File is larger than expected. Expected: {expected_size}, Got: {actual_size}")
        data = data[:expected_size]

    # 데이터를 numpy 배열로 변환합니다.
    image_array = np.frombuffer(data, dtype=dtype)

    # 배열의 크기를 예상 픽셀 수와 비교하여 확인합니다.
    num_expected_pixels = width * height * num_images
    if image_array.size != num_expected_pixels:
        raise ValueError(f"Image size mismatch: expected {num_expected_pixels} pixels but got {image_array.size}")

    # numpy 배열을 (num_images, height, width) 형태로 변환합니다.
    image_array = image_array.reshape((num_images, height, width))
    return image_array

def apply_threshold(image_array, threshold_value, min, max):
    """
    임계값을 기준으로 픽셀 값을 이진git config --list화합니다.

    Parameters:
        image_array (numpy.ndarray): 16-bit 이미지 데이터 배열.
        threshold_value (int): 픽셀 값에 적용할 임계값.

    Returns:
        numpy.ndarray: 임계값 처리된 이미지 배열 (16-bit).
    """
    # 픽셀 값이 임계값보다 크면 65535 (흰색), 그렇지 않으면 0 (검은색)으로 설정합니다.
    thresholded_image = np.where(image_array > threshold_value, max, min)
    return thresholded_image

def save_image(image_array, file_path, dtype=np.uint16):
    """
    numpy 배열을 이미지 파일로 저장합니다.

    Parameters:
        image_array (numpy.ndarray): 저장할 이미지 데이터 배열.
        file_path (str): 저장할 파일 경로.
        dtype (np.dtype): 이미지 데이터 타입 (기본값: np.uint16).
    """
    # numpy 배열을 PIL 이미지 객체로 변환합니다.
    image = Image.fromarray(image_array.astype(dtype))
    # 이미지를 파일로 저장합니다.
    image.save(file_path)

def main():
    """
    전체 이미지 처리 과정을 실행합니다.
    """
    file_path = '0008.raw'  # RAW 이미지 파일 경로
    width = 1512  # 이미지의 너비
    height = 1512  # 이미지의 높이
    num_images = 1  # 이미지의 개수
    threshold_value = 15000  # 임계값

    # RAW 이미지 불러오기
    try:
        raw_images = load_raw_image(file_path, width, height, num_images)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return  # 오류 발생 시 프로그램 종료

    # 첫 번째 슬라이스를 가져옵니다.
    first_image = raw_images[0]

    # 원본 데이터의 최소값과 최대값을 출력하여 이미지의 범위를 확인합니다.
    print(f"Original Image Min: {np.min(first_image)}, Max: {np.max(first_image)}")

    # 임계값 처리 적용
    thresholded_image = apply_threshold(first_image, threshold_value, np.min(first_image), np.max(first_image))

    # 결과를 16-bit TIFF 이미지로 저장
    save_image(thresholded_image, 'thresholded_image_16bit.tiff')

    # 원본 이미지 (16-bit TIFF) 저장 (디버깅 용)
    save_image(first_image, 'original_image_16bit.tiff')
if __name__ == "__main__":
    main()

