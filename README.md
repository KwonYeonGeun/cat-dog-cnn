# 고양이·개 이미지 분류

AI 응용 과제입니다.

- `cat_dog_cnn.py` : TensorFlow/Keras
- `cat_dog_cnn_pytorch.py` : PyTorch
- 입력 크기 160, 학습 데이터에만 좌우 반전·회전·확대
- 검증 손실이 개선되지 않으면 학습을 멈추고 가장 나은 가중치를 되돌림

학습 이미지와 labelImg 도구는 넣지 않았습니다. Keras 스크립트는 `train/cats`, `train/dogs` 폴더가 있어야 실행됩니다.
