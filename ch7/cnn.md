Chapter 7. 합성곱 신경망CNN
===========================

Convolutional Neural Network는 이미지 인식과 음성 인식등에서 사용

Affine - Activation 에서 Conv - Activation - Pooling 계층으로 변경
출력계층에선 기존 DNN 계층 방식 그대로 사용 가능.

지금까지 완전 연결 신경망에서는 데이터의 형상Dimension이 무시된다.
ex) MNIST 사용시 height * weight * channel로 1차원 데이터로 변경하여 입력

+ 특징 맵Feature map : 합성 곱 계층의 입출력 데이터

## 합성 곱 연산

이미지 처리에서 말하는 필터 연산.

1. 필터 연산

필터(커널)의 윈도우를 일정 간격이동해 가며 마스킹

2. 패딩Padding

외곽에 패딩을 채워둔다. 0, 주변 값, 풀링 등 패딩을 크게 하면 출력 크기가 커진다.

3. 스트라이드

이동하는 위치의 간격을 말한다. 스트라이드를 키우면 출력크기가 작아진다.

+ 출력 크기는 패딩,스트라이드와 관계

필터는 C * FH * FW 형태를 여러개 사용한다.<br>
즉 4차원 필터를 사용하여 3차원 출력데이터를 얻는다.

+ C * H * W Conv FN * C * FH * FW => FN * OH * OW

## 풀링 계층

공간의 크기를 줄인다.<br>
따라서 학습 X, 채널 수 변동 X, 데이터 변화에 따른 영향 축소.

Max Pool, Average Pool, ...

[conv.py](conv.py)

## CNN 구현하기

Convoultion과 Pooling 계층은 이미지를 행렬로 전개하여 구현한다.

Conv1 - ReLU - Pool - Affine - ReLU - Affine - SoftmaxWithLoss

를 적용 시킨다

[network.py](network.py)

## CNN 시각화

각각의 계층 필터들을 시각화 해본다면

초기 필터는 에지, 블롭 등의 원시적인 정보를 추출하고
뒤로 갈수록 점점 고급 시각 정보가 인식된다.

대표적인 CNN으로

1. LeNet<br>
2. AlexNet<br>
