Chapter 3. 신경망
====================

## 신경망
입력층 - 은닉층 - 출력층 구조
### h(x) - Activation Function - 뉴런의 활성화 결정

1. Step Function
Threshold 기준으로 0, 1 이분법 구조

2. Sigmoid Function
h(x) = 1/( 1 + e ^ -x)
0~1 사이의 값이 나온다.

#### 비 선형 함수를 사용하는 이유는 은닉층 표현 위해

3. ReLU Function
h(x) = 1 if x > 0
     = 0 else 

4. Softmax Function
A = [ ... ]
h(x) = e ^ A[k] / Sum(e ^ A)
+ Overflow 문제
h(x) = C * e ^ A[k] / C * Sum(e ^ A) 
     = e ^ (A[k] + log C) / Sum(e ^ (A + log C))
     = e ^ (A[k] + C') / Sum(e ^ (A + C'))
C' = -1 * max(입력 신호)
Softmax는 다 분류 문제에서 확률이 된다.

[activation.py](activation.py)

## 신경망의 구현
x = [ ... ] 1차원 벡터

w = [ ... ] n차원 매트릭스 
    [ ... ] 입력층 길이 x 출력층 길이

output = (x * w) + bias

## Batch
Batch Size 만큼 처리한다.

[neural.py](neural.py)

> MNIST Dataset은 다음 장에서 한다.