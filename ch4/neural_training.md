Chapter 4. 신경망 학습
=================

+ 훈련 데이터, 시험 데이터로 나눠 최적의 모델을 찾는다.
+ Overfitting - 훈련 데이터에 과적합되어 범용적 성능이 떨어질 때

## 손실 함수 (Cost , Loss Function)
Cost를 최소화 하자
손실 함수의 값을 작게 하는 매개 변수의 미분을 구한다.
미분 값에 따라 변화.

1. 평균 제곱 오차 Mean squared error, MSE
E = Sum((Yk-Tk)^2) / 2

2. 교차 엔트로치 Cross entropy error, CSE
E = - ln Y     if   d=1
  = - ln (1-Y) elif d=0
  = - Sum(T * log Y)

[cost.py](cost.py)

## 미니 배치
E = - 1/N * Sum( Sum( tnk * log(ynk)))
훈련 데이터중 일부만 골라 학습을 수행한다.

## 수치 미분
+ 미분 : 변화량
근사로 구한 접선.
이때 접선을 근사로 구하면 rounding error 와 float의 표현 한계 때문에 제대로 나타나지 않는다.

+ 중심 차분 : 수치 미분의 오차를 줄이기 위해 사용

+ 편미분 : 변수가 여럿인 함수에 대한 미분

[diff.py](diff.py)

