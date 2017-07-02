Chapter 4. 신경망 학습
=================

+ 훈련 데이터, 시험 데이터로 나눠 최적의 모델을 찾는다.
+ Overfitting - 훈련 데이터에 과적합되어 범용적 성능이 떨어질 때

## 손실 함수 (Cost , Loss Function)
Cost를 최소화 하자

1. 평균 제곱 오차 Mean squared error, MSE
E = Sum((Yk-Tk)^2) / 2

2. 교차 엔트로치 Cross entropy error, CSE
E = - ln Y     if   d=1
  = - ln (1-Y) elif d=0
  = - Sum(T * log Y)