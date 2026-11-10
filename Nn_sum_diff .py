# -*- coding: utf-8 -*-
"""
3자리 정수 두 개(a, b)를 입력받아
  - 합(a + b)
  - 차(a - b)
를 예측하도록 학습하는 신경망 프로그램 (numpy로 직접 구현)

신경망 구조
  입력층   : 2개 노드 (a, b)
  은닉층1  : 20개 노드 (ReLU)
  은닉층2  : 20개 노드 (ReLU)
  출력층   : 2개 노드 (합, 차) - 선형(회귀)
"""

import numpy as np

np.random.seed(42)

# ----------------------------
# 1. 데이터 정규화 관련 상수
# ----------------------------
# 3자리 정수 범위: 100 ~ 999
IN_MIN, IN_MAX = 100, 999
SUM_MIN, SUM_MAX = 200, 1998      # 가능한 합의 범위
DIFF_MIN, DIFF_MAX = -899, 899    # 가능한 차의 범위


def normalize_input(a, b):
    """입력값을 0~1 사이로 정규화"""
    x1 = (a - IN_MIN) / (IN_MAX - IN_MIN)
    x2 = (b - IN_MIN) / (IN_MAX - IN_MIN)
    return np.array([x1, x2])


def normalize_output(s, d):
    """정답(합, 차)을 -1~1 사이로 정규화"""
    y1 = 2 * (s - SUM_MIN) / (SUM_MAX - SUM_MIN) - 1
    y2 = 2 * (d - DIFF_MIN) / (DIFF_MAX - DIFF_MIN) - 1
    return np.array([y1, y2])


def denormalize_output(y):
    """정규화된 출력을 실제 값(합, 차)으로 복원"""
    y1, y2 = y[0], y[1]
    s = (y1 + 1) / 2 * (SUM_MAX - SUM_MIN) + SUM_MIN
    d = (y2 + 1) / 2 * (DIFF_MAX - DIFF_MIN) + DIFF_MIN
    return s, d


# ----------------------------
# 2. 신경망 정의 (numpy로 직접 구현)
# ----------------------------
class NeuralNetwork:
    def __init__(self, input_size=2, hidden_size=20, output_size=2, lr=0.01):
        self.lr = lr
        # He 초기화
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, hidden_size))
        self.W3 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros((1, output_size))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_deriv(x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)

        self.z3 = self.a2 @ self.W3 + self.b3  # 출력층은 선형(회귀 문제)
        return self.z3

    def backward(self, X, y_true, y_pred):
        m = X.shape[0]

        # 출력층 (MSE loss의 미분)
        dz3 = (y_pred - y_true) * 2 / m
        dW3 = self.a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        # 은닉층 2
        da2 = dz3 @ self.W3.T
        dz2 = da2 * self.relu_deriv(self.z2)
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # 은닉층 1
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_deriv(self.z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # 파라미터 업데이트 (경사하강법)
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train_step(self, X, y):
        y_pred = self.forward(X)
        loss = np.mean((y_pred - y) ** 2)
        self.backward(X, y, y_pred)
        return loss

    def predict(self, X):
        return self.forward(X)


# ----------------------------
# 3. 학습 데이터 생성
# ----------------------------
def make_batch(batch_size):
    a = np.random.randint(IN_MIN, IN_MAX + 1, size=batch_size)
    b = np.random.randint(IN_MIN, IN_MAX + 1, size=batch_size)

    X = np.stack([(a - IN_MIN) / (IN_MAX - IN_MIN),
                  (b - IN_MIN) / (IN_MAX - IN_MIN)], axis=1)

    s = a + b
    d = a - b
    Y = np.stack([2 * (s - SUM_MIN) / (SUM_MAX - SUM_MIN) - 1,
                  2 * (d - DIFF_MIN) / (DIFF_MAX - DIFF_MIN) - 1], axis=1)

    return X, Y


# ----------------------------
# 4. 학습 실행
# ----------------------------
def train(epochs=500000, batch_size=64, lr=0.01):
    net = NeuralNetwork(input_size=2, hidden_size=20, output_size=2, lr=lr)

    print("학습을 시작합니다...")
    for epoch in range(1, epochs + 1):
        X, Y = make_batch(batch_size)
        loss = net.train_step(X, Y)

        if epoch % 500 == 0 or epoch == 1:
            print(f"  epoch {epoch:5d}  loss = {loss:.6f}")

    print("학습이 완료되었습니다.\n")
    return net


def evaluate(net, n_samples=5):
    """학습된 모델의 정확도를 몇 개 샘플로 확인"""
    print("=== 검증 (테스트 샘플) ===")
    correct_close = 0
    for _ in range(n_samples):
        a = np.random.randint(IN_MIN, IN_MAX + 1)
        b = np.random.randint(IN_MIN, IN_MAX + 1)
        x = normalize_input(a, b).reshape(1, -1)
        y_pred = net.predict(x)[0]
        pred_sum, pred_diff = denormalize_output(y_pred)
        real_sum, real_diff = a + b, a - b
        print(f"  a={a}, b={b} | 실제 합={real_sum}, 예측 합={pred_sum:.1f} "
              f"| 실제 차={real_diff}, 예측 차={pred_diff:.1f}")
    print()


# ----------------------------
# 5. 사용자 입력을 받아 예측
# ----------------------------
def get_3digit_int(prompt):
    while True:
        try:
            v = int(input(prompt))
            if 100 <= v <= 999:
                return v
            else:
                print("  -> 3자리 정수(100~999)를 입력해 주세요.")
        except ValueError:
            print("  -> 숫자를 입력해 주세요.")


def main():
    net = train(epochs=500000, batch_size=64, lr=0.01)
    evaluate(net, n_samples=5)

    print("이제 직접 3자리 정수 두 개를 입력해서 예측해 보세요. (종료: 'q' 입력)")
    while True:
        raw = input("\n계속 하시겠습니까? (엔터=계속, q=종료): ")
        if raw.strip().lower() == 'q':
            print("프로그램을 종료합니다.")
            break

        a = get_3digit_int("첫 번째 3자리 정수를 입력하세요: ")
        b = get_3digit_int("두 번째 3자리 정수를 입력하세요: ")

        x = normalize_input(a, b).reshape(1, -1)
        y_pred = net.predict(x)[0]
        pred_sum, pred_diff = denormalize_output(y_pred)

        real_sum = a + b
        real_diff = a - b

        print(f"\n입력값: a={a}, b={b}")
        print(f"  실제 합(a+b)   = {real_sum}   /  신경망 예측 합   = {pred_sum:.2f}")
        print(f"  실제 차(a-b)   = {real_diff}   /  신경망 예측 차   = {pred_diff:.2f}")


if __name__ == "__main__":
    main()