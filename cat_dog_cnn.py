"""
고양이 vs 개 이미지 분류 CNN 모델 학습 스크립트
===================================================

[전제 조건: 데이터 폴더 구조]
train/
 ├── cats/
 │     ├── cat001.jpg
 │     ├── cat002.jpg
 │     └── ...
 └── dogs/
       ├── dog001.jpg
       ├── dog002.jpg
       └── ...

- 하나의 train 폴더 안에 클래스별 하위 폴더를 두면,
  Keras가 자동으로 클래스를 인식하고 train/validation을 나눠줍니다.
- DATA_DIR 변수만 본인 환경에 맞게 수정하면 바로 실행됩니다.

[적용한 설정]
- 모든 이미지 크기 통일: IMG_SIZE (기본 160x160)
- 활성화 함수: LeakyReLU
- 가중치 초기화: He 초기화 (he_normal)
- 학습률: 0.001 (Adam 옵티마이저 기준, 현재 가장 널리 쓰이는 기본값)
- 은닉층 구성: GPU가 없는 환경을 고려해 가볍게 설계
  (Conv 블록 4개 + GlobalAveragePooling → Dense 1개), 파라미터 수 최소화
- 실시간 Loss 출력: 매 배치(batch)마다 콘솔 출력 + 실시간 갱신 그래프
- 조기 종료(Early Stopping): Loss가 더 이상 개선되지 않으면 자동 중단
- 학습 종료 후 Loss 곡선을 이미지 파일로 저장
"""

import os
import matplotlib
matplotlib.use("TkAgg")  # 실시간 그래프 창을 띄우기 위한 백엔드.
                          # 화면이 없는 서버 환경이라면 "Agg"로 바꾸고
                          # 실시간 창 표시 없이 최종 그래프 저장만 하세요.
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------------------------------------------
# 1. 기본 설정
# ----------------------------------------------------------------------
DATA_DIR = "train"          # 본인 데이터 경로로 수정하세요.
IMG_SIZE = (160, 160)       # 모든 사진을 이 크기로 통일해서 입력합니다.
BATCH_SIZE = 32
LEARNING_RATE = 1e-3        # Adam + 1e-3: 현재 가장 널리 쓰이는 기본 학습률
MAX_EPOCHS = 100            # 조기 종료가 실제 종료 시점을 결정하므로 크게 설정
VALIDATION_SPLIT = 0.2
SEED = 42

# ----------------------------------------------------------------------
# 2. 데이터셋 로드 (이미지 크기 자동 통일 + 정규화)
# ----------------------------------------------------------------------
train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",   # 고양이=0, 개=1 형태의 이진 분류
)

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

class_names = train_ds.class_names
print(f"클래스: {class_names}")

# 픽셀값 0~255 -> 0~1 정규화
normalization_layer = layers.Rescaling(1.0 / 255)

# 가벼운 데이터 증강 (과적합 방지, GPU 없이도 부담 적음)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(
    lambda x, y: (normalization_layer(data_augmentation(x)), y),
    num_parallel_calls=AUTOTUNE,
).prefetch(AUTOTUNE)

val_ds = val_ds.map(
    lambda x, y: (normalization_layer(x), y),
    num_parallel_calls=AUTOTUNE,
).prefetch(AUTOTUNE)

# ----------------------------------------------------------------------
# 3. CNN 모델 정의 (LeakyReLU + He 초기화, CPU 친화적인 경량 구조)
# ----------------------------------------------------------------------
he_init = keras.initializers.HeNormal()


def conv_block(x, filters):
    x = layers.Conv2D(
        filters, (3, 3), padding="same", kernel_initializer=he_init
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    return x


inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = conv_block(inputs, 16)   # 160x160 -> 80x80
x = conv_block(x, 32)        # 80x80  -> 40x40
x = conv_block(x, 64)        # 40x40  -> 20x20
x = conv_block(x, 128)       # 20x20  -> 10x10

x = layers.GlobalAveragePooling2D()(x)   # Flatten 대신 사용해 파라미터 수를 크게 줄임
x = layers.Dense(64, kernel_initializer=he_init)(x)
x = layers.LeakyReLU(negative_slope=0.1)(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# ----------------------------------------------------------------------
# 4. 콜백: 실시간 배치 loss 출력 + 실시간 그래프 갱신 + 조기 종료
# ----------------------------------------------------------------------
class RealtimeLossPlot(keras.callbacks.Callback):
    """매 배치마다 loss를 출력하고, 매 epoch마다 그래프를 실시간 갱신합니다."""

    def __init__(self):
        super().__init__()
        self.batch_losses = []
        self.epoch_train_losses = []
        self.epoch_val_losses = []

    def on_train_begin(self, logs=None):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7, 4))

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None:
            self.batch_losses.append(loss)
            print(f"  batch {batch:>4} - loss: {loss:.4f}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch_train_losses.append(logs.get("loss"))
        self.epoch_val_losses.append(logs.get("val_loss"))

        self.ax.clear()
        self.ax.plot(self.epoch_train_losses, label="train loss", marker="o")
        self.ax.plot(self.epoch_val_losses, label="val loss", marker="o")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("실시간 학습 Loss")
        self.ax.legend()
        self.ax.grid(alpha=0.3)
        plt.pause(0.01)

    def on_train_end(self, logs=None):
        plt.ioff()


realtime_plot_cb = RealtimeLossPlot()

# Loss가 개선되지 않으면 학습을 자동 종료 (patience 만큼 기다렸다가 중단)
early_stop_cb = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,             # 8 epoch 동안 개선 없으면 종료
    restore_best_weights=True,
    verbose=1,
)

# 학습률 자동 감소 (loss 정체 시 학습률을 낮춰 더 세밀하게 수렴하도록 보조)
reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1,
)

# ----------------------------------------------------------------------
# 5. 학습 실행
# ----------------------------------------------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=MAX_EPOCHS,
    callbacks=[realtime_plot_cb, early_stop_cb, reduce_lr_cb],
)

# ----------------------------------------------------------------------
# 6. 최종 Loss 그래프 저장
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss", marker="o")
plt.plot(history.history["val_loss"], label="Validation Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("최종 학습 Loss 곡선")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")
print("최종 loss 그래프를 loss_curve.png 로 저장했습니다.")
plt.show()

# ----------------------------------------------------------------------
# 7. 테스트 데이터 평가
# ----------------------------------------------------------------------
# 테스트 폴더도 train과 동일한 구조를 가정합니다.
# test/
#  ├── cats/
#  └── dogs/
TEST_DIR = "test"   # 본인 테스트 데이터 경로로 수정하세요.

if os.path.isdir(TEST_DIR):
    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,   # 예측 결과와 실제 라벨을 순서대로 비교하기 위해 섞지 않음
    )
    test_class_names = test_ds.class_names

    test_ds_norm = test_ds.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=AUTOTUNE,
    ).prefetch(AUTOTUNE)

    print("\n===== 테스트 데이터 평가 =====")
    test_loss, test_acc = model.evaluate(test_ds_norm, verbose=1)
    print(f"테스트 Loss: {test_loss:.4f}")
    print(f"테스트 Accuracy: {test_acc:.4f}")

    # 이미지 하나씩 예측값 vs 실제값 비교 (혼동 행렬 계산용)
    y_true, y_pred = [], []
    for images, labels in test_ds_norm:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().flatten().astype(int).tolist())
        y_pred.extend((preds.flatten() > 0.5).astype(int).tolist())

    # 혼동 행렬 (외부 라이브러리 없이 직접 계산)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    print("\n[혼동 행렬] (클래스 순서:", test_class_names, ")")
    print(f"                예측:{test_class_names[0]:<8} 예측:{test_class_names[1]:<8}")
    print(f"실제:{test_class_names[0]:<10} {tn:<15} {fp:<15}")
    print(f"실제:{test_class_names[1]:<10} {fn:<15} {tp:<15}")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"\nPrecision({test_class_names[1]}): {precision:.4f}")
    print(f"Recall({test_class_names[1]}):    {recall:.4f}")
    print(f"F1-score:                {f1:.4f}")

    # 테스트 이미지 몇 장을 실제/예측 라벨과 함께 시각화하여 저장
    plt.figure(figsize=(10, 10))
    shown = 0
    for images, labels in test_ds.take(1):
        for i in range(min(9, images.shape[0])):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            pred_prob = model.predict(
                normalization_layer(tf.expand_dims(images[i], 0)), verbose=0
            )[0][0]
            pred_label = test_class_names[int(pred_prob > 0.5)]
            true_label = test_class_names[int(labels[i].numpy()[0])]
            color = "green" if pred_label == true_label else "red"
            plt.title(f"실제:{true_label} / 예측:{pred_label}", color=color)
            plt.axis("off")
            shown += 1
    plt.tight_layout()
    plt.savefig("test_predictions_sample.png", dpi=150, bbox_inches="tight")
    print("\n샘플 예측 결과 이미지를 test_predictions_sample.png 로 저장했습니다.")
    plt.show()
else:
    print(f"\n'{TEST_DIR}' 폴더가 없어 테스트 평가를 건너뜁니다. "
          f"테스트 데이터를 준비한 뒤 TEST_DIR 경로를 확인해주세요.")

# ----------------------------------------------------------------------
# 8. 모델 및 가중치 저장
# ----------------------------------------------------------------------
model.save("cat_dog_model.keras")
print("\n모델 전체(구조+가중치)를 cat_dog_model.keras 로 저장했습니다.")

model.save_weights("cat_dog_model.weights.h5")
print("가중치 값만 별도로 cat_dog_model.weights.h5 로 저장했습니다.")
print("(나중에 동일한 모델 구조를 만든 뒤 model.load_weights('cat_dog_model.weights.h5') 로 불러올 수 있습니다.)")
