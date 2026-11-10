"""
고양이 vs 개 이미지 분류 CNN 모델 학습 스크립트 (PyTorch 버전)
================================================================
※ TensorFlow는 아직 Python 3.14를 지원하지 않아서(2026년 9월 기준),
   Python 3.14를 정식 지원하는 PyTorch(2.10+)로 작성했습니다.

[전제 조건: 데이터 폴더 구조] - torchvision ImageFolder 규칙을 따릅니다.
train/
 ├── cats/  (cat001.jpg, cat002.jpg, ...)
 └── dogs/  (dog001.jpg, dog002.jpg, ...)
test/
 ├── cats/
 └── dogs/

- 폴더 이름이 알파벳순으로 클래스 인덱스가 됩니다: cats=0, dogs=1
- DATA_DIR / TEST_DIR 변수만 본인 환경에 맞게 수정하면 바로 실행됩니다.

[설치]
pip install torch torchvision matplotlib scikit-learn

[적용한 설정]
- 모든 이미지 크기 통일: IMG_SIZE (기본 160x160)
- 활성화 함수: LeakyReLU
- 가중치 초기화: He 초기화 (kaiming_normal_, leaky_relu 기준)
- 학습률: 0.001 (Adam 옵티마이저 기준, 현재 가장 널리 쓰이는 기본값)
- 은닉층 구성: GPU가 없는 환경을 고려해 가볍게 설계
  (Conv 블록 4개 + GlobalAveragePooling), 파라미터 수 최소화
- 실시간 Loss 출력: 매 배치(batch)마다 콘솔 출력 + 실시간 갱신 그래프
- 조기 종료(Early Stopping): validation loss가 더 이상 개선되지 않으면 자동 중단
- 학습 종료 후 Loss 곡선을 이미지 파일로 저장
- 테스트 데이터 평가: accuracy / 혼동행렬 / precision / recall / f1
- 가중치 저장: state_dict를 별도 파일로 저장
"""

import os
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("TkAgg")  # 실시간 그래프 창을 띄우기 위한 백엔드.
                          # 화면이 없는 서버 환경이라면 "Agg"로 바꾸세요.
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. 기본 설정
# ----------------------------------------------------------------------
DATA_DIR = r"AI 응용\cat_dog_dataset\train"          # 본인 학습 데이터 경로로 수정하세요.
TEST_DIR = r"AI 응용\cat_dog_dataset\test"           # 본인 테스트 데이터 경로로 수정하세요.
IMG_SIZE = 160               # 모든 사진을 이 크기(정사각형)로 통일해서 입력합니다.
BATCH_SIZE = 32
LEARNING_RATE = 1e-3         # Adam + 1e-3: 현재 가장 널리 쓰이는 기본 학습률
MAX_EPOCHS = 100             # 조기 종료가 실제 종료 시점을 결정하므로 크게 설정
VALIDATION_SPLIT = 0.2
EARLY_STOP_PATIENCE = 8      # 8 epoch 동안 val loss 개선 없으면 종료
SEED = 42

torch.manual_seed(SEED)

# CPU 환경 기본. GPU가 있으면 자동으로 사용합니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

# ----------------------------------------------------------------------
# 2. 데이터셋 로드 (이미지 크기 자동 통일 + 정규화 + 가벼운 증강)
# ----------------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),   # 모든 이미지 크기 통일
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),                     # 0~255 -> 0~1 자동 정규화
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

full_train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
class_names = full_train_dataset.classes
print(f"클래스: {class_names}  (0={class_names[0]}, 1={class_names[1]})")

val_size = int(len(full_train_dataset) * VALIDATION_SPLIT)
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED),
)
# validation 데이터에는 증강을 적용하지 않도록 transform을 별도로 지정
val_dataset.dataset = datasets.ImageFolder(DATA_DIR, transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------------------------------------------------
# 3. CNN 모델 정의 (LeakyReLU + He 초기화, CPU 친화적인 경량 구조)
# ----------------------------------------------------------------------
class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(3, 16),     # 160x160 -> 80x80
            conv_block(16, 32),    # 80x80  -> 40x40
            conv_block(32, 64),    # 40x40  -> 20x20
            conv_block(64, 128),   # 20x20  -> 10x10
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Flatten 대신 -> 파라미터 수 절감
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),   # 로짓(logit) 출력. sigmoid는 손실함수에서 처리.
        )

        self._init_weights()

    def _init_weights(self):
        # He 초기화 (LeakyReLU에 맞춰 nonlinearity='leaky_relu' 지정)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x  # 로짓 반환 (sigmoid 미적용)


model = CatDogCNN().to(device)
print(model)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"학습 가능한 파라미터 수: {n_params:,}")

criterion = nn.BCEWithLogitsLoss()  # sigmoid + BCE를 안정적으로 한번에 처리
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
)

# ----------------------------------------------------------------------
# 4. 실시간 loss 시각화 준비
# ----------------------------------------------------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(7, 4))
train_loss_history = []
val_loss_history = []


def update_live_plot():
    ax.clear()
    ax.plot(train_loss_history, label="train loss", marker="o")
    ax.plot(val_loss_history, label="val loss", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("실시간 학습 Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.pause(0.01)


# ----------------------------------------------------------------------
# 5. 학습 루프 (배치마다 loss 출력 + 조기 종료)
# ----------------------------------------------------------------------
best_val_loss = float("inf")
best_model_state = copy.deepcopy(model.state_dict())
patience_counter = 0

for epoch in range(1, MAX_EPOCHS + 1):
    # ----- Train -----
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        print(f"  epoch {epoch:>3} - batch {batch_idx:>4} - loss: {loss.item():.4f}")

    train_loss = running_loss / len(train_loader.dataset)

    # ----- Validation -----
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
    val_loss = val_running_loss / len(val_loader.dataset)

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    update_live_plot()

    print(f"[Epoch {epoch}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    scheduler.step(val_loss)

    # ----- 조기 종료 판단 -----
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  -> val_loss 개선 없음 ({patience_counter}/{EARLY_STOP_PATIENCE})")
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"조기 종료: {EARLY_STOP_PATIENCE} epoch 동안 val_loss 개선 없음.")
            break

# 가장 좋았던 시점의 가중치로 복원
model.load_state_dict(best_model_state)
plt.ioff()

# ----------------------------------------------------------------------
# 6. 최종 Loss 그래프 저장
# ----------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(train_loss_history, label="Train Loss", marker="o")
plt.plot(val_loss_history, label="Validation Loss", marker="o")
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
if os.path.isdir(TEST_DIR):
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_class_names = test_dataset.classes

    model.eval()
    test_running_loss = 0.0
    y_true, y_pred = [], []
    sample_images_shown = []

    with torch.no_grad():
        for images, labels in test_loader:
            images_dev = images.to(device)
            labels_dev = labels.float().unsqueeze(1).to(device)

            outputs = model(images_dev)
            loss = criterion(outputs, labels_dev)
            test_running_loss += loss.item() * images_dev.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)

            y_true.extend(labels.numpy().astype(int).tolist())
            y_pred.extend(preds.tolist())

            if len(sample_images_shown) < 9:
                for i in range(min(9 - len(sample_images_shown), images.size(0))):
                    sample_images_shown.append(
                        (images[i], int(labels[i].item()), int(preds[i]))
                    )

    test_loss = test_running_loss / len(test_loader.dataset)
    test_acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    print("\n===== 테스트 데이터 평가 =====")
    print(f"테스트 Loss: {test_loss:.4f}")
    print(f"테스트 Accuracy: {test_acc:.4f}")

    # 혼동 행렬 (외부 라이브러리 없이 직접 계산)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    print(f"\n[혼동 행렬] (클래스 순서: {test_class_names})")
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
    for i, (img_tensor, true_idx, pred_idx) in enumerate(sample_images_shown):
        ax_i = plt.subplot(3, 3, i + 1)
        img_np = img_tensor.permute(1, 2, 0).numpy()  # C,H,W -> H,W,C
        plt.imshow(img_np)
        true_label = test_class_names[true_idx]
        pred_label = test_class_names[pred_idx]
        color = "green" if true_label == pred_label else "red"
        plt.title(f"실제:{true_label} / 예측:{pred_label}", color=color)
        plt.axis("off")
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
torch.save(model.state_dict(), "cat_dog_model_weights.pth")
print("\n가중치 값을 cat_dog_model_weights.pth 로 저장했습니다.")
print("(나중에 CatDogCNN() 을 만든 뒤 model.load_state_dict(torch.load("
      "'cat_dog_model_weights.pth')) 로 불러올 수 있습니다.)")

# 구조까지 포함한 전체 체크포인트도 함께 저장 (선택 사항)
torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": class_names,
    "img_size": IMG_SIZE,
}, "cat_dog_model_full.pth")
print("모델 구조 정보를 포함한 체크포인트를 cat_dog_model_full.pth 로 저장했습니다.")
