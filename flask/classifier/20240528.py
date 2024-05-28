import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn as nn

# 데이터 변환 정의
def get_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # 데이터 증강
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # 사전 훈련된 모델의 정규화 값 사용
    ])
    return transform

# 데이터 로더 설정
data_dir = r'C:\Users\hyukkyo\repo\fishdex-back\flask\classifier\a_large_scale_fish_dataset\Fish_Dataset\Fish_Dataset'
BATCH_SIZE = 32

transform = get_transform()
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 사전 훈련된 모델 로드 및 수정
class FishClassifier(nn.Module):
    def __init__(self, out_dim):
        super(FishClassifier, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, out_dim)

    def forward(self, x):
        return self.base_model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FishClassifier(out_dim=9).to(device)

# 고정 레이어 설정
for param in model.base_model.parameters():
    param.requires_grad = False

# 마지막 레이어만 학습 가능하게 설정
model.base_model.fc.weight.requires_grad = True
model.base_model.fc.bias.requires_grad = True

# 손실 함수 및 최적화 함수 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)

# 학습 및 평가 루프
def train_model():
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')

    return model

if __name__ == "__main__":
    trained_model = train_model()
    torch.save(trained_model.state_dict(), r'C:\Users\hyukkyo\repo\fishdex-back\flask\fish_classifier.pth')
