import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from model_training.image_transform import get_transform
from cnn import CNN
import torch.nn as nn

def train_model():
    # 데이터셋 경로 및 변환 설정
    data_dir = r'C:\Users\hyukkyo\ITE4055\server\dataset\a-large-scale-fish-dataset\Fish_Dataset\Fish_Dataset'  # 데이터셋 경로로 변경
    BATCH_SIZE = 32  # 배치 크기를 줄임. 32로 한번만 더해봐도 될듯

    transform = get_transform()

    # 데이터 로딩
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # GPU 사용 가능 여부 확인 및 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # input_dim = dataset[0][0].shape[0] # Number of channels in the image (3)
    # output_dim = len(classes) # Number of classes (9)

    model = CNN(3, 9).to(device)

    # 손실 함수 및 최적화 함수 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # 모델 훈련
    num_epochs = 10

    for epoch in range(num_epochs):
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
    torch.save(trained_model.state_dict(), r'C:\Users\hyukkyo\ITE4055\server\fish_classifier.pth')