from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

app = Flask(__name__)

# 이미지 변환 정의 (모델 학습 시 사용한 것과 동일)
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

transform = get_transform()

# 모델 정의 및 로드
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
model.load_state_dict(torch.load(r'C:\Users\hyukkyo\repo\fishdex-back\flask\fish_classifier.pth'))
model.eval()

# 클래스 이름 정의 (예: 각 클래스의 이름)
classes = [
    "Black Sea Sprat",
    "Gilt-Head Bream",
    "Hourse Mackerel",
    "Red Mullet",
    "Red Sea Bream",
    "Sea Bass",
    "Shrimp",
    "Striped Red Mullet",
    "Trout"
]

# # 이미지 전처리 함수
# def preprocess_image(image):
#     transform = image_transform.get_transform()
#     # Image 모듈을 사용하여 Werkzeug의 FileStorage 객체를 PIL Image로 변환
#     image = Image.open(image)
#     image = transform(image)
#     return image.unsqueeze(0)  # 배치 차원 추가

# API 엔드포인트
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # 이미지 파일 받기
#         image_file = request.files['image']
        
#         # 이미지 전처리
#         input_tensor = preprocess_image(image_file)

#         # 모델 예측
#         with torch.no_grad():
#             output = model(input_tensor)

#         # 결과 반환
#         probabilities = torch.nn.functional.softmax(output, dim=1).numpy()[0]
#         predicted_class = int(torch.argmax(output))
#         print(predicted_class)
#         print(fish_species[predicted_class])

#         result = {
#             'species': fish_species[predicted_class],
#             # 'probabilities': probabilities.tolist()
#         }
#         print(result)
#         return jsonify(result)

#     except Exception as e:
#         print(e)
#         return jsonify({'error': str(e)})
    
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        
        predicted_class = classes[predicted.item()]

    # return jsonify({'class': predicted_class})
    return jsonify({'species': predicted_class})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
