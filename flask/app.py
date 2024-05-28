from PIL import Image
from flask import Flask, request, jsonify
import torch
from model_training import cnn, image_transform

app = Flask(__name__)

# 모델 로드
model = cnn.CNN(3, 9)  # 예제에서는 input_dim이 3 (RGB 이미지)이고 output_dim이 9로 가정
model.load_state_dict(torch.load(r'C:\Users\hyukkyo\repo\fishdex-back\flask\20240528.pth', map_location='cpu'))
model.eval()

fish_species = [
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

# 이미지 전처리 함수
def preprocess_image(image):
    transform = image_transform.get_transform()
    # Image 모듈을 사용하여 Werkzeug의 FileStorage 객체를 PIL Image로 변환
    image = Image.open(image)
    image = transform(image)
    return image.unsqueeze(0)  # 배치 차원 추가

# API 엔드포인트
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 이미지 파일 받기
        image_file = request.files['image']
        
        # 이미지 전처리
        input_tensor = preprocess_image(image_file)

        # 모델 예측
        with torch.no_grad():
            output = model(input_tensor)

        # 결과 반환
        probabilities = torch.nn.functional.softmax(output, dim=1).numpy()[0]
        predicted_class = int(torch.argmax(output))
        print(predicted_class)
        print(fish_species[predicted_class])

        result = {
            'species': fish_species[predicted_class],
            # 'probabilities': probabilities.tolist()
        }
        print(result)
        return jsonify(result)

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
