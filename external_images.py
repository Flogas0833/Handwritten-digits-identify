import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

MODEL_FILENAME = "Trained.pth"
INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 64
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('L')
    except FileNotFoundError:
        return None
    except Exception:
        return None

    transform_pipeline = transforms.Compose([
        transforms.Resize((28, 28)),
        #transforms.functional.invert,
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        img_tensor = transform_pipeline(img)
    except Exception:
        return None

    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def predict_digit(image_path, model, device):
    model.eval()
    img_tensor = preprocess_image(image_path)
    if img_tensor is None:
        return None, None, None

    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted_class = torch.max(outputs.data, 1)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_prob = probabilities[0, predicted_class.item()].item()

    return predicted_class.item(), predicted_prob, img_tensor.cpu().squeeze(0)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FILENAME)

    loaded_model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)

    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy file mô hình tại '{model_path}'.")
        exit()

    try:
        loaded_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        loaded_model.to(DEVICE)
    except Exception:
        print("Lỗi khi tải mô hình đã huấn luyện.")
        exit()

    image_to_predict = "4.png"
    image_full_path = os.path.join(script_dir, image_to_predict)
    if not os.path.exists(image_full_path):
        print(f"Lỗi: File ảnh '{image_full_path}' không tồn tại.")
        exit()

    predicted_digit, probability, preprocessed_tensor = predict_digit(image_full_path, loaded_model, DEVICE)

    if predicted_digit is not None and preprocessed_tensor is not None:
        print(f"\nẢnh: {image_full_path}")
        print(f"Mô hình dự đoán là số: {predicted_digit}")
        print(f"Với xác suất: {probability*100:.2f}%")

        plt.figure(figsize=(8,4))

        plt.subplot(1, 2, 1)
        try:
            original_img = Image.open(image_full_path)
            plt.imshow(original_img, cmap='gray' if original_img.mode == 'L' else None)
            plt.title("Ảnh Gốc")
            plt.axis('off')
        except Exception:
            pass

        plt.subplot(1, 2, 2)
        img_to_show = preprocessed_tensor.numpy()
        plt.imshow(img_to_show.squeeze(), cmap='gray')
        plt.title(f"Ảnh Tiền Xử Lý\nDự đoán: {predicted_digit}")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    elif predicted_digit is None:
        print("Không thể thực hiện dự đoán trên ảnh cung cấp.")