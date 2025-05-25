import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image # Thư viện Pillow để xử lý ảnh
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Cấu hình ---
MODEL_FILENAME = "Trained.pth" # Tên file mô hình đã lưu (từ file huấn luyện)
INPUT_SIZE = 28 * 28           # Phải giống với lúc huấn luyện
HIDDEN_SIZE = 64               # Phải giống với lúc huấn luyện
NUM_CLASSES = 10               # Phải giống với lúc huấn luyện
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Định nghĩa lại cấu trúc mô hình (phải giống hệt với mô hình đã huấn luyện) ---
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

# --- Hàm tiền xử lý ảnh ---
def preprocess_image(image_path):
    """
    Tải ảnh, chuyển đổi sang ảnh xám, resize về 28x28,
    chuyển sang tensor, và chuẩn hóa giống như dữ liệu MNIST.
    """
    try:
        img = Image.open(image_path).convert('L') # Mở ảnh và chuyển sang ảnh xám
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file ảnh tại '{image_path}'")
        return None
    except Exception as e:
        print(f"Lỗi khi mở ảnh '{image_path}': {e}")
        return None

    # Các phép biến đổi:
    # 1. Resize về 28x28
    # 2. Invert màu (để chữ thành trắng, nền thành đen, giả sử ảnh gốc là chữ đen nền trắng)
    # 3. Chuyển sang Tensor (giá trị pixel từ 0-1)
    # 4. Chuẩn hóa (giống như khi huấn luyện MNIST: mean=0.1307, std=0.3081)
    transform_pipeline = transforms.Compose([
        transforms.Resize((28, 28)),
        # transforms.functional.invert, # Đảo ngược màu: chữ đen -> trắng, nền trắng -> đen
        transforms.ToTensor(),        # Chuyển sang Tensor, pixel [0,1].
        transforms.Normalize((0.1307,), (0.3081,)) # Chuẩn hóa
    ])

    try:
        img_tensor = transform_pipeline(img)
    except Exception as e:
        print(f"Lỗi khi tiền xử lý ảnh '{image_path}': {e}")
        return None

    # Thêm chiều batch (mô hình mong đợi [batch_size, C, H, W])
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

# --- Hàm dự đoán ---
def predict_digit(image_path, model, device):
    model.eval() # Đặt mô hình ở chế độ đánh giá

    # Tiền xử lý ảnh
    img_tensor = preprocess_image(image_path)
    if img_tensor is None:
        return None, None, None # Thêm None cho preprocessed_tensor

    img_tensor = img_tensor.to(device)

    with torch.no_grad(): # Không tính gradient
        outputs = model(img_tensor)
        # outputs là logits, lấy argmax để có lớp dự đoán
        _, predicted_class = torch.max(outputs.data, 1)

        # Lấy xác suất (softmax) nếu muốn
        probabilities = torch.softmax(outputs, dim=1)
        predicted_prob = probabilities[0, predicted_class.item()].item()

    return predicted_class.item(), predicted_prob, img_tensor.cpu().squeeze(0)


# --- Chương trình chính ---
if __name__ == "__main__":
    # Đường dẫn đến mô hình đã lưu (trong cùng thư mục với script này)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FILENAME)

    # 1. Khởi tạo mô hình
    loaded_model = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)

    # 2. Tải các trọng số đã lưu
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy file mô hình tại '{model_path}'.")
        print("Hãy đảm bảo bạn đã chạy file huấn luyện để tạo ra file 'Trained.pth' và file đó nằm cùng thư mục với script này.")
        exit()

    try:
        # map_location=DEVICE giúp tải mô hình lên CPU nếu nó được lưu trên GPU và ngược lại
        loaded_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        loaded_model.to(DEVICE)
        print(f"Đã tải mô hình từ: {model_path}")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        exit()

    # 3. Đường dẫn đến ảnh bạn muốn dự đoán
    # Ảnh mặc định là "2.png" và được giả định nằm trong cùng thư mục với script này.
    # Bạn có thể thay đổi đường dẫn này hoặc truyền vào từ command line.
    image_to_predict = "4.png" # <<< TÊN ẢNH MẶC ĐỊNH ĐÃ THAY ĐỔI

    # Kiểm tra xem file ảnh có tồn tại không
    image_full_path = os.path.join(script_dir, image_to_predict) # Giả sử ảnh nằm cùng thư mục
    if not os.path.exists(image_full_path):
        print(f"Lỗi: File ảnh '{image_full_path}' không tồn tại.")
        exit()

    # 4. Thực hiện dự đoán
    predicted_digit, probability, preprocessed_tensor = predict_digit(image_full_path, loaded_model, DEVICE)

    if predicted_digit is not None and preprocessed_tensor is not None:
        print(f"\nẢnh: {image_full_path}")
        print(f"Mô hình dự đoán là số: {predicted_digit}")
        print(f"Với xác suất: {probability*100:.2f}%")

        # Hiển thị ảnh gốc và ảnh đã tiền xử lý
        plt.figure(figsize=(8,4)) # Tăng kích thước figure một chút

        plt.subplot(1, 2, 1)
        try:
            original_img = Image.open(image_full_path)
            plt.imshow(original_img, cmap='gray' if original_img.mode == 'L' else None)
            plt.title("Ảnh Gốc")
            plt.axis('off')
        except Exception as e_show_orig:
            print(f"Không thể hiển thị ảnh gốc: {e_show_orig}")


        plt.subplot(1, 2, 2)
        img_to_show = preprocessed_tensor.numpy() # Chuyển về NumPy array
        plt.imshow(img_to_show.squeeze(), cmap='gray')
        plt.title(f"Ảnh Tiền Xử Lý\nDự đoán: {predicted_digit}")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    elif predicted_digit is None:
        print("Không thể thực hiện dự đoán do lỗi trong quá trình xử lý ảnh.")