import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image # <<< THÊM: Để tải ảnh ngoài

# --- Bước 0: Cấu hình thiết bị (CPU hoặc GPU nếu có) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Bước 1: Định nghĩa các tham số ---
batch_size = 50
epochs = 20 # Giữ nguyên hoặc giảm để test nhanh
learning_rate = 0.0008
input_size = 28 * 28
hidden_size = 64
num_classes = 10
MODEL_SAVE_FILENAME = "Trained.pth"

# --- Bước 2: Tải và chuẩn bị dữ liệu MNIST ---
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# --- Bước 3: Xây dựng mô hình Mạng Nơ-ron Đơn giản (MLP) ---
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

model = NeuralNet(input_size, hidden_size, num_classes).to(device)
print("\nCấu trúc mô hình:")
print(model)

# --- Bước 4: Định nghĩa Hàm mất mát và Trình tối ưu hóa ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- Bước 5: Huấn luyện mô hình ---
print("\nBắt đầu huấn luyện...")
total_steps = len(train_loader)
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{total_steps}], Loss: {loss.item():.4f}')
    avg_train_loss_epoch = running_loss / total_steps
    print(f'Epoch [{epoch+1}/{epochs}] completed. Average Training Loss: {avg_train_loss_epoch:.4f}')

print("Huấn luyện hoàn tất!")

# --- LƯU MÔ HÌNH SAU KHI HUẤN LUYỆN ---
current_script_directory = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(current_script_directory, MODEL_SAVE_FILENAME)
torch.save(model.state_dict(), model_save_path)
print(f"Mô hình đã được lưu tại: {model_save_path}")

# --- Bước 6: Đánh giá mô hình trên tập kiểm tra MNIST ---
print("\nĐánh giá mô hình trên tập MNIST test...")
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    test_loss_mnist = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss_mnist += loss.item()
        _, predicted_classes = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted_classes == labels).sum().item()
    accuracy_mnist = 100 * correct / total
    avg_test_loss_mnist = test_loss_mnist / len(test_loader)
    print(f'Độ chính xác trên {total} ảnh MNIST kiểm tra: {accuracy_mnist:.2f}%')
    print(f'Mất mát trung bình trên tập MNIST kiểm tra: {avg_test_loss_mnist:.4f}')

# --- Bước 7: Dự đoán và Hiển thị một vài ví dụ từ MNIST test ---
# (Giữ nguyên phần này như code gốc của bạn)
print("\nDự đoán một vài ảnh từ tập MNIST kiểm tra:")
dataiter = iter(test_loader)
images_mnist_display, labels_mnist_display = next(dataiter)
images_mnist_for_display = images_mnist_display.cpu()
labels_mnist_for_display = labels_mnist_display.cpu()

model.eval()
with torch.no_grad():
    outputs_mnist_display = model(images_mnist_display.to(device))
    _, predicted_mnist_display = torch.max(outputs_mnist_display.data, 1)
    predicted_mnist_on_cpu = predicted_mnist_display.cpu()

plt.figure(figsize=(12, 5))
for i in range(min(20, images_mnist_for_display.shape[0])):
    plt.subplot(4, 5, i + 1)
    img_display = images_mnist_for_display[i].squeeze()
    plt.imshow(img_display, cmap = 'gray')
    plt.title(f"Pred: {predicted_mnist_on_cpu[i].item()}\nTrue: {labels_mnist_for_display[i].item()}",
              color=("green" if predicted_mnist_on_cpu[i] == labels_mnist_for_display[i] else "red"))
    plt.axis('off')
plt.tight_layout()
plt.suptitle("Ví dụ từ tập MNIST Test", fontsize=16) # Thêm tiêu đề cho figure này
plt.show()


# --- THÊM MỚI: KIỂM TRA VỚI ẢNH NGOÀI (ví dụ: 2.png) ---
print("\nKiểm tra với ảnh ngoài (ví dụ: 2.png)...")

# --- Hàm tiền xử lý ảnh ngoài (GIỐNG HỆT external_images.py) ---
def preprocess_external_image(image_path_external):
    try:
        img_pil = Image.open(image_path_external).convert('L')
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file ảnh tại '{image_path_external}'")
        return None
    except Exception as e:
        print(f"Lỗi khi mở ảnh '{image_path_external}': {e}")
        return None

    transform_external = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.functional.invert,
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    try:
        img_tensor_external = transform_external(img_pil)
    except Exception as e:
        print(f"Lỗi khi tiền xử lý ảnh '{image_path_external}': {e}")
        return None
    return img_tensor_external.unsqueeze(0) # Thêm chiều batch

# Đường dẫn đến ảnh ngoài để kiểm tra
# HÃY TẠO FILE "2.png" (chữ số 2, chữ đen nền trắng) TRONG CÙNG THƯ MỤC VỚI SCRIPT NÀY
external_image_filename = "4.png"
external_image_path = os.path.join(current_script_directory, external_image_filename)

if not os.path.exists(external_image_path):
    print(f"Không tìm thấy file '{external_image_path}'. Vui lòng tạo file ảnh này.")
else:
    # Sử dụng mô hình `model` đã được huấn luyện (không cần tải lại `Trained.pth` vì model đang ở trong bộ nhớ)
    model.eval() # Đảm bảo model ở chế độ đánh giá
    
    input_tensor_external = preprocess_external_image(external_image_path)

    if input_tensor_external is not None:
        input_tensor_external = input_tensor_external.to(device)
        with torch.no_grad():
            outputs_external = model(input_tensor_external)
            probabilities_external = torch.softmax(outputs_external, dim=1)
            confidence_external, predicted_class_external = torch.max(probabilities_external, 1)

        print(f"Dự đoán cho ảnh '{external_image_filename}': {predicted_class_external.item()}")
        print(f"Với độ tự tin: {confidence_external.item()*100:.2f}%")

        # Hiển thị ảnh gốc và ảnh đã tiền xử lý
        try:
            original_img_pil = Image.open(external_image_path)
            preprocessed_img_display = input_tensor_external.cpu().squeeze().numpy()

            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(original_img_pil, cmap='gray' if original_img_pil.mode == 'L' else None)
            plt.title(f"Ảnh Gốc: {external_image_filename}")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(preprocessed_img_display, cmap='gray')
            plt.title(f"Ảnh Tiền Xử Lý\nDự đoán: {predicted_class_external.item()}")
            plt.axis('off')
            plt.suptitle(f"Dự đoán cho ảnh ngoài: {external_image_filename}", fontsize=16)
            plt.show()
        except Exception as e_display:
            print(f"Lỗi khi hiển thị ảnh ngoài: {e_display}")
    else:
        print(f"Không thể dự đoán cho '{external_image_filename}' do lỗi tiền xử lý.")

print("\n--- Hoàn tất tất cả các bước ---")