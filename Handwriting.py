import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from multiprocessing import freeze_support # Thêm dòng này

# --- Bước 0: Cấu hình thiết bị (CPU hoặc GPU nếu có) ---
# (Có thể để ở đây hoặc trong if __name__ == '__main__')
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Đang sử dụng GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU không khả dụng, đang sử dụng CPU.")

# --- Định nghĩa lớp mô hình và các hàm có thể để ở ngoài ---
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

def imshow(img, title=""): # Hàm này cũng có thể để ngoài
    img = img.cpu()
    img = img * 0.3081 + 0.1307
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)).squeeze(), cmap='gray')
    plt.title(title)


# --- Phần code chính thực thi ---
if __name__ == '__main__':
    # Thêm dòng này, đặc biệt quan trọng cho Windows khi tạo executable
    # Dù không tạo executable, nó vẫn là good practice cho multiprocessing trên Windows
    freeze_support()

    # --- Bước 1: Định nghĩa các tham số ---
    batch_size = 100
    epochs = 10
    learning_rate = 0.001
    input_size = 28 * 28
    hidden_size = 64
    num_classes = 10
    MODEL_SAVE_FILENAME = "Trained_GPU_Aug.pth"

    # --- Bước 2: Tải và chuẩn bị dữ liệu MNIST ---
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    # Kiểm tra device một lần nữa bên trong if __name__ == '__main__' để num_workers được set đúng
    current_device_type = device.type
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2 if current_device_type == 'cuda' else 0,
                              pin_memory=True if current_device_type == 'cuda' else False)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=2 if current_device_type == 'cuda' else 0,
                             pin_memory=True if current_device_type == 'cuda' else False)

    # Chuyển mô hình lên device
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    print("\nCấu trúc mô hình:")
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nBắt đầu huấn luyện trên thiết bị: {device.type.upper()}...")
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
        print(f'Epoch [{epoch+1}/{epochs}] completed. Average Training Loss: {running_loss/total_steps:.4f}')
    print("Huấn luyện hoàn tất!")

    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(current_script_directory, MODEL_SAVE_FILENAME)
    torch.save(model.state_dict(), model_save_path)
    print(f"Mô hình đã được lưu tại: {model_save_path}")

    print("\nĐánh giá mô hình...")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        test_loss = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted_classes = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted_classes == labels).sum().item()
        accuracy = 100 * correct / total
        average_test_loss = test_loss / len(test_loader)
        print(f'Độ chính xác trên {total} ảnh kiểm tra: {accuracy:.2f}%')
        print(f'Mất mát trung bình trên tập kiểm tra: {average_test_loss:.4f}')

    print("\nDự đoán một vài ảnh từ tập kiểm tra:")
    images_vis, labels_vis = next(iter(test_loader))
    images_for_display = images_vis.cpu()
    labels_for_display = labels_vis.cpu()
    model.eval()
    with torch.no_grad():
        outputs_vis = model(images_vis.to(device))
        _, predicted_vis = torch.max(outputs_vis.data, 1)
        predicted_on_cpu = predicted_vis.cpu()
    plt.figure(figsize=(12, 5))
    num_images_to_show = min(20, batch_size)
    for i in range(num_images_to_show):
        if i >= images_for_display.shape[0]: break
        plt.subplot(4, 5, i + 1)
        img = images_for_display[i].squeeze()
        plt.imshow(img, cmap='gray')
        plt.title(f"Pred: {predicted_on_cpu[i].item()}\nTrue: {labels_for_display[i].item()}",
                  color=("green" if predicted_on_cpu[i] == labels_for_display[i] else "red"))
        plt.axis('off')
    plt.tight_layout()
    plt.show()