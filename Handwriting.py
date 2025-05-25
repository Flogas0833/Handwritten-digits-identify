import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader # Để tạo các batch dữ liệu
import matplotlib.pyplot as plt
import numpy as np

# --- Bước 0: Cấu hình thiết bị (CPU hoặc GPU nếu có) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Bước 1: Định nghĩa các tham số ---
batch_size = 64
epochs = 10
learning_rate = 0.001
input_size = 28 * 28 # Kích thước ảnh MNIST
hidden_size = 64 # Số lượng nơ-ron trong lớp ẩn 
num_classes = 10

# --- Bước 2: Tải và chuẩn bị dữ liệu MNIST ---
# Định nghĩa các phép biến đổi cho dữ liệu đầu vào
transform = transforms.Compose([
    transforms.ToTensor(), # Chuyển ảnh PIL hoặc NumPy (H, W, C) thành Tensor (C, H, W) và chuẩn hóa pixel về khoảng [0.0, 1.0]
    transforms.Normalize((0.1307,), (0.3081,)) # Chuẩn hóa Tensor với giá trị trung bình và độ lệch chuẩn.
                       # (0.1307,) và (0.3081,) là giá trị trung bình và độ lệch chuẩn thường được tính toán trước cho tập MNIST.
])

# Tải dữ liệu huấn luyện
train_dataset = datasets.MNIST(root = './data',  # Thư mục lưu dữ liệu
                               train = True,     # Lấy tập huấn luyện
                               download = True,  # Tải về nếu chưa có
                               transform = transform) # Áp dụng phép biến đổi

# Tải dữ liệu kiểm tra
test_dataset = datasets.MNIST(root = './data',
                              train = False,    # Lấy tập kiểm tra
                              download = True,
                              transform = transform)

# Tạo DataLoaders để quản lý việc tạo batch và xáo trộn dữ liệu
train_loader = DataLoader(dataset = train_dataset,
                          batch_size = batch_size,
                          shuffle = True) # Xáo trộn dữ liệu huấn luyện sau mỗi epoch

test_loader = DataLoader(dataset = test_dataset,
                         batch_size = batch_size,
                         shuffle = False) # Không cần xáo trộn dữ liệu kiểm tra

# --- Bước 3: Xây dựng mô hình Mạng Nơ-ron Đơn giản (MLP) ---
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten() # Làm phẳng ảnh từ [batch, 1, 28, 28] thành [batch, 784]
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()    
        self.fc2 = nn.Linear(hidden_size, num_classes) 

    def forward(self, x):
        x = self.flatten(x)    # Làm phẳng ảnh
        out = self.fc1(x)      # Lớp ẩn
        out = self.relu(out)   # Kích hoạt ReLU
        out = self.fc2(out)    # Lớp đầu ra (logits)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)
print("\nCấu trúc mô hình:")
print(model)

# --- Bước 4: Định nghĩa Hàm mất mát và Trình tối ưu hóa ---
criterion = nn.CrossEntropyLoss() # Hàm mất mát: CrossEntropyLoss kết hợp nn.LogSoftmax() và nn.NLLLoss()

# Trình tối ưu hóa: Adam 
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

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
    print(f'Epoch [{epoch+1}/{epochs}] completed. Average Training Loss: {running_loss/total_steps:.4f}')

print("Huấn luyện hoàn tất!")

# --- Bước 6: Đánh giá mô hình trên tập kiểm tra ---
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

        # Lấy lớp có xác suất cao nhất làm dự đoán
        # outputs.data chứa tensor logits
        _, predicted_classes = torch.max(outputs.data, 1) # 1 là chiều của các lớp
        total += labels.size(0) # Số lượng mẫu trong batch
        correct += (predicted_classes == labels).sum().item() # Số dự đoán đúng

    accuracy = 100 * correct / total
    average_test_loss = test_loss / len(test_loader)
    print(f'Độ chính xác trên {total} ảnh kiểm tra: {accuracy:.2f}%')
    print(f'Mất mát trung bình trên tập kiểm tra: {average_test_loss:.4f}')


# --- Bước 7: Dự đoán và Hiển thị một vài ví dụ ---
def imshow(img, title=""):
    # Unnormalize nếu cần (ví dụ này đã normalize)
    img = img * 0.3081 + 0.1307 # std, mean của MNIST
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)).squeeze(), cmap='gray') # Squeeze để bỏ chiều kênh nếu là ảnh xám
    plt.title(title)
    plt.show()

print("\nDự đoán một vài ảnh từ tập kiểm tra:")
dataiter = iter(test_loader)
images, labels = next(dataiter)
images_for_display = images.cpu() # Lấy một vài ảnh để hiển thị
labels_for_display = labels.cpu()

model.eval()
with torch.no_grad():
    outputs = model(images.to(device))
    _, predicted = torch.max(outputs.data, 1)
    predicted_on_cpu = predicted.cpu()

# Hiển thị 20 ảnh đầu tiên và dự đoán
plt.figure(figsize=(12, 5))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    # img = images_for_display[i] * 0.3081 + 0.1307 # Unnormalize
    img = images_for_display[i].squeeze() # Bỏ chiều kênh (1)
    plt.imshow(img, cmap = 'gray')
    plt.title(f"Pred: {predicted_on_cpu[i].item()}\nTrue: {labels_for_display[i].item()}",
              color=("green" if predicted_on_cpu[i] == labels_for_display[i] else "red"))
    plt.axis('off')
plt.tight_layout()
plt.show()