import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# --- Bước 1: Tải và chuẩn bị dữ liệu MNIST ---
print("Đang tải dữ liệu MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data, mnist.target.astype(int)

print(f"Kích thước dữ liệu gốc: X={X.shape}, y={y.shape}")

# --- Giới hạn số lượng mẫu để tăng tốc độ chạy thử ---
num_samples = 10000
X = X[:num_samples]
y = y[:num_samples]
print(f"Kích thước dữ liệu sau khi giới hạn (demo): X={X.shape}, y={y.shape}")


# Bước 2: Tiền xử lý dữ liệu
# SVM rất nhạy cảm với thang đo của dữ liệu, cần chuẩn hóa.
# StandardScaler sẽ chuyển đổi dữ liệu sao cho có trung bình = 0 và độ lệch chuẩn = 1.
print("Đang chuẩn hóa dữ liệu...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Bước 3: Chia tập dữ liệu thành huấn luyện và kiểm tra
# train_size=0.8 nghĩa là 80% cho huấn luyện, 20% cho kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Kích thước tập huấn luyện: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Kích thước tập kiểm tra: X_test={X_test.shape}, y_test={y_test.shape}")

# --- Bước 4: Xây dựng và huấn luyện mô hình Soft-margin SVM với RBF Kernel ---
print("\nĐang khởi tạo mô hình SVM...")
# SVC (Support Vector Classifier) là lớp cho SVM phân loại.
# kernel='rbf': Sử dụng RBF Kernel (Non-linear SVM)
# C: Tham số regularization (Soft-margin SVM). Giá trị nhỏ hơn -> biên mềm hơn, ít bị overfitting.
# gamma: Tham số cho RBF kernel. Giá trị nhỏ hơn -> ảnh hưởng rộng hơn, biên mượt hơn.
# Chúng ta sẽ thử nghiệm với các giá trị C và gamma cụ thể.
# Thường cần tinh chỉnh các tham số này bằng GridSearchCV hoặc RandomizedSearchCV.

# Ví dụ về một cấu hình có thể hoạt động tốt:
svm_model = SVC(kernel='rbf', C=10, gamma=0.001, random_state=42, verbose=True)

print("Đang huấn luyện mô hình SVM (có thể mất rất nhiều thời gian)...")
svm_model.fit(X_train, y_train)
print("Huấn luyện hoàn tất!")

# --- Bước 5: Đánh giá mô hình ---
print("\nĐánh giá mô hình trên tập kiểm tra...")
y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác trên tập kiểm tra: {accuracy*100:.2f}%")

print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_pred))

# --- Bước 6: Dự đoán và Hiển thị một vài ví dụ (Tùy chọn) ---
print("\nDự đoán một vài ảnh từ tập kiểm tra:")

# Chọn ngẫu nhiên một vài mẫu từ tập kiểm tra
indices = np.random.choice(len(X_test), 10, replace=False)
sample_images = X_test[indices]
sample_true_labels = y_test[indices]
sample_predictions = y_pred[indices]

plt.figure(figsize=(12, 5))
for i, idx in enumerate(indices):
    plt.subplot(1, 10, i + 1)
    # Vì ảnh đã được làm phẳng và chuẩn hóa, cần reshape lại 28x28
    # và unnormalize ngược lại để hiển thị (tùy thuộc vào cách chuẩn hóa)
    # StandardScaler không dễ dàng unnormalize như min-max scaler,
    # nhưng ta có thể hiển thị ảnh sau khi đã làm phẳng và chuẩn hóa.
    # Để hiển thị đúng, bạn cần unnormalize bằng cách đảo ngược StandardScaler:
    # (image_scaled * std_dev) + mean. Tuy nhiên, ví dụ này chỉ hiển thị giá trị đã scale.
    # Để hiển thị ảnh thô ban đầu, ta cần lấy từ X_test_raw nếu có, hoặc unscale từ X_scaled.
    # Trong ví dụ này, chỉ hiển thị giá trị pixel đã được scale.
    img_display = X_test[idx].reshape(28, 28)
    plt.imshow(img_display, cmap='gray')
    plt.title(f"P: {sample_predictions[i]}\nT: {sample_true_labels[i]}",
              color=("green" if sample_predictions[i] == sample_true_labels[i] else "red"))
    plt.axis('off')
plt.tight_layout()
plt.show()

# Để hiển thị ảnh gốc từ MNIST trước khi scale, bạn có thể làm như sau:
# X_raw_for_display, y_raw_for_display = mnist.data, mnist.target.astype(int)
# plt.figure(figsize=(12, 5))
# for i, idx in enumerate(indices):
#     plt.subplot(1, 10, i + 1)
#     original_image = X_raw_for_display[indices[i]].reshape(28, 28)
#     plt.imshow(original_image, cmap='gray')
#     plt.title(f"P: {sample_predictions[i]}\nT: {sample_true_labels[i]}",
#               color=("green" if sample_predictions[i] == sample_true_labels[i] else "red"))
#     plt.axis('off')
# plt.tight_layout()
# plt.show()
