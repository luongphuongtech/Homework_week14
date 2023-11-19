import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Đọc dữ liệu từ tệp 'SPECTF.dat'
with open('SPECTF.dat', 'r') as file:
    data = file.read()

data_lines = data.split('\n')
labels = []
features = []

# Xử lý dữ liệu
for line in data_lines:
    values = line.split(',')
    labels.append(int(values[0]))
    features.append(list(map(int, values[1:])))

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Huấn luyện mô hình cây quyết định
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Đánh giá mô hình trên tập kiểm tra
accuracy = clf.score(X_test, y_test)
print(f'Accuracy on test set: {accuracy}')

