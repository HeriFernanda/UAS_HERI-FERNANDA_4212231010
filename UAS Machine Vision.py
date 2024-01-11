#klasifikasi karakter tulisan tangan pada dataset MNIST menggunakan HOG Feature Extraction dan Support Vector Machine (SVM)

# Import library
import tensorflow as tf 
import numpy as np 
from tensorflow.keras import datasets #library untuk membangun dan melatih pembelajaran dengan model tensorflow
from skimage.feature import hog #library untuk klasifikasi dan deteksi objek
from sklearn.svm import SVC #library untuk klasifikasi kelas svm
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix #library untuk mengevaluasi peforma model pembelajaran 

# Load data
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data() #untuk meload data pada mnist dataset

# Ekstraksi Fitur HOG
hog_features_train = [] #inisialisasi variabel data latih
for image in x_train: #perulangan untuk setiap citra dalam data latih
    fd = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3,3)) #untuk mengekstrasi fitur HOG dari sebuah image
    hog_features_train.append(fd) #menambahkan fitur HOG dari sebuah citra ke dalam list hog_features_train.

hog_features_train = np.array(hog_features_train) #untuk merubah hog_features_train kedalam numpy array

# Ekstraksi fitur untuk data test
hog_features_test = [] #inisialisasi variabel data testing
for image in x_test: #perulangan untuk setiap citra dalam data testing
    fd = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3,3)) #untuk mengekstrasi fitur HOG dari sebuah image
    hog_features_test.append(fd)  #menambahkan fitur HOG dari sebuah citra ke dalam list hog_features_test.
    
hog_features_test = np.array(hog_features_test) #untuk merubah hog_features_test kedalam numpy array

# Buat model SVM dan latih
svm_model = SVC(gamma='scale') #membuat instance model svc dari sklearn dengan parameter gamma='scale'
svm_model.fit(hog_features_train, y_train) #proses training model svc dengan data fitur dan label dari dataset latih

# Prediksi dan evaluasi
svm_predictions = svm_model.predict(hog_features_test) #untuk memprediksi dataset testing menggunakan model svm yang sudah dilatih


print("Confusion Matrix:", confusion_matrix(y_test,svm_predictions)) # menampilkan hasil prediksi confusion matrix
print("Accuracy:", accuracy_score(y_test, svm_predictions)) #menampilkan hasil prediksi accuracy
print("Precision:", precision_score(y_test, svm_predictions, average='weighted')) #menampilkan hasil prediksi precision