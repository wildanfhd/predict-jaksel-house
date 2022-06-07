# UAS Pengantar Kecerdasan Buatan
# 1. Via Google Drive
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
# Import library untuk membuat model
from sklearn.tree import DecisionTreeRegressor
# Import library untuk menghitung MAE (Mean Absolute Error)
from sklearn.metrics import mean_absolute_error
# Import library untuk melakukan Split Data
from sklearn.model_selection import train_test_split

# Membuat Path ke File 'HARGA RUMAH JAKSEL.xlsx'
# 2 Opsi path : 
# 1. Google Drive
rumah_jaksel_file_path_gd = '/content/drive/MyDrive/Dataset/HARGA RUMAH JAKSEL.xlsx'

# 2. Local Files
# local_rumah_jaksel_file_path = '../namaFolder/namaFile.xlsx'

# Membaca data 
data_rumah_jaksel = pd.read_excel(rumah_jaksel_file_path_gd)

# Membuang nilai/value yang hilang
data_rumah_jaksel = data_rumah_jaksel.dropna(axis=0) 

# Review Data
print('\n===========================REVIEW DATA==========================')
print('================================================================')
print(data_rumah_jaksel.describe())
print('================================================================')
# Melihat kolom-kolom yang ada pada data
print('Kolom-kolom : ', data_rumah_jaksel.columns)
print('================================================================')

# Menentukan dan mendefinisikan Target yang akan diprediksi
y = data_rumah_jaksel.HARGA
print(np.mean(y))

# Menentukan dan mendefinisikan Feature yang akan menjadi variabel pendukung untuk memprediksi
rumah_jaksel_features = ['LT', 'LB', 'JKT', 'JKM']
X = data_rumah_jaksel[rumah_jaksel_features]
# Melihat 5 Data teratas
print('\n5 Data teratas : ') 
print(X.head())
print('================================================================')

# Membuat Model dengan Algoritma Decision Tree Model
# Mengatur random_state dengan nilai 1 agar setiap kali kita menjalankan program hasilnya akan tetap sama
rumah_jaksel_model = DecisionTreeRegressor(random_state=1)

# Fit Model yang sudah dibuat
rumah_jaksel_model.fit(X, y)

# Membuat Prediksi 
print('\nMembuat prediksi untuk 5 rumah di Daerah Jakarta Selatan')
print(X.head())
print('\nPrediksi Harga Rumah : ')
print(rumah_jaksel_model.predict(X.head()))
print('\nHarga Rumah Sesungguhnya : ')
print(y.head())
print('================================================================')

# Menampung harga prediksi rumah
prediksi_harga_rumah = rumah_jaksel_model.predict(X)

# Menghitung Nilai MAE (Mean Absolute Error) untuk mengevaluasi data
mae_value = mean_absolute_error(y, prediksi_harga_rumah)
print('\nNilai MAE(Mean Absolute Error) pada prediksi sebelum Split Data :', mae_value)

# Splitting Data menjadi 2 bagian :
#  1. Training Data - Digunakan untuk fit model
#  2. Validation Data - Digunakan untuk menghitung Mean Absolute Error(MAE)

# Membuat training dan validation data pada masing-masing variabel
# dengan menggunakan train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)

# Membuat Model
rumah_jaksel_model = DecisionTreeRegressor()

# Fit Model yang sudah dibuat
rumah_jaksel_model.fit(train_X, train_y)

# Mendapatkan harga prediksi dari validation data
val_predictions = rumah_jaksel_model.predict(val_X)

# Menghitung Nilai MAE (Mean Absolute Error) pada validation data untuk mengevaluasi data
val_mae = mean_absolute_error(val_predictions, val_y)
print('\nNilai MAE(Mean Absolute Error) pada prediksi Setelah Split Data : ', val_mae)

# Membuat Prediksi setelah mendapatkan harga prediksi pada validation data
print('\nMembuat prediksi untuk 5 rumah setelah mendapatkan harga prediksi pada validation data')
print(val_X.head())
print('\nPrediksi Harga Rumah : ')
print(rumah_jaksel_model.predict(val_X.head()))
print('\nHarga Rumah Sesungguhnya : ')
print(val_y.head())
