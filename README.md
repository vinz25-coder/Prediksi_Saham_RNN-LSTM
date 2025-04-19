# Prediksi Harga Penutupan Emiten Saham PT. Bank Rakyat Indonesia (Persero) Tbk. (BBRI) Dengan Recurrent Neural Network (RNN) dan Long Short-Term Memory (LSTM)

Dataset terkait harga saham diperoleh melalui situs web **Yahoo Finance** (https://finance.yahoo.com/). Model prediktif telah dijalankan dan diubah menjadi sebuah aplikasi web yang memungkinkan pengguna untuk menggunakan model tersebut dengan menginput: Kode Emiten Saham, Start Date, End Date, dan Memilih Model prediksi RNN dan LSTM . 


### Penulis

- Nama             : Evindo Amanda Riza
- Program Studi    : Teknik Informatika
- Npm              : 218160018
- Perguruan Tinggi : Universitas Medan Area


## Features

1. **Data Collection**:
   - Dataset harga saham untuk **BBRI.JK** diperoleh menggunakan **Yahoo Finance API**.
   - Dataset meliputi harga seperti **Open**, **High**, **Low**,**Close**, dan **Volume** secara harian.

2. **Preprocessing**:
   - Pembersihan data 
   - Splitting data : Training, Validation dan Testing
   - Scalling data : MinMaxScaler
   - Pembentukan Sliding Window 

3. **Models**:
   - **RNN** 
   - **LSTM** 

4. **Hyperparameter Tuning**:
   - **RandomizedSearchCV** digunakan dalam pencarian kombinasi terbaik tiap model, diantaranya:
     - Unit/Neuron   : 64, 128, 256
     - Layer         : 1, 2
     - Dropout       : 0.2, 0.3
     - Learning rate : 0.001, 0.0001
     - Epoch         : 100
     - Batch Size    : 64
     - optimizer     : Adam
     - loss          : MSE
     - metric        : RMSE, MSE, MAE

5. **Visualization**:
   - Plotting Prediction vs Real Price
   - Plotting Training loss vs Validation loss

---

## Important Requirements
- Python == `3.12.7`
- TensorFlow == `2.18.0`
- Scikit-Learn == `1.5.0`
- Pandas == `2.2.3`
- NumPy == `2.0.2`
- Matplotlib == `3.10.1`