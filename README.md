## Prediksi Harga Penutupan Emiten Saham PT. Bank Rakyat Indonesia (Persero) Tbk. (BBRI) Dengan Recurrent Neural Network (RNN) dan Long Short-Term Memory (LSTM)

Dataset terkait harga saham diperoleh melalui situs web **Yahoo Finance**: (https://finance.yahoo.com/quote/BBRI.JK/) dengan cara scrapping dataset. Hasil prediksi juga ditampilkan dalam sebuah aplikasi web interaktif yaitu **Streamlit** yang memungkinkan pengguna menggunakannya dengan menginput: Kode Emiten Saham, Start Date, End Date, dan memilih model prediksi RNN dan LSTM. 

---
### Penulis:

- Nama             : Evindo Amanda Riza
- Program Studi    : Teknik Informatika
- Perguruan Tinggi : Universitas Medan Area
- Npm              : 218160018

---
### Tahapan:

1. **Pengumpulan Data**:
   - Dataset harga saham **BBRI.JK** diperoleh melalui **Yahoo Finance API**.
   - Dataset meliputi harga seperti **Open**, **High**, **Low**,**Close**, dan **Volume** secara harian.

2. **Praproses Data**:
   - Pembersihan data 
   - Splitting data : Training (70%), Validation (15%) dan Testing (15%)
   - Scalling data : MinMaxScaler
   - Pembentukan Sliding Window : 30 Timestep

3. **Models**:
   - **RNN** 
   - **LSTM** 

4. **Hyperparameter Tuning**:
   - Metode : **RandomizedSearchCV** 

5. **Visualisasi**:
   - Plotting Prediction vs Real Price
   - Plotting Training loss vs Validation loss
   - App Web : Streamlit

---
### Instalasi: 

- Clone the repository:
   ```bash
   git clone https://github.com/vinz25-coder/Prediksi_Saham_RNN-LSTM.git
   ```
- Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Run the Streamlit app:
  ```bash
  streamlit run main.py
  ```

---
### Persyaratan Sistem Penting / Important Requirements:

- python == `3.12.7`
- tensorFlow == `2.18.0`
- scikit-learn == `1.5.0`
- pandas == `2.2.3`
- numPy == `2.0.2`
- matplotlib == `3.10.1`
- streamlit == `1.44.1`
- yfinance == `0.2.55`
