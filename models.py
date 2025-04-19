import math
import pandas as pd
import tensorflow as tf
import os
import plotly.io as pio
import plotly.graph_objects as go
import streamlit as st
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from services import load_data


# =========================================
#  Fungsi Ambil Data Saham
# =========================================
@st.cache_data
def prediction(ticker, start_date, end_date):
    """Mengambil data saham dan mengembalikan dataframe dengan hanya kolom 'Date' dan 'Close'."""

    if ticker == "BBRI.JK":
        df = pd.read_excel("datasets/BBRI_2010-2025.xlsx")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]
    else:
        df, _ = load_data(ticker, start_date, end_date)

    # Pastikan hanya kolom Date dan Close yang digunakan
    df = df[['Date', 'Close']].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df

# =========================================
#  Fungsi Split Data Training, Validation, Testing
# =========================================

def split_data(df):
    length_data = len(df)

    # Rasio pembagian data
    train_len = math.ceil(length_data * 0.7)  # 70% data untuk training
    val_len = int(length_data * 0.15)   # 15% data untuk validation
    test_len = length_data - (train_len + val_len)  # Sisa 15% untuk testing

    # Ambil kolom tanggal sebelum membagi data
    dates = df.index  # Simpan index datetime
    
    # Pisahkan dataset dengan iloc dan copy untuk menghindari warning
    train_data = df.iloc[:train_len].copy()
    val_data = df.iloc[train_len:train_len + val_len].copy()
    test_data = df.iloc[train_len + val_len:].copy()

    return train_data, val_data, test_data


# =========================================
# Fungsi Plot Data Split
# =========================================     

def plot_data_split(df):
    """Menampilkan plot pembagian data menjadi Training, Validation, dan Testing."""
    pio.templates.default = "plotly_dark"
    
    # Menghapus MultiIndex pada kolom jika ada
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
    
    # Pastikan 'Date' ada di kolom dan bukan index
    df = df.reset_index() if "Date" not in df.columns else df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Tentukan nama kolom harga saham yang benar
    price_column = "Close" if "Close" in df.columns else df.columns[1]
    if price_column not in df.columns:
        raise ValueError("Kolom harga saham tidak ditemukan.")
    
    # Bagi data menjadi Training, Validation, dan Testing
    train_set, val_set, test_set = split_data(df)
    
    # Tambahkan kolom kategori untuk mempermudah pewarnaan
    for dataset, label in zip([train_set, val_set, test_set], ["Training", "Validation", "Testing"]):
        dataset["Data Type"] = label
    
    # Gabungkan semua dataset kembali
    combined_df = pd.concat([train_set, val_set, test_set], ignore_index=True)
    
    # Plotting Dataset
    fig = go.Figure()
    # 🟣 Training Data, 🟡 Validation Data, 🔴 Testing Data
    colors = {"Training": "#9966CC", "Validation": "gold", "Testing": "red"}
    for dataset in [train_set, val_set, test_set]:
        fig.add_trace(go.Scatter(
            x=dataset["Date"],
            y=dataset[price_column],
            mode="lines",
            name=dataset["Data Type"].iloc[0],
            line=dict(color=colors[dataset["Data Type"].iloc[0]], width=2)
        ))
    
    # Tambahkan kustomisasi tampilan
    fig.update_traces(
        line=dict(width=2), 
        hoverinfo="x+y", 
        hovertemplate="📅 Tanggal: %{x|%Y-%m-%d}<br> Close Price: %{y:,.2f}"
    )

    fig.update_layout(
        title=" Pembagian Data Training, Validation, dan Testing",
        xaxis_title="Tanggal",
        yaxis_title="Harga",
        height=650,
        hovermode="x",
        xaxis=dict(
            showgrid=False, gridcolor="rgba(200, 200, 200, 0.4)",
            showline=False,linewidth=1, linecolor="grey",
            rangeslider=dict(visible=True, bgcolor="rgba(255,255,255,0.1)"),
        ),
        yaxis=dict(
            fixedrange=False,
            showgrid=True,
            zeroline=True, 
            zerolinewidth=1,
            zerolinecolor="rgba(150, 150, 150, 0.2)"
        ),
        dragmode="zoom"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tampilkan ringkasan data di Streamlit
    total_data = len(df)
    train_size = len(train_set)
    val_size = len(val_set)
    test_size = len(test_set)

    # Tampilkan informasi tambahan di Streamlit dalam bentuk grid
    train_pct = (train_size / total_data) * 100
    val_pct = (val_size / total_data) * 100
    test_pct = (test_size / total_data) * 100

    st.markdown("<h5> Ringkasan Data</h5>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="📊 Total Data", value=total_data)
    with col2:
        st.metric(label="🟣 Training Data", value=train_size, delta=f"{train_pct:,.2f}%")
    with col3:
        st.metric(label="🟡 Validation Data", value=val_size, delta=f"{val_pct:,.2f}%")
    with col4:
        st.metric(label="🔴 Testing Data", value=test_size, delta=f"{test_pct:,.2f}%")

# =========================================
#  Fungsi Normalisasi dan Sliding Window                                                                                        
# =========================================

def sliding_window(data, time_step):
    """Membentuk dataset dengan teknik sliding window."""
    x, y = [], []
    for i in range(time_step, len(data)):
        x.append(data[i - time_step:i, 0])
        y.append(data[i, 0])
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    y = np.reshape(y, (y.shape[0], 1))
    return x, y

def preprocess_data(df, train_len, val_len, test_len, time_step=30):
    sc = MinMaxScaler(feature_range=(0, 1))
    train_set = df.iloc[:train_len]
    train_scaled = sc.fit_transform(train_set)
    x_train, y_train = sliding_window(train_scaled, time_step)
    
    val_set = df.iloc[train_len - time_step:train_len + val_len]
    val_scaled = sc.transform(val_set)
    x_val, y_val = sliding_window(val_scaled, time_step)
    
    test_set = df.iloc[-(test_len + time_step):]
    test_scaled = sc.transform(test_set)
    x_test, _ = sliding_window(test_scaled, time_step)

    # Simpan tanggal untuk bagian testing saja
    dates_test = df.index[-test_len:]

    close_col = [col for col in df.columns if "Close" in col][0]
    y_test = df[[close_col]].iloc[-test_len:, :].values
    
    return {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
        "scaler": sc,
        "dates": dates_test  # Tambahkan dates
    }

# =========================================
#  Fungsi Load Model & Training Results
# =========================================
   
MODEL_PATHS = {
    "RNN": os.path.join("model_checkpoints", "best_RNN_model.keras"),
    "LSTM": os.path.join("model_checkpoints", "best_LSTM_model.keras")
}

TRAINING_RESULTS_PATHS = {
    "RNN": os.path.join("training_results", "RNN_loss_history.csv"),
    "LSTM": os.path.join("training_results", "LSTM_loss_history.csv")
}

def load_best_model(model_type):
    """Memuat model terbaik berdasarkan tipe yang diberikan."""
    model_path = MODEL_PATHS.get(model_type)

    if model_path is None:
        raise ValueError("Model type harus 'RNN' atau 'LSTM'")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' tidak ditemukan.")
    
    logging.info(f"Memuat model {model_type} terbaik yang telah disimpan...")
    return tf.keras.models.load_model(model_path)

def load_training_results(model_type):
    """Memuat hasil training berdasarkan model type."""
    results_path = TRAINING_RESULTS_PATHS.get(model_type)

    if results_path is None or not os.path.exists(results_path):
        raise FileNotFoundError(f"File training results untuk {model_type} tidak ditemukan.")
    
    try:
        training_results = pd.read_csv(results_path)
        return training_results
    except Exception as e:
        logging.error(f"Error memuat file training results: {e}")
        return None
    


