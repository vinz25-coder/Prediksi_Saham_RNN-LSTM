import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date
from streamlit_option_menu import option_menu
from services import load_data, validation_input, plot_data, plot_volume, statistics
from models import prediction, split_data, plot_data_split, load_best_model, load_training_results, preprocess_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error


# ===================================================================================
st.set_page_config(layout="wide", page_title="DeepStock AI", page_icon="🚀")

# Sidebar
def sidebar():
    st.sidebar.markdown("<h1 style='text-align: center; font-size: 30px;'>"
                        "<b>Deep</b><b style='color: #9966CC;'>Stock</b></h1>", 
                        unsafe_allow_html=True)
    st.sidebar.title("Menu")

    # Pilihan saham
    stocks = ["BBCA.JK", "BBNI.JK", "BBRI.JK", "BMRI.JK"]
    
    # Menyimpan dan memuat selected_stock dalam session_state 
    if "selected_stock" not in st.session_state:
        st.session_state["selected_stock"] = "-- Pilih Saham --"
    selected_stock = st.sidebar.selectbox(
        "Pilih Saham", 
        ["-- Pilih Saham --"] + stocks, 
        index=(["-- Pilih Saham --"] + stocks).index(st.session_state["selected_stock"])
    )
    st.session_state["selected_stock"] = selected_stock
    
    # Menyimpan dan memuat start_date dalam session_state  
    if "start_date" not in st.session_state:
        st.session_state["start_date"] = date.today()
    start_date = st.sidebar.date_input(
        "Tanggal Mulai",
        value=st.session_state["start_date"],  
        min_value=date(2010, 1, 1),
        max_value=date.today()
    )

    # Menyimpan dan memuat end_date dalam session_state 
    if "end_date" not in st.session_state:
        st.session_state["end_date"] = date.today() 
    end_date = st.sidebar.date_input(
        "Tanggal Akhir",
        value=st.session_state["end_date"],
        min_value=start_date,  
        max_value=date.today()  
    )
    st.session_state["end_date"] = end_date  

    selected_stocks_comparison = st.sidebar.multiselect("Pilih Saham Untuk Perbandingan", stocks)

    return selected_stock, start_date, end_date, selected_stocks_comparison

def main():
    selected_stock, start_date, end_date, selected_stocks_comparison = sidebar()

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;700&display=swap');

        h3 {
            font-size: 26px !important;
            font-weight: bold !important;
            font-family: 'IBM Plex Sans', sans-serif !important;
        }

        .stat-text {
            font-size: 20px !important;
            font-weight: bold !important;
            font-family: 'IBM Plex Sans', sans-serif !important;
        }
        
        body, html {
            font-family: 'IBM Plex Sans', sans-serif !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        "<h1 style='text-align: center; margin-top: -30px;'>"
        "Deep<span style='color: #9966CC;'>Stock</span> Prediction AI 📈"
        "</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center;'>"
        "<span style='font-weight: bold;'>Deep</span>"
        "<span style='color: #9966CC; font-weight: bold;'>Stock</span> "
        "adalah aplikasi web sederhana prediksi harga saham menggunakan DeepLearning: "
        "<span style='color: #FF5733; font-weight: bold;'>RNN</span> & "
        "<span style='color: #33C1FF; font-weight: bold;'>LSTM</span>"
        "</p>",
        unsafe_allow_html=True
    )
    
    # ====================== NAVIGASI ======================
    selected_tab = option_menu(
        menu_title=None,
        options=["Dataframes", "Charts", "Statistics", "Prediction",  "Comparison"],
        icons=["table", "bar-chart", "calculator", "graph-up-arrow", "arrow-down-up"],
        menu_icon="📊",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"class": "menu-container", "padding": "8px"}, 
            "icon": {"font-size": "16px"},  
            "nav-link": {"font-size": "14px", "text-align": "center", "margin": "1px"},
            "nav-link-selected": {"background-color": "#9966CC", "color": "white"},
        }
    )

    # ====================== Validasi Input & load data ========================
    validation_input(selected_stock, start_date, end_date)

    with st.spinner(f"Mengambil data {selected_stock} dari {start_date} sampai {end_date}..."):
        data, stock_info = load_data(selected_stock, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        
    if data is None or data.empty:
        st.warning(f"⚠️ Tidak ada data untuk {selected_stock} dari {start_date} sampai {end_date}. Coba pilih tanggal lain.")
        st.stop()
    st.toast("✅ Data berhasil dimuat!")

    # ====================== DATAFRAMES ===========================
    if selected_tab == "Dataframes":
        st.markdown(
            f"<h3>📆 Data Historis: <span style='color:#9966CC; font-weight:bold;'>{selected_stock}</span> ({start_date} - {end_date})</h3>", unsafe_allow_html=True)
        st.markdown("<hr style='border: 1px solid #333; margin:-1px 0;'>", unsafe_allow_html=True)
        data = data.applymap(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
        df = pd.DataFrame(data)
        st.dataframe(df)
       
     # ====================== CHARTS ======================
    elif selected_tab == "Charts":
        st.markdown(
            f"<h3>📊 Grafik Harga Saham: <span style='color:#9966CC; font-weight:bold;'>{selected_stock}</span> ({start_date} - {end_date})</h3>", 
        unsafe_allow_html=True
        )
        st.markdown("<hr style='border: 1px solid #333; margin:-1px 0;'>", unsafe_allow_html=True)
        plot_data(data)  
        plot_volume(data)  

    # ====================== STATISTICS ======================
    elif selected_tab == "Statistics":
        st.markdown(
            f"<h3>🔢 Statistik: <span style='color:#9966CC; font-weight:bold;'>{selected_stock}</span> ({start_date} - {end_date})</h3>", 
            unsafe_allow_html=True
        )
        st.markdown("<hr style='border: 1px solid #333; margin:-1px 0;'>", unsafe_allow_html=True)
        statistics(data, stock_info, selected_stock)
    
    # ====================== PREDICTION ======================
    elif selected_tab == "Prediction":
        def display_prediction(selected_stock, start_date, end_date):
            st.markdown(
                f"""<h3>🚀 Prediksi Model 
                <span style="color:#FF5733;">RNN</span> 
                & <span style="color:#33C1FF;">LSTM</span> : <span style='color:#9966CC;'>{selected_stock}</span> ({start_date} - {end_date})</h3>""",  
                unsafe_allow_html=True
            )
            st.markdown("<hr style='border: 1px solid #333; margin:-1px 0;'>", unsafe_allow_html=True)
    
        def load_stock_data(selected_stock, start_date, end_date):
            with st.spinner("🔍 Mengambil Data..."):
                df = prediction(selected_stock, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                if df is None or df.empty:
                    st.warning("⚠️ Data tidak ditemukan!")
                    st.stop()
                fig = plot_data_split(df) 
                st.divider() 
                return df
        
        def preprocess_and_display_data(df):
            st.markdown("<h5> Pembentukan Pola Sliding Window</h5>", unsafe_allow_html=True)

            train_data, val_data, test_data = split_data(df)
            data = preprocess_data(df, len(train_data), len(val_data), len(test_data), time_step=30)
            
            # Normalisasi
            x_train_df = pd.DataFrame(data['x_train'][:, :, 0]).applymap(lambda x: f"{x:.4f}")
            x_train_df["y_train"] = data['y_train'].flatten()
            
            st.markdown("**Normalisasi Data**")
            st.dataframe(x_train_df)
            return train_data, val_data, test_data, data
       
        def display_model(model_name):
            try:
                
                if model_name not in st.session_state:
                    st.session_state[model_name] = load_best_model(model_name)
                model = st.session_state[model_name]
                
                with st.expander(f"📑 Arsitektur {model_name}"):
                    model_summary = []
                    model.summary(print_fn=lambda x: model_summary.append(x))
                    st.code("\n".join(model_summary), language="plaintext")
                
                training_results = load_training_results(model_name)
                if training_results is not None and not training_results.empty:
                    with st.expander(f"📑 Hasil Training {model_name}"):
                        st.write(training_results)
                        if {'epoch', 'training_loss', 'validation_loss'}.issubset(training_results.columns):
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=training_results['epoch'], y=training_results['training_loss'], mode='lines', name='Training Loss'))
                            fig.add_trace(go.Scatter(x=training_results['epoch'], y=training_results['validation_loss'], mode='lines', name='Validation Loss'))
                            fig.update_layout(title=f"Training Loss {model_name}", 
                                              xaxis_title='Epoch', 
                                              yaxis_title='Loss',
                                              height = 550,
                                              )
                            st.plotly_chart(fig)          
                else:
                    st.warning(f"❌ Tidak ada hasil training untuk {model_name}.")
                return model
            except Exception as e:
                st.error(f"❌ Gagal memuat {model_name}: {e}")
                return None
            
        def predict_prices(model_name, model, data, scaler):
            x_test, y_test, dates = data['x_test'], data['y_test'], data['dates']
            test_len = len(y_test)
            
            y_pred = model.predict(x_test)
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            
            df_pred = pd.DataFrame({
                "Close": y_pred
            }, index=pd.to_datetime(dates[-len(y_pred):]))
            
            return df_pred
        
        def evaluate_model(y_true, y_pred):
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            return mse, rmse, mae

        def plot_stock_data(train, val, test, pred, model_option):
            close_col = next((col for col in train.columns if "Close" in col), None)
            
            if not close_col:
                st.error("❌ Tidak ditemukan kolom harga penutupan dalam dataset!")
                return None
            
            # Pastikan pred_df menggunakan kolom yang sama
            pred_col = "Close" if "Close" in pred.columns else close_col  

            # Set colors based on selected model
            if model_option == "RNN":
                pred_color = 'red'  
            elif model_option == "LSTM":
                pred_color = 'red'  

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=train.index, y=train[close_col], mode='lines', name='Training', line=dict(color='#9966CC')))
            fig.add_trace(go.Scatter(x=val.index, y=val[close_col], mode='lines', name='Validation', line=dict(color='gold')))
            fig.add_trace(go.Scatter(x=test.index, y=test[close_col], mode='lines', name='Testing', line=dict(color='#636EFA')))
            fig.add_trace(go.Scatter(x=pred.index, y=pred[pred_col], mode='lines', name=f'Prediksi {model_option}', line=dict(color=pred_color, dash='dot')))
            
            # Tambahkan kustomisasi tampilan
            fig.update_traces(
                line=dict(width=1), 
                hoverinfo="x+y", 
                hovertemplate="📅 Date: %{x|%Y-%m-%d}<br> Close Price: %{y:,.2f}"
            )
            fig.update_layout(
                title="Visualisasi Training, Validation, Testing, dan Prediksi",
                xaxis_title="Date",
                yaxis_title="Harga Saham",
                height=650,
                hovermode="x",
                xaxis=dict(
                    showgrid=False, gridcolor="rgba(200, 200, 200, 0.4)",
                    showline=False, linewidth=1, linecolor="grey",
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
            st.divider()
            return fig


        def focus_testing(test, pred, model_name="Model"):
            fig = go.Figure()

            # Mendeteksi kolom Close secara otomatis
            close_col = next((col for col in test.columns if "Close" in col), None)

            if not close_col:
                st.error("❌ Tidak ditemukan kolom harga penutupan dalam dataset!")
                return None

            pred_col = "Close" if "Close" in pred.columns else close_col  

            # Set colors based on selected model
            if model_name == "RNN":
                pred_color = 'red'  
            elif model_name == "LSTM":
                pred_color = 'red'  
  
            
            # Fokus hanya pada data Testing dan Prediksi
            fig.add_trace(go.Scatter(x=test.index, y=test[close_col], mode='lines', name='Testing', line=dict(color='#636EFA', width=3)))
            fig.add_trace(go.Scatter(x=pred.index, y=pred[pred_col], mode='lines', name=f'Prediksi {model_name}', line=dict(color=pred_color, width=4, dash='dot')))
            
            # Tambahkan kustomisasi tampilan
            fig.update_traces(
                line=dict(width=1), 
                hoverinfo="x+y", 
                hovertemplate="📅 Date: %{x|%Y-%m-%d}<br> Close Price: %{y:,.2f}"
            )
            fig.update_layout(
                title=f"Prediksi pada Model {model_name}",
                xaxis_title="Date",
                yaxis_title="Harga Saham",
                height=650,
                hovermode="x",
                xaxis=dict(
                    showgrid=False, gridcolor="rgba(200, 200, 200, 0.4)",
                    showline=False, linewidth=2, linecolor="grey",
                    rangeslider=dict(visible=True, bgcolor="rgba(255,255,255,0.1)"),
                ),
                yaxis=dict(
                    fixedrange=False,
                    showgrid=True,
                    zeroline=True, 
                    zerolinewidth=2,
                    zerolinecolor="rgba(150, 150, 150, 0.2)"
                ),
                dragmode="zoom"
            )
            return fig
        
        def plot_comparison(test, pred_rnn, pred_lstm):
            close_col = next((col for col in test.columns if "Close" in col), None)

            if not close_col:
                st.error("❌ Tidak ditemukan kolom harga penutupan dalam dataset!")
                return None
            
            # Pastikan prediksi juga menggunakan kolom yang sama
            pred_col_rnn = "Close" if "Close" in pred_rnn.columns else close_col
            pred_col_lstm = "Close" if "Close" in pred_lstm.columns else close_col

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=test.index, y=test[close_col], mode='lines', name='Testing', line=dict(color='#636EFA')))
            fig.add_trace(go.Scatter(x=pred_rnn.index, y=pred_rnn[pred_col_rnn], mode='lines', name='Prediksi RNN', line=dict(color='red', dash='dot')))
            fig.add_trace(go.Scatter(x=pred_lstm.index, y=pred_lstm[pred_col_lstm], mode='lines', name='Prediksi LSTM', line=dict(color='yellow', dash='dot')))

            # Tambahkan kustomisasi tampilan
            fig.update_traces(
                line=dict(width=1), 
                hoverinfo="x+y", 
                hovertemplate="📅 Date: %{x|%Y-%m-%d}<br> Close Price: %{y:,.2f}"
            )
            fig.update_layout(
                title="RNN vs LSTM",
                xaxis_title="Tanggal",
                yaxis_title="Harga Saham",
                height=750,
                hovermode="x",
                xaxis=dict(
                showgrid=False, gridcolor="rgba(200, 200, 200, 0.4)",
                showline=False,linewidth=2, linecolor="grey",
                rangeslider=dict(visible=True, bgcolor="rgba(255,255,255,0.1)"),
                ),
                yaxis=dict(
                    fixedrange=False,
                    showgrid=True,
                    zeroline=True, 
                    zerolinewidth=2,
                    zerolinecolor="rgba(150, 150, 150, 0.2)"
                ),
                dragmode="zoom"
            )
            return fig

        def reset_predictions(selected_stock, start_date, end_date):
            if "last_selected_stock" not in st.session_state or st.session_state["last_selected_stock"] != selected_stock or \
            "last_start_date" not in st.session_state or st.session_state["last_start_date"] != start_date or \
            "last_end_date" not in st.session_state or st.session_state["last_end_date"] != end_date:
                st.session_state.pop("pred_rnn", None)
                st.session_state.pop("pred_lstm", None)

            st.session_state["last_selected_stock"] = selected_stock
            st.session_state["last_start_date"] = start_date
            st.session_state["last_end_date"] = end_date

        # Tampilkan Header
        display_prediction(selected_stock, start_date, end_date)
        # Load Data Saham
        df = load_stock_data(selected_stock, start_date, end_date)
        # Tampilkan Data yang Diproses
        train_data, val_data, test_data, data = preprocess_and_display_data(df)
        
        # Simpan Scaler ke session_state
        st.session_state["scaler_RNN"] = data["scaler"]
        st.session_state["scaler_LSTM"] = data["scaler"]

        reset_predictions(selected_stock, start_date, end_date)
        
        with st.container():
            st.markdown("<h5> Pilih Model yang Ingin Ditampilkan</h5>", unsafe_allow_html=True)
            model_option = st.radio(
                "Pilih Model :",
                options=["RNN", "LSTM", "RNN vs LSTM"],
                index=None,  
                horizontal=True
            )
            st.divider()
    
        # Menampilkan hasil prediksi berdasarkan pilihan radio button
        if model_option == "RNN":
            st.markdown("**Model Arsitektur dan Training Loss RNN**")
            
            with st.spinner("🔍 Memuat Model RNN..."):
                model_rnn = display_model("RNN")
                scaler_rnn = st.session_state.get("scaler_RNN")
                
                if model_rnn and scaler_rnn:
                    pred_rnn = predict_prices("RNN", model_rnn, data, scaler_rnn)
                    st.session_state["pred_rnn"] = pred_rnn  

                    # ✅ Deteksi otomatis nama kolom
                    close_col = next((col for col in test_data.columns if "Close" in col), "Close")
                    if "Close" in pred_rnn.columns and close_col != "Close":
                        pred_rnn = pred_rnn.rename(columns={"Close": close_col})

                    # Evaluasi Model
                    mse_rnn, rmse_rnn, mae_rnn = evaluate_model(test_data[close_col].values, pred_rnn[close_col].values)

                    # Tampilkan Hasil Evaluasi
                    st.divider()
                    st.markdown("**Evaluasi Model RNN**")
                    st.write(f"- **MSE**: {mse_rnn:,.2f}")
                    st.write(f"- **RMSE**: {rmse_rnn:,.2f}")
                    st.write(f"- **MAE**: {mae_rnn:,.2f}")

                    # Plot Grafik
                    fig_rnn = plot_stock_data(train_data, val_data, test_data, pred_rnn, model_option="RNN")
                    fig_rnn_focus = focus_testing(test_data, pred_rnn, model_name="RNN")
                    st.plotly_chart(fig_rnn)
                    st.plotly_chart(fig_rnn_focus)
                
        elif model_option == "LSTM":
            st.markdown("**Model Arsitektur dan Training Loss LSTM**")
            with st.spinner("🔍 Memuat Model LSTM..."):
                model_lstm = display_model("LSTM")
                scaler_lstm = st.session_state.get("scaler_LSTM")

                if model_lstm and scaler_lstm:
                    pred_lstm = predict_prices("LSTM", model_lstm, data, scaler_lstm)
                    st.session_state["pred_lstm"] = pred_lstm

                    # ✅ Deteksi otomatis nama kolom
                    close_col = next((col for col in test_data.columns if "Close" in col), "Close")
                    if "Close" in pred_lstm.columns and close_col != "Close":
                        pred_lstm = pred_lstm.rename(columns={"Close": close_col})

                    # Evaluasi Model
                    mse_lstm, rmse_lstm, mae_lstm = evaluate_model(test_data[close_col].values, pred_lstm[close_col].values)

                    # Tampilkan Hasil Evaluasi
                    st.divider()
                    st.markdown("**Evaluasi Model LSTM**")
                    st.write(f"- **MSE**: {mse_lstm:,.2f}")
                    st.write(f"- **RMSE**: {rmse_lstm:,.2f}")
                    st.write(f"- **MAE**: {mae_lstm:,.2f}")

                    # Plot Grafik
                    fig_lstm = plot_stock_data(train_data, val_data, test_data, pred_lstm, model_option="LSTM")
                    fig_lstm_focus = focus_testing(test_data, pred_lstm, model_name="LSTM")
                    st.plotly_chart(fig_lstm)
                    st.plotly_chart(fig_lstm_focus)
                    
        elif model_option == "RNN vs LSTM":
            if "pred_rnn" not in st.session_state or "pred_lstm" not in st.session_state:
                st.warning("⚠️ Silakan jalankan prediksi RNN dan LSTM terlebih dahulu.")
                st.divider()
            else:
                pred_rnn = st.session_state["pred_rnn"]
                pred_lstm = st.session_state["pred_lstm"]

                # ✅ Deteksi otomatis kolom harga penutupan
                close_col = next((col for col in test_data.columns if "Close" in col), "Close")

                # ✅ Rename prediksi kalau perlu
                if "Close" in pred_rnn.columns and close_col != "Close":
                    pred_rnn = pred_rnn.rename(columns={"Close": close_col})
                if "Close" in pred_lstm.columns and close_col != "Close":
                    pred_lstm = pred_lstm.rename(columns={"Close": close_col})

                # Plot Perbandingan
                fig_comparison = plot_comparison(test_data, pred_rnn, pred_lstm)
                st.plotly_chart(fig_comparison)

                # Evaluasi
                actual_prices = test_data[close_col].values.squeeze()
                predicted_rnn = pred_rnn[close_col].values.squeeze()
                predicted_lstm = pred_lstm[close_col].values.squeeze()

                mse_rnn, rmse_rnn, mae_rnn = evaluate_model(actual_prices, predicted_rnn)
                mse_lstm, rmse_lstm, mae_lstm = evaluate_model(actual_prices, predicted_lstm)

                st.divider()
                st.markdown("<h5>  Perbandingan Evaluasi Model </h5>", unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**RNN :**")
                    st.write(f"- **MSE**: {mse_rnn:,.2f}")
                    st.write(f"- **RMSE**: {rmse_rnn:,.2f}")
                    st.write(f"- **MAE**: {mae_rnn:,.2f}")
                with col2:
                    st.markdown("**LSTM :**")
                    st.write(f"- **MSE**: {mse_lstm:,.2f}")
                    st.write(f"- **RMSE**: {rmse_lstm:,.2f}")
                    st.write(f"- **MAE**: {mae_lstm:,.2f}")

                # Tabel Perbandingan
                st.divider()
                st.markdown("**Hasil Prediksi RNN vs LSTM**", unsafe_allow_html=True)

                df_comparison = pd.DataFrame({
                    "Date": test_data.index.strftime('%Y-%m-%d'),
                    "Actual Price": actual_prices,
                    "Predicted RNN": predicted_rnn,
                    "Predicted LSTM": predicted_lstm
                })

                df_comparison.iloc[:, 1:] = df_comparison.iloc[:, 1:].applymap(
                    lambda x: "{:,.2f}".format(x) if isinstance(x, (int, float)) else x
                )
                st.dataframe(df_comparison, height=350, use_container_width=True)

                # ====================== Future Prediction Section ======================
                st.divider()
                st.markdown("<h5>Prediksi Harga Saham</h5>", unsafe_allow_html=True)

                # Input jumlah hari prediksi
                future_days = st.number_input("Masukkan Hari: (1-7 hari)", min_value=1, max_value=7, value=1, step=1)
                predict_button = st.button("Prediksi RNN dan LSTM")

                if predict_button:
                    with st.spinner("🚀 Memprediksi masa depan..."):

                        # Load model dan scaler
                        model_rnn = load_best_model("RNN")
                        model_lstm = load_best_model("LSTM")
                        scaler_rnn = st.session_state.get("scaler_RNN")
                        scaler_lstm = st.session_state.get("scaler_LSTM")

                        if model_rnn and model_lstm and scaler_rnn and scaler_lstm:
                            # Ambil dan skalakan 30 hari terakhir
                            last_30_days = df[close_col][-30:].values.reshape(-1, 1)
                            scaled_rnn = scaler_rnn.transform(last_30_days)
                            scaled_lstm = scaler_lstm.transform(last_30_days)

                            pred_scaled_rnn, pred_scaled_lstm = [], []

                            # Fungsi untuk prediksi iteratif
                            def predict_future(model, scaled_data, future_days):
                                predictions = []
                                for _ in range(future_days):
                                    x_input = np.array(scaled_data[-30:]).reshape(1, 30, 1)
                                    pred = model.predict(x_input, verbose=0)
                                    predictions.append(pred[0][0])
                                    scaled_data = np.append(scaled_data[1:], pred[0][0])
                                return predictions

                            # Prediksi
                            pred_scaled_rnn = predict_future(model_rnn, scaled_rnn, future_days)
                            pred_scaled_lstm = predict_future(model_lstm, scaled_lstm, future_days)

                            # Kembalikan ke skala asli
                            pred_rnn = scaler_rnn.inverse_transform(np.array(pred_scaled_rnn).reshape(-1, 1))
                            pred_lstm = scaler_lstm.inverse_transform(np.array(pred_scaled_lstm).reshape(-1, 1))

                            # Buat DataFrame hasil prediksi
                            def create_prediction_df(model_name, predictions):
                                return pd.DataFrame({
                                    "Prediksi Hari Ke": [f"Hari ke-{i+1}" for i in range(future_days)],
                                    f"Predicted Close ({model_name})": [f"{p:,.2f}" for p in predictions.flatten()]
                                })

                            df_future_rnn = create_prediction_df("RNN", pred_rnn)
                            df_future_lstm = create_prediction_df("LSTM", pred_lstm)
                            df_combined_future = pd.merge(df_future_rnn, df_future_lstm, on="Prediksi Hari Ke")
                            df_combined_future.index = range(1, len(df_combined_future) + 1)

                            # Tampilkan hasil
                            st.success(f"✅ Berhasil memprediksi {future_days} hari ke depan menggunakan model RNN dan LSTM!")
                            st.dataframe(df_combined_future)

                            # Persiapan visualisasi
                            df.index = pd.to_datetime(df.index)
                            test_data = test_data[close_col].values  
                            test_dates = df.index[-len(test_data):].strftime('%Y-%m-%d').values
                            last_date = df.index[-1]
                            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, future_days + 1)]

                            df_test = pd.DataFrame({"Tanggal": test_dates, "Harga Aktual": test_data})
                            df_pred_rnn = pd.DataFrame({"Tanggal": future_dates, "Prediksi RNN": pred_rnn.flatten()})
                            df_pred_lstm = pd.DataFrame({"Tanggal": future_dates, "Prediksi LSTM": pred_lstm.flatten()})
                            df_pred_rnn["Prediksi Hari Ke"] = df_pred_lstm["Prediksi Hari Ke"] = [f"{i+1}" for i in range(future_days)]

                            # Plot
                            fig = go.Figure()

                            fig.add_trace(go.Scatter(
                                x=df_test["Tanggal"], y=df_test["Harga Aktual"],
                                mode='lines', name='Data Aktual', line=dict(color='#636EFA', width=1),
                                hovertemplate='Date: %{x}<br>Close: %{y:,.2f}<extra></extra>'
                            ))
                            fig.add_trace(go.Scatter(
                                x=df_pred_rnn["Tanggal"], y=df_pred_rnn["Prediksi RNN"],
                                mode='lines', name=f'Prediksi RNN {future_days} Hari', line=dict(color='red', width=1),
                                customdata=df_pred_rnn["Prediksi Hari Ke"],
                                hovertemplate='<br>Prediksi Hari Ke: %{customdata}<br>Close: %{y:,.2f}<extra></extra>'
                            ))
                            fig.add_trace(go.Scatter(
                                x=df_pred_lstm["Tanggal"], y=df_pred_lstm["Prediksi LSTM"],
                                mode='lines', name=f'Prediksi LSTM {future_days} Hari', line=dict(color='#33C1FF', width=1),
                                customdata=df_pred_lstm["Prediksi Hari Ke"],
                                hovertemplate='<br>Prediksi Hari Ke: %{customdata}<br>Close: %{y:,.2f}<extra></extra>'
                            ))

                            fig.update_layout(
                                title=f"Harga Aktual dan Prediksi {future_days} Hari Ke Depan Menggunakan Model RNN dan LSTM",
                                xaxis_title="Tanggal", yaxis_title="Harga Saham",
                                legend=dict(x=1.05, y=1, xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0)', bordercolor='Black'),
                                template='plotly_white',
                                height=500
                            )

                            st.plotly_chart(fig, use_container_width=True)                        
                            
    # ====================== COMPARISON ======================
    elif selected_tab == "Comparison":
        st.markdown(
            f"<h3>🔄 Perbandingan Harga Saham: ({start_date} - {end_date})</h3>", 
            unsafe_allow_html=True
        )
        st.markdown("<hr style='border: 1px solid #333; margin:-1px 0;'>", unsafe_allow_html=True)

        if len(selected_stocks_comparison) < 2:
            st.warning("⚠️ Pilih setidaknya dua saham untuk perbandingan!")
        else:
            tickers = selected_stocks_comparison
            with st.spinner("🔍 Mengambil data saham..."):
                data = yf.download(tickers, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))['Close']

            if data.empty:
                st.warning("⚠️ Tidak ada data yang tersedia untuk saham yang dipilih.")
            else:
                # Buat Figure
                fig = go.Figure()
                for ticker in tickers:
                    fig.add_trace(go.Scatter(x=data.index, 
                                            y=data[ticker], 
                                            mode='lines', 
                                            name=ticker,
                                            hovertemplate="📅 Date: %{x|%Y-%m-%d}<br> Close: %{y:,.2f}",
                                            text=ticker,
                                            line=dict(width=1)
                                          ))
                # Layout
                fig.update_layout(title='Perbandingan Harga Saham',
                    xaxis_title='Tanggal',
                    yaxis_title='Harga Saham',               
                    template="plotly_dark",
                    width=1100, 
                    height=700,
                    hovermode="x",
                    xaxis=dict(
                        showgrid=False, gridcolor="rgba(200, 200, 200, 0.4)",
                        showline=False, linewidth=1, linecolor="grey",
                        rangeslider=dict(visible=True, bgcolor="rgba(255,255,255,0.1)"),
                        rangeselector=dict(
                            buttons=[
                                dict(count=1, label="1D", step="day", stepmode="backward"),
                                dict(count=7, label="1W", step="day", stepmode="backward"),
                                dict(count=1, label="1M", step="month", stepmode="backward"),
                                dict(count=1, label="1Y", step="year", stepmode="backward"),
                                dict(step="all", label="ALL")
                            ]
                        )
                    ),
                    yaxis=dict(
                        showgrid=True, gridcolor="rgba(200, 200, 200, 0.4)",
                        zeroline=True, zerolinewidth=1, zerolinecolor="rgba(150, 150, 150, 0.2)", 
                        fixedrange=False
                    ),  
                    dragmode="zoom",
                    margin=dict(l=50, r=50, t=100, b=50)
                )
                st.plotly_chart(fig, use_container_width=True)

# Copyright 
st.markdown("""
    <div style='position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
                font-size: 14px; color: #A9A9A9; text-align: center; width: 100%;'>
        © 2025 DeepStock Prediction AI. All Rights Reserved.
    </div>
""", unsafe_allow_html=True)

# ====================== JALANKAN APLIKASI ======================
if __name__ == "__main__":
    main()