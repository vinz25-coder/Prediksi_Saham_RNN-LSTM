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

# buat sidebar
def sidebar():
    st.sidebar.markdown("<h1 style='text-align: center; font-size: 30px;'>"
                        "<b>Deep</b><b style='color: #9966CC;'>Stock</b></h1>", 
                        unsafe_allow_html=True)
    st.sidebar.title("Menu")

    # Pilihan saham
    stocks = ["BBCA.JK", "BBRI.JK"]
    
    # Sidebar dan menyimpan selected_stock dalam session_state jika belum ada
    if "selected_stock" not in st.session_state:
        st.session_state["selected_stock"] = "-- Pilih Saham --"
    selected_stock = st.sidebar.selectbox(
        "Pilih Saham untuk Prediksi", 
        ["-- Pilih Saham --"] + stocks, 
        index=(["-- Pilih Saham --"] + stocks).index(st.session_state["selected_stock"])
    )
    st.session_state["selected_stock"] = selected_stock
    
    # Set start_date jika belum ada di session_state dan memilih start_date
    if "start_date" not in st.session_state:
        st.session_state["start_date"] = date.today()
    start_date = st.sidebar.date_input(
        "Start Date",
        value=st.session_state["start_date"],  
        min_value=date(2010, 1, 1),
        max_value=date.today()
    )

    # Set end_date ke tanggal hari ini set default ke tanggal hari ini
    if "end_date" not in st.session_state:
        st.session_state["end_date"] = date.today() 
    end_date = st.sidebar.date_input(
        "End Date",
        value=st.session_state["end_date"],
        min_value=start_date,  
        max_value=date.today()  
    )
    st.session_state["end_date"] = end_date  

    selected_stocks_comparison = st.sidebar.multiselect("Pilih Saham untuk Perbandingan", stocks)

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
        "Deep<span style='color: #9966CC;'>Stock</span> Forecast AI 📈"
        "</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center;'>"
        "<span style='font-weight: bold;'>Deep</span>"
        "<span style='color: #9966CC; font-weight: bold;'>Stock</span> "
        "adalah aplikasi web sederhana prediksi harga saham menggunakan RNN & LSTM."
        "</p>",
        unsafe_allow_html=True
    )
    
    # ====================== MENU NAVIGASI ======================
    selected_tab = option_menu(
        menu_title=None,
        options=["Dataframes", "Plots", "Statistics", "Forecasting", "Comparison"],
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
            f"<h3>📆 Historical Data: <span style='color:#9966CC; font-weight:bold;'>{selected_stock}</span> ({start_date} - {end_date})</h3>", unsafe_allow_html=True)
        st.markdown("<hr style='border: 1px solid #333; margin:-1px 0;'>", unsafe_allow_html=True)
        # Ubah angka menjadi format string dengan 2 desimal
        data = data.applymap(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
        df = pd.DataFrame(data)
        st.dataframe(df)
       
     # ====================== TAMPILKAN CHART PADA "PLOTS" ======================
    elif selected_tab == "Plots":
        st.markdown(
            f"<h3>📊 Plotting Chart: <span style='color:#9966CC; font-weight:bold;'>{selected_stock}</span> ({start_date} - {end_date})</h3>", 
        unsafe_allow_html=True
        )
        st.markdown("<hr style='border: 1px solid #333; margin:-1px 0;'>", unsafe_allow_html=True)
        plot_data(data)  
        plot_volume(data)  

    # ====================== Statistics ======================
    elif selected_tab == "Statistics":
        st.markdown(
            f"<h3>🔢 Statistics: <span style='color:#9966CC; font-weight:bold;'>{selected_stock}</span> ({start_date} - {end_date})</h3>", 
            unsafe_allow_html=True
        )
        st.markdown("<hr style='border: 1px solid #333; margin:-1px 0;'>", unsafe_allow_html=True)
        statistics(data, stock_info, selected_stock)
    
    # ======================  Forecasting ======================
    elif selected_tab == "Forecasting":
        def display_prediction(selected_stock, start_date, end_date):
            """Menampilkan judul prediksi harga saham."""
            st.markdown(
                f"<h3>🎯 Prediksi Harga Saham: <span style='color:#9966CC;'>{selected_stock}</span> ({start_date} - {end_date})</h3>",  
                unsafe_allow_html=True
            )
            st.markdown("<hr style='border: 1px solid #333; margin:-1px 0;'>", unsafe_allow_html=True)
    
        def load_stock_data(selected_stock, start_date, end_date):
            """Mengambil dan memproses data saham."""
            with st.spinner("📊 Mengambil Data..."):
                df = prediction(selected_stock, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                if df is None or df.empty:
                    st.warning("⚠️ Data tidak ditemukan!")
                    st.stop()
                fig = plot_data_split(df) 
                st.divider() 
                return df
        
        def preprocess_and_display_data(df):
            """Melakukan preprocessing dan menampilkan data pelatihan."""
            st.markdown("<h5> Pembentukan Sliding Window</h5>", unsafe_allow_html=True)

            train_data, val_data, test_data = split_data(df)
            data = preprocess_data(df, len(train_data), len(val_data), len(test_data), time_step=30)
            
            # Normalisasi
            x_train_df = pd.DataFrame(data['x_train'][:, :, 0]).applymap(lambda x: f"{x:.4f}")
            x_train_df["y_train"] = data['y_train'].flatten()
            
            st.markdown("**🔍 Scaled Training Data (First 10 Rows)**")
            st.dataframe(x_train_df.head(10))
            
            # Denormalisasi
            x_train_original = data['scaler'].inverse_transform(data['x_train'][:, :, 0])
            y_train_original = data['scaler'].inverse_transform(data['y_train'])
            df_x_train_original = pd.DataFrame(x_train_original).applymap(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if isinstance(x, (int, float)) else x)
            df_x_train_original["y_train"] = y_train_original
            
            st.markdown("**🔍 Denormalized Training Data (First 10 Rows)**", unsafe_allow_html=True)
            st.dataframe(df_x_train_original.head(10))
            st.divider() 
            return train_data, val_data, test_data, data
       
        def display_model(model_name):
            """Menampilkan model dan hasil training."""
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
            """Melakukan prediksi harga saham dengan model yang dipilih."""
            x_test, y_test, dates = data['x_test'], data['y_test'], data['dates']
            test_len = len(y_test)
            
            y_pred = model.predict(x_test)
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            
            df_pred = pd.DataFrame({
                "Close": y_pred
            }, index=pd.to_datetime(dates[-len(y_pred):]))
            
            return df_pred
        
        def evaluate_model(y_true, y_pred):
            """Menghitung MSE, RMSE, dan MAE untuk model."""
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            return mse, rmse, mae

        def plot_stock_data(train, val, test, pred):
            # Mendeteksi kolom harga penutupan
            close_col = next((col for col in train.columns if "Close" in col), None)
            
            if not close_col:
                st.error("❌ Tidak ditemukan kolom harga penutupan dalam dataset!")
                return None
            
            # Pastikan pred_df menggunakan kolom yang sama
            pred_col = "Close" if "Close" in pred.columns else close_col  

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=train.index, y=train[close_col], mode='lines', name='Training', line=dict(color='#9966CC')))
            fig.add_trace(go.Scatter(x=val.index, y=val[close_col], mode='lines', name='Validation', line=dict(color='gold')))
            fig.add_trace(go.Scatter(x=test.index, y=test[close_col], mode='lines', name='Testing', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=pred.index, y=pred[pred_col], mode='lines', name='Prediksi', line=dict(color='#00CED1', dash='dot')))
            
            # Tambahkan kustomisasi tampilan
            fig.update_traces(
                line=dict(width=2), 
                hoverinfo="x+y", 
                hovertemplate="📅 Tanggal: %{x|%Y-%m-%d}<br> Close Price: %{y:,.2f}"
            )
            fig.update_layout(
                title="Visualisasi Training, Validation, Testing, dan Prediksi",
                xaxis_title="Date",
                yaxis_title="Close Price",
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

            # Fokus hanya pada data Testing dan Prediksi
            fig.add_trace(go.Scatter(x=test.index, y=test[close_col], mode='lines', name='Testing', line=dict(color='red', width=3)))
            fig.add_trace(go.Scatter(x=pred.index, y=pred[pred_col], mode='lines', name=f'Prediksi {model_name}', line=dict(color='#00CED1', width=4, dash='dot')))
            
            # Tambahkan kustomisasi tampilan
            fig.update_traces(
                line=dict(width=2), 
                hoverinfo="x+y", 
                hovertemplate="📅 Tanggal: %{x|%Y-%m-%d}<br> Close Price: %{y:,.2f}"
            )
            fig.update_layout(
                title=f"Prediksi pada Model {model_name}",
                xaxis_title="Date",
                yaxis_title="Close Price",
                height=650,
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
        
        def plot_comparison(test, pred_rnn, pred_lstm):
            close_col = next((col for col in test.columns if "Close" in col), None)

            if not close_col:
                st.error("❌ Tidak ditemukan kolom harga penutupan dalam dataset!")
                return None
            
            # Pastikan prediksi juga menggunakan kolom yang sama
            pred_col_rnn = "Close" if "Close" in pred_rnn.columns else close_col
            pred_col_lstm = "Close" if "Close" in pred_lstm.columns else close_col

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=test.index, y=test[close_col], mode='lines', name='Testing', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=pred_rnn.index, y=pred_rnn[pred_col_rnn], mode='lines', name='Prediksi RNN', line=dict(color='Yellow', dash='dot')))
            fig.add_trace(go.Scatter(x=pred_lstm.index, y=pred_lstm[pred_col_lstm], mode='lines', name='Prediksi LSTM', line=dict(color='DeepSkyBlue', dash='dot')))

            # Tambahkan kustomisasi tampilan
            fig.update_traces(
                line=dict(width=2), 
                hoverinfo="x+y", 
                hovertemplate="📅 Tanggal: %{x|%Y-%m-%d}<br> Close Price: %{y:,.2f}"
            )
            fig.update_layout(
                title="RNN vs LSTM",
                xaxis_title="Date",
                yaxis_title="Close Price",
                height=650,
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

        # 1️⃣ Tampilkan Header
        display_prediction(selected_stock, start_date, end_date)

        # 2️⃣ Load Data Saham
        df = load_stock_data(selected_stock, start_date, end_date)
        
        # 3️⃣ Tampilkan Data yang Diproses
        train_data, val_data, test_data, data = preprocess_and_display_data(df)
        
        # 4️⃣ Simpan Scaler ke session_state
        st.session_state["scaler_RNN"] = data["scaler"]
        st.session_state["scaler_LSTM"] = data["scaler"]
        
        with st.container():
            st.markdown("<h5> Pilih Model yang Ingin Ditampilkan</h5>", unsafe_allow_html=True)
            model_option = st.radio(
                "Pilih Model :",
                options=["RNN", "LSTM", "RNN vs LSTM"],
                index=0,  
                horizontal=True
            )
            st.divider()
    
        # Menampilkan hasil prediksi berdasarkan pilihan radio button
        if model_option == "RNN":
            st.markdown("**🔍 Model Arsitektur dan Training Loss RNN**")
            
            with st.spinner("🔄 Prediksi harga dengan RNN..."):
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
                    fig_rnn = plot_stock_data(train_data, val_data, test_data, pred_rnn)
                    fig_rnn_focus = focus_testing(test_data, pred_rnn, model_name="RNN")
                    st.plotly_chart(fig_rnn)
                    st.plotly_chart(fig_rnn_focus)
                
        elif model_option == "LSTM":
            st.markdown("**🔍 Model Arsitektur dan Training Loss LSTM**")
            with st.spinner("🔄 Prediksi harga dengan LSTM..."):
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
                    fig_lstm = plot_stock_data(train_data, val_data, test_data, pred_lstm)
                    fig_lstm_focus = focus_testing(test_data, pred_lstm, model_name="LSTM")
                    st.plotly_chart(fig_lstm)
                    st.plotly_chart(fig_lstm_focus)
                    
        elif model_option == "RNN vs LSTM":
            if "pred_rnn" in st.session_state and "pred_lstm" in st.session_state:
                pred_rnn = st.session_state["pred_rnn"]
                pred_lstm = st.session_state["pred_lstm"]

                # ✅ Deteksi otomatis kolom harga penutupan
                close_col = next((col for col in test_data.columns if "Close" in col), "Close")

                # ✅ Rename prediksi jika kolomnya hanya 'Close' tapi test_data pakai 'Close_XXX'
                if "Close" in pred_rnn.columns and close_col != "Close":
                    pred_rnn = pred_rnn.rename(columns={"Close": close_col})
                if "Close" in pred_lstm.columns and close_col != "Close":
                    pred_lstm = pred_lstm.rename(columns={"Close": close_col})

                # Plot perbandingan prediksi
                fig_comparison = plot_comparison(test_data, pred_rnn, pred_lstm)
                st.plotly_chart(fig_comparison)

                # Evaluasi Model
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

                # Tabel perbandingan hasil prediksi
                with st.container():
                    st.divider()
                    st.markdown("**Hasil Prediksi RNN vs LSTM**", unsafe_allow_html=True)

                    df_comparison = pd.DataFrame({
                        "Date": test_data.index.strftime('%Y-%m-%d'),
                        "Actual Price": actual_prices,
                        "Predicted RNN": predicted_rnn,
                        "Predicted LSTM": predicted_lstm
                    })

                    # Format angka
                    df_comparison.iloc[:, 1:] = df_comparison.iloc[:, 1:].applymap(
                        lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                    )

                    st.dataframe(df_comparison, height=400, use_container_width=True)
                st.divider()



    # ====================== "Comparison" ======================
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
            with st.spinner("📊 Mengambil data saham..."):
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
                                            ))
                # Layout
                fig.update_layout(title='Perbandingan Harga Saham',
                    xaxis_title='Tanggal',
                    yaxis_title='Harga',               
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
        © 2025 Evindo Amanda. All Rights Reserved.
    </div>
""", unsafe_allow_html=True)

# ====================== JALANKAN APLIKASI ======================
if __name__ == "__main__":
    main()