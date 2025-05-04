import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio

@st.cache_data
def load_data(ticker, start_date, end_date):
    def get_additional_info(ticker):
        try:
            stock_ticker = yf.Ticker(ticker)
            stock_info = stock_ticker.info  
            return {
                "Market Cap": stock_info.get("marketCap"),
                "52-Week High": stock_info.get("fiftyTwoWeekHigh"),
                "52-Week Low": stock_info.get("fiftyTwoWeekLow"),
                "P/E Ratio": stock_info.get("trailingPE"),
                "EPS": stock_info.get("trailingEps"),
                "Dividend Yield": stock_info.get("dividendYield"),
                "Revenue": stock_info.get("totalRevenue"),
                "Net Income": stock_info.get("netIncomeToCommon"),
                "Profit Margin": stock_info.get("profitMargins"),
                "ROE": stock_info.get("returnOnEquity"),
            }
        except Exception as e:
            st.error(f"⚠️ Tidak bisa mengambil informasi tambahan untuk {ticker}: {e}")
            return {}

     # Jika ticker adalah BBRI.JK, maka ambil dari file lokal
    if ticker == "BBRI.JK":
        try:
            df = pd.read_excel("datasets/BBRI_2010-2025.xlsx")
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
            df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]
            df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

            additional_info = get_additional_info(ticker)
            return df, additional_info

        except Exception as e:
            st.error(f"❌ Gagal membaca file lokal BBRI: {str(e)}")
            return None, None

    # Jika bukan BBRI.JK, ambil data dari Yahoo Finance
    try:
        stock = yf.download(ticker, start=start_date, end=end_date)

        if stock.empty:
            all_data = yf.download(ticker, period="max")
            if all_data.empty:
                st.error(f"⚠️ Tidak ada data historis untuk {ticker}!")
                return None, None

            nearest_date = all_data.index[all_data.index.get_indexer([pd.to_datetime(start_date)], method='nearest')[0]]
            stock = all_data.loc[[nearest_date]]

        stock.index.name = "Date"
        stock.reset_index(inplace=True)

        available_columns = ['Date', 'Open', 'Low', 'High', 'Close', 'Volume']
        if 'Adj Close' in stock.columns:
            available_columns.append('Adj Close')

        stock = stock[[col for col in available_columns if col in stock.columns]]
        stock["Date"] = pd.to_datetime(stock["Date"]).dt.strftime("%Y-%m-%d")
        stock = stock.sort_values("Date")

        additional_info = get_additional_info(ticker)
        return stock, additional_info

    except Exception as e:
        st.error(f"❌ Gagal mengambil data untuk {ticker}: {str(e)}")
        return None, None
    
def validation_input(selected_stock, start_date, end_date):
    """Validation input yang diberikan pengguna."""
    
    # Validasi jika saham belum dipilih
    if selected_stock == "-- Pilih Saham --":
        st.info("❗ Silahkan pilih saham sebelum melanjutkan!")
        st.stop()

    # Validasi rentang tanggal salah
    if start_date > end_date:
        st.warning("⚠️ Silahkan pilih rentang tanggal yang valid. Tanggal akhir harus setelah tanggal mulai!")
        st.stop()

def plot_data(data):
    pio.templates.default = "plotly_dark"  
    data.columns = data.columns.map(lambda x: x[0] if isinstance(x, tuple) else x)  
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data['Date'], 
        y=data['Close'], 
        mode='lines',
        line=dict(color='#9966CC', width=2), 
        name="Close Price"
    ))
    
    hover_text = "<b>📅 Date:</b> %{x|%Y-%m-%d}<br>"  
    hover_text += "<b> Close:</b> %{y:,.2f}<br>"  
    if "Open" in data.columns:
        hover_text += "<b> Open:</b> %{customdata[0]:,.2f}<br>"  
    if "High" in data.columns:
        hover_text += "<b> High:</b> %{customdata[1]:,.2f}<br>"  
    if "Low" in data.columns:
        hover_text += "<b> Low:</b> %{customdata[2]:,.2f}<br>" 
    if "Volume" in data.columns:
        hover_text += "<b> Volume:</b> %{customdata[3]:,.0f}"  

    # Customdata agar informasi ditampilkan di hover
    custom_data_cols = [col for col in ["Open", "High", "Low", "Volume"] if col in data.columns]
    fig.update_traces(
        line=dict(width=1),
        hoverinfo="x+y",
        hovertemplate=hover_text,
        customdata=data[custom_data_cols].values if custom_data_cols else None
    )
    
    # Layout
    fig.update_layout(
        title=" Harga Saham Sepanjang Waktu",
        xaxis_title="Tanggal",
        yaxis_title="Harga Saham ",
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
        margin=dict(l=50, r=50, t=100, b=50), 
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_volume(data):
    """Menampilkan grafik volume saham"""
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Date'], 
        y=data['Volume'], 
        mode='lines',
        line=dict(color='#9966CC', width=1), 
        name="Volume",
        hovertemplate='%{y:,.0f}'
    ))

    fig.update_layout(
        title=" Volume Saham Sepanjang Waktu",
        height=520,  
        xaxis=dict(
            title="Tanggal",
            showgrid=False,
            gridcolor="rgba(200, 200, 200, 0.4)",
            showline=False,
            linewidth=0.4,
            linecolor="grey",
            rangeslider=dict(visible=True, bgcolor="rgba(255,255,255,0.1)"), 
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1D", step="day", stepmode="backward"),
                    dict(count=7, label="1W", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL")
                ]
            ),
        ),
        yaxis=dict(
            title="Volume Perdagangan",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.4)",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="rgba(150, 150, 150, 0.2)", 
            fixedrange=False,
        ),
        hovermode="x unified",
        dragmode="zoom",
    )
    st.plotly_chart(fig, use_container_width=True)

def format_value(value, is_percentage=False):
    if value is None or value == "N/A":  
        return "-"
    try:
        value = float(value)
        if is_percentage:
            return f"{value * 100:.2f}%"  
        if value >= 1_000_000_000_000:
            return f"{value / 1_000_000_000_000:,.2f}T"
        elif value >= 1_000_000_000:
            return f"{value / 1_000_000_000:,.2f}B"
        elif value >= 1_000_000:
            return f"{value / 1_000_000:,.2f}M"
        else:
            return f"{value:,.2f}"
    except ValueError:
        return "-"

def statistics(data, stock_info, selected_stock):
    """Menghitung dan menampilkan statistik harga saham serta informasi tambahan."""
    
    if data is None or data.empty:
        st.warning("⚠️ Data tidak tersedia.")
        return

    # Menghitung statistik harga saham
    highest_price = float(data["High"].max())
    lowest_price = float(data["Low"].min())
    avg_price = float(data["Close"].mean())
    close_price = float(data["Close"].iloc[-1])
    open_price = float(data["Open"].iloc[0])
    price_change = ((close_price - open_price) / open_price) * 100
    avg_volume = float(data["Volume"].mean())
    max_volume = float(data["Volume"].max())

    # Menampilkan statistik saham
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="⭑ Harga Tertinggi", value=f"{highest_price:,.2f}")
        st.metric(label="⭑ Harga Terendah", value=f"{lowest_price:,.2f}")
        st.metric(label="⭑ Harga Rata-rata", value=f"{avg_price:,.2f}")

    with col2:
        st.metric(label="⭑ Harga Pembukaan", value=f"{open_price:,.2f}")
        st.metric(label="⭑ Harga Penutupan Terakhir", value=f"{close_price:,.2f}")
        st.metric(label="⭑ Perubahan Harga (%)", value=f"{price_change:,.2f} %")
        

    with col3:
        st.metric(label="⭑ Rata-rata Volume", value=format_value(avg_volume))
        st.metric(label="⭑ Volume Tertinggi", value=format_value(max_volume))
    
    # Menampilkan informasi saham 
    if stock_info:
        st.markdown(f"<h3>🏛️ Informasi Saham: <span style='color:#9966CC; font-weight:bold;'>{selected_stock}</span></h3>", unsafe_allow_html=True)
        st.markdown("<hr style='border: 1px solid #333; margin:-1px 0;'>", unsafe_allow_html=True)
        
        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric(label="⭑ 52-Week Low", value=format_value(stock_info.get("52-Week Low")))
            st.metric(label="⭑ Dividend Yield", value=format_value(stock_info.get("Dividend Yield")))
            st.metric(label="⭑ Profit Margin", value=format_value(stock_info.get("Profit Margin"), is_percentage=True))
            st.metric(label="⭑ Return on Equity (ROE)", value=format_value(stock_info.get("ROE"), is_percentage=True))

        with col5:
            st.metric(label="⭑ 52-Week High", value=format_value(stock_info.get("52-Week High")))
            st.metric(label="⭑ P/E Ratio", value=format_value(stock_info.get("P/E Ratio")))
            st.metric(label="⭑ Earnings Per Share (EPS)", value=format_value(stock_info.get("EPS")))

        with col6:
            st.metric(label="⭑ Market Cap", value=format_value(stock_info.get("Market Cap")))
            st.metric(label="⭑ Revenue", value=format_value(stock_info.get("Revenue")))
            st.metric(label="⭑ Net Income", value=format_value(stock_info.get("Net Income")))
            