import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import base64
import time
import json
from datetime import datetime

# Fungsi untuk menyimpan record maintenance
def save_maintenance_records(records):
    with open('maintenance_records.json', 'w') as f:
        json.dump(records, f)

# Fungsi untuk memuat record maintenance
def load_maintenance_records():
    try:
        with open('maintenance_records.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Fungsi untuk memuat data dari file yang diunggah
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col="Date")  # Muat file CSV yang diunggah
        df.index = pd.to_datetime(df.index)  # Konversi index ke datetime
        return df
    else:
        return None

# Fungsi untuk memberikan link download file CSV
def get_table_download_link(df):
    csv = df.to_csv(index=True)  # Pastikan kolom Date disertakan
    b64 = base64.b64encode(csv.encode()).decode()  # Encode dalam format base64
    href = f'<a href="data:file/csv;base64,{b64}" download="updated_data.csv">Download updated CSV file</a>'
    return href

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard-Preventive Maintenance PLTU Gunung Salak",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul halaman dashboard
st.title("Preventive Maintenance Dashboard")

# Bagian sidebar
with st.sidebar:
    st.title('📈 Dashboard-Preventive Maintenance PLTU Gunung Salak')

    # Opsi untuk mengunggah file CSV
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    # Model selection
    selected_model = 'Model_lstm.h5'  # Using single LSTM model

    # Slider untuk pengaturan
    n_day = st.slider("Days of prediction :", 1, 30)
    sample = st.slider("Sample :", 1, 30)
    
    # Tambahkan pengaturan threshold
    st.subheader("Threshold Settings")
    enable_thresholds = st.checkbox("Enable Safety Thresholds", value=True)
    if enable_thresholds:
        col1, col2 = st.columns(2)
        with col1:
            threshold_percent_upper = st.number_input("Upper Threshold (%)", min_value=1, max_value=100, value=10)
        with col2:
            threshold_percent_lower = st.number_input("Lower Threshold (%)", min_value=1, max_value=100, value=10)
    
    check_box = st.checkbox(label="Display Table of Prediction")

# Fungsi untuk melakukan prediksi
def prediction(uploaded_file, selected_model, n_day, sample):
    df = load_data(uploaded_file)
    if df is None:
        st.warning("Please upload a CSV file.")
        return

    data = df.filter(['Close'])
    dataset = data.values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Dapatkan jumlah data untuk pelatihan
    training_data_len = int(np.ceil(len(dataset) * .95))

    # Muat model yang dipilih
    model = load_model(selected_model)

    test_data = scaled_data[training_data_len - 60:, :]

    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Siapkan data latih dan valid
    train = data[:training_data_len]
    valid = data[training_data_len:].copy()
    valid['Predictions'] = predictions

    # Prediksi untuk hari berikutnya
    last_60_days = scaled_data[-60:]
    for day in range(n_day):
        x_future = np.array([last_60_days])
        x_future = np.reshape(x_future, (x_future.shape[0], x_future.shape[1], 1))

        # Prediksi untuk hari berikutnya
        predicted_value = model.predict(x_future)
        predicted_value = scaler.inverse_transform(predicted_value)

        # Tambahkan prediksi ke dataset valid sebagai prediksi masa depan
        next_date = valid.index[-1] + pd.DateOffset(1)  # Tambah 1 hari
        valid.loc[next_date] = [np.nan, predicted_value[0][0]]  # Append predicted value

        # Update last_60_days dengan prediksi terbaru
        new_value_scaled = scaler.transform(predicted_value)
        last_60_days = np.append(last_60_days[1:], new_value_scaled, axis=0)  # Shift window

    # Plot hasil
    fig = go.Figure()

    # Plot data latih
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Train Close', line=dict(color='blue')))

    # Plot data valid
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], mode='lines', name='Valid Close', line=dict(color='red')))

    # Plot prediksi
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predictions', line=dict(color='green')))

    # Tambahkan threshold lines jika diaktifkan
    if enable_thresholds:
        # Hitung nilai threshold berdasarkan data terakhir
        last_actual_price = valid['Close'].dropna().iloc[-1]
        upper_threshold = last_actual_price * (1 + threshold_percent_upper/100)
        lower_threshold = last_actual_price * (1 - threshold_percent_lower/100)

        # Tambahkan garis threshold
        fig.add_hline(y=upper_threshold, line_dash="dash", line_color="red", 
                     annotation_text=f"Upper Threshold (+{threshold_percent_upper}%)")
        fig.add_hline(y=lower_threshold, line_dash="dash", line_color="red",
                     annotation_text=f"Lower Threshold (-{threshold_percent_lower}%)")

    # Update layout
    fig.update_layout(
        title='Parameter pump Prediction with Safety Thresholds',
        width=1000,
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
        xaxis_title='Date',
        yaxis_title='Value',
        legend_title='Legend'
    )

    # Tampilkan grafik di Streamlit
    st.plotly_chart(fig)

    # Tampilkan tabel hasil prediksi
    if check_box:
        st.dataframe(valid)
        st.markdown(get_table_download_link(valid), unsafe_allow_html=True)
    else:
        st.dataframe(df, width=1000, height=500)
        st.markdown(get_table_download_link(df), unsafe_allow_html=True)

    # Tampilkan informasi threshold jika diaktifkan
    if enable_thresholds:
        st.subheader("Threshold Information")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Upper Safety Threshold: {upper_threshold:.2f}")
        with col2:
            st.info(f"Lower Safety Threshold: {lower_threshold:.2f}")
        
        # Tambahkan sistem peringatan untuk threshold
        st.subheader("Threshold Alerts")
        
        # Cek prediksi terakhir
        latest_predictions = valid['Predictions'].tail(n_day)
        latest_date = valid.index[-1]
        
        # Container untuk alert
        alert_container = st.container()
        
        with alert_container:
            # Cek setiap prediksi untuk threshold violations
            for date, pred_value in zip(latest_predictions.index[-n_day:], latest_predictions[-n_day:]):
                if pred_value > upper_threshold:
                    st.error(f"⚠️ WARNING: Predicted value ({pred_value:.2f}) on {date.strftime('%Y-%m-%d')} exceeds upper threshold ({upper_threshold:.2f})!")
                elif pred_value < lower_threshold:
                    st.warning(f"⚠️ ALERT: Predicted value ({pred_value:.2f}) on {date.strftime('%Y-%m-%d')} is below lower threshold ({lower_threshold:.2f})!")
            
            # Tambahkan ringkasan status
            if any(latest_predictions > upper_threshold):
                st.error("🔴 Critical: Some predictions exceed upper safety threshold!")
            elif any(latest_predictions < lower_threshold):
                st.warning("🟡 Warning: Some predictions are below lower safety threshold!")
            else:
                st.success("✅ All predictions are within safe threshold limits.")

# Panggil fungsi prediksi
prediction(uploaded_file, selected_model, n_day, sample)

# Tambahkan section untuk maintenance record
st.markdown("---")
st.header("Maintenance Records")

# Load existing records
maintenance_records = load_maintenance_records()

# Form untuk menambah record maintenance baru
with st.form("maintenance_form"):
    st.subheader("Add New Maintenance Record")
    maintenance_date = st.date_input("Maintenance Date", datetime.today())
    maintenance_type = st.selectbox("Maintenance Type", 
        ["Preventive Maintenance", "Corrective Maintenance", "Breakdown Maintenance"])
    component = st.text_input("Component Name")
    description = st.text_area("Maintenance Description")
    technician = st.text_input("Technician Name")
    status = st.selectbox("Status", ["Completed", "In Progress", "Scheduled"])
    
    submit_button = st.form_submit_button("Add Record")
    
    if submit_button:
        new_record = {
            "date": maintenance_date.strftime("%Y-%m-%d"),
            "type": maintenance_type,
            "component": component,
            "description": description,
            "technician": technician,
            "status": status
        }
        maintenance_records.append(new_record)
        save_maintenance_records(maintenance_records)
        st.success("Maintenance record added successfully!")

# Tampilkan tabel maintenance records
if maintenance_records:
    st.subheader("Maintenance History")
    df_maintenance = pd.DataFrame(maintenance_records)
    df_maintenance['date'] = pd.to_datetime(df_maintenance['date'])
    df_maintenance = df_maintenance.sort_values('date', ascending=False)
    
    # Filter records
    col1, col2 = st.columns(2)
    with col1:
        filter_type = st.multiselect("Filter by Type", 
            options=df_maintenance['type'].unique(),
            default=df_maintenance['type'].unique())
    with col2:
        filter_status = st.multiselect("Filter by Status",
            options=df_maintenance['status'].unique(),
            default=df_maintenance['status'].unique())
    
    filtered_df = df_maintenance[
        (df_maintenance['type'].isin(filter_type)) &
        (df_maintenance['status'].isin(filter_status))
    ]
    
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download maintenance records
    csv = filtered_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="maintenance_records.csv">Download Maintenance Records</a>'
    st.markdown(href, unsafe_allow_html=True)
else:
    st.info("No maintenance records found. Add your first record above.")

