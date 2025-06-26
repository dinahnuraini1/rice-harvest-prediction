import gdown
import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import time
import joblib
import os
import altair as alt
import tempfile

# Fungsi untuk memuat objek dari file pickle
# def load_pickle(file_path):
#     with open(file_path, 'rb') as file:
#         obj = pickle.load(file)
#     return obj


# STREAMLIT
def main():
    # Menampilkan gambar menggunakan st.image dengan pengaturan width
   
       
    # Sidebar Menu
    st.sidebar.image("panen.png", width=200) 
   # Menambahkan judul besar di sidebar
    st.sidebar.markdown("<h2 style='font-size: 24px;'> select menu</h2>", unsafe_allow_html=True)

    # Pilihan menu di sidebar tanpa label "Pilih Menu"
    menu = st.sidebar.selectbox(
        "",
        ["Home", "Load Data", "Preprocessing", "Random Forest Modelling","Random Forest + PSO Modelling", "Predictions"]
    )
    if menu == "Home":
        st.markdown("""
            <div style='text-align: center; padding: 50px 0;'>
                <h1 style='font-size: 50px; margin-bottom: 10px; color: #2E7D32;'>🌾 Welcome To 🌾</h1>
                <h2 style='font-size: 40px; margin-bottom: 20px;'>Rice Harvest Prediction System</h2>
                <h4 style='color: #555;'>Using <span style='color: #1E88E5;'>Random Forest Regression</span> + 
                <span style='color: #F9A825;'>Particle Swarm Optimization</span></h4>
            </div>
            <div style='text-align: left; padding: 0 40px; font-size: 18px;'>
            <p><strong>Sistem ini memiliki beberapa menu utama di antaranya:</strong></p>
            <ol>
                <li><strong>Load Data</strong> – Untuk memuat dataset kelompok tani dari Desa Mojorayung. Berformat Comma Separated Values (CSV) </li>
                <li><strong>Preprocessing</strong> – Untuk melakukan preprocessing data</li>
                <li><strong>Random Forest Modelling</strong> – Untuk melatih model menggunakan Random Forest </li>
                <li><strong>Random Forest Modelling + PSO </strong> – Untuk melatih model menggunakan Random Forest dan PSO </li>
                <li><strong>Predictions </strong> – Untuk memprediksi hasil panen berdasarkan input pengguna. Tersedia 6 fitur diantaranya:</li>
                    <ul>
                            <li>Luas Tanam (HA)</li>
                            <li>Pupuk Urea (KG)</li>
                            <li>Pupuk NPK (KG)</li>
                            <li>Pupuk Organik (KG)</li>
                            <li>Jumlah Bibit (KG)</li>
                            <li>Varietas Padi</li>
                    </ul>
            </ol>
            </div>
        """, unsafe_allow_html=True)

    
    # Inisialisasi session_state
    if "data" not in st.session_state:
        st.session_state["data"] = None
        st.session_state["selected_features"] = None
        st.session_state["X"] = None
        st.session_state["y"] = None
        st.session_state["scaler_X"] = None
        st.session_state["scaler_y"] = None

    if menu == "Load Data":
        st.header("1. Load Data")
        uploaded_file = st.file_uploader("Upload file CSV", type=['csv'])
        
        if uploaded_file is not None:
            # Baca data
            data = pd.read_csv(uploaded_file)
            
            # Ambil hanya kolom yang dibutuhkan
            required_columns = [
                'luas_tanam', 'urea', 'npk', 'organik',
                'jumlah_bibit', 'varietas', 'hasil_panen'
            ]
            missing_cols = [col for col in required_columns if col not in data.columns]
            if missing_cols:
                st.error(f"Kolom berikut tidak ditemukan dalam data: {', '.join(missing_cols)}")
            else:
                data = data[required_columns]
                st.session_state["data"] = data
                st.session_state["X"] = data.drop(columns=["hasil_panen"]).values
                st.session_state["y"] = data["hasil_panen"].values

                st.success("✅ Data berhasil dimuat.")
                st.write("Data yang diunggah:")
                st.dataframe(data.head())

                # Tambahkan grafik naik-turun hasil panen
                st.subheader("📈 Grafik Hasil Panen")

                # Pastikan index urut dari 0
                grafik_df = data.copy()
                grafik_df["Indeks"] = range(len(grafik_df))
                grafik_df = grafik_df.rename(columns={"hasil_panen": "Hasil Panen (Ton)"})

                # Buat grafik Altair
                chart = alt.Chart(grafik_df).mark_line(color='#88cc88', strokeWidth=3).encode(
                    x=alt.X('Indeks', title='Index Data'),
                    y=alt.Y('Hasil Panen (Ton)', title='Hasil Panen Padi (Ton)')
                ).properties(
                    width=700,
                    height=400
                )

                st.altair_chart(chart, use_container_width=True)


    elif menu == "Preprocessing":
        st.header("2. Preprocessing")

        if "data" not in st.session_state or st.session_state["data"] is None or st.session_state["data"].empty:
            st.warning("Harap upload data terlebih dahulu di menu 'Load Data'.")
        else:
            df = st.session_state["data"].copy()

            # === Fungsi Deteksi Outlier dengan IQR ===
            def detect_outliers_iqr(series):
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                return series[(series < lower_bound) | (series > upper_bound)]

            # === Fungsi Imputasi Mean secara Iteratif ===
            def imputasi_mean(data, column):
                while True:
                    outliers = detect_outliers_iqr(data[column])
                    if len(outliers) == 0:
                        break
                    mean_val = data[column][~data.index.isin(outliers.index)].mean()
                    data.loc[outliers.index, column] = mean_val
                return data

            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

            # === Ringkasan Awal (Sebelum Imputasi) ===
            st.markdown("###### 📊 Ringkasan Data Sebelum Preprocessing")

            null_count = df.isnull().sum()
            zero_count = (df[numeric_cols] == 0).sum()
            outlier_count = {col: len(detect_outliers_iqr(df[col])) for col in numeric_cols}

            summary_before = pd.DataFrame({
                "Data Kosong": null_count,
                "Data bernilai 0": zero_count,
                "Outlier": pd.Series(outlier_count)
            }).fillna(0).astype(int)

            st.dataframe(summary_before)

            # === Tangani Outlier dengan Imputasi Mean (Iteratif) ===
            df_outlier_handled = df.copy()

            for col in numeric_cols:
                df_outlier_handled = imputasi_mean(df_outlier_handled, col)

            # Simpan data yang telah dibersihkan dari outlier
            st.session_state["data"] = df_outlier_handled

            # === Ringkasan Setelah Penanganan Outlier ===
            st.markdown("###### 📊 Ringkasan Data Setelah Penanganan Outlier (dengan imputasi Mean)")

            null_after = df_outlier_handled.isnull().sum()
            zero_after = (df_outlier_handled[numeric_cols] == 0).sum()
            outlier_after = {col: len(detect_outliers_iqr(df_outlier_handled[col])) for col in numeric_cols}

            summary_after = pd.DataFrame({
                "Data Kosong": null_after,
                "Data bernilai 0": zero_after,
                "Outlier": pd.Series(outlier_after)
            }).fillna(0).astype(int)

            st.dataframe(summary_after)

            # ===== One-Hot Encoding untuk 'varietas' =====
            st.markdown("###### 📊 One-Hot Encoding untuk Kolom varietas")
            try:
                if "varietas" in df.columns:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded_data = encoder.fit_transform(df[["varietas"]])
                    feature_names = encoder.get_feature_names_out(["varietas"])
                    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)
                    df_encoded = pd.concat([df_outlier_handled.drop(columns=["varietas"]), encoded_df], axis=1)
                    # Simpan hasil encoding & encoder
                    st.session_state["df_processed"] = df_encoded
                    st.session_state["one_hot_encoders"] = {"varietas": encoder}
                    st.session_state["one_hot_encoded"] = True

                    # st.success("One-Hot Encoding berhasil dilakukan untuk kolom 'varietas'.")
                    st.dataframe(df_encoded.head())
                else:
                    st.warning("Kolom 'varietas' tidak ditemukan dalam data.")
                    df_encoded = df.copy()
            except Exception as e:
                st.error(f"Terjadi kesalahan saat One-Hot Encoding: {e}")
                df_encoded = df.copy()

            # ===== Normalisasi =====
            st.markdown(" ###### 📊Normalisasi Fitur dan Target dengan MinMax Scaler")
            try:
                df_normalized = df_encoded.copy()

                # Pisahkan kolom numerik, kecuali hasil_panen
                kolom_normalisasi = df_normalized.select_dtypes(include=['int64', 'float64']).columns.tolist()
                if 'hasil_panen' in kolom_normalisasi:
                    kolom_normalisasi.remove("hasil_panen")

                # Normalisasi X
                scaler_X = MinMaxScaler()
                df_normalized[kolom_normalisasi] = scaler_X.fit_transform(df_normalized[kolom_normalisasi])

                # Normalisasi y
                scaler_y = MinMaxScaler()
                if "hasil_panen" in df_normalized.columns:
                    df_normalized["hasil_panen"] = scaler_y.fit_transform(df_normalized[["hasil_panen"]])

                # Simpan hasil
                st.session_state["scaler_X"] = scaler_X
                st.session_state["scaler_y"] = scaler_y
                st.session_state["df_normalized"] = df_normalized
                st.session_state["normalized"] = True

                # Simpan X dan y
                st.session_state["X"] = df_normalized.drop(columns=["hasil_panen"])
                st.session_state["y"] = df_normalized["hasil_panen"]

                # st.success("Normalisasi berhasil dilakukan.")
                st.write("Data setelah Normalisasi:")
                st.dataframe(df_normalized.head())
                # === Grafik Hasil Panen Setelah Semua Preprocessing ===

                st.markdown("###### 📈 Grafik Hasil Panen Padi Setelah Preprocessing")
                df_chart = df_normalized.reset_index()[["index","hasil_panen"]]
                chart = alt.Chart(df_chart).mark_line(color="#6ABF69").encode(
                    x=alt.X("index",title="Index Data"),
                    y=alt.Y("hasil_panen", title="Hasil Panen Padi (Ton)")
                ).properties(
                    width=700,
                    height=400,
                )

                st.altair_chart(chart, use_container_width=True)

            except Exception as e:
                st.error(f"Terjadi kesalahan saat normalisasi: {e}")

                


    elif menu == "Random Forest Modelling":
        st.header("3. Random Forest Modelling")

        if "X" not in st.session_state or "y" not in st.session_state:
            st.warning("Harap lakukan preprocessing terlebih dahulu.")
        elif "normalized" not in st.session_state or not st.session_state["normalized"]:
            st.warning("⚠️ Harap lakukan normalisasi data terlebih dahulu sebelum melanjutkan ke pemodelan Random Forest.")
        else:
            # Mapping rasio -> test_size dan file model
            rasio_opsi = {
                "50:50": {"test_size": 0.5, "drive_id": "1Cybn-fysrB5iKi-C97cY1K1msrVHd9t0"},
                "60:40": {"test_size": 0.4, "drive_id": "1s18g6ejJHYHBvEG2TMVadPH2Jy25Aasn"},
                "70:30": {"test_size": 0.3, "drive_id": "1Wpui9thkor3rKsttbQpkMu67BM2ZigGL"}, 
                "80:20": {"test_size": 0.2, "drive_id": "1vp1nBE3DKBoGpojDRY4V53V4LFmx2P_i"},
                "90:10": {"test_size": 0.1, "drive_id": "1p0dtljJfEEoZTyEeMMywcgWKG0bJkMAB"},
            }

            # Pilihan rasio dari dropdown
            selected_rasio_label = st.selectbox("Pilih rasio data latih dan uji:", list(rasio_opsi.keys()))
            selected_rasio = rasio_opsi[selected_rasio_label]
            test_size = selected_rasio["test_size"]
            drive_id = selected_rasio["drive_id"]

             # Split data hanya untuk info, bukan buat latih ulang
            X = st.session_state["X"]
            y = st.session_state["y"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

             # Hitung jumlah data
            total_data = len(st.session_state["X"])
            train_count = int((1 - selected_rasio["test_size"]) * total_data)
            test_count = int(selected_rasio["test_size"] * total_data)

            st.info(f"Jumlah data latih: {train_count}")
            st.info(f"Jumlah data uji: {test_count}")

            
            # Unduh file model jika belum ada
            model_dir = "model"
            os.makedirs(model_dir, exist_ok=True)
            model_path = f"{model_dir}/model_rf_{selected_rasio_label.replace(':', '')}.pkl"
 

            tab1, tab2 = st.tabs(["📂 Hasil Load Model", "🛠️ Input Manual RF"])
            with tab1:
                if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    with st.spinner("🔽 Mengunduh model dari Google Drive..."):
                        try:
                            url = f"https://drive.google.com/uc?id={drive_id}"
                            gdown.download(url, model_path, quiet=False, fuzzy=True)
                        except Exception as e:
                            st.error(f"Gagal mengunduh model: {e}")

                if os.path.exists(model_path):
                    try:
                        with open(model_path, "rb") as f:
                            model_data = pickle.load(f)
                        model_rf = model_data.get("model")
                        params = model_data.get("params", {})
                        mape_train = model_data.get("mape_train")
                        mape_test = model_data.get("mape_test")

                        if model_rf and mape_train is not None and mape_test is not None:
                            # Tampilkan parameter model dalam input field yang tidak bisa diubah (read-only)
                            st.subheader("📌 Parameter Model Random Forest")
                            st.markdown(f"**Jumlah pohon (n_estimators):** {params.get('n_estimators', 0)}")
                            st.markdown(f"**Kedalaman maksimum pohon (max_depth):** {params.get('max_depth', 0)}")
                            st.markdown(f"**Fitur maksimum (max_features):** {params.get('max_features', 0)}")
                            # st.success("Model berhasil dimuat!")
                            st.write(f"📊 MAPE Training: **{mape_train:.2f}%**")
                            st.write(f"📊 MAPE Testing : **{mape_test:.2f}%**")
                            # Tambahan: jika MAPE < 10%, sarankan untuk optimasi
                            # if mape_test > 10:
                            #     st.warning("📈 MAPE Testing > 10%. Lakukan optimasi menggunakan PSO.")
                        else:
                            st.error("Beberapa parameter model tidak ditemukan dalam fileee.")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memuat model: {e}")
                else:
                    st.error("File model tidak ditemukan.")
            with tab2:
                st.subheader("🔧 Latih Ulang Random Forest (Manual)")
                n_estimators = st.number_input("Jumlah pohon (n_estimators)", min_value=1, max_value=1000, value=1)
                max_depth = st.number_input("Kedalaman maksimum pohon (max_depth)", min_value=1, max_value=20, value=1)
                max_features = st.slider("Fitur maksimum (max_features)", min_value=0.4, max_value=1.0, value=0.4, step=0.1)

                if st.button("Latih Model"):
                    try:
                        # X = st.session_state["X"]
                        # y = st.session_state["y"]
                        scaler_y = st.session_state["scaler_y"]

                        # Split ulang dengan rasio yang dipilih
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )

                        rf = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            max_features=max_features,
                            random_state=42
                        )
                        rf.fit(X_train, y_train)

                        # Prediksi (normalisasi)
                        y_pred_train = rf.predict(X_train).reshape(-1, 1)
                        y_pred_test = rf.predict(X_test).reshape(-1, 1)

                        # Denormalisasi target & prediksi
                        y_train_denorm = scaler_y.inverse_transform(y_train.to_numpy().reshape(-1, 1))
                        y_test_denorm = scaler_y.inverse_transform(y_test.to_numpy().reshape(-1, 1))
                        y_pred_train_denorm = scaler_y.inverse_transform(y_pred_train)
                        y_pred_test_denorm = scaler_y.inverse_transform(y_pred_test)


                        # Hitung MAPE
                        mape_train = mean_absolute_percentage_error(y_train_denorm, y_pred_train_denorm) * 100
                        mape_test = mean_absolute_percentage_error(y_test_denorm, y_pred_test_denorm) * 100

                        st.success("✅ Model berhasil dilatih ulang.")
                        st.write(f"📊 MAPE Training: **{mape_train:.2f}%**")
                        st.write(f"📊 MAPE Testing : **{mape_test:.2f}%**")

                        # Evaluasi kualitas MAPE
                        # if mape_test < 10:
                        #     st.success("🟢 Kategori MAPE: **Sangat Baik**")
                        # elif 10 <= mape_test <= 20:
                        #     st.info("🔵 Kategori MAPE: **Baik**")
                        # elif 20 < mape_test <= 50:
                        #     st.warning("🟠 Kategori MAPE: **Cukup**")
                        # else:
                        #     st.error("🔴 Kategori MAPE: **Buruk**")

                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat pelatihan: {e}")



    elif menu == "Random Forest + PSO Modelling":
        st.header("4. Random Forest + PSO Modelling")

        if "X" not in st.session_state or "y" not in st.session_state:
            st.warning("Harap lakukan preprocessing terlebih dahulu.")
        elif "normalized" not in st.session_state or not st.session_state["normalized"]:
            st.warning("⚠️ Harap lakukan normalisasi data terlebih dahulu sebelum melanjutkan ke pemodelan Random Forest.")

        else:
            # st.info("Silakan lakukan proses optimasi Random Forest menggunakan PSO di sini.")

            # Mapping rasio ke file model hasil optimasi
            rasio_opsi_pso = {
                "50:50": "1CGxsRkMXRNVMAa6vvH5c_Si1G5o2dPO_",
                "60:40": "1Qynex6zbi-ljxxgHESnWushrL5YOjwQ1",
                "70:30": "1YetscjK9lYWeuPZIT2aNXOs33tasBv2C",
                "80:20": "1DzaGfxAdg1ohNPxss_pLn9ukTlYJJJxh",
                "90:10": "1LZqDyupjcoY_RO3BFFE7McREHv2A2P01",
            }

            # Dropdown untuk pilih rasio
            selected_rasio_label = st.selectbox("Pilih rasio data latih dan uji:", list(rasio_opsi_pso.keys()))
            file_ref = rasio_opsi_pso[selected_rasio_label]

            # Hitung dan tampilkan jumlah data train-test
            total_data = len(st.session_state["X"])
            test_size = {
                "50:50": 0.5,
                "60:40": 0.4,
                "70:30": 0.3,
                "80:20": 0.2,
                "90:10": 0.1
            }[selected_rasio_label]
            train_count = int((1 - test_size) * total_data)
            test_count = int(test_size * total_data)

            st.info(f"Jumlah data latih: {train_count}")
            st.info(f"Jumlah data uji: {test_count}")

            model_dir = "model"
            os.makedirs(model_dir, exist_ok=True)
            model_path_pso = f"rfpso_{selected_rasio_label.replace(':', '').replace('/', '')}.pkl"

            
            tab1_pso, tab2_pso = st.tabs(["📂 Hasil Optimasi PSO", "📌 Parameter Model PSO"])
            with tab1_pso:
                
                # Jika file belum ada, unduh dari Google Drive
                if not os.path.exists(model_path_pso) or os.path.getsize(model_path_pso) == 0:
                    with st.spinner("🔽 Mengunduh model hasil PSO dari Google Drive..."):
                        try:
                            url = f"https://drive.google.com/uc?id={file_ref}"
                            gdown.download(url, model_path_pso, quiet=False, fuzzy=True)
                        except Exception as e:
                            st.error(f"Gagal mengunduh model dari Google Drive: {e}")
                              
                # Cek dan load file model PSO
                if os.path.exists(model_path_pso):
                    try:
                        with open(model_path_pso, "rb") as f:
                            model_data = pickle.load(f)
                            

                        params = model_data.get("params", {})
                        model_rf_pso = model_data.get("model")  # tetap string, cukup untuk ditampilkan
                        mape_train = model_data.get("mape_train")
                        mape_test = model_data.get("mape_test")

                        if model_rf_pso is not None and isinstance(params, dict) and isinstance(mape_train, (float, int)) and isinstance(mape_test, (float, int)):
                            st.subheader("📌 Parameter PSO")
                            st.markdown(f"**Partikel : 100**")
                            st.markdown(f"**Iterasi: 50**")
                            st.markdown(f"**C1: 1.49618**")
                            st.markdown(f"**C2: 1.49618**")
                            st.markdown(f"**Inertia: 0.7**")

                            st.subheader("📌 Parameter Hasil Optimasi (PSO)")             
                            st.markdown(f"**Jumlah pohon (n_estimators):** {params.get('n_estimators', 0)}")
                            st.markdown(f"**Kedalaman maksimum pohon (max_depth):** {params.get('max_depth', 0)}")
                            st.markdown(f"**Fitur maksimum (max_features):** {params.get('max_features', 0)}")

                            st.write(f"📊 MAPE Training: **{mape_train:.2f}%**")
                            st.write(f"📊 MAPE Testing : **{mape_test:.2f}%**")

                            # Evaluasi kategori berdasarkan nilai MAPE Testing
                            if mape_test < 10:
                                st.success("🎯 Nilai MAPE Testing dalam kategori **SANGAT BAIK**")
                            elif 10 <= mape_test < 20:
                                st.success("✅ Nilai MAPE Testing dalam kategori **BAIK** ")
                            elif 20 <= mape_test <= 50:
                                st.warning("⚠️ Nilai MAPE Testing dalam kategori **CUKUP BAIK**")
                            else:
                                st.error("❌ Nilai MAPE Testing dalam kategori **BURUK** ")

                        else:
                            st.error("Parameter model atau nilai MAPE tidak ditemukaNn.")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memuat model: {e}")
                else:
                    st.error("File model hasil optimasi PSO tidak ditemukan.")
            with tab2_pso:
                def run_pso_rf_optimization(X_train, X_test, y_train, y_test, scaler_y, bounds, n_particles, n_iterations, c1, c2, inertia):

                    # --- Inisialisasi Partikel ---
                    def initialize_particles():
                        particles = []
                        velocities = []
                        for _ in range(n_particles):
                            n_estimators = np.random.randint(bounds["n_estimators"][0], bounds["n_estimators"][1] + 1)
                            max_depth = np.random.randint(bounds["max_depth"][0], bounds["max_depth"][1] + 1)
                            max_features = np.random.uniform(bounds["max_features"][0], bounds["max_features"][1])
                            particles.append([n_estimators, max_depth, max_features])
                            velocities.append(np.zeros(3))
                        return np.array(particles), np.array(velocities)

                    # --- Fungsi Evaluasi: kembalikan MAPE Test ---
                    def evaluate(particle):
                        n_estimators = int(particle[0])
                        max_depth = int(particle[1])
                        max_features = float(particle[2])

                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            max_features=max_features,
                            random_state=42
                        )
                        model.fit(X_train, y_train)

                        y_pred_train = model.predict(X_train).reshape(-1, 1)
                        y_pred_test = model.predict(X_test).reshape(-1, 1)

                        y_train_denorm = scaler_y.inverse_transform(y_train.to_numpy().reshape(-1, 1))
                        y_test_denorm = scaler_y.inverse_transform(y_test.to_numpy().reshape(-1, 1))
                        y_pred_train_denorm = scaler_y.inverse_transform(y_pred_train)
                        y_pred_test_denorm = scaler_y.inverse_transform(y_pred_test)

                        mape = mean_absolute_percentage_error(y_test_denorm, y_pred_test_denorm) * 100
                        return mape, model, y_pred_train_denorm, y_pred_test_denorm, y_train_denorm, y_test_denorm

                    # --- PSO Start ---
                    particles, velocities = initialize_particles()
                    p_best = particles.copy()
                    p_best_scores = np.array([evaluate(p)[0] for p in particles])
                    g_best = p_best[np.argmin(p_best_scores)]
                    g_best_score = min(p_best_scores)

                    best_model = None
                    best_y_train = None
                    best_y_test = None
                    best_y_train_true = None
                    best_y_test_true = None

                    for i in range(n_iterations):
                        for j in range(n_particles):
                            # Update kecepatan dan posisi
                            r1, r2 = np.random.rand(3), np.random.rand(3)
                            velocities[j] = (
                                inertia * velocities[j] +
                                c1 * r1 * (p_best[j] - particles[j]) +
                                c2 * r2 * (g_best - particles[j])
                            )
                            particles[j] += velocities[j]

                            # Clipping agar dalam batas
                            particles[j][0] = np.clip(particles[j][0], bounds["n_estimators"][0], bounds["n_estimators"][1])
                            particles[j][1] = np.clip(particles[j][1], bounds["max_depth"][0], bounds["max_depth"][1])
                            particles[j][2] = np.clip(particles[j][2], bounds["max_features"][0], bounds["max_features"][1])

                            # Evaluasi ulang
                            score, model, y_train_pred, y_test_pred, y_train_true, y_test_true = evaluate(particles[j])

                            if score < p_best_scores[j]:
                                p_best[j] = particles[j]
                                p_best_scores[j] = score

                                if score < g_best_score:
                                    g_best = particles[j]
                                    g_best_score = score
                                    best_model = model
                                    best_y_train = y_train_pred
                                    best_y_test = y_test_pred
                                    best_y_train_true = y_train_true
                                    best_y_test_true = y_test_true

                    # Hitung ulang MAPE untuk best
                    mape_train = mean_absolute_percentage_error(best_y_train_true, best_y_train) * 100
                    mape_test = mean_absolute_percentage_error(best_y_test_true, best_y_test) * 100

                    best_params = {
                        "n_estimators": int(g_best[0]),
                        "max_depth": int(g_best[1]),
                        "max_features": float(g_best[2])
                    }

                    return best_params, mape_train, mape_test

                st.subheader("🛠️ Optimasi Manual Random Forest dengan PSO")

                st.markdown("### ⚙️ Parameter PSO")
                partikel = st.number_input("Jumlah Partikel", min_value=10, max_value=150, value=100, step=10)
                iterasi = st.number_input("Jumlah Iterasi", min_value=10, max_value=200, value=50, step=10)
                c1 = st.number_input("Koefisien Learning C1", min_value=0.0, max_value=3.0, value=1.49618, step=0.1)
                c2 = st.number_input("Koefisien Learning C2", min_value=0.0, max_value=3.0, value=1.49618, step=0.1)
                inertia = st.slider("Inertia Weight", min_value=0.1, max_value=1.0, value=0.7, step=0.05)

                if st.button("🚀 Jalankan Optimasi PSO"):
                    try:
                        X = st.session_state["X"]
                        y = st.session_state["y"]
                        scaler_y = st.session_state["scaler_y"]

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                        # Batasan parameter RF
                        bounds = {
                            "n_estimators": (1, 1000),
                            "max_depth": (1, 20),
                            "max_features": (0.4, 1.0)
                        }

                        # Fungsi optimasi PSO untuk RF
                        best_params, mape_train, mape_test = run_pso_rf_optimization(
                            X_train, X_test, y_train, y_test, scaler_y,
                            bounds, partikel, iterasi, c1, c2, inertia
                        )

                        st.success("✅ Optimasi selesai!")
                        st.write(f"📊 MAPE Training: **{mape_train:.2f}%**")
                        st.write(f"📊 MAPE Testing : **{mape_test:.2f}%**")

                        st.markdown("### 📌 Parameter Terbaik Hasil PSO")
                        st.write(f"- n_estimators: {best_params['n_estimators']}")
                        st.write(f"- max_depth: {best_params['max_depth']}")
                        st.write(f"- max_features: {best_params['max_features']:.2f}")

                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat optimasi: {e}")



    elif menu == "Predictions":
        st.header("5. Prediksi Hasil Panen Padi")
    
        # === 1. One-Hot Encoder untuk Varietas ===
        def create_default_varietas_encoder():
            list_varietas = ["Serang Bentis", "Ciherang", "Toyo Arum", "Inpari 32", "Inpari 13"]
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoder.fit(pd.DataFrame(list_varietas, columns=["varietas"]))
            return encoder
    
        if "one_hot_encoders" not in st.session_state:
            st.session_state["one_hot_encoders"] = {}
    
        if "varietas" not in st.session_state["one_hot_encoders"]:
            st.session_state["one_hot_encoders"]["varietas"] = create_default_varietas_encoder()
    
        encoder = st.session_state["one_hot_encoders"]["varietas"]
    
        # === 2. Load model dari Google Drive ===
        if "model_rf_pso_best" not in st.session_state:
            drive_id = "1LZqDyupjcoY_RO3BFFE7McREHv2A2P01"
            url = f"https://drive.google.com/uc?id={drive_id}"
    
            try:
                with st.spinner("🔽 Mengunduh model dari Google Drive..."):
                    with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
                        gdown.download(url, tmp.name, quiet=True, fuzzy=True)
                        tmp.seek(0)
                        model_data = pickle.load(tmp)
                        st.write("📦 DEBUG: Isi model_data:", model_data)
    
                        model = model_data.get("model", None)
                        if model is not None:
                            st.session_state["model_rf_pso_best"] = model
                            st.success("✅ Model berhasil dimuat!")
                        else:
                            st.session_state["model_rf_pso_best"] = None
                            st.error("❌ Model tidak ditemukan dalam file.")
            except Exception as e:
                st.session_state["model_rf_pso_best"] = None
                st.error(f"❌ Gagal memuat model: {e}")
    
        # === 3. Input Fitur ===
        st.subheader("Masukkan Nilai Fitur:")
        luas_tanam = st.number_input("Luas Tanam (HA)", min_value=0.0)
        urea = st.number_input("Pupuk Urea (KG)", min_value=0.0)
        npk = st.number_input("Pupuk NPK (KG)", min_value=0.0)
        organik = st.number_input("Pupuk Organik (KG)", min_value=0.0)
        jumlah_bibit = st.number_input("Jumlah Bibit (KG)", min_value=0.0)
    
        varietas_padi = st.selectbox(
            "Varietas Padi",
            ["Serang Bentis", "Ciherang", "Toyo Arum", "Inpari 32", "Inpari 13"]
        )
    
        if st.button("Prediksi Hasil Panen"):
            try:
                # === 4. Siapkan DataFrame input ===
                input_dict = {
                    "luas_tanam": luas_tanam,
                    "urea": urea,
                    "npk": npk,
                    "organik": organik,
                    "jumlah_bibit": jumlah_bibit,
                    "varietas": varietas_padi
                }
                input_df = pd.DataFrame([input_dict])
    
                # === 5. One-hot encoding varietas ===
                encoded = encoder.transform(input_df[["varietas"]])
                encoded_df = pd.DataFrame(
                    encoded, columns=encoder.get_feature_names_out(["varietas"])
                )
                input_df.drop(columns=["varietas"], inplace=True)
                input_df = pd.concat([input_df, encoded_df], axis=1)
    
                # === 6. Pastikan semua fitur lengkap dan urut ===
                final_features = [
                    "luas_tanam", "urea", "npk", "organik", "jumlah_bibit",
                    "varietas_Ciherang", "varietas_Inpari 13", "varietas_Inpari 32",
                    "varietas_Serang Bentis", "varietas_Toyo Arum"
                ]
                for col in final_features:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[final_features]
    
                # === 7. Ambil model dan lakukan prediksi ===
                model = st.session_state.get("model_rf_pso_best")
                if model is None:
                    st.warning("⚠️ Model belum tersedia.")
                    st.stop()
    
                hasil = model.predict(input_df.values).reshape(-1, 1)
    
                st.success(f"🌾 Prediksi Hasil Panen Padi Adalah: **{hasil[0][0]:,.2f}** Ton")
    
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")

    


if __name__ == "__main__":
    main()
