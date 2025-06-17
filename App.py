import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Klasifikasi Bunga Iris", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Iris.csv")
    df.drop("Id", axis=1, inplace=True)
    return df

df = load_data()

# Navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih halaman:", ["1. Dataset & Visualisasi", "2. Pelatihan Model", "3. Prediksi"])

# Halaman 1: Dataset, Karakteristik, Visualisasi
if page == "1. Dataset & Visualisasi":
    st.title("📄 Dataset Iris & Visualisasi")

    st.subheader("Dataset")
    st.write(df.head())

    st.markdown(f"Jumlah data: **{df.shape[0]}** baris, **{df.shape[1]}** kolom")

    st.subheader("Karakteristik Dataset")
    st.markdown("**Info Singkat**")
    buffer = df.dtypes.to_frame("Tipe").join(df.isnull().sum().to_frame("Null")).T
    st.dataframe(buffer)

    st.markdown("**Deskripsi Statistik**")
    st.dataframe(df.describe())

    st.subheader("Visualisasi")
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        sns.boxplot(data=df, x="species", y="sepal_length", ax=ax1)
        ax1.set_title("Sepal Length per Species")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="species", ax=ax2)
        ax2.set_title("Petal Length vs Petal Width")
        st.pyplot(fig2)

# Halaman 2: Pelatihan Model
elif page == "2. Pelatihan Model":
    st.title("🤖 Pelatihan Model Klasifikasi")

    X = df.drop("species", axis=1)
    y = df["species"]

    test_size = st.slider("Ukuran data uji (%)", 10, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    st.success(f"Akurasi Model: **{accuracy_score(y_test, y_pred)*100:.2f}%**")

    with st.expander("Lihat Laporan Klasifikasi"):
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

# Halaman 3: Prediksi
elif page == "3. Prediksi":
    st.title("🌼 Prediksi Jenis Bunga Iris")

    st.markdown("Masukkan fitur bunga:")

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])

    if st.button("Prediksi"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(df.drop("species", axis=1), df["species"])
        prediction = model.predict(input_data)[0]
        st.success(f"Jenis bunga yang diprediksi: **{prediction}**")
