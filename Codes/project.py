


import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

st.set_page_config(layout='wide',initial_sidebar_state ='expanded',page_title="MiulCar",page_icon="🚗")


if 'CarPricePredict' not in st.session_state:
    st.session_state['CarPricePredict'] = False


#SIDEBAR
st.sidebar.header('USER INPUT FEATURES')
def user_input_features():
    selected_brand = st.sidebar.selectbox('Brand', ["Audi", "BMW", "Mercedes", "Hyundai", "Opel", "Skoda", "Toyota"])
    selected_model = st.sidebar.selectbox('Model', ["200", "220", "1 Series", "2 Series", "3 Series", "4 Series", "5 Series", "A Class", "A1", "A2", "A3", "A4", "A5", "A6","Accent", "Astra", "Auris", "B Class", "CL Class", "CLA Class", "CLK Class", "CLK", "CLS Class", "Corolla", "Corolla X", "E Class", "G Class", "Getz", "GLA Class", "I20", "I30", "Insignia", "Karoq", "Kodiaq", "M4", "M5", "Mokka", "Mokka X", "Octavia", "Prius", "Q7", "SuperB", "Tucson", "Verso", "X1", "X2", "X5", "X6", "Yaris", "Yeti"])
    selected_year = st.sidebar.selectbox("Araç Üretim Yılını Seçiniz", ["1998", "1999", "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"])
    selected_Km = st.sidebar.slider("Kilometre Bilgisi Giriniz", 0,250000, 1)
    selected_fuelType = st.sidebar.selectbox("Yakıt Türünü Seçiniz", ["Diesel", "Benzin", "Hybrid", "Other"])
    selected_yakittuketimi = st.sidebar.selectbox("Motor Hacmi", ["1.0", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "2.0", "2.1", "2.2", "2.7", "2.9", "3.0", "3.5", "4.0", "4.1", "4.3", "5.0", "5.4", "6.0", "6.2"])
    selected_vites = st.sidebar.selectbox("Vites Türünü Seçiniz", ["Automatic", "Manual", "Semi-Auto", "Other"])
    selected_motor = st.sidebar.slider("Ortalama Yakıt Tükrtimi", 1.0,15.0, 0.1)

    data = {
        'brand': selected_brand,
        'model': selected_model,
        'year': int(selected_year),
        'transmission': selected_vites,
        'Km': selected_Km,
        'fuelType': selected_fuelType,
        'Km/L': selected_yakittuketimi,
        'engineSize': selected_motor}
    features = pd.DataFrame(data, index=[0])

    return features



image = Image.open('WhatsApp Image 2022-09-23 at 14.31.10.jpeg')
st.image(image,width=900)

#Seçilen Değerleri DF yapıp göster
input_df = user_input_features()
st.header('User Choices')
st.write(input_df)

# Veriyi okutmak
df = pd.read_excel('final_car_dataset(1).xlsx', index_col=0)

# Model Kur
median = df["Km/L"].median()
df["Km/L"].fillna(median, inplace=True)



def grab_col_names(dataframe, cat_th=55, car_th=75):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes != ["float64", "int64"]]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes == ["float64", "int64"]]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes != ["float64", "int64"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes == ["float64", "int64"]]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, cat_but_car, num_cols = grab_col_names(df)

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range

    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def featureengineering(dataframe):
    # Car Year
    dataframe["Recent_Year"] = 2022
    dataframe["Car_Age"] = dataframe["Recent_Year"] - dataframe["year"]

    dataframe.drop("Recent_Year", axis=1, inplace=True)

    # Car Age Category
    dataframe.loc[dataframe["Car_Age"] <= 2, "Yaş_Sınıfı"] = "Genç"
    dataframe.loc[(dataframe["Car_Age"] > 2) & (dataframe["Car_Age"] <= 5), "Yaş_Sınıfı"] = "Orta"
    dataframe.loc[(dataframe["Car_Age"] > 5) & (dataframe["Car_Age"] <= 10), "Yaş_Sınıfı"] = "Orta-Yaşlı"
    dataframe.loc[dataframe["Car_Age"] > 10, "Yaş_Sınıfı"] = "Yaşlı"

    # Km variables

    dataframe.loc[(dataframe["Km"] < 10000) & (dataframe["year"] > 2018), "NewClass"] = "New"
    dataframe.loc[
        (dataframe["Km"] < 10000) & (dataframe["year"] > 2016) & (dataframe["year"] < 2019), "NewClass"] = "New_Good"
    dataframe.loc[(dataframe["Km"] < 10000) & (dataframe["year"] <= 2016), "NewClass"] = "New_VeryGood"
    dataframe.loc[
        (dataframe["Km"] >= 10000) & (dataframe["Km"] < 100000) & (dataframe["year"] > 2018), "NewClass"] = "Med_Good"
    dataframe.loc[(dataframe["Km"] >= 10000) & (dataframe["Km"] < 100000) & (dataframe["year"] > 2016) & (
            dataframe["year"] <= 2018), "NewClass"] = "Med_Verygood"
    dataframe.loc[
        (dataframe["Km"] >= 10000) & (dataframe["Km"] < 100000) & (dataframe["year"] <= 2016), "NewClass"] = "Med_Super"
    dataframe.loc[
        (dataframe["Km"] >= 100000) & (dataframe["Km"] < 200000) & (dataframe["year"] > 2018), "NewClass"] = "Old_Bad"
    dataframe.loc[(dataframe["Km"] >= 100000) & (dataframe["Km"] < 200000) & (dataframe["year"] <= 2018) & (
            dataframe["year"] > 2016), "NewClass"] = "Old_Normal"
    dataframe.loc[
        (dataframe["Km"] >= 100000) & (dataframe["Km"] < 200000) & (dataframe["year"] <= 2016), "NewClass"] = "Old_Good"
    dataframe.loc[(dataframe["Km"] >= 200000) & (dataframe["Km"] < 500000) & (
                dataframe["year"] > 2018), "NewClass"] = "Bad_Badd"  # 1 tane var
    dataframe.loc[(dataframe["Km"] >= 200000) & (dataframe["Km"] < 500000) & (dataframe["year"] <= 2018) & (
            dataframe["year"] > 2016), "NewClass"] = "Bad_Normal"
    dataframe.loc[(dataframe["Km"] >= 200000) & (dataframe["Km"] < 500000) & (
                dataframe["year"] <= 2016), "NewClass"] = "Bad_Normal"
    dataframe.loc[(dataframe["Km"] >= 500000), "NewClass"] = "Bad"

    return dataframe

df = featureengineering(df)



def one_hot_encoder(dataframe, drop_first=False):
    dataframe = pd.get_dummies(dataframe,  drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df)



#Scaling
#num_cols = [col for col in num_cols if "TotalPrice" not in col]
#df_encode[num_cols]= RobustScaler(df_encode[num_cols])

y = df["TotalPrice"]
X = df.drop(["TotalPrice"], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=112)
reg_model = RandomForestRegressor()
reg_fitted_model = reg_model.fit(X_train, y_train)


st.header('Price Prediction Result')
if st.button('Predict Price'):
    #st.header("Sales Price Prediction of Your Car")
    #st.write(input_df)

    #st.write("input df after feature engineering")
    #input_df = featureengineering(input_df)
    #st.dataframe(input_df)

    #st.write("input df after one hot encoding")
    #input_df = one_hot_encoder(input_df)
    #st.dataframe(input_df)

    #st.write("test df")
    test_df = pd.DataFrame(columns=X.columns)
    #st.dataframe(test_df)

    #fill test df with input df
    for col in test_df.columns:
        if col in input_df.columns:
            test_df[col] = input_df[col]
        else:
            test_df[col] = 0

    #st.write("test df after filling")
    #st.dataframe(test_df)

    st.write("# RESULTS")
    result = reg_fitted_model.predict(test_df)
    st.metric("Sales Price", result[0], "TL")



st.markdown(
    """
    <style>
    .main {
    background-color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)
