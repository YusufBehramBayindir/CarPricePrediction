################################################
# Car Price Prediction Machine Learning Pipeline II
################################################

import joblib
import pandas as pd
import numpy as np
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve,\
    RandomizedSearchCV



################################################
# Helper Functions
################################################

# Data Preprocessing & Feature Engineering

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(tail))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("##################### Describe #####################")
    print(dataframe.describe().T)

def grab_col_names(dataframe, cat_th=10, car_th=70):
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
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


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

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def carprice_data_prep(dataframe):

    median = dataframe["Km/L"].median()
    dataframe["Km/L"].fillna(median, inplace=True)

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
    dataframe.loc[(dataframe["Km"] < 10000) & (dataframe["year"] > 2016) & (dataframe["year"] < 2019), "NewClass"] = "New_Good"
    dataframe.loc[(dataframe["Km"] < 10000) & (dataframe["year"] <= 2016), "NewClass"] = "New_VeryGood"
    dataframe.loc[(dataframe["Km"] >= 10000) & (dataframe["Km"] < 100000) & (dataframe["year"] > 2018), "NewClass"] = "Med_Good"
    dataframe.loc[(dataframe["Km"] >= 10000) & (dataframe["Km"] < 100000) & (dataframe["year"] > 2016) & (
                dataframe["year"] <= 2018), "NewClass"] = "Med_Verygood"
    dataframe.loc[(dataframe["Km"] >= 10000) & (dataframe["Km"] < 100000) & (dataframe["year"] <= 2016), "NewClass"] = "Med_Super"
    dataframe.loc[(dataframe["Km"] >= 100000) & (dataframe["Km"] < 200000) & (dataframe["year"] > 2018), "NewClass"] = "Old_Bad"
    dataframe.loc[(dataframe["Km"] >= 100000) & (dataframe["Km"] < 200000) & (dataframe["year"] <= 2018) & (
                dataframe["year"] > 2016), "NewClass"] = "Old_Normal"
    dataframe.loc[(dataframe["Km"] >= 100000) & (dataframe["Km"] < 200000) & (dataframe["year"] <= 2016), "NewClass"] = "Old_Good"
    dataframe.loc[(dataframe["Km"] >= 200000) & (dataframe["Km"] < 500000) & (dataframe["year"] > 2018), "NewClass"] = "Bad_Badd"  # 1 tane var
    dataframe.loc[(dataframe["Km"] >= 200000) & (dataframe["Km"] < 500000) & (dataframe["year"] <= 2018) & (
                dataframe["year"] > 2016), "NewClass"] = "Bad_Normal"
    dataframe.loc[(dataframe["Km"] >= 200000) & (dataframe["Km"] < 500000) & (dataframe["year"] <= 2016), "NewClass"] = "Bad_Normal"
    dataframe.loc[(dataframe["Km"] >= 500000), "NewClass"] = "Bad"

    # Transmission and fuel type

    dataframe.loc[(dataframe["transmission"] == "Automatic") & (dataframe["fuelType"] == "Diesel"), "T_Fuel"] = "A"
    dataframe.loc[(dataframe["transmission"] == "Automatic") & (dataframe["fuelType"] == "Petrol"), "T_Fuel"] = "B"
    dataframe.loc[(dataframe["transmission"] == "Automatic") & (dataframe["fuelType"] == "Hybrid"), "T_Fuel"] = "C"
    dataframe.loc[(dataframe["transmission"] == "Automatic") & (dataframe["fuelType"] == "Other"), "T_Fuel"] = "C"
    dataframe.loc[(dataframe["transmission"] == "Manual") & (dataframe["fuelType"] == "Diesel"), "T_Fuel"] = "F"
    dataframe.loc[(dataframe["transmission"] == "Manual") & (dataframe["fuelType"] == "Petrol"), "T_Fuel"] = "E"
    dataframe.loc[(dataframe["transmission"] == "Manual") & (dataframe["fuelType"] == "Hybrid"), "T_Fuel"] = "G"
    dataframe.loc[(dataframe["transmission"] == "Manual") & (dataframe["fuelType"] == "Other"), "T_Fuel"] = "H"
    dataframe.loc[(dataframe["transmission"] == "Semi_Auto") & (dataframe["fuelType"] == "Diesel"), "T_Fuel"] = "I"
    dataframe.loc[(dataframe["transmission"] == "Semi_Auto") & (dataframe["fuelType"] == "Petrol"), "T_Fuel"] = "J"
    dataframe.loc[(dataframe["transmission"] == "Semi_Auto") & (dataframe["fuelType"] == "Hybrid"), "T_Fuel"] = "K"
    dataframe.loc[(dataframe["transmission"] == "Semi_Auto") & (dataframe["fuelType"] == "Other"), "T_Fuel"] = "L"
    dataframe.loc[(dataframe["transmission"] == "Other") & (dataframe["fuelType"] == "Diesel"), "T_Fuel"] = "M"
    dataframe.loc[(dataframe["transmission"] == "Other") & (dataframe["fuelType"] == "Petrol"), "T_Fuel"] = "N"
    dataframe.loc[(dataframe["transmission"] == "Other") & (dataframe["fuelType"] == "Hybrid"), "T_Fuel"] = "O"
    dataframe.loc[(dataframe["transmission"] == "Other") & (dataframe["fuelType"] == "Other"), "T_Fuel"] = "P"


    # Car model

    dataframe.loc[dataframe["model"] == " A5", "Arac_Tipi"] = "Sedan"
    dataframe.loc[dataframe["model"] == " A3", "Arac_Tipi"] = "Sedan"
    dataframe.loc[dataframe["model"] == " CL Class", "Arac_Tipi"] = "Sedan"
    dataframe.loc[dataframe["model"] == " CLA Class", "Arac_Tipi"] = "Sedan"
    dataframe.loc[dataframe["model"] == " CLS Class", "Arac_Tipi"] = "Sedan"
    dataframe.loc[dataframe["model"] == " CLC Class", "Arac_Tipi"] = "Sedan"
    dataframe.loc[dataframe["model"] == " CLK", "Arac_Tipi"] = "Sedan"
    dataframe.loc[dataframe["model"] == " 220", "Arac_Tipi"] = "Sedan"
    dataframe.loc[dataframe["model"] == " 200", "Arac_Tipi"] = "Sedan"
    dataframe.loc[dataframe["model"] == " B Class", "Arac_Tipi"] = "Hatchback"
    dataframe.loc[dataframe["model"] == " GLA Class", "Arac_Tipi"] = "SUV"
    dataframe.loc[dataframe["model"] == " G Class", "Arac_Tipi"] = "SUV"
    dataframe.loc[dataframe["model"] == " Corolla", "Arac_Tipi"] = "Sedan"
    dataframe.loc[dataframe["model"] == " Yaris", "Arac_Tipi"] = "SUV"
    dataframe.loc[dataframe["model"] == " X6", "Arac_Tipi"] = "SUV"
    dataframe.loc[dataframe["model"] == " M5", "Arac_Tipi"] = "Sedan"
    dataframe.loc[dataframe["model"] == " Mokka X", "Arac_Tipi"] = "SUV"
    dataframe.loc[dataframe["model"] == " Astra", "Arac_Tipi"] = "Sedan"
    dataframe.loc[dataframe["model"] == " Crossland X", "Arac_Tipi"] = "SUV"
    dataframe.loc[dataframe["model"] == " Insignia", "Arac_Tipi"] = "Sedan"
    dataframe.loc[dataframe["model"] == " I30", "Arac_Tipi"] = "Hatchback"
    dataframe.loc[dataframe["model"] == " I20", "Arac_Tipi"] = "Hatchback"
    dataframe.loc[dataframe["model"] == " Tucson", "Arac_Tipi"] = "SUV"
    dataframe.loc[dataframe["model"] == " Accent", "Arac_Tipi"] = "Sedan"
    dataframe.loc[dataframe["model"] == " Getz", "Arac_Tipi"] = "Hatchback"
    dataframe.loc[dataframe["model"] == " A6", "Arac_Tipi"] = "Sedan"
    dataframe.loc[dataframe["model"] == " A3", "Arac_Tipi"] = "Sedan"
    dataframe.loc[dataframe["model"] == " A5", "Arac_Tipi"] = "Hatchback"
    dataframe.loc[dataframe["model"] == " A4", "Arac_Tipi"] = "Sedan"
    dataframe.loc[dataframe["model"] == " A1", "Arac_Tipi"] = "Hatchback"
    dataframe.loc[dataframe["model"] == " Q7", "Arac_Tipi"] = "SUV"
    dataframe.loc[dataframe["model"] == " A2", "Arac_Tipi"] = "Hatchback"
    dataframe.loc[dataframe["model"] == " Superb", "Arac_Tipi"] = "Sedan"
    dataframe.loc[dataframe["model"] == " Octavia", "Arac_Tipi"] = "Hatchback"
    dataframe.loc[dataframe["model"] == " Yeti", "Arac_Tipi"] = "SUV"
    dataframe.loc[dataframe["model"] == " Kodiaq", "Arac_Tipi"] = "SUV"
    dataframe.loc[dataframe["model"] == " E Class", "Arac_Tipi"] = "Sedan"
    dataframe.loc[dataframe["model"] == " A Class", "Arac_Tipi"] = "Hatchback"
    dataframe.loc[dataframe["model"] == " Karoq", "Arac_Tipi"] = "SUV"
    dataframe.loc[dataframe["model"] == " Auris", "Arac_Tipi"] = "Hatchback"
    dataframe.loc[dataframe["model"] == " Prius", "Arac_Tipi"] = "Hatchback"
    dataframe.loc[dataframe["model"] == ' Verso', "Arac_Tipi"] = "MPV"
    dataframe.loc[dataframe["model"] == ' 4 Series', "Arac_Tipi"] = "Coupe"
    dataframe.loc[dataframe["model"] == ' 3 Series', "Arac_Tipi"] = "Sedan"
    dataframe.loc[dataframe["model"] == " X2", "Arac_Tipi"] = "SUV"
    dataframe.loc[dataframe["model"] == ' 5 Series', "Arac_Tipi"] = "Sedan"
    dataframe.loc[dataframe["model"] == ' M4', "Arac_Tipi"] = "Coupe"
    dataframe.loc[dataframe["model"] == " X5", "Arac_Tipi"] = "SUV"
    dataframe.loc[dataframe["model"] == ' 2 Series', "Arac_Tipi"] = "Coupe"
    dataframe.loc[dataframe["model"] == " X1", "Arac_Tipi"] = "SUV"
    dataframe.loc[dataframe["model"] == ' 1 Series', "Arac_Tipi"] = "Hatchback"


    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=5, car_th=70)

    cat_cols = [col for col in cat_cols if "TotalPrice" not in col]

    df = one_hot_encoder(dataframe, cat_cols, drop_first=True)

    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=70)

    num_cols = [col for col in num_cols if "TotalPrice" not in col]
    X_scaled = StandardScaler().fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

    y = df["TotalPrice"]
    X = df.drop(["ID", "Km", "Km/L", "TotalPrice"], axis=1)

    return X, y

# Base Models
def base_models(X, y, scoring="neg_root_mean_squared_error"):
    print("Base Models....")
    models = [('LR', LinearRegression()),
              ("Ridge", Ridge()),
              ("Lasso", Lasso()),
              ("ElasticNet", ElasticNet()),
              ('CART', DecisionTreeRegressor()),
              ('RF', RandomForestRegressor()),
              ('GBM', GradientBoostingRegressor()),
              ("XGBoost", XGBRegressor(objective='reg:squarederror')),
              ("LightGBM", LGBMRegressor()),
              ("CatBoost", CatBoostRegressor(verbose=False))]

    for name, model in models:
        rmse_results = np.mean(np.sqrt(-cross_val_score(model, X, y, cv=10, scoring=scoring)))
        print(f"scoring: {round(rmse_results.mean(), 4)} ({name}) ")



# Hyperparameter Optimization

rf_params = {"max_depth": [5, 8, 15, None],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [200, 500]}

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500, 1000],
              "subsample": [1, 0.5, 0.7]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]}

models = [("RF", RandomForestRegressor(), rf_params),
          ('XGBoost', XGBRegressor(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
          ('GBM', GradientBoostingRegressor(), gbm_params)]


def hyperparameter_optimization(X, y, cv=3, scoring="neg_root_mean_squared_error"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, model, params in models:
        print(f"########## {name} ##########")
        rmse_results = np.mean(np.sqrt(-cross_val_score(model, X, y, cv=10, scoring=scoring)))
        print(f"{scoring} (Before): {round(rmse_results.mean(), 4)}")

        gs_best = GridSearchCV(model, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = model.set_params(**gs_best.best_params_)

        rmse_results = np.mean(np.sqrt(-cross_val_score(model, X, y, cv=10, scoring=scoring)))
        print(f"{scoring} (After): {round(rmse_results.mean(), 4)}")

        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


# Stacking & Ensemble Learning
def voting_regressor(best_models, X, y):
    print("Voting Regressor...")
    voting_clfp = VotingRegressor(estimators=[('GBM', best_models["GBM"]), ('RF', best_models["RF"]),
                                              ('XGBoost', best_models["XGBoost"])]).fit(X, y)
    rmse_results = np.mean(np.sqrt(-cross_val_score(voting_clfp, X, y, cv=10, scoring="neg_root_mean_squared_error")))
    print(f"RMSE: {round(rmse_results)}")
    return voting_clfp



################################################
# Pipeline Main Function
################################################

def main():
    df = pd.read_excel("/Users/mervekirisci/VBO/datasets/final_car_dataset.xlsx")
    X, y = carprice_data_prep(df)
    base_models(X, y)
    best_models = hyperparameter_optimization(X, y)
    voting_clfp = voting_regressor(best_models, X, y)
    joblib.dump(voting_clfp, "voting_clfp.pkl")
    return voting_clfp

if __name__ == "__main__":
    print("İşlem başladı")
    main()
