################################################
# Miuul-Car Sales Price Prediction
################################################

import warnings
import pydotplus
import joblib
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skompiler import skompile
import graphviz
from sklearn.svm import SVC, SVR
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, LocalOutlierFactor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz, export_text
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve,\
    RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier,\
    RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score,  classification_report,\
    roc_auc_score, plot_roc_curve, confusion_matrix, median_absolute_error, r2_score
import joblib


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from pandas.core.common import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Reading the data set
df = pd.read_excel("datasets/final_car_dataset.xlsx")
df.head()
##############################
# EDA
##############################

#1. General Picture
# 2. Categorical Variable Analysis
# 3. Numerical Variable Analysis
# 4. Target Variable Analysis
# 5. Correlation Analysis


######################################
#1. General Picture
######################################

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("##################### Describe #####################")
    print(dataframe.describe().T)
check_df(df)

# Determination of categorical and numerical variables
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
        car_th: int, optional
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

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols,  num_cols, cat_but_car = grab_col_names(df)

cat_cols
# ['brand', 'model', 'transmission', 'fuelType']

num_cols
# ['ID', 'year', 'Km', 'Km/L', 'engineSize', 'TotalPrice']

cat_but_car
# []

# Examining the number of unique values of the model variable
df["model"].nunique() #50

######################################
# 2. Categorical Variable Analysis
######################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("###########################################")
    if plot:
        sns.countplot(x = dataframe[col_name], data= dataframe)
        plt.show()

for col in cat_cols:
        cat_summary(df, col, plot=True)

################
# 3. Numerical Variable Analysis
################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")

for col in num_cols:
    num_summary(df, col)

################
# 4. Target Variable Analysis
################

# For categorical variables

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}),
          end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "TotalPrice", col)

################
# 5. Correlation Analysis
################

corr = df[num_cols].corr()
corr

# Display of correlations
sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

#Highly correlated variables;

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdPu")
        plt.show()
    return drop_list

high_correlated_cols(df, plot=False)
# []

#We NaN because there won't be a year called #2060. The maximum year in the dataset was thus 2020.
#df.loc[df.year > 2020, "year"] = np.nan
#df.isnull().sum()

median = df["Km/L"].median()
df["Km/L"].fillna(median, inplace=True)


######################################
# Data Preprocessing
######################################

# 1. Outlier Analysis
# 2. Missing Value Analysis
# 3. Feature Engineering
# 4. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
# 5. Feature Scaling

################
# 1. Outlier Analysis
################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


low_limit, up_limit = outlier_thresholds(df, "TotalPrice")
print("Low Limit : {0} Up Limit : {1}".format(low_limit, up_limit))


# We check for outlier
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    if col != "TotalPrice":
      print(col, check_outlier(df, col))

# ID False
# year True
# Km True
# Km/L True
# engineSize True

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    if col != "TotalPrice":
        replace_with_thresholds(df, col)


################
# 2. Missing Value Analysis
################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)


################
# 3. Feature Engineering
################

# Calculation of vehicle age

df["Recent_Year"]= 2022
df["Car_Age"] = df["Recent_Year"] - df["year"]

df.drop("Recent_Year", axis=1, inplace=True)


# Segmentation of vehicles by age

df.loc[df["Car_Age"] <= 2, "Yaş_Sınıfı"] = "Genç"
df.loc[(df["Car_Age"] > 2) & (df["Car_Age"] <= 5), "Yaş_Sınıfı"] = "Orta"
df.loc[(df["Car_Age"] > 5) & (df["Car_Age"] <= 10),"Yaş_Sınıfı"] = "Orta-Yaşlı"
df.loc[df["Car_Age"] > 10, "Yaş_Sınıfı"] = "Yaşlı"

# Segmentation of the km variable
df.loc[(df["Km"] < 10000) & (df["year"] >2018 ),"NewClass"] = "New"
df.loc[(df["Km"] < 10000) & (df["year"] >2016 )& (df["year"]<2019),"NewClass"] = "New_Good"
df.loc[(df["Km"] < 10000) & (df["year"]<=2016 ),"NewClass"] = "New_VeryGood"
df.loc[(df["Km"] >= 10000) &(df["Km"] < 100000) & (df["year"] >2018 ),"NewClass"] = "Med_Good"
df.loc[(df["Km"] >= 10000) &(df["Km"] < 100000) & (df["year"] >2016 )& (df["year"] <=2018 ),"NewClass"] = "Med_Verygood"
df.loc[(df["Km"] >= 10000) &(df["Km"] < 100000) & (df["year"] <=2016 ),"NewClass"] = "Med_Super"
df.loc[(df["Km"] >= 100000) &(df["Km"] < 200000) & (df["year"] >2018 ),"NewClass"] = "Old_Bad"
df.loc[(df["Km"] >= 100000) &(df["Km"] < 200000) & (df["year"] <=2018 )& (df["year"] >2016 ),"NewClass"] = "Old_Normal"
df.loc[(df["Km"] >= 100000) &(df["Km"] < 200000) & (df["year"] <=2016 ),"NewClass"] = "Old_Good"
df.loc[(df["Km"] >=200000) &(df["Km"] < 500000) & (df["year"] >2018 ),"NewClass"] = "Bad_Badd" #1 tane var
df.loc[(df["Km"] >=200000) &(df["Km"] < 500000) & (df["year"] <=2018 )& (df["year"] >2016 ),"NewClass"] = "Bad_Normal"
df.loc[(df["Km"] >=200000) &(df["Km"] < 500000) & (df["year"] <=2016 ),"NewClass"] = "Bad_Normal"
df.loc[(df["Km"] >=500000) ,"NewClass"] = "Bad"


################
# 4. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
################

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df.head()

ohe_cols = [col for col in df.columns if 46 >= df[col].nunique() > 2]
# Name the columns that have undergone one-hot encoding with ohe_cols and return variables with more than 46 unique variables.
# the reason for inclusion is that it is actually a categorical variable like the model, but it is not considered categorical
# because it is numerically more is to prevent.

df = one_hot_encoder(df, ohe_cols, drop_first=True)

df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)



##################################
# Modeling
##################################

y = df["TotalPrice"]
X = df.drop(["ID", "Km", "Km/L", "TotalPrice"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=46)

# Establishing a base model by targeting RMSE values
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


for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

"""
RMSE: 145702.0265 (LR) 
RMSE: 146694.1822 (Ridge) 
RMSE: 147082.0864 (Lasso) 
RMSE: 202171.5603 (ElasticNet) 
RMSE: 198149.8448 (CART) 
RMSE: 170460.6202 (RF) 
RMSE: 159978.1666 (GBM) 
RMSE: 156168.0841 (XGBoost) 
RMSE: 179071.7553 (LightGBM) 
RMSE: 143356.2924 (CatBoost) 
"""

# Automated Hyperparameter Optimization

catboost_model= CatBoostRegressor(random_state=17, verbose=False)

XgBoost_model=XGBRegressor()

gbm_model = GradientBoostingRegressor(random_state=17)

XgBoost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

catboost_params= {"learning_rate":[0.1, 0.01],
                  "iterations" :[100,500],
                  "depth" : [2,5]}


# CatBoost Review

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)

catboost_best_grid.best_params_
# {'depth': 5, 'iterations': 500, 'learning_rate': 0.1}

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)
catboost_model.score(X_train, y_train)
# 0.9505360464864221


# Test R-Squared
catboost_model.score(X_test, y_test)
# 0.9019436741562201
y_pred=catboost_model.predict(X_train)
#MSE
mean_squared_error(y_train, y_pred)
# 5099703200.855955
#MAE
mean_absolute_error(y_train, y_pred)
# 52383.23176592799
#R^2
r2_score(y_train, y_pred)
#  0.9505360464864221

# XGBoost Review

XgBoost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [2, 8],
                  "n_estimators": [100, 200]}

XgBoost_best_grid = GridSearchCV(XgBoost_model, XgBoost_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)
catboost_best_grid.best_params_
# {'depth': 5, 'iterations': 500, 'learning_rate': 0.1}

XgBoost_final = XgBoost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)
XgBoost_model.score(X_train, y_train)
# 0.9461072985466522

# Test R-Squared
XgBoost_model.score(X_test, y_test)
# 0.8956857623943747
y_pred=XgBoost_model.predict(X_train)
#MSE
mean_squared_error(y_train, y_pred)
# 5556304391.014135
#MAE
mean_absolute_error(y_train, y_pred)
# 54309.13027115604
#R^2
r2_score(y_train, y_pred)
# 0.9461072985466522


# Random Forest Analysis

rf_model = RandomForestRegressor(random_state=17)
rf_model.get_params()

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 400]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_
"""{'max_depth': 8,
 'max_features': 'auto',
 'min_samples_split': 8,
 'n_estimators': 400}
"""
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)
y_pred = rf_model.predict(X_train)


# Train R-Squared
rf_model.score(X_train, y_train)
# 0.21254443880142204
# Test R-Squared
rf_model.score(X_test, y_test)
# : 0.22439313203078745
#MSE
mean_squared_error(y_train, y_pred)
#  10874924569.773489
#MAE
mean_absolute_error(y_train, y_pred)
#  61252.381919756224
#R^2
r2_score(y_train, y_pred)
#  0.8849918794580349


# Examining the importance levels of the variables
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)


def val_curve_params(model, X, y, param_name, param_range, scoring="mean", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

val_curve_params(rf_final, X, y, "max_depth", range(1, 11), scoring="rmse")

X.columns

# Prediction for a New Observation

X.columns
random_user = X.sample(1, random_state=45)
xgboost_model.predict(random_user)

import pickle
pickle.dump(xgboost_final, open('CarPickle.pkl', 'wb'))

