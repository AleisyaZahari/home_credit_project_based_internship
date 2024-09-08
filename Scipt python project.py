# %% [markdown]
# # import library

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import altair as alt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# %%
!gdown 1BlF8VU9dPAzhl_4QXeDzmRinsmRbfdNv
!gdown 1utatxybdRFBfTkm2YuuJnJEyTYXZXWX8

# %% [markdown]
# #Read dataset

# %%
df_train = pd.read_csv('application_train.csv')
df_test = pd.read_csv('application_test.csv')

# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# %% [markdown]
# # Exploratory Data analisis

# %%
df = df_train.copy()

# %%
df.head()

# %%
df.info(verbose=True)

# %%


# %% [markdown]
# ## Data quality check

# %%
list_items = []
for col in df.columns:
  list_items.append([col, df[col].dtype, df[col].isna().sum(), 100*df[col].isna().sum()/len(df[col]), df[col].nunique(), df[col].unique()[:5]])
desc_df = pd.DataFrame(data=list_items, columns= 'Feature, Data Type, Null, Null %, Unique, Unique Sample'.split(','))
desc_df

# %%
null_thresh = 0.45
list_null_above_thresh = []

for col in df.columns:
  if df[col].isna().sum() / len(df[col]) > null_thresh:
    list_null_above_thresh.append(col)

print("Columns with nulls above", null_thresh * 100, "%:", list_null_above_thresh)


# %%
# cek data duplicate
print('Jumlah data duplicate: ', df.duplicated().sum())

# %% [markdown]
# ## Descriptive statistic

# %%
df.describe().T

# %%
df.describe(exclude= np.number).T

# %%
nums = df.select_dtypes(include=[np.number]).columns
cats = df.select_dtypes(exclude=[np.number]).columns

# %% [markdown]
# ### Univariate analisis

# %%
df.head()

# %%
unique_counts = df.nunique()

unique_counts_df = pd.DataFrame(unique_counts, columns=['Unique_Count'])

unique_counts_df

# %%
CATS = ['TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
       'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE','FLAG_MOBIL',
       'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
       'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START',
       'REG_REGION_NOT_LIVE_REGION',	'REG_REGION_NOT_WORK_REGION',	'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
       'REG_CITY_NOT_WORK_CITY',	'LIVE_CITY_NOT_WORK_CITY',
       'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'FONDKAPREMONT_MODE',
       'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE',
       'FLAG_DOCUMENT_2',	'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4',	'FLAG_DOCUMENT_5',	'FLAG_DOCUMENT_6',
       'FLAG_DOCUMENT_7',	'FLAG_DOCUMENT_8',	'FLAG_DOCUMENT_9',	'FLAG_DOCUMENT_10',
       'FLAG_DOCUMENT_11',	'FLAG_DOCUMENT_12',	'FLAG_DOCUMENT_13',	'FLAG_DOCUMENT_14',	'FLAG_DOCUMENT_15',
       'FLAG_DOCUMENT_16',	'FLAG_DOCUMENT_17',	'FLAG_DOCUMENT_18',	'FLAG_DOCUMENT_19',	'FLAG_DOCUMENT_20',	'FLAG_DOCUMENT_21']

numerical_features = [col for col in df.columns if col not in CATS]

NUMS = df[numerical_features].columns

# %%
print(cats)

# %%
unique_counts = df.nunique()

cats_2 = unique_counts[unique_counts < 5].index.tolist()

print("Columns with less than 5 unique values:", cats_2)

# %%
cats_1 = cats_2

color = '#9eccc8'

sns.set_style("whitegrid")

fig, axes = plt.subplots(len(cats_1)//2, 2, figsize=(15, 25))
# fig, axes = plt.subplots(6, 2, figsize=(25, 25))

axes = axes.flatten()

for i, cat in enumerate(cats_1):
    sns.countplot(x=cat, data=df, ax=axes[i], color=color)
    axes[i].set_title(f'Countplot of {cat}')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Count')
    plt.subplots_adjust(top=0.85)

    for p in axes[i].patches:
        axes[i].annotate(format(p.get_height(), '.0f'),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha = 'center', va = 'center',
                         xytext = (0, 10),
                         textcoords = 'offset points')

plt.tight_layout()
plt.show()

# %%
decode_map = {0: "Good Payment History", 1: "Bad Payment History"}
def decode_sentiment(label):
    return decode_map[int(label)]

df['TARGET'] = df['TARGET'].apply(lambda x: decode_sentiment(x))


# %%
target_agg = (df[['TARGET']].groupby("TARGET").agg(COUNT=("TARGET","count")).sort_values(by=["COUNT"],ascending=False).reset_index())

target_agg.style.background_gradient(cmap='Blues')

# %%
# menghitung target

gpp = df['TARGET'].value_counts(normalize=True)
gpp.reset_index().style.background_gradient(cmap='Blues')

# %%
labels = ['Good Payment History', 'Bad Payment History']

churn_rate = df['TARGET'].value_counts()
explode = (0.05, 0)
text_props = {'color': 'black', 'weight': 'bold'}

plt.figure(figsize=(7, 5))
plt.pie(churn_rate, labels=labels, autopct='%1.1f%%', startangle=90, explode = explode, colors = ['#E2F4C5', '#58A399'], textprops=text_props)
plt.axis('equal')
plt.title('Customers that has difficulties in payment', fontweight='bold', pad=20)
plt.show()

# %%
color = ['#E2F4C5', '#58A399']

# %%
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1)

fig, axes = plt.subplots(2, 2, figsize=(15, 15))

list_cats = ['CODE_GENDER', 'NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']

for i, cat in enumerate(list_cats):
    sns.countplot(x=cat, data=df, ax=axes[i//2, i%2], palette = 'dark:#9eccc8', hue = 'TARGET')  # Efficient indexing for subplots
    axes[i//2, i%2].set_title(f'Customers payment histories by {cat}')  # Set subplot title

plt.tight_layout()

plt.show()

# %%
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1)

fig, axes = plt.subplots(2, 2, figsize=(15, 25))

list_cats = ['NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_TYPE_SUITE']
for i, cat in enumerate(list_cats):
    sns.countplot(x=cat, data=df, ax=axes[i//2, i%2], palette = 'dark:#9eccc8', hue = 'TARGET')  # Efficient indexing for subplots
    axes[i//2, i%2].set_title(f'Customers payment histories by {cat}')  # Set subplot title

plt.tight_layout()

plt.show()

# %%
plt.figure(figsize=(15,10))

fig = sns.countplot(x='NAME_HOUSING_TYPE', data = df, hue = 'TARGET', palette = 'ch:start=0.2,rot=-.3_r')
plt.title('Customers payment Histories by Housing Type\n', fontweight='bold', fontsize=14)
plt.xlabel('\nHousing Type', fontsize=12)

# %%

plt.figure(figsize=(15,10))

fig = sns.countplot(x='WEEKDAY_APPR_PROCESS_START', data = df, hue = 'TARGET', palette = 'ch:start=0.2,rot=-.3_r')
plt.title('Customers Payment Histories By Process Day\n', fontweight='bold', fontsize=14)
plt.xlabel('\nProcess Day', fontsize=12)

# %% [markdown]
# ### Bivariate analisis
# numerik dan terget

# %%
df[NUMS].head(10)

# %% [markdown]
# ## OUTLIER, KORELASI

# %%


# %%
df[NUMS].head()

# %%
print(f'Total rows: {len(df)}')
Kolom_numerik1 = NUMS

outlier = []
no_outlier = []
is_outlier = []
low_lim = []
high_lim = []


filtered_entries = np.array([True] * len(df))
for col in Kolom_numerik1:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    low_limit = Q1 - (IQR * 1.5)
    high_limit = Q3 + (IQR * 1.5)

    #filter outlier
    filter_outlier = ((df[col] >= low_limit) & (df[col] <= high_limit))
    outlier.append(len(df[~filter_outlier]))
    no_outlier.append(len(df[filter_outlier]))
    is_outlier.append(df[col][~filter_outlier].any())
    low_lim.append(low_limit)
    high_lim.append(high_limit)

    filtered_entries = ((df[col] >= low_limit) & (df[col] <= high_limit)) & filtered_entries

print("Outlier All Data :", len(df[~filtered_entries]))
print("Not Outlier All Data :", len(df[filtered_entries]))
print()

pd.DataFrame({
    "Column Name":Kolom_numerik1,
    "is Outlier": is_outlier,
    "Lower Limit": low_lim,
    "Upper Limit": high_lim,
    "Outlier":outlier,
    "No Outlier":no_outlier
})

# %%
import scipy.stats as stats
numeric_data = df[NUMS]

shapiro_results = {}
for col in numeric_data.columns:
    shapiro_results[col] = stats.shapiro(numeric_data[col])

# Output results
for col, result in shapiro_results.items():
    print(f"{col}: p-value = {result.pvalue}, Is Normally Distributed? {result.pvalue > 0.05}")


# %%
normal_columns = []
non_normal_columns = []

# Perform Shapiro-Wilk test for normality on each numeric column
for col in numeric_data.columns:
    p_value = stats.shapiro(numeric_data[col]).pvalue
    if p_value > 0.05:
        normal_columns.append(col)
    else:
        non_normal_columns.append(col)

# Output lists
print("Columns that are normally distributed:\n", normal_columns)
print("Columns that are not normally distributed\n:", non_normal_columns)

# %%
normal_columns = []
non_normal_columns = []

# Perform Shapiro-Wilk test for normality on each numeric column
for col in numeric_data.columns:
    p_value = stats.shapiro(numeric_data[col]).pvalue
    if p_value > 0.05:
        normal_columns.append(col)
    else:
        non_normal_columns.append(col)

# Output lists
print("Columns that are normally distributed:\n", normal_columns)
print("Columns that are not normally distributed\n:", non_normal_columns)

# %%
len(df[NUMS].nunique())

# %%
df_nums = df[NUMS]
plt.figure(figsize=(15,10))
sns.heatmap(df_nums.corr(), annot=True, fmt='.2f')

# %%
import pandas as pd

# Assuming you have df and NUMS defined
df_nums = df[NUMS]

# Calculate correlation matrix
corr_matrix = df_nums.corr()

# Exclude diagonal (self-correlation) with boolean indexing
corr_without_diag = corr_matrix.where(~np.triu(np.ones(corr_matrix.shape)).astype(bool))

# Sort by absolute correlation in descending order
sorted_corr = corr_without_diag.abs().stack().sort_values(ascending=False)

# Print top N correlations (adjust N as needed)
print("Highest Correlations (Excluding Same Column):")
print(sorted_corr)


# %%

corr_matrix = df_nums.corr()

threshold = 0.5

mask = (corr_matrix.abs() > threshold) & (corr_matrix != 1)

plt.figure(figsize=(20, 15))

sns.heatmap(corr_matrix[mask], annot=True, fmt='.2f', cmap='coolwarm')

plt.yticks(rotation=0)

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.title('Correlation Heatmap (Absolute Correlation > 0.5, Excluding Self-correlations)')
plt.show()


# %%
df[NUMS].head()


# %%
# categorical
from scipy.stats import chi2_contingency
cats2 = CATS
chi2_array, p_array = [], []
for column in cats2:

    crosstab = pd.crosstab(df[column], df['TARGET'])
    chi2, p, dof, expected = chi2_contingency(crosstab)
    chi2_array.append(chi2)
    p_array.append(p)

df_chi = pd.DataFrame({
    'Variable': cats2,
    'Chi-square': chi2_array,
    'p-value': p_array
})
df_chi.sort_values(by='Chi-square', ascending=False)

# %%
import pandas as pd
from scipy.stats import chi2_contingency

# Assuming your dataframe is called 'df' and the target variable is 'TARGET'
# Categorical columns are listed in 'CATS'

cats2 = CATS
alpha = 0.05  # Significance level (commonly used)
reject_list = []
chi2_array, p_array = [], []

for column in cats2:
    crosstab = pd.crosstab(df[column], df['TARGET'])
    chi2, p, dof, expected = chi2_contingency(crosstab)
    chi2_array.append(chi2)
    p_array.append(p)

    # Reject null hypothesis if p-value is less than alpha
    reject_list.append(p < alpha)

df_chi = pd.DataFrame({
    'Variable': cats2,
    'Chi-square': chi2_array,
    'p-value': p_array,
    'Reject H0': reject_list  # Add new column for hypothesis rejection
})

# Filter columns that reject the null hypothesis
rejected_cols = df_chi[df_chi['Reject H0'] == True]['Variable'].tolist()

# Print the list of columns rejecting the null hypothesis
print("Columns Rejecting Null Hypothesis:", rejected_cols)

# Sort remaining DataFrame by chi-square (optional)
print(df_chi.sort_values(by='Chi-square', ascending=False))


# %% [markdown]
# # Data preprocessing

# %% [markdown]
# ## drop kolom

# %%
df_fe = df.copy()

# %%
len(df_fe.columns)

# %%
#delete unnecessary column
delete_clmn_nums = [
    'CNT_CHILDREN', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    'DAYS_EMPLOYED', 'EXT_SOURCE_1', 'BASEMENTAREA_AVG',
    'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG',
    'FLOORSMIN_AVG',	'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG'	,'LIVINGAREA_AVG',
    'APARTMENTS_MODE',	'BASEMENTAREA_MODE',	'YEARS_BEGINEXPLUATATION_MODE',
    'YEARS_BUILD_MODE',	'COMMONAREA_MODE',	'ELEVATORS_MODE',	'ENTRANCES_MODE',
    'FLOORSMAX_MODE',	'FLOORSMIN_MODE',	'LANDAREA_MODE',	'LIVINGAPARTMENTS_MODE',
    'LIVINGAREA_MODE',	'NONLIVINGAPARTMENTS_MODE',	'NONLIVINGAREA_MODE',
    'APARTMENTS_MEDI',	'BASEMENTAREA_MEDI',	'YEARS_BEGINEXPLUATATION_MEDI',
    'YEARS_BUILD_MEDI',	'COMMONAREA_MEDI',	'ELEVATORS_MEDI',	'ENTRANCES_MEDI',
    'FLOORSMAX_MEDI',	'FLOORSMIN_MEDI',	'LANDAREA_MEDI',	'LIVINGAPARTMENTS_MEDI',
    'LIVINGAREA_MEDI',	'NONLIVINGAPARTMENTS_MEDI',	'NONLIVINGAREA_MEDI',	'TOTALAREA_MODE']
del_clmn_cats = [
    'FLAG_DOCUMENT_21','FLAG_DOCUMENT_17', 'LIVE_REGION_NOT_WORK_REGION',
    'FLAG_DOCUMENT_4', 'FLAG_EMAIL', 'FLAG_DOCUMENT_19','FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_5' ,'FLAG_CONT_MOBILE','FLAG_DOCUMENT_10',
    'FLAG_DOCUMENT_12', 'FLAG_MOBIL', 'FLAG_DOCUMENT_20']

# %%
df_fe = df.drop(columns = delete_clmn_nums)

# %%
df_fe = df.drop(columns = del_clmn_cats)

# %%
print(list_null_above_thresh)

# %%
df_fe = df.drop(columns = list_null_above_thresh)

# %%
df_fe = df_fe.drop('SK_ID_CURR', axis=1)

# %%
df_fe.head()

# %%
df_fe['TARGET'].value_counts()

# %%
df_fe.columns

# %%
len(df_fe.columns)

# %% [markdown]
# ##REPLACE NULL

# %%
list_items = []
for col in df_fe.columns:
  list_items.append([col, df_fe[col].dtype, df_fe[col].isna().sum(), 100*df_fe[col].isna().sum()/len(df_fe[col]), df_fe[col].nunique(), df_fe[col].unique()[:5]])
desc_df = pd.DataFrame(data=list_items, columns= 'Feature, Data Type, Null, Null %, Unique, Unique Sample'.split(','))
desc_df

# %%
print('Missing values status:', df_fe.isnull().values.any())
nvc = pd.DataFrame(df_fe.isnull().sum().sort_values(), columns=['Total Null Values'])
nvc['Percentage'] = (nvc['Total Null Values']/df_fe.shape[0])*100
nvc.sort_values(by=['Percentage'], ascending=False).reset_index()

# %%
nums = df_fe.select_dtypes(exclude = 'object')
cats = df_fe.select_dtypes(include = 'object')
print(" Kategorik:")
print(cats.columns)
print("\n Numerik:")
print(nums.columns)

# %%
list_items = []
for col in cats.columns:
  list_items.append([col, df_fe[col].dtype, df_fe[col].isna().sum(), 100*df_fe[col].isna().sum()/len(df_fe[col]), df_fe[col].nunique(), df_fe[col].unique()[:5]])
desc_df = pd.DataFrame(data=list_items, columns= 'Feature, Data Type, Null, Null %, Unique, Unique Sample'.split(','))
desc_df

# %%
df_fe.head()

# %%
list_items = []
for col in nums.columns:
  list_items.append([col, df_fe[col].dtype, df_fe[col].isna().sum(), 100*df_fe[col].isna().sum()/len(df_fe[col]), df_fe[col].nunique(), df_fe[col].unique()[:5]])
desc_df = pd.DataFrame(data=list_items, columns= 'Feature, Data Type, Null, Null %, Unique, Unique Sample'.split(','))
desc_df

# %%
cats_col = ['TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR',
       'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
       'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
       'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE',
       'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',
       'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
       'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
       'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
       'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
       'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
       'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
       'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
       'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
       'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',]
nums_col = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
       'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
       'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START',
       'EXT_SOURCE_2',
       'EXT_SOURCE_3', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
       'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
       'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',
       'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
       'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
       'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']

# %%
# For Numerical columns
# replace missing value with "Median"

for col in df_fe[nums_col]:
    df_fe[col] = df[col].fillna(df_fe[col].median())
df_fe.isnull().sum()

# For Categorical columns
# replace missing value with "Mode"

for col in df_fe[cats_col]:
    df_fe[col] = df_fe[col].fillna(df_fe[col].mode().iloc[0])
df_fe.isnull().sum()

# %% [markdown]
# ## feature encoding

# %%
df_fe[cats_col].head(10)

# %%
df_fe['OCCUPATION_TYPE'].unique()

# %%
#label encoding
cats_col_label = ['FLAG_OWN_CAR',	'FLAG_OWN_REALTY']
prefix = ['gender', 'own_car', 'own_realty']
encoder = LabelEncoder()

for col in cats_col_label:
  df_fe[col] = encoder.fit_transform(df_fe[col])

print(f"Label encoded columns: {cats_col_label}")

# %%
df_fe.head()

# %%
df_fe.describe().T

# %%
mapping = {"Good Payment History": 0, "Bad Payment History": 1}

df_fe['TARGET'] = df_fe['TARGET'].replace(mapping)

# %%
df_fe.head()

# %%
df_fe1 = df_fe.copy()

# %%
df_fe1.head()

# %%
df_fe1['DAYS_BIRTH'] = df_fe1['DAYS_BIRTH']/-.1

df_fe1['DAYS_EMPLOYED'] = df_fe1['DAYS_EMPLOYED']/-.1
df_fe1['DAYS_REGISTRATION'] = df_fe1['DAYS_REGISTRATION']/-.1
df_fe1['DAYS_ID_PUBLISH'] = df_fe1['DAYS_ID_PUBLISH']/-.1
df_fe1['DAYS_LAST_PHONE_CHANGE'] = df_fe1['DAYS_LAST_PHONE_CHANGE']/-.1

# %% [markdown]
# ## one hot ENCODING

# %%
df_fe1['ORGANIZATION_TYPE'].nunique()

# %%
df_fe1 = df_fe1.drop(columns = 'ORGANIZATION_TYPE')

# %%
df_fe1.head()

# %%
COLUMN = [
    'NAME_CONTRACT_TYPE',	'CODE_GENDER', 'NAME_TYPE_SUITE',
    'NAME_INCOME_TYPE',	'NAME_EDUCATION_TYPE',	'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START']
prefix = ['contract', 'gender', 'type_suite', 'income', 'education', 'family_status', 'houseing_type', 'occupation', 'weekday_process_start']
df_fe1 = pd.get_dummies(df_fe1, columns=COLUMN, prefix=prefix)
df_fe1.head()

# %% [markdown]
# ## STANDARSCALLE, MINMAX SCALLER

# %%
print(normal_columns)
print(non_normal_columns)

# %%
# Get existing columns
existing_columns = list(df_fe1.columns)

normal_columns = [col for col in normal_columns if col in existing_columns]
non_normal_columns = [col for col in non_normal_columns if col in existing_columns]

print(f"Normal Columns: {normal_columns}")
print(f"Non-Normal Columns: {non_normal_columns}")


# %%
# standar scaler
scaler = StandardScaler()

# transform data non normal
df_fe1[non_normal_columns] = scaler.fit_transform(df_fe1[non_normal_columns])

# minmax scaler
minmax = MinMaxScaler()

# transform data normal
df_fe1[normal_columns] = minmax.fit_transform(df_fe1[normal_columns])

# %% [markdown]
# #modelling

# %% [markdown]
# ## SPLIT DATA

# %%
X = df_fe1.drop('TARGET', axis=1)
y = df_fe['TARGET']

# %%
# bagi data menjadi data train dan data test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
print('Data train: ',X_train.shape, y_train.shape)

# %% [markdown]
# ## Modeling

# %%
# logistic regression
logreg = LogisticRegression()

# fit model
logreg.fit(X_train, y_train)

# %%
# decision tree
tree = DecisionTreeClassifier()

# fit model
tree.fit(X_train, y_train)

# %%
from sklearn.ensemble import RandomForestClassifier

randf = RandomForestClassifier()
randf.fit(X_train, y_train)

# %% [markdown]
# ## evaluation

# %%
#model evaluation

def eval_classification(model):
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    print("Accuracy (Test Set): %.2f" % accuracy_score(y_test, y_pred))
    print("Accuracy (Train set): %.2f" % accuracy_score(y_train, y_pred_train))
    print("\nPrecision (Test Set): %.2f" % precision_score(y_test, y_pred))
    print("Precision (Train Set): %.2f" % precision_score(y_train, y_pred_train))
    print("\nRecall (Test Set): %.2f" % recall_score(y_test, y_pred))
    print("Recall (Train Set): %.2f" % recall_score(y_train, y_pred_train))
    print("\nF1-Score (Test Set): %.2f" % f1_score(y_test, y_pred))
    print("F1-Score (Train Set): %.2f" % f1_score(y_train, y_pred_train))

def roc_auc_eval(model):
    y_pred_proba = model.predict_proba(X_test)
    y_pred_proba_train = model.predict_proba(X_train)

    print("\nroc_auc (test-proba): %.2f" % roc_auc_score(y_test, y_pred_proba[:, 1]))
    print("roc_auc (train-proba): %.2f" % roc_auc_score(y_train, y_pred_proba_train[:, 1]))

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix

eval_classification(logreg)
roc_auc_eval(logreg)

# %%
eval_classification(tree)
roc_auc_eval(tree)

# %%
eval_classification(randf)
roc_auc_eval(randf)

# %% [markdown]
# ## grid search

# %%
from sklearn.model_selection import GridSearchCV

# %%
param_grid = {
    'C': [0.1, 1, 10, 100],
}

grid_logreg = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy')
grid_logreg.fit(X_train, y_train)

# %%
best_params = grid_logreg.best_params_

print("Best Hyperparameters:")
print(best_params)

# %%
best_logreg = LogisticRegression(C=best_params['C'])
best_logreg.fit(X_train, y_train)

# %%
feature_importances = best_logreg.coef_[0]

# Create a DataFrame to display feature importance
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print the feature importance
print(importance_df)

# %%
top5_importance_df = importance_df.head(5)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(top5_importance_df['Feature'], top5_importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 5 Features Importance')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance at the top
plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 4))

sns.histplot(df_fe, x='AMT_ANNUITY', bins=20, hue='TARGET', palette=['#9eccc8', '#b92a27'], multiple='stack', ax=ax)
max_churn_value = df_fe['TARGET'].max()

plt.figtext(0.5, 1.0,
           "Jumlah anuitas 20k-30k mempengaruhi pembayaran",
           ha='center', va='center', fontsize=16, weight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()


