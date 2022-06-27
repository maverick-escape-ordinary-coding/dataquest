# %%
# Library for data manipulation
import pandas as pd
import numpy as np

# Library for plotting
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


# Library for Feature Engineering
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from scipy.stats import skew, kurtosis

# Library for data shuffle
from sklearn.utils import shuffle

# Library for fitting data
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Library for model evaluation
from sklearn.metrics import r2_score, mean_absolute_error

# Library to store Trained models (Temporary purpose)
import pickle
from datetime import timezone
import datetime

# %%
# Path to file
file_path = "/data/pp-complete.csv"

# Column names
columns = ["transaction_unique_id","sale_price","transfer_date","postcode","type","age","duration","paon","saon","street","locality","area","district","county","ppd_category","record_status"]

# %%
%%time

# Step: Import data (Assuming datatype as string to fasten the process of importing data because pandas takes lot of time to check the datatype)
data = pd.read_csv(file_path, 
                   names = columns,
                   encoding='utf-8',
                   skip_blank_lines=True,
                   dtype=str
                   )

# Step: Backup data
data_backup = data.copy()

# %% [markdown]
# On a first glance,
# - Need to check if there are missing values
# - Need to change the data types of the corresponding columns (Sales Price, Transfer date, type)
# - Need to split the transfer date into year and month
# - Standardise the column values

# %% [markdown]
# There are no missing transactions

# %%
# Step: Check if any transaction with sale price is missing
data["sale_price"].isnull().sum()

# %%
# Step: Standardise the required column values to uppercase
data[['area','duration','type']] = data[['area','duration','type']].apply(lambda each_value: each_value.str.upper())

# %%
# Step: Convert sale_price datatype to floating point
data.sale_price = data.sale_price.astype('float')

# %% [markdown]
# The standard deviation of sale prices are in the range of $10^5$. This looks to be highly variable. 

# %%
# Step: Get statistical description of sale_price
data.sale_price.describe()

# %%
# Step: Convert to datetime format 
data.transfer_date = pd.to_datetime(data.transfer_date)

# %%
# Step: Checking percentage of data with other property types 
(data[data.type == 'O'].shape[0] / data.shape[0]) * 100

# %% [markdown]
# As there are less than 1.5% of data with other property, Its best to remove those data as they cannot significantly influence the outcome. 

# %%
# Step: Drop data with other property type
data.drop(data[data.type == 'O'].index, inplace=True)

# %%
# Step: Change data type of transfer_date to Datetime
data['transfer_date_converted'] = pd.to_datetime(data['transfer_date'], infer_datetime_format=True)

# Step: Extract datetime attribute(s) month name from 'transfer_date_converted'
data['transfer_date_converted_month_name'] = data['transfer_date_converted'].dt.month_name()

# Step: Extract datetime attribute(s) year from 'transfer_date_converted'
data['transfer_date_converted_year'] = data['transfer_date_converted'].dt.year

# %%
# Temporary data for plotting with specific years.
data_visualise = data[['sale_price','transfer_date_converted_year','area']]
data_visualise.transfer_date_converted_year.isin([1995,2005,2015,2022])

# %%
# Plotting to check sales price of different cities for 1995, 2005, 2015 and 2022
fig = px.line(data.sort_values(by=['transfer_date_converted_year'], ascending=[True]), x='transfer_date_converted_year', y='sale_price', color='area')
fig

# %%
# # Most popular houses in UK
# fig = px.violin(data_copy, x='type', y='sale_price')
# fig

# %%
# Temporary data for plotting having London area with property sales price more than 20000
data_london = data.loc[(data['area'].isin(['LONDON'])) & (data['sale_price'] > 20000)]

# %%
# Most popular houses in london
fig = px.box(data_london, x='type', y='sale_price')
fig

# %%
# Temporary data for plotting having london area based properties sold during the year 1995, 2005, 2015 and 2022
data_london_year = data_london.loc[data_london.transfer_date_converted_year.isin([1995,2005,2015,2022])]

data_london_year

# %%
# Plotting property sales prices in London for specific years
fig = px.line(data_london_year.sort_values(by=['transfer_date_converted_year'], ascending=[True]), x='transfer_date_converted_year', y='sale_price')
fig

# %%
# Step: Considering only specific column of interest
data = data[["sale_price","transfer_date_converted_year","type","duration","area"]]

# %%
# Step: Change data type of property type to Categorical/Factor
data['type_converted'] = data['type'].astype('category')

# Step: Change data type of duration to Categorical/Factor
data['duration_converted'] = data['duration'].astype('category')

# %%
# Step: Drop raw column property type and duration and hold only those that have datatype converted
data.drop(['type','duration'],axis = 1, inplace=True)

# %% [markdown]
# #### Feature Engineering
# 
# Plan is to Encode all the columns to numerical. This will help the model to fit better.

# %%
# Step: Checking how many categories are there
for each_column in data.select_dtypes(include='category').keys().to_list():
    print(each_column, ':', len(data[each_column].unique()))

# %% [markdown]
# We need to apply one-hot encoding on these multicategorical variables

# %%
# Step: Encoding the categorical columns
data_linear = pd.get_dummies(data, columns=data.select_dtypes(include='category').keys().to_list())

# %%
# Step: Checking the data count of area 
data_linear.area.value_counts().sort_values(ascending=False).head(20)

# %%
# Step: Encoding area other than London to 0
data_linear.area.replace(list(data_linear.area.loc[~(data_linear.area == 'LONDON')].unique()), 0, inplace=True)


# %%
# Step: Encoding area London to 1
data_linear.area.replace('LONDON',1, inplace=True)

# %%
# Step: Correlation to check multicollinearity
mask = np.zeros_like(data_linear.corr(), dtype=bool)
mask[np.triu_indices_from(mask)] = True
# heatmap
sns.heatmap(data_linear.corr()*100, 
           cmap='RdBu_r', 
           annot = True, 
           mask = mask)

# %% [markdown]
# Need to check the data for skewness. This can influence the result of the model. We need to have Gaussian-like distribution.

# %%
# Step: Skewness
skew(data_linear.sale_price, bias=False)

# %% [markdown]
# The value suggests data is highly skewed. Next let's check if outliers are propogating this skewness.

# %%
# Step: Kurtosis
kurtosis(data_linear.sale_price, fisher=False)

# %% [markdown]
# The value is leptokurtic. Suggests we have heavy outliers.

# %%
# Step: Log transformation
np.log(data_linear.sale_price).skew()

# %% [markdown]
# After log transformation, It is still not exactly guassian

# %%


# Step: Applying numerical transformers 
pt = PowerTransformer()
qt = QuantileTransformer(output_distribution='normal')
plt.figure(figsize=(20,30))
j = 1
array = np.array(data_linear['sale_price']).reshape(-1, 1)
y = pt.fit_transform(array)
x = qt.fit_transform(array)
plt.subplot(3,3,j)
sns.distplot(array, bins = 50, kde = True)
plt.title(f"Original Distribution for {'sale_price'}")
plt.subplot(3,3,j+1)
sns.distplot(x, bins = 50, kde = True)
plt.title(f"Quantile Transform for {'sale_price'}")
plt.subplot(3,3,j+2)
sns.distplot(y, bins = 50, kde = True)
plt.title(f"Power Transform for {'sale_price'}")
j += 3

# %% [markdown]
# Quantile Transformer shows a better distribution

# %%
# Step: Split data for purchases prior to 2019 as train and the rest as test data
train = data_linear[data_linear['transfer_date_converted_year'] < 2019]
test = data_linear[data_linear['transfer_date_converted_year'] >= 2019]

# Step: Shuffling to remove bias
train_shuffle, test_shuffle = shuffle(train, test, random_state = 1234)

# Step: Training and Testing sets
X_train = train_shuffle.loc[:, train_shuffle.columns != 'sale_price']
y_train = train_shuffle['sale_price']
X_test = test_shuffle.loc[:, test_shuffle.columns != 'sale_price']
y_test = test_shuffle['sale_price']

# %% [markdown]
# - Decided to try linear models and tree-based models for this regression problem. 
# - Applying regularised linear regression (Lasso, Ridge and Elastic Net), Random forest, gradient-boosted trees.  
# - Instead of working on each model seperately, it would be more efficient to construct a pipeline object to fit. 

# %%
# Step: Constructing model pipelines
pipelines = {
    'lasso' : make_pipeline(QuantileTransformer(output_distribution='normal'),
              Lasso(random_state=123)),
    'ridge' : make_pipeline(QuantileTransformer(output_distribution='normal'),
              Ridge(random_state=123)),
    'enet' :  make_pipeline(QuantileTransformer(output_distribution='normal'),
              ElasticNet(random_state=123)),
    'rf' :    make_pipeline(
              RandomForestRegressor(random_state=123)),
    'gb' :    make_pipeline(
              GradientBoostingRegressor(random_state=123))
}

# %%
# Step: Using GridSearch to estimate best parameters and train all the models with 5 fold cross validation

lasso_hyperparameters = {
    'lasso__alpha' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
ridge_hyperparameters = {
    'ridge__alpha' : [0.001, 0.005, 0.01, 0.1, 0.5, 1, 5, 10]}
enet_hyperparameters = { 
    'elasticnet__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 1, 5, 10], 
    'elasticnet__l1_ratio' : [0.1, 0.3, 0.5, 0.7, 0.9]}
rf_hyperparameters = {
     'randomforestregressor__n_estimators' : [100, 200],
     'randomforestregressor__min_samples_leaf' : [1, 3, 5, 10]}
gb_hyperparameters = {
      'gradientboostingregressor__n_estimators' : [100, 200],
      'gradientboostingregressor__learning_rate' : [0.05, 0.1, 0.2],
      'gradientboostingregressor__max_depth' : [1, 3, 5]}

hyperparameters = {'lasso':lasso_hyperparameters, 'ridge':ridge_hyperparameters, 'enet': enet_hyperparameters, 'rf': rf_hyperparameters, 'gb': gb_hyperparameters}
fitted_models = {}
for name, pipeline in pipelines.items():
    model = GridSearchCV(pipeline,
                         hyperparameters[name],
                         cv=5, 
                         n_jobs=-1)
    model.fit(X_train, y_train)
    fitted_models[name] = model

# %%
# Step: Locally saving all the trained models
date_timezone = datetime.datetime.now(timezone.utc)
utc_time = date_timezone.replace(tzinfo=timezone.utc)
utc_timestamp = int(utc_time.timestamp())
model_file = open('house_price_prediction_model_'+str(utc_timestamp),'ab')
pickle.dump(fitted_models, model_file)

# %%
# Step: Checking the training performance and those best parameters
for name, model in fitted_models.items():
    print(name, model.best_score_, model.best_params_)

# %% [markdown]
# Looking at the Training Scores, the random forest and gradient boosting regressor are the best performing models at 85% and 86.4% respectively.

# %%
# Step: Checking test data performance and the R^2 of the models
for name, model in fitted_models.items():
   pred = model.predict(X_test)
   print(name)
   print('R^2:', r2_score(y_test, pred))
   print('MAE:', mean_absolute_error(y_test, pred))
   print()                            

# %% [markdown]
# The winning algorithm is the gradient boosting regressor with the best $R^2$ score of 0.67


