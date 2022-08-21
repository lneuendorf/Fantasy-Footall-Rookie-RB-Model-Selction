## Written by: Luke Neuendorf

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import random as rand
# regression algorithm imports
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.svm import SVR

# load data
df = pd.read_csv('rb_data.csv')

# split data
x_all = df.iloc[:, 4:-3] # features
y_all = df.iloc[:, -1:] # labels (rb fantasy ppg years 1-3)
train_x = df.iloc[122:, 4:-3] # features with at least 3 nfl years
train_y = df.iloc[122:, -1:] # y data with at least 3 nfl years

##########################
### DATA PREPROCESSING ###
##########################

print("Data Preprocessing:")
print("\tNum features before preprocessing:", len(x_all.columns))
print("\tNum of features after deleting columns with ...") 

# drop features with missing data greater than threshold
missing_threshold = .10
i = 0
while i < len(train_x.columns):
    if((train_x[train_x.columns[i]].isnull().sum()/len(train_x.index)) > missing_threshold):
        train_x = train_x.drop(train_x.columns[i], axis=1)
        x_all = x_all.drop(x_all.columns[i], axis=1)
    else:
        i += 1
print("\t\tmore than ", missing_threshold*100, "% null values: ",str(len(train_x.columns)),"\n", sep="",end="")

# remove highly collinear features, keeping the feature w/strongest correlation to y-var
corr_thres = .70
i = j_flag = i_flag = 0
while i < len(train_x.columns):
    j = i+1
    while j < len(train_x.columns):
        if (abs(train_x[train_x.columns[i]].corr(train_x[train_x.columns[j]])) > corr_thres):
            if abs(train_x[train_x.columns[i]].corr(train_y[train_y.columns[0]])) <= abs(train_x[train_x.columns[j]].corr(train_y[train_y.columns[0]])):
                train_x = train_x.drop(train_x.columns[i], axis=1)
                x_all = x_all.drop(x_all.columns[i], axis=1)
                i_flag = 1
                break
            else:
                train_x = train_x.drop(train_x.columns[j], axis=1)
                x_all = x_all.drop(x_all.columns[j], axis=1)
                j_flag = 1
        if j_flag == 0:
            j += 1
        j_flag = 0
    if i_flag == 0:
        i += 1
    i_flag = 0
print("\t\tcorrelation coefficient lower than ",corr_thres,": ",str(len(train_x.columns)),"\n", sep="",end="")

# drop remaining rows w/ null values
train_x.isna().sum()
train_joined = train_x.join(train_y)
train_joined = train_joined.dropna().reset_index(drop=True)
train_x = train_joined.iloc[:, :-1]
train_y = train_joined.iloc[:, -1:]

# shuffle data rows
train_joined = train_joined.sample(frac = 1)

######################
### VISUALIZE DATA ###
######################

# target variable bar chart
print("\nLearning About the Data:","\n", sep="",end="")
plt.rcParams["figure.figsize"] = (10, 10)
plt.hist(train_y['Tot'], bins = 50)
plt.xlabel('NFL Total PPR Pts in First 3 Years')
plt.ylabel('Count')
plt.title('Distribution of NFL RB PPR Pts')

# Correlation Heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(abs(train_joined.corr()), annot=True)

print("\tSkew of target variable: ",round(train_y['Tot'].skew(),4),"\n", sep="",end="")
print("\tThe positve skew value means the target distribution is right-skewed.","\n", sep="",end="")
print("\tA correlation heat map and target var distribution bar chart have also been created.","\n", sep="",end="")

#######################
### MODEL SELECTION ###
#######################

print("\nNested K-Fold CV:","\n", sep="",end="")

# baseline model
print("\tBaseline Model (mean of y-var): ", sep="",end="")
print(round(train_y.mean(),4).to_string()[-8:],"\n", sep="",end="")

def nested_kfold_cv(df, model, param_grid={}, outer_folds=5, inner_folds=5, standardize=0):
    """
    Parameters
    ----------
    df : DataFrame
        dataframe with format [x_train : y_train]
    model : SciKit-Learn Wrapper Object
        supervised learning algorithm to use
    param_grid : dict
        sequence of hyperparemters to perform gridsearch cv on in inner loop
    outer_folds : int
        number of outer folds
    inner_folds : int
        number of inner folds
    standardize : int
        1 to standardize, 0 to skip standardization

    Returns
    -------
    None.
    """
    
    # initialize variables outside of for loops to expand scopes
    df_split = []
    mae_list = []
    df_copy = df.reset_index(drop=True)
    rand.seed(0)
    
    # standardize if needed
    if(standardize):
        scaler = StandardScaler().fit(df.iloc[:,:-1])
        x_std = pd.DataFrame(scaler.transform(df.iloc[:,:-1]), columns = df.iloc[:,:-1].columns)
        df_copy = x_std.join(df.iloc[:,-1:])
    
    # spilt df into k outer folds
    fold_size = int(len(df_copy) / outer_folds) # rounds down
    for x in range(outer_folds):
        df_fold = pd.DataFrame()
        # randomaly select rows to add to each fold
        while len(df_fold) < fold_size:
            i = rand.randrange(len(df_copy))
            df_fold = df_fold.append(df_copy.iloc[i:i+1, :])
            df_copy = df_copy.drop(index=i, axis=0).reset_index(drop=True)
        df_split.append(df_fold)

    # perform nested cv, calculating MAE on outer test fold
    for x in range(outer_folds):
        print('\t\tInner Fold ' + str(x+1) )
        test_outer = df_split[x]
        train_outer = pd.DataFrame()

        # hyperparameter tuning on inner folds
        grid = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=inner_folds, refit=True)
        for y in range(outer_folds):
            if(y != x) : train_outer = pd.concat([train_outer,df_split[y]], ignore_index=True)
        for z in range(inner_folds):
            grid.fit(train_outer.iloc[:, :-1], train_outer.iloc[:, -1:].values.ravel())
        ypred = grid.predict(test_outer.iloc[:, :-1])
        mae = mean_absolute_error(test_outer.iloc[:, -1:], ypred)
        print('\t\t\tMAE: ' + str(round(mae,4)) + '\n\t\t\tBest Hyperparameters:')
        for key in grid.best_params_:
            print('\t\t\t\t' + str(key) + ': ' + str(grid.best_params_[key]))
        mae_list.append(mae)

    print('\n\t\tAverage Outer Fold MAE\'s: ' + str(round(sum(mae_list)/len(mae_list),4)))


# specify number of inner/outer folds to use in CV
outer_folds = 5
inner_folds = 5

# DTree
print('\n\tDecision Tree')
model = DecisionTreeRegressor()
param_grid = {'max_depth' : [2,3,4,5],
              'min_samples_leaf':[10,15,20],
              'ccp_alpha': np.arange(0, 1, 0.01)}
nested_kfold_cv(train_joined, model, param_grid, outer_folds, inner_folds, standardize=0)

# Elastic Net
print('\n\tElastic Net')
model = ElasticNet()
param_grid = {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
              'l1_ratio': np.arange(0.01, 1, 0.01)}
nested_kfold_cv(train_joined, model, param_grid, outer_folds, inner_folds, standardize=1)

# XGBoost
print('\n\tXGBoost')
model = XGBRegressor(verbosity=1)
param_grid = {'max_depth': [2],
              'learning_rate': [0.01],
              'n_estimators': [200,300],
              'subsample': [0.7,0.9,1],
              'gamma': [1,10,100]}
nested_kfold_cv(train_joined, model, param_grid, outer_folds, inner_folds, standardize=0)

# Linear Regression
print('\n\tLinear Regression')
model = LinearRegression()
param_grid = {'positive': [True, False]}
nested_kfold_cv(train_joined, model, param_grid, outer_folds, inner_folds, standardize=1)

# RANSAC Regression
print('\n\tRANSAC Regression')
model = RANSACRegressor(LinearRegression())
param_grid = {'min_samples': [.5, .6, .7, .8, .9, 1],
              'max_trials': [200]}
nested_kfold_cv(train_joined, model, param_grid, outer_folds, inner_folds, standardize=1)

# Gaussian Regression
print('\n\tGaussian Process Regressor')
kernel = DotProduct() + WhiteKernel()
model = GaussianProcessRegressor(kernel=kernel)
param_grid = {}
nested_kfold_cv(train_joined, model, param_grid, outer_folds, inner_folds, standardize=1)

# SVR
print('\n\tSupport Vector Regression')
model = SVR()
param_grid = {'kernel': ['linear', 'poly', 'rbf'],
              'C':[1, 1.5, 3, 6, 10],
              'epsilon':[0.1,0.2,0.3,0.5]}
nested_kfold_cv(train_joined, model, param_grid, outer_folds, inner_folds, standardize=1)



