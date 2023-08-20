import copy
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
import pickle
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import kendalltau

##############################################################################
pd.set_option('display.max_columns', None)


########################################################################################################################
############################################ Helper Functions ##########################################################
########################################################################################################################

def replace_nulls_and_zeros_with_avg(df,column_name, avg):

    df_copy = df.copy()
    df_copy[column_name] = df_copy[column_name].replace([0, np.nan], np.nan).fillna(avg)

    return df_copy

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def replace_nulls_and_zeros_with_avg_in_y(df, column_name):
    df_copy = df.copy()

    avg = df_copy[column_name].replace([0, np.nan], np.nan).mean()

    df_copy[column_name] = df_copy[column_name].replace([0, np.nan], np.nan).fillna(avg)

    return df_copy

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def one_hot_encode(df, column_name):
    df.loc[:, column_name] = df[column_name].apply(json.loads)
    genre_lists = df[column_name].apply(lambda m: [d['name'] for d in m])

    mlb = MultiLabelBinarizer()
    mlb.fit(genre_lists)

    genre_matrix = mlb.transform(genre_lists)
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_,)
    genre_df['index_col']=df.index
    genre_df.set_index('index_col',inplace=True)
    #print(genre_df)
    df = pd.concat([df, genre_df], axis=1)
    # Drop the original column
    #df = df.dropna(subset=[column_name])
    df = df.drop(columns=[column_name])

    #print(df)
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def dropcol(df, column_name):
    df = df.drop(columns=[column_name]).copy()
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def encode_column(df, column):

    le = LabelEncoder()
    df[f'{column}_encoded'] = le.fit_transform(df[column])
    df.drop(column, axis=1, inplace=True)
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def filter_frequent_items(df, column_name, n):

    filtered_df = df.copy()

    column_data = filtered_df[column_name]

    item_counts = column_data.value_counts()

    frequent_items = item_counts[item_counts >= n].index.tolist()

    filtered_df[column_name] = filtered_df[column_name].where(filtered_df[column_name].isin(frequent_items), other=None)

    return filtered_df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def replace_null_with_string(df, column_name, string):
    new_df = df.copy()
    new_df[column_name].fillna(string, inplace=True)
    return new_df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def split_date(df, date_column):

    df[['month', 'day', 'year']] = df[date_column].str.split('/', expand=True)
    df = df.drop(columns=[date_column, 'day'])
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def correlation_analysis2(X, Y):
    import pandas as pd

    # Combine X and Y into a single DataFrame
    df = pd.concat([X, Y], axis=1)

    # Compute the correlation matrix
    corr_matrix = df.corr()

    # Only keep correlations between X and Y
    corr_xy = corr_matrix.iloc[:-1, -1]

    # Return the correlation matrix
    return corr_xy
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#reading data
data = pd.read_csv('movies-regression-dataset.csv')
x=data.iloc[:,:19]

y=data.iloc[:,19:20]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
t_y = replace_nulls_and_zeros_with_avg_in_y(y_train, 'vote_average')
training_avg = y_train['vote_average'].replace([0, np.nan], np.nan).mean()

########################################################################################################################
####################################### preprocessing train Function ###################################################
########################################################################################################################

def preprocess_data_train(train_data):
    X_train = train_data.copy()


    columns_to_replace = ['budget', 'revenue', 'runtime', 'vote_count', 'viewercount']


    for column in columns_to_replace:
        avg = X_train[column].replace([0, np.nan], np.nan).mean()
        X_train = replace_nulls_and_zeros_with_avg(X_train, column, avg)


    X_train = split_date(X_train, 'release_date')

    X_train = one_hot_encode(X_train, 'genres')


    col_to_move = 'original_language'
    X_train.insert(len(X_train.columns) - 1, col_to_move, X_train.pop(col_to_move))

    X_train = replace_null_with_string(X_train, 'original_language', 'en')
    X_train = encode_column(X_train, 'original_language')

    columns_to_filter = ['spoken_languages', 'keywords', 'production_countries', 'production_companies']
    for column in columns_to_filter:
        X_train = filter_frequent_items(X_train, column, 10)
        X_train[column] = X_train[column].fillna('[]')
        X_train = one_hot_encode(X_train, column)


    correlation_analysis2(X_train,y_train)
    # Drop homepage and id columns (unique values)
    X_train = dropcol(X_train, 'homepage')
    X_train = dropcol(X_train, 'id')

    # Drop status, tagline, title, and overview columns
    X_train = dropcol(X_train, 'status')
    X_train = dropcol(X_train, 'tagline')
    X_train = dropcol(X_train, 'title')
    X_train = dropcol(X_train, 'overview')
    X_train = dropcol(X_train, 'original_title')

    return X_train, X_test

########################################################################################################################
####################################### preprocessing test Function ####################################################
########################################################################################################################

def preprocess_data_test(train_data, test_data):
    train_data = pd.DataFrame(train_data[0])
    test_data = pd.DataFrame(test_data)
    X_train = train_data.copy()
    X_test = test_data.copy()

    columns_to_replace = ['budget', 'revenue', 'runtime', 'vote_count', 'viewercount']
    for column in columns_to_replace:
        avg =X_train[column].replace([0, np.nan], np.nan).mean()
        X_test = replace_nulls_and_zeros_with_avg(X_test, column, avg)


    X_test = split_date(X_test, 'release_date')

    X_test = one_hot_encode(X_test, 'genres')


    col_to_move = 'original_language'
    X_test.insert(len(X_test.columns) - 1, col_to_move, X_test.pop(col_to_move))

    X_test = replace_null_with_string(X_test, 'original_language', 'en')
    X_test = encode_column(X_test, 'original_language')

    columns_to_filter = ['spoken_languages', 'keywords', 'production_countries', 'production_companies']
    for column in columns_to_filter:
       # X_test = filter_frequent_items(X_test, column, 10)
        X_test[column] = X_test[column].fillna('[]')
        X_test = one_hot_encode(X_test, column)

    df1_columns = X_train.columns.tolist()
    df2_columns = X_test.columns.tolist()
    common_columns = list(set(df1_columns) & set(df2_columns))
    X_test = X_test.drop([col for col in df2_columns if col not in common_columns], axis=1)

    return X_test

########################################################################################################################
####################################### feature selection Function #####################################################
########################################################################################################################

def spearman_feature_selection(df, data, target, threshold=0.5):

    import pandas as pd
    from scipy.stats import spearmanr

    original_cols = df.columns.tolist()

    data = pd.DataFrame(data)

    correlations = {}
    for i, column in enumerate(data.columns):
        correlations[original_cols[i]] = spearmanr(data[column], target)[0]

    selected_features = [col for col in correlations.keys() if abs(correlations[col]) > threshold]


    selected_features = df[selected_features]

    #print(selected_features)
    return selected_features


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def anova_feature_selection(X, y, num_features):
    y = np.ravel(y)
    # Calculate ANOVA F-values and p-values for each feature
    f_values, p_values = f_classif(X, y)

    # Rank features by F-value and select top features
    ranked_features = np.argsort(f_values)[::-1]
    selected_indices = ranked_features[:num_features]
    selected_features = X.iloc[:, selected_indices]

    #print(selected_features)
    return selected_features

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def normalize_feature(df):
    copy_df=copy.deepcopy(df)
    scaler=MinMaxScaler()
    scaler.fit(copy_df)
    return pd.DataFrame(scaler.transform(copy_df),columns=df.columns)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def linear_regression(X_train, X_test, y_train, y_test):

  regr = LinearRegression()
  regr.fit(X_train, y_train)
  y_pred = regr.predict(X_test)

  coef = regr.coef_
  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)

  return regr, coef, mse, r2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def polynomial_regression(X_train, X_test, y_train, y_test, degree=2):

    poly_features = PolynomialFeatures(degree=degree)

    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    regr = LinearRegression()

    regr.fit(X_train_poly, y_train)
    y_pred = regr.predict(X_test_poly)
    coef = regr.coef_
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return regr, coef, mse, r2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def ridge_regression(X_train, X_test, y_train, y_test, alpha=1.0):
    regr = Ridge(alpha=alpha)
    y_train = np.ravel(y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return regr, mse, r2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def random_forest_regression(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=None):
    regr = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    y_train = np.ravel(y_train)
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return regr, mse, r2
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def decision_tree_regression(X_train, X_test, y_train, y_test, max_depth=None):
    regr = DecisionTreeRegressor(max_depth=max_depth)

    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return regr, mse, r2


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_regression(x, y):
    plt.figure()

    for i in range(x.shape[1]):
        xi = x.iloc[:, i]

        a, b = np.polyfit(xi, y, 1)
        plt.scatter(xi, y)

        plt.plot(xi, a * xi + b, color='red')

        plt.xlabel('{}'.format(x.columns[i]))  # Use the column name as the x-axis label
        plt.ylabel('vote_average')

        plt.show()

########################################################################################################################
############################################ Main Function #############################################################
########################################################################################################################

#splitting data
new_traind_x=preprocess_data_train(X_train)
new_tested_x=preprocess_data_test(new_traind_x,X_test)

################## getting average of y_train to use it in replacing the null values in y_test

avg = t_y['vote_average'].replace([0, np.nan], np.nan).mean()
##save
pickle.dump(avg, open("y_train_avg.pkl", 'wb'))
y_test = replace_nulls_and_zeros_with_avg(y_test, 'vote_average', avg)
######################################

#################### saving each column mean to use it in the test script

new_traind_x = pd.DataFrame(new_traind_x[0])
column_means = new_traind_x.mean(axis=0)
column_means = column_means.to_dict()
pickle.dump(column_means, open("x_train_all_columns_avg.pkl", 'wb'))
#####################################

#################### feature selection ###########################
x_corr=new_traind_x.iloc[:,:7].values
x_corrdf=new_traind_x.iloc[:,:7]
x_an=new_traind_x.iloc[:,7:]

#spearman feature selection for { " numerical input variable" and " numerical output variable " }
selected_feature_corr=spearman_feature_selection(x_corrdf, x_corr, t_y, threshold=0.1)

# anova feature selection for { " categorical input variable" and " numerical output variable " }
selected_feature_an=anova_feature_selection(x_an, t_y, 13)
#################################################################

################################### concatenation between the selected features
final_df = pd.concat([selected_feature_corr, selected_feature_an], axis=1)
final_df = dropcol(final_df, 'Português')
final_df = dropcol(final_df, 'Documentary')

df1_columns = final_df.columns.tolist()
df2_columns = new_tested_x.columns.tolist()
common_columns = list(set(df1_columns) & set(df2_columns))
new_tested_x=new_tested_x.drop([col for col in df2_columns if col not in common_columns], axis=1)
new_tested_x = new_tested_x.reindex(columns=final_df.columns)

# save the final train data
pickle.dump(final_df, open("training_data.pkl", 'wb'))


#######################################

regr_linear_regression, coef_linear_regression, mse_linear_regression, r2_linear_regression=linear_regression(final_df, new_tested_x, t_y, y_test)

# save the model to disk
pickle.dump(regr_linear_regression, open("linear_reg_model.pkl", 'wb'))


regr_random_forest, mse_random_forest, r2_random_forest=random_forest_regression(final_df, new_tested_x, t_y, y_test, n_estimators=100, max_depth=None)
pickle.dump(regr_random_forest, open("random_forest_model.pkl", 'wb'))

regr_decision_tree, mse_decision_tree, r2_decision_tree=decision_tree_regression(final_df, new_tested_x, t_y, y_test, max_depth=None)
pickle.dump(regr_decision_tree, open("decision_tree_regression_model.pkl", 'wb'))

regr_ridge_regression, mse_ridge_regression, r2_ridge_regression=ridge_regression(final_df, new_tested_x, t_y, y_test, alpha=1.0)
pickle.dump(regr_ridge_regression, open("ridge_regression_model.pkl", 'wb'))


print("linear regression accuracy : "  )
print(r2_linear_regression)
print("linear regression mse : " )
print(mse_linear_regression)
print("random forest regression accuracy : ")
print(r2_random_forest)
print("random forest regression mse" )
print(mse_random_forest)
print("ridge regression accuracy : ")
print(r2_ridge_regression)
print("ridge regression mse : " )
print(mse_ridge_regression)
