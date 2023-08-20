import copy
import json
import pandas as pd
import pickle
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

pd.set_option('display.max_columns', None)

def replace_nulls_and_zeros_with_avg(df,column_name, avg):

    df_copy = df.copy()
    df_copy[column_name] = df_copy[column_name].replace([0, np.nan], np.nan).fillna(avg)

    return df_copy

################################################################################

def replace_nulls_and_zeros_with_avg_in_y(df, column_name):
    df_copy = df.copy()

    avg = df_copy[column_name].replace([0, np.nan], np.nan).mean()

    df_copy[column_name] = df_copy[column_name].replace([0, np.nan], np.nan).fillna(avg)

    return df_copy

###############################################################################

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

#############################################################################

def dropcol(df, column_name):
    df = df.drop(columns=[column_name]).copy()
    return df

#############################################################################

def encode_column(df, column):
    le = LabelEncoder()
    df[f'{column}_encoded'] = le.fit_transform(df[column])
    df.drop(column, axis=1, inplace=True)
    df.rename(columns={f'{column}_encoded': column}, inplace=True)
    return df
#############################################################################

def filter_frequent_items(df, column_name, n):

    filtered_df = df.copy()

    column_data = filtered_df[column_name]

    item_counts = column_data.value_counts()

    frequent_items = item_counts[item_counts >= n].index.tolist()

    filtered_df[column_name] = filtered_df[column_name].where(filtered_df[column_name].isin(frequent_items), other=None)

    return filtered_df

#############################################################################

def replace_null_with_string(df, column_name, string):
    new_df = df.copy()
    new_df[column_name].fillna(string, inplace=True)
    return new_df

#############################################################################

def split_date(df, date_column):

    df[['month', 'day', 'year']] = df[date_column].str.split('/', expand=True)
    df = df.drop(columns=[date_column, 'day'])
    return df

#############################################################################

def preprocess_data_train(train_data,train_y):
    X_train = train_data.copy()
    Y_train =train_y.copy()

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

    # Drop homepage and id columns (unique values)
    X_train = dropcol(X_train, 'homepage')
    X_train = dropcol(X_train, 'id')

    # Drop status, tagline, title, and overview columns
    X_train = dropcol(X_train, 'status')
    X_train = dropcol(X_train, 'tagline')
    X_train = dropcol(X_train, 'title')
    X_train = dropcol(X_train, 'overview')
    X_train = dropcol(X_train, 'original_title')
    mode = fill_nulls_with_mode(Y_train, 'Rate')
    Y_train['Rate'].fillna(mode, inplace=True)
    Y_train = encode_column(Y_train, 'Rate')
    return X_train,Y_train
########################################################################
def fill_nulls_with_mode(df, column):
    mode = df[column].mode()[0]
    #df[column].fillna(mode, inplace=True)
    return mode
############################################################################

def preprocess_data_test(train_data, test_data,train_y,test_y):
    X_train = train_data.copy()
    X_test = test_data.copy()
    Y_train=train_y.copy()
    Y_test=test_y.copy()
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
    #print(X_test)
    mode = fill_nulls_with_mode(Y_train, 'Rate')
    Y_test['Rate'].fillna(mode, inplace=True)
    Y_test = encode_column(Y_test, 'Rate')
    return X_test,Y_test

def anova_feature_selection(X, y, k):
    y = np.ravel(y)
    # Select the top k features based on ANOVA F-value
    f_values, p_values = f_classif(X, y)

    # Rank features by F-value and select top features
    ranked_features = np.argsort(f_values)[::-1]
    selected_indices = ranked_features[:k]
    selected_features = X.iloc[:, selected_indices]

    # print(selected_features)
    return selected_features
############################################################################
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def chi_squared_feature_selection(X, y, k):

    # Select the top k features based on chi-squared test
    skb = SelectKBest(chi2, k=k)
    X_new = skb.fit_transform(X, y)

    # Get the indices of the selected features
    feature_indices = skb.get_support(indices=True)

    # Get the names of the selected features
    feature_names = list(X.columns[feature_indices])

    # Create a DataFrame with the selected features and their names
    X_new = pd.DataFrame(X_new, columns=feature_names)

    return X_new
#############################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def random_forest_classifier(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=None, random_state=None):
    y_train = y_train.iloc[:, 0]
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Random_Forest_Classifier Accuracy:", accuracy)
    return rf, accuracy

#############################################################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
def knn(X_train, y_train, X_test, y_test, n_neighbors=5):
    # Initialize KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Train the model on the training data and record the total training time
    start_train_time = time.time()
    knn.fit(X_train, y_train)
    end_train_time = time.time()
    training_time = end_train_time - start_train_time

    # Make predictions on the test data and record the total test time
    start_test_time = time.time()
    y_pred = knn.predict(X_test)
    end_test_time = time.time()
    test_time = end_test_time - start_test_time

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("KNN Classifier Accuracy:", accuracy)

    # Plot bar graphs for accuracy, training time, and test time
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(["Accuracy", "Training Time", "Test Time"], [accuracy, training_time, test_time])
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Time / Accuracy")
    ax.set_title("KNN Classifier Performance")
    plt.show()

    return knn, accuracy, training_time, test_time
#############################################################################
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
def svm_classifier(X_train, y_train, X_test, y_test, C=1.0, kernel='rbf', random_state=None):
    y_train = y_train.iloc[:, 0]
    svm = SVC(C=C, kernel=kernel, random_state=random_state)

    # Train the model on the training data and measure training time
    start_train_time = time.time()
    svm.fit(X_train, y_train)
    end_train_time = time.time()
    train_time = end_train_time - start_train_time

    # Make predictions on the test data and measure test time
    start_test_time = time.time()
    y_pred = svm.predict(X_test)
    end_test_time = time.time()
    test_time = end_test_time - start_test_time

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("svm_classifier Accuracy:", accuracy)

    # Plot the training time, testing time, and accuracy using bar graphs
    x_labels = ['Training Time', 'Testing Time', 'Accuracy']
    y_values = [train_time, test_time, accuracy]
    colors = ['green', 'blue', 'orange']
    plt.bar(x_labels, y_values, color=colors)
    plt.title('SVM Classifier')
    plt.xlabel('Metrics')
    plt.ylabel('Time/Accuracy')
    plt.show()

    return svm
#############################################################################
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def gaussian_nb(X_train, y_train, X_test, y_test):
    # Initialize Gaussian Naive Bayes Classifier
    gnb = GaussianNB()

    # Train the model on the training data and record the total training time
    start_train_time = time.time()
    gnb.fit(X_train, y_train)
    end_train_time = time.time()
    training_time = end_train_time - start_train_time

    # Make predictions on the test data and record the total test time
    start_test_time = time.time()
    y_pred = gnb.predict(X_test)
    end_test_time = time.time()
    test_time = end_test_time - start_test_time

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Gaussian Accuracy:", accuracy)

    # Plot bar graphs for accuracy, training time, and test time
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(["Accuracy", "Training Time", "Test Time"], [accuracy, training_time, test_time])
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Time / Accuracy")
    ax.set_title("Gaussian Naive Bayes Classifier Performance")
    plt.show()

    return gnb, accuracy, training_time, test_time
#############################################################################
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import time
import matplotlib.pyplot as plt

def decision_tree(X_train, y_train, X_test, y_test, max_depth=None):
    # Initialize Decision Tree Classifier
    dt = DecisionTreeClassifier(max_depth=max_depth)

    # Train the model on the training data and record the total training time
    start_train_time = time.time()
    dt.fit(X_train, y_train)
    end_train_time = time.time()
    training_time = end_train_time - start_train_time

    # Make predictions on the test data and record the total test time
    start_test_time = time.time()
    y_pred = dt.predict(X_test)
    end_test_time = time.time()
    test_time = end_test_time - start_test_time

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Decision Tree Accuracy:", accuracy)

    # Plot bar graphs for accuracy, training time, and test time
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(["Accuracy", "Training Time", "Test Time"], [accuracy, training_time, test_time])
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Time / Accuracy")
    ax.set_title("Decision Tree Classifier Performance")
    plt.show()

    return dt, accuracy, training_time, test_time

#############################################################################
pd.set_option('display.max_columns', None)
data = pd.read_csv('movies-classification-dataset.csv')
x=data.iloc[:,:19]
y=data.iloc[:,19:20]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
X_train,y_train=preprocess_data_train(X_train,y_train)
pickle.dump(y_train, open("y_train.pkl", 'wb'))

X_test,y_test=preprocess_data_test(X_train,X_test,y_train,y_test)
#############
mode = fill_nulls_with_mode(y_train, 'Rate')
pickle.dump(mode, open("classification_y_train_avg.pkl", 'wb'))
#############
column_means = X_train.mean(axis=0)
column_means = column_means.to_dict()
pickle.dump(column_means, open("x_train_all_columns_avg.pkl", 'wb'))
#############
X_an=X_train.iloc[:,:7]
X_chi=X_train.iloc[:,7:]

selected_an=anova_feature_selection(X_an, y_train, 3)
selected_chi=chi_squared_feature_selection(X_chi, y_train, 9)

selected_chi = selected_chi.reset_index(drop=True)
selected_an = selected_an.reset_index(drop=True)

final_df = pd.concat([selected_chi, selected_an], axis=1)
pickle.dump(final_df, open("final_x_train.pkl", 'wb'))


df1_columns = final_df.columns.tolist()
df2_columns = X_test.columns.tolist()
common_columns = list(set(df1_columns) & set(df2_columns))
X_test=X_test.drop([col for col in df2_columns if col not in common_columns], axis=1)
X_test =X_test.reindex(columns=final_df.columns)


dt, dt_accuracy, dt_training_time, dt_test_time = decision_tree(final_df, y_train, X_test, y_test, max_depth=None)
pickle.dump(dt, open("train_decision_tree_model.pkl", 'wb'))

gnb, gnb_accuracy, gnb_training_time, gnb_test_time = gaussian_nb(final_df, y_train, X_test, y_test)
pickle.dump(gnb, open("gaussian_model.pkl", 'wb'))

knn, knn_accuracy, knn_training_time, knn_test_time = knn(final_df, y_train, X_test, y_test, n_neighbors=5)
pickle.dump(knn, open("knn_model.pkl", 'wb'))

rf, rf_accuracy = random_forest_classifier(final_df, y_train, X_test, y_test, n_estimators=100, max_depth=None, random_state=None)
pickle.dump(rf, open("random_forest_classifier_model.pkl", 'wb'))

svm = svm_classifier(final_df, y_train, X_test, y_test, C=1.0, kernel='rbf', random_state=None)
pickle.dump(svm, open("svm_classifier_model.pkl", 'wb'))