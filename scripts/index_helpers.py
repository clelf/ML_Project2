# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd


'''
Example: How to use
df = read_and_merge_segmented_data()
df = index_df_by_person(df)
X_train, X_test, y_train, y_test = train_test_split_on_index(features = df.drop("Label", axis=1), 
                                                             label = df["Label"])

'''

'''
Function Goal: Reads (segmented )labels and features and merge into dataframe - optionally include data
Function Input: segmentation: False if "no segmentation", True if "fine" or "coarse". By default, True
                fine_segmentation: True = import fine segmentation data - False = import coarse segmentation data
                exlude_expert: True = excludes column for which expert labeled the data
                exlude_meta_data: True = exludes meta data
Function Output: df - merged DataFrame
                
'''
def read_and_merge_data(segmentation = True, fine_segmentation=True, exclude_expert=True, exclude_meta_data=True):
    
    if segmentation:
      ## Set path
      if(fine_segmentation):
          PATH_labels = Path('../data/labels_fine_segmentation.csv')
          PATH_features = Path('../data/features_fine_segmentation.csv')
      else:
          PATH_labels = Path('../data/labels_coarse_segmentation.csv')
          PATH_features = Path('../data/features_coarse_segmentation.csv')
    else:
      PATH_features = Path("../data/features_no_segmentation.csv")
      PATH_labels = Path("../data/labels_no_segmentation.csv")
    
    ## Read data
    labels_segmentation = pd.read_csv(PATH_labels, header=0, index_col=0)
    features_raw = pd.read_csv(PATH_features, header=0, index_col=0)
    
    ## Translate floats of categorical to int
    labels_segmentation["Label"] = labels_segmentation["Label"].astype(int)
    
    ## Drop "File_Name" from features_raw because labels already has it. Will be merged
    features_segmentation = features_raw.drop('File_Name', axis=1)
    
    ## Merge features and labels to same dataset
    df = pd.merge(labels_segmentation, features_segmentation, left_index=True, right_index=True)
    
    ## List of features to exlude from the features dataframe
    drop_list = []
    if(exclude_expert):
        drop_list = drop_list + ['Expert']
    if(exclude_meta_data):
        drop_list = drop_list + ['Age', 'Gender', 'Resp_Condition', 'Symptoms']
        
    ## Drop labels from list
    if(len(drop_list) != 0):
        df = df.drop(drop_list, axis=1)
    
    return df


'''
Function Goal: Create hierarchical index based on column "File_Name", splitted by person and n_recording
Function Input: df: DataFrame to index
Function Output: df: Indexed DataFrame
P.S: Keep labels and features in same dataframe to avoid having to match later. numerical indices 
are removed with this function
'''
def index_df_by_person(df):
    
    ## Create new dataframe split by '_', one column per value
    File_Name_split = df["File_Name"].str.split('_', expand=True)
    
    ## Rename columns - Done to improve clarity of index headers in the final df
    File_Name_split.rename(columns={0:'File_Name_split', 1:'File_n_recording'}, inplace=True)
    
    ## Merge in splitted columns based on their numerical index
    df = pd.merge(df, File_Name_split, left_index=True, right_index=True)
    
    ## Create hierarchical index based on splitted column
    df = df.set_index(['File_Name_split', 'File_n_recording'])
    
    ## Drop splitted column
    df = df.drop('File_Name', axis=1)
    
    return df


'''
Function Goal: Create a training and testing set based on index - USed here to split sets by people
Function Input: features: Dataframe of features
                label: DataFrame of labels 
                level: Which level of the hierarchical index to split on. Default is 0
                test_size: size of test set
                random_state: int, for reproducability
Function Output: X_train, X_test, y_train, y_test - split by index

'''
def train_test_split_on_index(features, label, level=0, test_size=0.2, random_state = 42):
    
    ## Split indices into training and testing
    X_train_ind, X_test_ind, y_train_ind, y_test_ind = train_test_split(features.index.levels[level], 
                                                                        label.index.levels[level],
                                                                        test_size=test_size, 
                                                                        random_state=random_state)
    
    ## Slice features by split indices (persons)
    X_train = features.loc[X_train_ind]
    X_test = features.loc[X_test_ind]
    y_train = label.loc[y_train_ind]
    y_test = label.loc[y_test_ind]
    return X_train, X_test, y_train, y_test

'''
Function Goal: translate categorical features from float to int
'''
def categorical_float_to_int(df):
    categorical_features = df.drop('Label',axis=1).columns[:19]
    df[categorical_features] = df[categorical_features].astype(int)
    return df

'''
Function Goal: Take columns starting with EEPD and translate them to dummy variables
'''
def categorical_to_dummy(df):
    categorical_features = df.drop('Label',axis=1).columns[:19]
    df = pd.get_dummies(df, columns=categorical_features)
    return df


def separate_expert(df):
    '''
    Function Goal: separate a data set into three data sets according to the expert
    Input: a data set (DataFrame) that should contain a column labeled as 'Expert' and a column containing the labels
    Output: 3 datasets containing their respective labels, and without the 'Expert' column
    '''
    df1 = df[df['Expert']==1.0]
    df2 = df[df['Expert']==2.0]
    df3 = df[df['Expert']==3.0]
    
    return df1, df2, df3

