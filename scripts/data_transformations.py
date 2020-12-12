# -*- coding: utf-8 -*-
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
'''
Function Goal: Center and scale features without modifying label
Function Input: df: contains label and features
Function Output: df_normalized: df of normalized features
'''
def scenter_scale_features(df):
    
    ## list of features
    features = df.columns[1:]
    
    ## Normalize features
    df[features] = (df[features]-df[features].mean())/df[features].std()
    
    return df


'''
Function Goal: Power transform skewed featuers using box-cox method
Function Input: df: contains label and features
                skew_threshold: sets the skew_threshold which determines which features are to be transformed
                x_shift: sets the shift in order to do a log(x + x_shift) transformation to avoid log(0)
Function Output: df: contains label with power transformed features
                 pt: fitted power transform object, to be used on transforming new data
'''
def power_transform_skewed_features(df, skew_threshold=2, x_shift=1):
    
    ## Calculate skew and filter by skew threshold
    skewed = df.skew()
    skewed_features = skewed[(skewed > skew_threshold ) | (skewed < -skew_threshold)]
    
    ## Create a copy dataframe to make transformations on
    df_transform = df.copy()
    
    ## Shift data by 1 to avoid log(0)
    df_transform[skewed_features.index] = df_transform[skewed_features.index] + x_shift
    
    ## Fit power transformer
    pt = PowerTransformer(method='box-cox', standardize=False).fit(df_transform[skewed_features.index])
    
    ## Power transform skewed features
    df[skewed_features.index] = pt.transform(df_transform[skewed_features.index])
    
    return df, pt

#reduces dimensionality after Feature Correlation
def reduce_dimension(df, treshold):
    #Compute the correlation matrix
    corr = df.drop('Label', axis=1).corr()
    i = corr[(corr > treshold) | (corr < -treshold) ].fillna(0)
    i = i[i<1.0]
    i = i[i>-1.0]
    new = i**2
    new[new.isnull()==True]=0
    sumnew=new.sum(axis=0)
    ind_=sumnew[sumnew>1]
    ind_=ind_.index[1:]
    final=df.drop(ind_,axis=1)
    return final

def test_LogReg(df):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('Label', axis=1), df['Label'], test_size=0.2)
    LR = LogisticRegression(max_iter=5000).fit(X_train, y_train.to_numpy().squeeze())
    predLR = LR.predict(X_test)
    accuracyLR = LR.score(X_test, y_test.to_numpy().squeeze())
    return accuracyLR