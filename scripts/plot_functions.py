# -*- coding: utf-8 -*-
import seaborn as sns
import matplotlib.pyplot as plt

'''
Function Goal: Boxplot all 66 features
Function Input: df: contains label and features
'''

def boxplot_features(df):
    features = df.drop('Label', axis=1)
    features_name = features.columns
    
    fig, ax = plt.subplots(3, 1, figsize=(45,30))
    
    g1 = sns.boxplot(data=features[features_name[:22]], palette="Blues", ax=ax[0])
    g1.set_xticklabels(g1.get_xticklabels(),rotation=45);
    
    g2 = sns.boxplot(data=features[features_name[22:44]], palette="Blues", ax=ax[1])
    g2.set_xticklabels(g2.get_xticklabels(),rotation=45);
    
    g3 = sns.boxplot(data=features[features_name[44:66]], palette="Blues", ax=ax[2])
    g3.set_xticklabels(g3.get_xticklabels(),rotation=45);