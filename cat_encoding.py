"""
Encoding Methodologies

Method 1: Label encoding
Method 2: OneHot encoding
Method 3: Feature Hashing
Method 4: Target encoding
Method 6: K-Fold target Encoding

"""
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def cat_encoding(data,feature_name,target_feature,method):
    """
    params:
    data: the dataset
    feature_name: categorical feature
    target_feature: Target variable
    method: label_encoding,target_encoding
    """
    df = data.copy()
    
    if method == "label_encoding":
        label = LabelEncoder()
        if(df[feature_name].dtype == 'object'):
            df[feature_name] = label.fit_transform(df[feature_name])
        else:
            df[feature_name] = feature_name
    
    # elif method == "one_hot_encoding":
    #     one = OneHotEncoder()
    #     if(df[feature_name].dtype == 'object'):
    #         df[feature_name] = one.fit_transform(df[feature_name])
    #     else:
    #         df[feature_name] = feature_name

    elif method == "target_encoding":
        if(df[feature_name].dtype == 'object'):
            target = dict(df.groupby(feature_name)[target_feature].agg('sum') / df.groupby(feature_name)[target_feature].agg('count'))
            df[feature_name] = df[feature_name].replace(target).values
        else:
            df[feature_name] = feature_name

    return df


