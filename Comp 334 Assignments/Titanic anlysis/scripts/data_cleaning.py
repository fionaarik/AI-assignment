# data_cleaning.py
import pandas as pd
import numpy as np

def clean_titanic_data(df):
    """Clean the Titanic dataset"""
    df_clean = df.copy()
    
    # Age: Fill with median
    df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
    
    # Cabin: Extract Deck and create missing indicator
    df_clean['Deck'] = df_clean['Cabin'].str[0]
    df_clean['Cabin_Missing'] = df_clean['Cabin'].isnull().astype(int)
    df_clean['Deck'] = df_clean['Deck'].fillna('Unknown')
    
    # Embarked: Fill with mode
    df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    return df_clean

if __name__ == "__main__":
    df = pd.read_csv('../data/train.csv')
    cleaned_df = clean_titanic_data(df)
    cleaned_df.to_csv('../data/train_cleaned.csv', index=False)
    print("Data cleaning complete. Saved to train_cleaned.csv")