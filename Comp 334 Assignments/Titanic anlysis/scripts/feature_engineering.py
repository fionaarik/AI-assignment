# feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df):
    """Create new features from Titanic dataset"""
    df_feat = df.copy()
    
    # Family size
    df_feat['FamilySize'] = df_feat['SibSp'] + df_feat['Parch'] + 1
    df_feat['IsAlone'] = (df_feat['FamilySize'] == 1).astype(int)
    
    # Title extraction
    df_feat['Title'] = df_feat['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
        'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
        'Capt': 'Rare', 'Sir': 'Rare'
    }
    df_feat['Title'] = df_feat['Title'].map(title_mapping).fillna('Rare')
    
    # Age groups
    def age_group(age):
        if pd.isna(age): return 'Unknown'
        if age < 12: return 'Child'
        elif age < 18: return 'Teen'
        elif age < 60: return 'Adult'
        else: return 'Senior'
    df_feat['AgeGroup'] = df_feat['Age'].apply(age_group)
    
    # Fare per person
    df_feat['FarePerPerson'] = df_feat['Fare'] / df_feat['FamilySize']
    
    # Log transformations
    df_feat['Fare_log'] = np.log1p(df_feat['Fare'])
    df_feat['Age_log'] = np.log1p(df_feat['Age'])
    
    # Family category
    def family_category(size):
        if size == 1: return 'Alone'
        elif size <= 4: return 'Small'
        else: return 'Large'
    df_feat['FamilyCategory'] = df_feat['FamilySize'].apply(family_category)
    
    return df_feat

if __name__ == "__main__":
    df = pd.read_csv('../data/train_cleaned.csv')
    df_feat = engineer_features(df)
    df_feat.to_csv('../data/train_featured.csv', index=False)
    print("Feature engineering complete. Saved to train_featured.csv")