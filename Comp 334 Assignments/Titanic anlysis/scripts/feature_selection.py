# feature_selection.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

def get_feature_importance(df, target_col='Survived'):
    """Calculate feature importance using Random Forest"""
    
    # Prepare data
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                'FamilySize', 'IsAlone', 'FarePerPerson', 'Title', 'Deck']
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df[features], columns=['Sex', 'Title', 'Deck'], drop_first=True)
    df_encoded = df_encoded.fillna(0)
    
    X = df_encoded
    y = df[target_col]
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return importance_df, X.columns

def select_top_features(importance_df, n=10):
    """Select top n features"""
    return importance_df.head(n)['Feature'].tolist()

if __name__ == "__main__":
    df = pd.read_csv('../data/train_featured.csv')
    importance_df, _ = get_feature_importance(df)
    
    print("Top 10 Features:")
    print(importance_df.head(10))
    
    selected = select_top_features(importance_df, 10)
    print(f"\nSelected features: {selected}")
    
    importance_df.to_csv('../data/feature_importance.csv', index=False)