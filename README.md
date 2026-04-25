# Titanic Survival Prediction - Assignment 2

## Approach
This project performs end-to-end data processing and feature engineering to predict survival on the Titanic.

## Features Engineered
- FamilySize (SibSp + Parch + 1)
- IsAlone (binary indicator for solo passengers)
- Title (Mr, Mrs, Miss, Master, Rare from Name)
- Deck (first letter of Cabin number)
- AgeGroup (Child, Teen, Adult, Senior)
- FarePerPerson (Fare / FamilySize)
- Fare_log and Age_log (log transformations)
- FamilyCategory (Alone, Small, Large)

## Data Cleaning Decisions
- Age: Median imputation
- Embarked: Mode imputation
- Cabin: Extracted Deck letter, created missing indicator
- Removed duplicates
- No columns dropped (preserved for feature engineering)

## Key Findings
1. Women (Mrs/Miss titles) had ~75% survival rate vs ~20% for men
2. 1st class passengers had ~63% survival vs 3rd class ~24%
3. Family size of 2-4 had highest survival rates
4. Children and seniors had different survival patterns than adults
5. Higher fare correlates with higher survival

## Instructions to Run
1. Install requirements: `pip install -r requirements.txt`
2. Place train.csv and test.csv in `data/` folder
3. Run notebooks/Titanic_Feature_Engineering.ipynb
4. Or run scripts in order:
   - `python scripts/data_cleaning.py`
   - `python scripts/feature_engineering.py`
   - `python scripts/feature_selection.py`
