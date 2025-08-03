
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from data_cleaning import clean_data

problem_to_task = {
    'sales decline': 'regression',
    'customer churn': 'classification',
    'fraud detection': 'classification',
    'employee retention': 'classification',
    'supply chain issue': 'regression',
    'financial loss': 'regression',
    'low productivity': 'regression',
    'customer complaints': 'classification',
    'market competition': 'classification',
    'data security concern': 'classification',
    'regulatory compliance risk': 'classification'
}

def detect_task_from_data(target_series):
    if target_series.nunique() <= 10 or target_series.dtype == 'object':
        return 'classification'
    elif target_series.dtype in ['float64', 'int64']:
        return 'regression'
    else:
        return 'classification'  # fallback



def train_and_predict(df, target_column, model):
    features_df = df.drop(columns=[target_column])
    target = df[target_column]

    cleaned_features_df = clean_data(features_df)

    X_train, X_test, y_train, y_test = train_test_split(cleaned_features_df, target, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return model, predictions

