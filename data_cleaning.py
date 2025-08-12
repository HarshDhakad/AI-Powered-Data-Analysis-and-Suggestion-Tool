import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def clean_data(features_df):
    
    df = features_df.copy()

    df.drop_duplicates(inplace=True)

    constant_col = [col for col in df.columns if df[col].nunique() == 1]
    df.drop(columns=constant_col, inplace=True)

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Step 1: Select categorical columns but avoid high-cardinality ones
    max_unique_values = 50  # adjust as needed
    categorical_cols = [
        col for col in df.select_dtypes(include=['object']).columns
        if df[col].nunique() <= max_unique_values
    ]

    # Drop columns that have too many unique values (like IDs, names, etc.)
    high_card_cols = [
        col for col in df.select_dtypes(include=['object']).columns
        if df[col].nunique() > max_unique_values
    ]
    if high_card_cols:
        print(f"Dropping high-cardinality columns: {high_card_cols}")
        df.drop(columns=high_card_cols, inplace=True)

    # Step 2: Fill missing categorical values
    for col in categorical_cols:
        df[col].fillna('Unknown', inplace=True)

    # Step 3: One-hot encode with sparse=True (memory efficient)
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(df[categorical_cols])

    # Keep it sparse or convert to DataFrame (if needed)
    encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(categorical_cols))

    df.drop(columns=categorical_cols, inplace=True)
    df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    df.reset_index(drop=True, inplace=True)

    return df  
