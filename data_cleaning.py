import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def clean_data(features_df):
    
    df = features_df.copy()

    df.drop_duplicates(inplace=True)

    constant_col = [col for col in df.columns if df[col].nunique() == 1]
    df.drop(columns=constant_col, inplace=True)

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    for col in categorical_cols:
        df[col].fillna('Unknown', inplace=True)

    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))

        df.drop(columns=categorical_cols, inplace=True)
        df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    df.reset_index(drop=True, inplace=True)

    return df  
