import pandas as pd
import json
import pickle
import os

print(os.getcwd())
with open('impute_dict.json', 'r') as f:
    data = json.load(f)

with open('scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

def check_cols_existence(df):
    if not isinstance(df, pd.DataFrame):
        return False

    if df.empty:
        return False

    required_columns = data["numerical_columns"] + data["categorical_columns"]
    if not all(col in df.columns for col in required_columns):
        return False

    else:
        return True



def primary_preprocess(df, cols_to_remove = data['cols_to_remove']):

    df = df.replace('?', pd.NA)
    cols_to_drop = [col for col in cols_to_remove if col in df.columns ]
    df = df.drop(columns=cols_to_drop)
    return df


def impute_from_dict(df, impute_dict=data):
    for col in impute_dict['cols_to_impute_with_mode']:
        if col in df.columns and col in impute_dict['mode']:
            df[col].fillna(impute_dict['mode'][col], inplace=True)

    for col in impute_dict['cols_to_impute_with_median']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if col in df.columns and col in impute_dict['median']:
            df[col].fillna(impute_dict['median'][col], inplace=True)

    for col in impute_dict['cols_to_impute_with_unknown']:
        df[col] = df[col].fillna('unknown')

    return df


def transform_categorical_data(df, encoders, categorical_columns):
    df_concat = pd.DataFrame()
    for col in categorical_columns:
        encoded_data = encoders[col].transform(df[[col]]).toarray()
        column_names = encoders[col].get_feature_names_out([col])
        encoded_df = pd.DataFrame(encoded_data, columns=column_names)
        df_concat = pd.concat([df_concat, encoded_df], axis=1)
    return df_concat


def transform_numerical_data(df, scalers, numerical_columns):
    df_normalized = pd.DataFrame()
    for col in numerical_columns:
        scaled_data = scalers[col].transform(df[[col]])
        scaled_df = pd.DataFrame(scaled_data, columns=[col])
        df_normalized = pd.concat([df_normalized, scaled_df], axis=1)
    return df_normalized


def transform_data(df):
    df = primary_preprocess(df, cols_to_remove=data['cols_to_remove'])
    for col in data["cols_to_impute_with_median"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df_imputed = impute_from_dict(df, data)
    df_scaled = transform_numerical_data(df_imputed, scalers, data["numerical_columns"])
    df_encoded = transform_categorical_data(df_imputed, encoders, data["categorical_columns"])
    x = pd.concat([df_scaled, df_encoded], axis=1)
    x.columns = ["".join(e for e in col if e.isalnum() or e == '_') for col in x.columns]
    return x
