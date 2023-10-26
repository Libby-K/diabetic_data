import pandas as pd
import joblib
from src.helper_functions import transform_data, check_cols_existence

class Predictor:

    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, df):
        if not check_cols_existence(df):
            return 'cant predict'
        x = transform_data(df)
        preds = self.model.predict(x)
        mapping = {0: 'NO', 1: '>30', 2: '<30'}
        converted_preds = [mapping[val] for val in preds]
        return converted_preds


if __name__ == '__main__':
    model_path = "src/xgb_selected_features_best_model.pkl"
    predictor = Predictor(model_path)
    # Can be replaced to read from database or accept dictionary and create dataframe from if
    input_data = pd.read_csv("src/test_raw_sample.csv")
    predictions = predictor.predict(input_data)
    print(predictions)



