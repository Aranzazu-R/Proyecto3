from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from app.models.bins import bins
from app.transformers import ColumnSelectorTransformer, BinningTransformer, WOETransformer

df = pd.read_csv('./app/models/datos.csv')

cols_to_keep = ['loan_amnt', 'term', 'int_rate', 'grade', 'sub_grade', 'home_ownership', 'annual_inc',
                'verification_status', 'pymnt_plan', 'purpose', 'addr_state', 'dti', 'delinq_2yrs', 'inq_last_6mths',
                'mths_since_last_delinq', 'open_acc', 'pub_rec', 'revol_util', 'initial_list_status', 'acc_now_delinq']

seed = 0

model = Pipeline([
    ('col selector', ColumnSelectorTransformer(columns=cols_to_keep)),
    ('bins', BinningTransformer(bins=bins)),
    ('woe', WOETransformer(columns=cols_to_keep)),
    ('xgb', XGBClassifier(n_estimators=100, random_state=seed))
])

x_train = df.loc[:, cols_to_keep]
y_train = df["status"]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)

y_train_nd = [1 if val == 0 else 0 for val in y_train.values]

# Entrenar el modelo
model.fit(x_train, y_train)

# guardar el modelo entrenado
with open("../model.pkl", "wb") as file:
    pickle.dump(model, file)

# cargar el modelo entrenado
with open("../model.pkl", 'rb') as file:
    model = pickle.load(file)

# hacer predicciones en el conjunto de prueba
y_test_nd = [1 if val == 0 else 0 for val in y_test.values]
y_pred = model.predict(x_test)

# calcular el accuracy del modelo
accuracy = sum(y_pred == y_test) / len(y_test)
print(f"Accuracy: {accuracy:.2f}")
