import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

# cargar el modelo entrenado
with open("./app/model.pkl", 'rb') as file:
    model = pickle.load(file)

df = pd.read_csv('./app/models/datos.csv')
cols_to_keep = ['loan_amnt', 'term', 'int_rate', 'grade', 'sub_grade', 'home_ownership', 'annual_inc',
                'verification_status', 'pymnt_plan', 'purpose', 'addr_state', 'dti', 'delinq_2yrs', 'inq_last_6mths',
                'mths_since_last_delinq', 'open_acc', 'pub_rec', 'revol_util', 'initial_list_status', 'acc_now_delinq']

x_train = df.loc[:, cols_to_keep]
y_train = df["status"]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)

pred_val: pd.DataFrame = x_test.iloc[[0], :]
print(pred_val.values)
# model.predict()
