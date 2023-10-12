import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.svm import SVR
from sklearn.metrics import r2_score
#uploaded_file = st.file_uploader("")
student_df = pd.read_csv('.\data\student-mat.csv',delimiter=';')

independent_vars = ['school', 'sex', 'age', 'address', 'studytime', 'absences', 'G1', 'G2']
regression = st.selectbox('What technique would you like to use?',
                          ('Linear Regression', 'Gradient Boosting Regressor', 'Support Vector Machine (SVM) Regression', 'Compare R² values'))

# Linear regression
encoder = ce.OrdinalEncoder(cols=['school', 'sex', 'address'])
df_encoded = encoder.fit_transform(student_df)

X = df_encoded[independent_vars]  
y = df_encoded['G3']

X_train_L, X_test_L, y_train_L, y_test_L = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train_L, y_train_L) 

# Gradient Boosting Regressor
student_df_GBR = student_df.copy()
X = df_encoded[independent_vars]
y = df_encoded['G3']

X_train_G, X_test_G, y_train_G, y_test_G = train_test_split(X, y, test_size=0.2, random_state=0)

gb_regressor = GradientBoostingRegressor(n_estimators=100, random_state=0)
gb_regressor.fit(X_train_G, y_train_G)

# Support Vector Machine (SVM) Regression
student_df_SVM = student_df.copy()
X = df_encoded[independent_vars]
y = df_encoded['G3']

X_train_S, X_test_S, y_train_S, y_test_S = train_test_split(X, y, test_size=0.2, random_state=0)

svr_regressor = SVR(kernel='rbf', C=1.0, epsilon=0.2)
svr_regressor.fit(X_train_S, y_train_S)


if regression == 'Linear Regression':
    y_pred_L = model.predict(X_test_L).round().astype(int)

    result_df_L = pd.DataFrame({'Actual': y_test_L[:25], 'Predicted': y_pred_L[:25]})
    st.table(result_df_L)

elif regression == 'Gradient Boosting Regressor':
    y_pred_G = gb_regressor.predict(X_test_G).round().astype(int)

    result_df = pd.DataFrame({'Actual': y_test_G[:25], 'Predicted': y_pred_G[:25]})
    st.table(result_df)

elif regression == 'Support Vector Machine (SVM) Regression':
    y_pred_S = svr_regressor.predict(X_test_S).round().astype(int)

    result_df = pd.DataFrame({'Actual': y_test_S[:25], 'Predicted': y_pred_S[:25]})
    st.table(result_df)

elif regression == 'Compare R² values':
    y_pred_L = model.predict(X_test_L).round().astype(int)
    r_squared_L = r2_score(y_test_L, y_pred_L)

    y_pred_G = gb_regressor.predict(X_test_G).round().astype(int)
    r_squared_G = r2_score(y_test_G, y_pred_G)

    y_pred_S = svr_regressor.predict(X_test_S).round().astype(int)
    r_squared_S = r2_score(y_test_S, y_pred_S)

    compare = {'Linear regression':[r_squared_L],'Gradient Boosting Regressor':[r_squared_G],'Support Vector Machine (SVM) Regression':[r_squared_S]}
    compare_df = pd.DataFrame(data=compare)
    st.text('Compare R² values:')
    st.table(compare_df)

# streamlit
st.title('Predict Final Score.')

new_student_features = {}
school = st.text_input('What school are you attending (For Gabriel Pereira insert GP, for Mousinho da Silveira insert MS): ').upper().strip()
sex = st.text_input('What sex are you (For female insert F, for male insert M): ').upper().strip()
age = st.text_input('How old are you: ')
if age != '':
    age = int(age)
address = st.text_input('Do you live in an urban (insert U) or a rural (insert R) neighboorhood: ').upper().strip()
studytime = st.text_input('How long do you study: ')
if studytime != '':
    studytime = int(studytime)
absences = st.text_input('How manny times were you absent: ')
if absences != '':
    absences = int(absences)
G1 = st.text_input('What was your first score (0-20): ')
if G1 != '':
    G1 = int(G1)
G2 = st.text_input('What was your second score (0-20): ')
if G2 != '':
    G2 = int(G2)

if school != '' and sex != '' and age != '' and address != '' and studytime != '' and absences != '' and G1 != '' and G2 != '':
    if school == 'GP':
        new_student_features['school'] = 1
    elif school == 'MS':
        new_student_features['school'] = 2

    if sex == 'F':
        new_student_features['sex'] = 1
    elif sex == 'M':
        new_student_features['sex'] = 2

    new_student_features['age'] = age
    if address == 'U':
        new_student_features['address'] = 1
    elif address == 'R':
        new_student_features['address'] = 2

    if studytime < 2:
        new_student_features['studytime'] = 1
    elif studytime < 5:
        new_student_features['studytime'] = 2
    elif studytime < 10:
        new_student_features['studytime'] = 3
    else:
        new_student_features['studytime'] = 4

    new_student_features['absences'] = absences
    new_student_features['G1'] = G1
    new_student_features['G2'] = G2

    new_student_df = pd.DataFrame([new_student_features])

    l_predicted = model.predict(new_student_df).round().astype(int)
    text = "With Linear regression I predict a final score of "+str(l_predicted[0])+'.'
    st.text(text)

    g_predicted = gb_regressor.predict(new_student_df).round().astype(int)
    text = "With Gradient Boosting Regressor I predict a final score of "+str(g_predicted[0])+'.'
    st.text(text)

    s_predicted = svr_regressor.predict(new_student_df).round().astype(int)
    text = "With Support Vector Machine (SVM) Regression I predict a final score of "+str(s_predicted[0])+'.'
    st.text(text)

#streamlit run ./AI/Homework/Task2/Homework_Task2_streamlit.py