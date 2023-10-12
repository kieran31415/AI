import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import category_encoders as ce

student_df = pd.read_csv('./data/student-mat.csv',delimiter=';')

independent_vars = ['school', 'sex', 'age', 'address', 'studytime', 'absences', 'G1', 'G2']

encoder = ce.OrdinalEncoder(cols=['school', 'sex', 'address'])
df_encoded = encoder.fit_transform(student_df)

X = df_encoded[independent_vars]  
y = df_encoded['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train) 

y_pred = model.predict(X_test).round().astype(int)

# streamlit
st.title('Predict Final Score with Linear regression.')

new_student_features = {}
school = st.text_input('What school are you attending (For Gabriel Pereira insert GP, for Mousinho da Silveira insert MS): ').upper().strip()
sex = st.text_input('What sex are you (For female insert F, for male insert M): ').upper().strip()
age = st.number_input('How old are you: ')
address = st.text_input('Do you live in an urban (insert U) or a rural (insert R) neighboorhood: ').upper().strip()
studytime = st.number_input('How long do you study: ')
studytime = int(studytime)
absences = st.number_input('How manny times were you absent: ')
G1 = st.number_input('What was your first score: ')
G2 = st.number_input('What was your second score: ')

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

predicted_G3 = model.predict(new_student_df)

st.text("Predicted final score:", predicted_G3[0])