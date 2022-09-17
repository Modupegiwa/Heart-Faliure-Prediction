# Exploratory Data Analysis On Heart Disease Data Set

## Heart Disease Data Set
This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. 
It contains 76 attributes, including the predicted attribute, but all published experiments refer to using a subset of 14 of them. 
This data set consists of several indicator variables and one target(Heart Disease Status), indicator variables includes the age, 
sex(gender), chest pain type, number of major vessels, etc.

It consists of 1025 patients with 14 features set:
1. age (Age in years): All patients are between the age 29 - 77 years
2. sex: (1= male, 0= female): Their are more males than females
3. cp (Chest Pain Type): [0= Asymptomatic, 1= Atypical angina, 2= Non-anginal pain, 3= Typical angina]
4. trestbps (Resting Blood Pressure in mm/hg)
5. chol (Serum Cholesterol in mg/dl)
6. fbs (Fasting Blood Sugar > 120mg/dl): [0= False, 1= True]
7. restecg (Resting Electrocardiographic Results): [0: showing probable or definite left ventricular hypertrophy by 
Estes'criteria, 1: normal, 2: having ST-T wave abnormality]
8. thalach (Maximum Heart Rate)
9. exang (Exercise Induced Angina): [1= True, 0= False]
10. oldpeak (ST depression induced by exercise relative to rest)
11. slope (Slope of the peak exercise ST segment): [0: downsloping, 1: flat, 2: upsloping]
12. ca (Number of major vessels): [0–3]
13. thal (Number of thal): [1= normal, 2= fixed, 3= reversible defect]
14. target (Heart disease status): [0= disease, 1= no disease]

### Summary Of Findings
In the exploration analysis:

I found out that male patients tend to have heart disease compared to female patients.

Patients with 0 major vessels tend to have heart disease while Patients who have a number of vessels between 1 to 3 tend not to have heart disease.

Most patients having typical angina, atypical angina and non-angina chest pain tend to have heart disease while patients with 
asymptomatic chest pain tend not to have heart disease.

Fasting blood sugar is an indicator for diabetes(if fbs > 120mg/dl). From the bar graph, There are a higher number of heart disease patients 
who isn't diabetic. This shows that fbs might not be a strong feature in showing the relationship between heart disease patients and non-disease patients.

Maximum heart rate ("thalach") and slope of the peak exercise("slope") shows a good positive correlation with Heart disease status("target").

"sex" and "oldpeak" shows a good negative correlation with heart disease status("target").

Serum cholesterol("chol") and resting blood pressure ("trestbps") show a low correlation with heart disease status ("target").
