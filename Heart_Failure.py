import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import skew,kurtosis
import pylab as py

df = pd.read_csv("heart.csv")
print(df.head())

# View Dimensions Of Dataset
print(df.shape)

# Print Dataset Info
print(df.info())
print("Dataset Info:")
print("Total Rows: ", df.shape[0])
print("Total Column: ", df.shape[1])

# Fix The Data Types
data = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'ca', 'thal']
df[data] = df[data].astype(object)

# Clean The Dataset
# Check for Null values
print(df.notnull().count())
# change values of column "ca" from 0 - 4 to 0 - 3
df.loc[df["ca"] == 4, "ca"] = np.NaN
df["ca"].unique()
# change values of column "thal" from 0 - 3 to 1 - 3
df.loc[df["thal"] == 0, "thal"] = np.NaN
df[df["thal"] == 0]
df["thal"].unique()

# Data Exploration

# GENDER
labels = ['Male', 'Female']
colors = ["#00008B", "#C1CDCD"]
order = df['sex'].value_counts().index
plt.figure(figsize = (12, 12))
plt.suptitle('Gender Distribution', fontweight = 'heavy', fontsize = '16',
            fontfamily = 'sans-serif', color = "black")
# Pie Chart
plt.subplot(1, 2, 1)
plt.title('Pie Chart', fontweight = 'bold', fontfamily = "sans-serif", color = 'black')
plt.pie(df["sex"].value_counts(), pctdistance = 0.7, autopct = '%.2f%%', labels = labels,
wedgeprops = dict(alpha = 0.8, edgecolor = "black"), textprops = {'fontsize': 12}, colors = colors)
centre = plt.Circle((0,0), 0.45, fc = "white", edgecolor = "black")
plt.gcf().gca().add_artist(centre)
# Histogram
countplt = plt.subplot(1, 2, 2)
plt.title("Histogram", fontweight = "bold", fontsize = 14, 
        fontfamily = "sans-serif", color = 'black')
ax = sns.countplot(x = "sex", data = df, order = order, edgecolor = "black", palette = colors)
for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 4.25, rect.get_height(),
    horizontalalignment="center", fontsize = 10, bbox = dict(facecolor = "none", edgecolor = "black",
    linewidth = 0.25, boxstyle = "round"))
plt.xlabel("Gender", fontweight = "bold", fontsize = 11, fontfamily = "sans-serif", color = "black")
plt.ylabel("Total", fontweight = "bold", fontsize = 11, fontfamily = "sans-serif", color = "black")
plt.xticks([0, 1], labels)
plt.grid(axis = "y", alpha = 0.4)
plt.show()
# Count Categorical Labels without counting null values
print("Sex Total:")
print(df.sex.value_counts(dropna = False))

# CHEST PAIN TYPE
labels = ['Asymptomatic', 'Non-Anginal Pain', 'Atypical Angina', 'Typical Angina']
colors = ["#6495ED", "#1874CD", "#009ACD", "#00688B"]
order = df['cp'].value_counts().index
plt.figure(figsize = (16, 8))
plt.suptitle('Chest Pain Type Distribution', fontweight = 'heavy', fontsize = '16',
            fontfamily = 'sans-serif', color = "black")
# Pie Chart
plt.subplot(1, 2, 1)
plt.title('Pie Chart', fontweight = 'bold', fontfamily = "sans-serif", color = 'black')
plt.pie(df["cp"].value_counts(), pctdistance = 0.7, autopct = '%.2f%%', labels = labels,
wedgeprops = dict(alpha = 0.8, edgecolor = "black"), textprops = {'fontsize': 12}, colors = colors)
centre = plt.Circle((0,0), 0.45, fc = "white", edgecolor = "black")
plt.gcf().gca().add_artist(centre)
# Histogram
countplt = plt.subplot(1, 2, 2)
plt.title("Histogram", fontweight = "bold", fontsize = 14, 
        fontfamily = "sans-serif", color = 'black')
ax = sns.countplot(x = "cp", data = df, order = order, edgecolor = "black", palette = colors)
for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 4.25, rect.get_height(),
    horizontalalignment="center", fontsize = 10, bbox = dict(facecolor = "none", edgecolor = "black",
    linewidth = 0.25, boxstyle = "round"))
plt.xlabel("Pain Type", fontweight = "bold", fontsize = 11, fontfamily = "sans-serif", color = "black")
plt.ylabel("Total", fontweight = "bold", fontsize = 11, fontfamily = "sans-serif", color = "black")
plt.xticks([0, 1, 2, 3], labels)
plt.grid(axis = "y", alpha = 0.4)
plt.show()
# Count Categorical Labels without counting null values
print("Chest Pain Type Total:")
print(df.cp.value_counts(dropna = False))

# Fasting Blood Sugar
labels = ['<120 mg/dl', '>120mgdl']
colors = ["#CD5B45", "#8B3E2F"]
order = df['fbs'].value_counts().index
plt.figure(figsize = (12, 12))
plt.suptitle('Fasting Blood Sugar Distribution', fontweight = 'heavy', fontsize = '16',
            fontfamily = 'sans-serif', color = "black")
# Pie Chart
plt.subplot(1, 2, 1)
plt.title('Pie Chart', fontweight = 'bold', fontfamily = "sans-serif", color = 'black')
plt.pie(df["fbs"].value_counts(), pctdistance = 0.7, autopct = '%.2f%%', labels = labels,
wedgeprops = dict(alpha = 0.8, edgecolor = "black"), textprops = {'fontsize': 12}, colors = colors)
centre = plt.Circle((0,0), 0.45, fc = "white", edgecolor = "black")
plt.gcf().gca().add_artist(centre)
# Histogram
countplt = plt.subplot(1, 2, 2)
plt.title("Histogram", fontweight = "bold", fontsize = 14, 
        fontfamily = "sans-serif", color = 'black')
ax = sns.countplot(x = "fbs", data = df, order = order, edgecolor = "black", palette = colors)
for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 4.25, rect.get_height(),
    horizontalalignment="center", fontsize = 10, bbox = dict(facecolor = "none", edgecolor = "black",
    linewidth = 0.25, boxstyle = "round"))
plt.xlabel("Fasting Blood Sugar", fontweight = "bold", fontsize = 11, fontfamily = "sans-serif", 
            color = "black")
plt.ylabel("Total", fontweight = "bold", fontsize = 11, fontfamily = "sans-serif", color = "black")
plt.xticks([0, 1], labels)
plt.grid(axis = "y", alpha = 0.4)
plt.show()
# Count Categorical Labels without counting null values
print("Fasting Blood Sugar Total:")
print(df.fbs.value_counts(dropna = False))

# RESTING ELECTROCARDIOGRAPHIC RESULTS
labels = ['1', '0', '2']
colors = ["#CDB870", "#8B814C", "#8B8970"]
order = df['restecg'].value_counts().index
plt.figure(figsize = (12, 12))
plt.suptitle('Resting Electrocardiographic Distribution', fontweight = 'heavy', fontsize = '16',
            fontfamily = 'sans-serif', color = "black")
# Pie Chart
plt.subplot(1, 2, 1)
plt.title('Pie Chart', fontweight = 'bold', fontfamily = "sans-serif", color = 'black')
plt.pie(df["restecg"].value_counts(), pctdistance = 0.7, autopct = '%.2f%%', labels = labels,
wedgeprops = dict(alpha = 0.8, edgecolor = "black"), textprops = {'fontsize': 12}, colors = colors)
centre = plt.Circle((0,0), 0.45, fc = "white", edgecolor = "black")
plt.gcf().gca().add_artist(centre)
# Histogram
countplt = plt.subplot(1, 2, 2)
plt.title("Histogram", fontweight = "bold", fontsize = 14, 
        fontfamily = "sans-serif", color = 'black')
ax = sns.countplot(x = "restecg", data = df, order = order, edgecolor = "black", palette = colors)
for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 4.25, rect.get_height(),
    horizontalalignment="center", fontsize = 10, bbox = dict(facecolor = "none", edgecolor = "black",
    linewidth = 0.25, boxstyle = "round"))
plt.xlabel("Resting Electrocardiographic", fontweight = "bold", fontsize = 11, fontfamily = "sans-serif", 
        color = "black")
plt.ylabel("Total", fontweight = "bold", fontsize = 11, fontfamily = "sans-serif", color = "black")
plt.grid(axis = "y", alpha = 0.4)
plt.show()
# Count Categorical Labels without counting null values
print("Resting Electrocardiograpic Results Total:")
print(df.restecg.value_counts(dropna = False))

# Exercise Induced Angina
labels = ['False', 'True']
colors = ["#CD5C5C", "#8B3A3A"]
order = df['exang'].value_counts().index
plt.figure(figsize = (12, 12))
plt.suptitle('Exercise Induced Angina Distribution', fontweight = 'heavy', fontsize = '16',
            fontfamily = 'sans-serif', color = "black")
# Pie Chart
plt.subplot(1, 2, 1)
plt.title('Pie Chart', fontweight = 'bold', fontfamily = "sans-serif", color = 'black')
plt.pie(df["exang"].value_counts(), pctdistance = 0.7, autopct = '%.2f%%', labels = labels,
wedgeprops = dict(alpha = 0.8, edgecolor = "black"), textprops = {'fontsize': 12}, colors = colors)
centre = plt.Circle((0,0), 0.45, fc = "white", edgecolor = "black")
plt.gcf().gca().add_artist(centre)
# Histogram
countplt = plt.subplot(1, 2, 2)
plt.title("Histogram", fontweight = "bold", fontsize = 14, 
        fontfamily = "sans-serif", color = 'black')
ax = sns.countplot(x = "exang", data = df, order = order, edgecolor = "black", palette = colors)
for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 4.25, rect.get_height(),
    horizontalalignment="center", fontsize = 10, bbox = dict(facecolor = "none", edgecolor = "black",
    linewidth = 0.25, boxstyle = "round"))
plt.xlabel("Exercise Induced Angina", fontweight = "bold", fontsize = 11, 
        fontfamily = "sans-serif", color = "black")
plt.ylabel("Total", fontweight = "bold", fontsize = 11, fontfamily = "sans-serif", color = "black")
plt.xticks([0, 1], labels)
plt.grid(axis = "y", alpha = 0.4)
plt.show()
# Count Categorical Labels without counting null values
print("Exercise Induced Angina Total:")
print(df.exang.value_counts(dropna = False))

# Slope Of The Peak Exercise
labels = ['Flat', 'Upsloping', 'Downsloping']
colors = ["#CD2626", "#8B1A1A", "#FF7D40"]
order = df['slope'].value_counts().index
plt.figure(figsize = (16, 8))
plt.suptitle('Slope Of The Peak Exercise Distribution', fontweight = 'heavy', fontsize = '16',
            fontfamily = 'sans-serif', color = "black")
# Pie Chart
plt.subplot(1, 2, 1)
plt.title('Pie Chart', fontweight = 'bold', fontfamily = "sans-serif", color = 'black')
plt.pie(df["slope"].value_counts(), pctdistance = 0.7, autopct = '%.2f%%', labels = labels,
wedgeprops = dict(alpha = 0.8, edgecolor = "black"), textprops = {'fontsize': 12}, colors = colors)
centre = plt.Circle((0,0), 0.45, fc = "white", edgecolor = "black")
plt.gcf().gca().add_artist(centre)
# Histogram
countplt = plt.subplot(1, 2, 2)
plt.title("Histogram", fontweight = "bold", fontsize = 14, 
        fontfamily = "sans-serif", color = 'black')
ax = sns.countplot(x = "slope", data = df, order = order, edgecolor = "black", palette = colors)
for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 4.25, rect.get_height(),
    horizontalalignment="center", fontsize = 10, bbox = dict(facecolor = "none", edgecolor = "black",
    linewidth = 0.25, boxstyle = "round"))
plt.xlabel("Slope", fontweight = "bold", fontsize = 11, fontfamily = "sans-serif", color = "black")
plt.ylabel("Total", fontweight = "bold", fontsize = 11, fontfamily = "sans-serif", color = "black")
plt.grid(axis = "y", alpha = 0.4)
plt.show()
# Count Categorical Labels without counting null values
print("Slope Total:")
print(df.slope.value_counts(dropna = False))

# Number Of Major Vessels
labels = ['0', '1', '2', '3']
colors = ["#8B7D6B", "#000000", "#CDB79E", "#FFE4C4"]
order = df['ca'].value_counts().index
plt.figure(figsize = (12, 12))
plt.suptitle('Number Of Major Vessels Distribution', fontweight = 'heavy', fontsize = '16',
            fontfamily = 'sans-serif', color = "black")
# Pie Chart
plt.subplot(1, 2, 1)
plt.title('Pie Chart', fontweight = 'bold', fontfamily = "sans-serif", color = 'black')
plt.pie(df["ca"].value_counts(), pctdistance = 0.7, autopct = '%.2f%%', labels = labels,
wedgeprops = dict(alpha = 0.8, edgecolor = "black"), textprops = {'fontsize': 12}, colors = colors)
centre = plt.Circle((0,0), 0.45, fc = "white", edgecolor = "black")
plt.gcf().gca().add_artist(centre)
# Histogram
countplt = plt.subplot(1, 2, 2)
plt.title("Histogram", fontweight = "bold", fontsize = 14, 
        fontfamily = "sans-serif", color = 'black')
ax = sns.countplot(x = "ca", data = df, order = order, edgecolor = "black", palette = colors)
for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 4.25, rect.get_height(),
    horizontalalignment="center", fontsize = 10, bbox = dict(facecolor = "none", edgecolor = "black",
    linewidth = 0.25, boxstyle = "round"))
plt.xlabel("Number Of Major Vessels", fontweight = "bold", fontsize = 11, 
            fontfamily = "sans-serif", color = "black")
plt.ylabel("Total", fontweight = "bold", fontsize = 11, fontfamily = "sans-serif", color = "black")
plt.grid(axis = "y", alpha = 0.4)
plt.show()
# Count Categorical Labels without counting null values
print("Number Of Major Vessels Total:")
print(df.ca.value_counts(dropna = False))

# Thal Distribution
labels = ['Fixed Defect', 'Reversible Defect', 'Normal']
colors = ["#FF4040", "#8B2323", "#8A360F"]
order = df['thal'].value_counts().index
plt.figure(figsize = (16, 8))
plt.suptitle('Thal Distribution', fontweight = 'heavy', fontsize = '16',
            fontfamily = 'sans-serif', color = "black")
# Pie Chart
plt.subplot(1, 2, 1)
plt.title('Pie Chart', fontweight = 'bold', fontfamily = "sans-serif", color = 'black')
plt.pie(df["thal"].value_counts(), pctdistance = 0.7, autopct = '%.2f%%', labels = labels,
wedgeprops = dict(alpha = 0.8, edgecolor = "black"), textprops = {'fontsize': 12}, colors = colors)
centre = plt.Circle((0,0), 0.45, fc = "white", edgecolor = "black")
plt.gcf().gca().add_artist(centre)
# Histogram
countplt = plt.subplot(1, 2, 2)
plt.title("Histogram", fontweight = "bold", fontsize = 14, 
        fontfamily = "sans-serif", color = 'black')
ax = sns.countplot(x = "thal", data = df, order = order, edgecolor = "black", palette = colors)
for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 4.25, rect.get_height(),
    horizontalalignment="center", fontsize = 10, bbox = dict(facecolor = "none", edgecolor = "black",
    linewidth = 0.25, boxstyle = "round"))
plt.xlabel("Number Of Thal", fontweight = "bold", fontsize = 11, fontfamily = "sans-serif", color = "black")
plt.ylabel("Total", fontweight = "bold", fontsize = 11, fontfamily = "sans-serif", color = "black")
plt.grid(axis = "y", alpha = 0.4)
plt.show()
# Count Categorical Labels without counting null values
print("Thal Total:")
print(df.thal.value_counts(dropna = False))

# Heart Disease Status
labels = ["Disease", "No Disease"]
colors = ["#00008B", "#C1CDCD"]
order = df['target'].value_counts().index
plt.figure(figsize = (12, 12))
plt.suptitle('Heart Disease Status Distribution', fontweight = 'heavy', fontsize = '16',
            fontfamily = 'sans-serif', color = "black")
# Pie Chart
plt.subplot(1, 2, 1)
plt.title('Pie Chart', fontweight = 'bold', fontfamily = "sans-serif", color = 'black')
plt.pie(df["target"].value_counts(), pctdistance = 0.7, autopct = '%.2f%%', labels = labels,
wedgeprops = dict(alpha = 0.8, edgecolor = "black"), textprops = {'fontsize': 12}, colors = colors)
centre = plt.Circle((0,0), 0.45, fc = "white", edgecolor = "black")
plt.gcf().gca().add_artist(centre)
# Histogram
countplt = plt.subplot(1, 2, 2)
plt.title("Histogram", fontweight = "bold", fontsize = 14, 
        fontfamily = "sans-serif", color = 'black')
ax = sns.countplot(x = "target", data = df, order = order, edgecolor = "black", palette = colors)
for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 4.25, rect.get_height(),
    horizontalalignment="center", fontsize = 10, bbox = dict(facecolor = "none", edgecolor = "black",
    linewidth = 0.25, boxstyle = "round"))
plt.xlabel("Heart Disease Status", fontweight = "bold", fontsize = 11, fontfamily = "sans-serif", color = "black")
plt.ylabel("Total", fontweight = "bold", fontsize = 11, fontfamily = "sans-serif", color = "black")
plt.xticks([0, 1], labels)
plt.grid(axis = "y", alpha = 0.4)
plt.show()
# Count Categorical Labels without counting null values
print("Heart Disease Status Total:")
print(df.target.value_counts(dropna = False))

# DESCRIPTIVE STATISTICS
print(df.select_dtypes(exclude = "object").describe().T)

# Continuous Column Distribution

# AGE
fig = plt.figure(figsize = (12, 12))
# Skewness & Kurtosis
print("Age Column Skewness & Kurtosis")
print("Skewness: ", round(skew(df["age"]), 3))
print("Kurtosis: ", round(kurtosis(df["age"]), 3))
# General Title
fig.suptitle("Age Distribution", fontweight = "bold", fontsize = 16, fontfamily = 'sans-serif',
            color = "black")
fig.subplots_adjust(top = 0.9)
# Histogram
plot1 = fig.add_subplot(1, 2, 2)
plt.title("Histogram Plot", fontweight = "bold", fontsize = 14, fontfamily = 'sans-serif', color = 'black')
sns.histplot(data = df, x = 'age', kde = True, color = "#104E8B")
plt.xlabel('Age', fontweight = 'normal', fontsize = 11, fontfamily = 'sans-serif', color = "black")
plt.ylabel('Total', fontweight = 'regular', fontsize = 11, fontfamily = "sans-serif", color = "black")
# Box Plot
plot2 = fig.add_subplot(1, 2, 1)
plt.title("Box Plot", fontweight = "bold", fontsize = 14, fontfamily = 'sans-serif', color = 'black')
sns.boxplot(data = df, y = 'age',  linewidth = 1.5, boxprops = dict(alpha = 0.8), color = "#104E8B")
plt.ylabel('Age', fontweight = 'regular', fontsize = 11, fontfamily = 'sans-serif', color = "black")
plt.show()
# Min, Max, Mean and Median Values
print('The Minumum, Maximum And Average Values: ')
print("Minimum: ", min(df.age))
print("Maximum: ", max(df.age))
print("Average: ", df.age.mean())
print("Median: ", df.age.median())

# Resting Blood Pressure In mmHg
fig = plt.figure(figsize = (12, 12))
# Skewness & Kurtosis
print("Resting Blood Pressure Column Skewness & Kurtosis")
print("Skewness: ", round(skew(df["trestbps"]), 3))
print("Kurtosis: ", round(kurtosis(df["trestbps"]), 3))
# General Title
fig.suptitle("Resting Blood Pressure Distribution", fontweight = "bold", fontsize = 16, 
fontfamily = 'sans-serif', color = "black")
fig.subplots_adjust(top = 0.9)
# Histogram
plot1 = fig.add_subplot(1, 2, 2)
plt.title("Histogram Plot", fontweight = "bold", fontsize = 14, fontfamily = 'sans-serif', color = 'black')
sns.histplot(data = df, x = 'trestbps', kde = True, color = "#00688B")
plt.xlabel('Total', fontweight = 'normal', fontsize = 11, fontfamily = 'sans-serif', color = "black")
plt.ylabel('Resting Blood Pressure', fontweight = 'regular', fontsize = 11, fontfamily = "sans-serif", 
            color = "black")
# Box Plot
plot2 = fig.add_subplot(1, 2, 1)
plt.title("Box Plot", fontweight = "bold", fontsize = 14, fontfamily = 'sans-serif', color = 'black')
sns.boxplot(data = df, y = 'trestbps',  linewidth = 1.5, boxprops = dict(alpha = 0.8), color = "#00688B")
plt.ylabel('Resting Blood Pressure', fontweight = "regular", fontsize = 11, fontfamily = 'sans-serif', 
            color = "black")
plt.show()
# Min, Max, Mean and Median Values
print('The Minumum, Maximum And Average Values: ')
print("Minimum: ", min(df.trestbps))
print("Maximum: ", max(df.trestbps))
print("Average: ", df.trestbps.mean())
print("Median: ", df.trestbps.median())

# Serum Cholestoral in mg/dl
fig = plt.figure(figsize = (12, 12))
# Skewness & Kurtosis
print("Serum Cholestoral Column Skewness & Kurtosis")
print("Skewness: ", round(skew(df["chol"]), 3))
print("Kurtosis: ", round(kurtosis(df["chol"]), 3))
# General Title
fig.suptitle("Serum Cholestoral Distribution", fontweight = "bold", fontsize = 16, fontfamily = 'sans-serif',
            color = "black")
fig.subplots_adjust(top = 0.9)
# Histogram
plot1 = fig.add_subplot(1, 2, 2)
plt.title("Histogram Plot", fontweight = "bold", fontsize = 14, fontfamily = 'sans-serif', color = 'black')
sns.histplot(data = df, x = 'chol', kde = True, color = "#009ACD")
plt.xlabel('Total', fontweight = 'normal', fontsize = 11, fontfamily = 'sans-serif', color = "black")
plt.ylabel('Serum Cholestoral', fontweight = 'regular', fontsize = 11, fontfamily = "sans-serif", 
            color = "black")
# Box Plot
plot2 = fig.add_subplot(1, 2, 1)
plt.title("Box Plot", fontweight = "bold", fontsize = 14, fontfamily = 'sans-serif', color = 'black')
sns.boxplot(data = df, y = 'chol',  linewidth = 1.5, boxprops = dict(alpha = 0.8), color = "#009ACD")
plt.ylabel('Serum Cholestoral', fontweight = 'regular', fontsize = 11, fontfamily = 'sans-serif', 
            color = "black")
plt.show()
# Min, Max And Average Values
print('The Minumum, Maximum, Mean And Median Values: ')
print("Minimum: ", min(df.chol))
print("Maximum: ", max(df.chol))
print("Average: ", df.chol.mean())
print("Median: ", df.chol.median())

# Maximum Heart Rate
fig = plt.figure(figsize = (12, 12))
# Skewness & Kurtosis
print("Maximum Heart Rate Colum Skewness & Kurtosis")
print("Skewness: ", round(skew(df["thalach"]), 3))
print("Kurtosis: ", round(kurtosis(df["thalach"]), 3))
# General Title
fig.suptitle("Maximum Heart Rate Distribution", fontweight = "bold", fontsize = 16, 
            fontfamily = 'sans-serif', color = "black")
fig.subplots_adjust(top = 0.9)
# Histogram
plot1 = fig.add_subplot(1, 2, 2)
plt.title("Histogram Plot", fontweight = "bold", fontsize = 14, fontfamily = 'sans-serif', color = 'black')
sns.histplot(data = df, x = 'thalach', kde = True, color = "#483D8B")
plt.xlabel('Total', fontweight = 'normal', fontsize = 11, fontfamily = 'sans-serif', color = "black")
plt.ylabel('Maximum Heart Rate', fontweight = 'regular', fontsize = 11, 
            fontfamily = "sans-serif", color = "black")
# Box Plot
plot2 = fig.add_subplot(1, 2, 1)
plt.title("Box Plot", fontweight = "bold", fontsize = 14, fontfamily = 'sans-serif', color = 'black')
sns.boxplot(data = df, y = 'thalach',  linewidth = 1.5, boxprops = dict(alpha = 0.8), color = "#483D8B")
plt.ylabel('Maximum Heart Rate', fontweight = 'regular', fontsize = 11, 
            fontfamily = 'sans-serif', color = "black")
plt.show()
# Min, Max And Average Values
print('The Minumum, Maximum, Mean And Median Values: ')
print("Minimum: ", min(df.thalach))
print("Maximum: ", max(df.thalach))
print("Average: ", df.thalach.mean())
print("Median: ", df.thalach.median())

# Old Peak
fig = plt.figure(figsize = (12, 12))
# Skewness & Kurtosis
print("Old Peak Column Skewness & Kurtosis")
print("Skewness: ", round(skew(df["oldpeak"]), 3))
print("Kurtosis: ", round(kurtosis(df["oldpeak"]), 3))
# General Title
fig.suptitle("Old Peak Distribution", fontweight = "bold", fontsize = 16, fontfamily = 'sans-serif',
            color = "black")
fig.subplots_adjust(top = 0.9)
# Histogram
plot1 = fig.add_subplot(1, 2, 2)
plt.title("Histogram Plot", fontweight = "bold", fontsize = 14, fontfamily = 'sans-serif', color = 'black')
sns.histplot(data = df, x = 'oldpeak', kde = True, color = "#3D59AB")
plt.xlabel('Total', fontweight = 'normal', fontsize = 11, fontfamily = 'sans-serif', color = "black")
plt.ylabel('Old Peak', fontweight = 'regular', fontsize = 11, fontfamily = "sans-serif", color = "black")
# Box Plot
plot2 = fig.add_subplot(1, 2, 1)
plt.title("Box Plot", fontweight = "bold", fontsize = 14, fontfamily = 'sans-serif', color = 'black')
sns.boxplot(data = df, y = 'oldpeak',  linewidth = 1.5, boxprops = dict(alpha = 0.8), color = "#3D59AB")
plt.ylabel('Old Peak', fontweight = 'regular', fontsize = 11, fontfamily = 'sans-serif', color = "black")
plt.show()
# Min, Max And Average Values
print('The Minumum, Maximum, Mean And Median Values: ')
print("Minimum: ", min(df.oldpeak))
print("Maximum: ", max(df.oldpeak))
print("Average: ", df.oldpeak.mean())
print("Median: ", df.oldpeak.median())

# EDA

# Heart Disease Distribution Based On Gender
labels = ["No Disease", "Disease"]
label_gender = np.array([0, 1])
label_gender2 = ["Female", "Male"]
colors = ["#CD919E", "#CDC9C9"]

# Bar Chart
ax = pd.crosstab(df.sex, df.target).plot(kind = 'bar', figsize = (8, 5), color = colors,
        edgecolor = "black", alpha = 0.85)
for rect in ax.patches:
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 1.25, rect.get_height(),
        horizontalalignment = "center", fontsize = 10)
plt.suptitle("Heart Disease Distribution Based On Gender", fontweight = "heavy",
        fontsize = 16, fontfamily = "sans-serif", color = "black")
plt.title("Male tend to have heart disease compared to Female", fontsize = 10, fontfamily = "sans-serif",
        loc = "left", color = "black")
plt.tight_layout(rect = [0, 0.04, 1, 1.025])
plt.xlabel('Gender', fontfamily = "sans-serif", fontweight = "bold", color = "black")
plt.xticks(label_gender, label_gender2, rotation = 0)
plt.grid(axis = "y", alpha = 0.4)
plt.grid(axis = "x", alpha = 0)
plt.legend(labels = labels, title = "Target", fontsize = 8, title_fontsize = 9, loc = "upper left",
        frameon = True)
plt.show()

# Heart Disease Distribution Based On Total Major Vessels
labels = ["No Disease", "Disease"]
colors = ["#CD919E", "#CDC9C9"]

# Bar Chart
ax = pd.crosstab(df.ca, df.target).plot(kind = "barh", figsize = (8, 5),color = colors, 
        edgecolor = "black", alpha = 0.85)
for rect in ax.patches:
        width, height = rect.get_width(), rect.get_height()
        x, y = rect.get_xy()
        ax.text(x+width/2, y+height/2, "{:.0f}".format(width), horizontalalignment = "center", 
        verticalalignment = "center")
plt.suptitle("Heart Disease Distribution Based On Total Major Vessels", fontweight = "heavy",
        fontsize = 16, fontfamily = "sans-serif", color = "black", x = 0.069, y =0.98, ha = "left")
plt.title("People with 0 major vessels tend to have heart diseases while People who have a number of vessels between 1 to 3 tend not to have heart diseases", 
        fontsize = 10, fontfamily = "sans-serif", loc = "left", color = "black")
plt.tight_layout(rect = [0, 0.04, 1, 1.025])
plt.xlabel('Total', fontfamily = "sans-serif", fontweight = "bold", color = "black")
plt.ylabel("Number Of Major Vessels", fontfamily = "sans-serif", fontweight = "bold", color = "black")
plt.yticks(rotation = 0)
plt.grid(axis = "x", alpha = 0.4)
plt.grid(axis = "y", alpha = 0)
plt.legend(labels = labels, title = "Target", fontsize = 8, title_fontsize = 9, loc = "upper right",
        frameon = True)
plt.show()

# Heart Disease Distribution Based On Chest Pain
labels = ["No Disease", "Disease"]
label_cp = np.array([0, 1, 2, 3])
label_cp2 = ["Asymptomatic", "Atypical Angina", "Non-Anginal Pain", "Typical Angina"]
colors = ["#CD919E", "#CDC9C9"]

# Bar Chart
ax = pd.crosstab(df.cp, df.target).plot(kind = 'bar', figsize = (8, 5), color = colors,
        edgecolor = "black", alpha = 0.85)
for rect in ax.patches:
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 1.25, rect.get_height(),
        horizontalalignment = "center", fontsize = 10)
plt.suptitle("Heart Disease Distribution Based On Chest Pain", fontweight = "heavy",
        fontsize = 16, fontfamily = "sans-serif", color = "black")
plt.tight_layout(rect = [0, 0.04, 1, 1.025])
plt.xlabel('Chest Pain', fontfamily = "sans-serif", fontweight = "bold", color = "black")
plt.ylabel('Total', fontfamily = "sans-serif", fontweight = "bold", color = "black")
plt.xticks(label_cp, label_cp2, rotation = 0)
plt.grid(axis = "y", alpha = 0.4)
plt.grid(axis = "x", alpha = 0)
plt.legend(labels = labels, title = "Target", fontsize = 8, title_fontsize = 9, loc = "upper right",
        frameon = True)
plt.show()

# Heart Disease Distribution Based On Fasting Blood Sugar
labels = ["No Disease", "Disease"]
label_fbs = np.array([0, 1])
label_fbs2 = ["<120mg/dl", ">120mg/dl"]
colors = ["#CD919E", "#CDC9C9"]

# Bar Chart
ax = pd.crosstab(df.fbs, df.target).plot(kind = 'bar', figsize = (8, 5), color = colors,
        edgecolor = "black", alpha = 0.85)
for rect in ax.patches:
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 1.25, rect.get_height(),
        horizontalalignment = "center", fontsize = 10)
plt.suptitle("Heart Disease Distribution Based On Fasting Blood Sugar", fontweight = "heavy",
        fontsize = 16, fontfamily = "sans-serif", color = "black")
plt.tight_layout(rect = [0, 0.04, 1, 1.025])
plt.xlabel('Fasting Blood Sugar', fontfamily = "sans-serif", fontweight = "bold", color = "black")
plt.ylabel('Total', fontfamily = "sans-serif", fontweight = "bold", color = "black")
plt.xticks(label_fbs, label_fbs2, rotation = 0)
plt.grid(axis = "y", alpha = 0.4)
plt.grid(axis = "x", alpha = 0)
plt.legend(labels = labels, title = "Target", fontsize = 8, title_fontsize = 9, loc = "upper right",
        frameon = True)
plt.show()


# CORRELATION
# Correlation Matrix
corr_matrix = round(df.corr(), 3)
print("Correlation Matrix: ")
print(corr_matrix)

# Correlation Map / Heat Map
plt.rcParams['figure.figsize'] =(15, 10)
sns.heatmap(df.corr(), annot=True, cmap='Blues', linewidths=5)
plt.suptitle('Correlation Between Variables', fontweight='heavy', 
             x=0.03, y=0.98, ha = "left", fontsize='16', fontfamily='sans-serif', 
             color= "black")
plt.show()