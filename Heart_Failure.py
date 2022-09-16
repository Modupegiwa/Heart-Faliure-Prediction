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

# View Dimensions Of Dataset
print(df.shape)

# Preview The Dataset
print(df.head())
print(df.info())

# Check for Null values
print(df.notnull().count())

# Print Dataset Info
print("Dataset Info:")
print("Total Rows: ", df.shape[0])
print("Total Column: ", df.shape[1])

# Fix The Data Types
data = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'ca', 'thal']
df[data] = df[data].astype(object)

# Data Exploration

# GENDER
labels = ['Male', 'Female']
colors = ["#CD919E", "#CDC9C9"]
order = df['sex'].value_counts().index
plt.figure(figsize = (12, 12))
plt.suptitle('Gender Distribution', fontweight = 'heavy', fontsize = '16',
            fontfamily = 'sans-serif', color = "black")

plt.subplot(1, 2, 1)
plt.title('Pie Chart', fontweight = 'bold', fontfamily = "sans-serif", color = 'black')
plt.pie(df["sex"].value_counts(), pctdistance = 0.7, autopct = '%.2f%%', labels = labels,
wedgeprops = dict(alpha = 0.8, edgecolor = "black"), textprops = {'fontsize': 12}, colors = colors)
centre = plt.Circle((0,0), 0.45, fc = "white", edgecolor = "black")
plt.gcf().gca().add_artist(centre)

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
plt.xticks([0, 1])
plt.grid(axis = "y", alpha = 0.4)
plt.show()
# Count Categorical Labels without counting null values
print("Sex Total:")
print(df.sex.value_counts(dropna = False))

# CHEST PAIN TYPE
labels = ['Type 0', 'Type 2', 'Type 1', 'Type 3']
colors = ["#6495ED", "#1874CD", "#009ACD", "#00688B"]
order = df['cp'].value_counts().index
plt.figure(figsize = (12, 12))
plt.suptitle('Chest Pain Type Distribution', fontweight = 'heavy', fontsize = '16',
            fontfamily = 'sans-serif', color = "black")

plt.subplot(1, 2, 1)
plt.title('Pie Chart', fontweight = 'bold', fontfamily = "sans-serif", color = 'black')
plt.pie(df["cp"].value_counts(), pctdistance = 0.7, autopct = '%.2f%%', labels = labels,
wedgeprops = dict(alpha = 0.8, edgecolor = "black"), textprops = {'fontsize': 12}, colors = colors)
centre = plt.Circle((0,0), 0.45, fc = "white", edgecolor = "black")
plt.gcf().gca().add_artist(centre)

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

plt.subplot(1, 2, 1)
plt.title('Pie Chart', fontweight = 'bold', fontfamily = "sans-serif", color = 'black')
plt.pie(df["restecg"].value_counts(), pctdistance = 0.7, autopct = '%.2f%%', labels = labels,
wedgeprops = dict(alpha = 0.8, edgecolor = "black"), textprops = {'fontsize': 12}, colors = colors)
centre = plt.Circle((0,0), 0.45, fc = "white", edgecolor = "black")
plt.gcf().gca().add_artist(centre)

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

plt.subplot(1, 2, 1)
plt.title('Pie Chart', fontweight = 'bold', fontfamily = "sans-serif", color = 'black')
plt.pie(df["exang"].value_counts(), pctdistance = 0.7, autopct = '%.2f%%', labels = labels,
wedgeprops = dict(alpha = 0.8, edgecolor = "black"), textprops = {'fontsize': 12}, colors = colors)
centre = plt.Circle((0,0), 0.45, fc = "white", edgecolor = "black")
plt.gcf().gca().add_artist(centre)

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
labels = ['1', '2', '0']
colors = ["#CD2626", "#8B1A1A", "#FF7D40"]
order = df['slope'].value_counts().index
plt.figure(figsize = (12, 12))
plt.suptitle('Slope Of The Peak Exercise Distribution', fontweight = 'heavy', fontsize = '16',
            fontfamily = 'sans-serif', color = "black")

plt.subplot(1, 2, 1)
plt.title('Pie Chart', fontweight = 'bold', fontfamily = "sans-serif", color = 'black')
plt.pie(df["slope"].value_counts(), pctdistance = 0.7, autopct = '%.2f%%', labels = labels,
wedgeprops = dict(alpha = 0.8, edgecolor = "black"), textprops = {'fontsize': 12}, colors = colors)
centre = plt.Circle((0,0), 0.45, fc = "white", edgecolor = "black")
plt.gcf().gca().add_artist(centre)

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
labels = ['0', '1', '2', '3', '4']
colors = ["#8B7D6B", "#000000", "#CDB79E", "#FFE4C4", "#CDC0B0"]
order = df['ca'].value_counts().index
plt.figure(figsize = (12, 12))
plt.suptitle('Number Of Major Vessels Distribution', fontweight = 'heavy', fontsize = '16',
            fontfamily = 'sans-serif', color = "black")

plt.subplot(1, 2, 1)
plt.title('Pie Chart', fontweight = 'bold', fontfamily = "sans-serif", color = 'black')
plt.pie(df["ca"].value_counts(), pctdistance = 0.7, autopct = '%.2f%%', labels = labels,
wedgeprops = dict(alpha = 0.8, edgecolor = "black"), textprops = {'fontsize': 12}, colors = colors)
centre = plt.Circle((0,0), 0.45, fc = "white", edgecolor = "black")
plt.gcf().gca().add_artist(centre)

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
labels = ['2', '3', '1', '0']
colors = ["#FF4040", "#8B2323", "#8A360F", "#8B3E2F"]
order = df['thal'].value_counts().index
plt.figure(figsize = (12, 12))
plt.suptitle('Thal Distribution', fontweight = 'heavy', fontsize = '16',
            fontfamily = 'sans-serif', color = "black")

plt.subplot(1, 2, 1)
plt.title('Pie Chart', fontweight = 'bold', fontfamily = "sans-serif", color = 'black')
plt.pie(df["thal"].value_counts(), pctdistance = 0.7, autopct = '%.2f%%', labels = labels,
wedgeprops = dict(alpha = 0.8, edgecolor = "black"), textprops = {'fontsize': 12}, colors = colors)
centre = plt.Circle((0,0), 0.45, fc = "white", edgecolor = "black")
plt.gcf().gca().add_artist(centre)

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

plt.subplot(1, 2, 1)
plt.title('Pie Chart', fontweight = 'bold', fontfamily = "sans-serif", color = 'black')
plt.pie(df["target"].value_counts(), pctdistance = 0.7, autopct = '%.2f%%', labels = labels,
wedgeprops = dict(alpha = 0.8, edgecolor = "black"), textprops = {'fontsize': 12}, colors = colors)
centre = plt.Circle((0,0), 0.45, fc = "white", edgecolor = "black")
plt.gcf().gca().add_artist(centre)

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
# Min, Max And Average Values
print('The Minumum, Maximum And Average Values: ')
print("Minimum: ", min(df.age))
print("Maximum: ", max(df.age))
print("Average: ", df.age.mean())

# Resting Blood Pressure In mmHg
fig = plt.figure(figsize = (12, 12))
# Skewness & Kurtosis
print("Resting Blood Pressure Column Skewness & Kurtosis")
print("Skewness: ", round(skew(df["trestbps"]), 3))
print("Kurtosis: ", round(kurtosis(df["trestbps"]), 3))
# General Title
fig.suptitle("Resting Blood Pressure Column Distribution", fontweight = "bold", fontsize = 16, 
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
# Min, Max And Average Values
print('The Minumum, Maximum And Average Values: ')
print("Minimum: ", min(df.restecg))
print("Maximum: ", max(df.restecg))
print("Average: ", df.restecg.mean())


# Serum Cholestoral in mg/dl
fig = plt.figure(figsize = (12, 12))
# Skewness & Kurtosis
print("Serum Cholestoral Column Skewness & Kurtosis")
print("Skewness: ", round(skew(df["chol"]), 3))
print("Kurtosis: ", round(kurtosis(df["chol"]), 3))
# General Title
fig.suptitle("Serum Cholestoral Column Distribution", fontweight = "bold", fontsize = 16, fontfamily = 'sans-serif',
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
print('The Minumum, Maximum And Average Values: ')
print("Minimum: ", min(df.chol))
print("Maximum: ", max(df.chol))
print("Average: ", df.chol.mean())

# Maximum Heart Rate
fig = plt.figure(figsize = (12, 12))
# Skewness & Kurtosis
print("Maximum Heart Rate Column Skewness & Kurtosis")
print("Skewness: ", round(skew(df["thalach"]), 3))
print("Kurtosis: ", round(kurtosis(df["thalach"]), 3))
# General Title
fig.suptitle("Maximum Heart Rate Column Distribution", fontweight = "bold", fontsize = 16, 
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
print('The Minumum, Maximum And Average Values: ')
print("Minimum: ", min(df.thalach))
print("Maximum: ", max(df.thalach))
print("Average: ", df.thalach.mean())

# Old Peak
fig = plt.figure(figsize = (12, 12))
# Skewness & Kurtosis
print("Old Peak Column Skewness & Kurtosis")
print("Skewness: ", round(skew(df["oldpeak"]), 3))
print("Kurtosis: ", round(kurtosis(df["oldpeak"]), 3))
# General Title
fig.suptitle("Old Peak Column Distribution", fontweight = "bold", fontsize = 16, fontfamily = 'sans-serif',
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
print('The Minumum, Maximum And Average Values: ')
print("Minimum: ", min(df.oldpeak))
print("Maximum: ", max(df.oldpeak))
print("Average: ", df.oldpeak.mean())

# EDA

# Heart Disease Distribution Based On Gender
labels = ["False", "True"]
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
plt.title("Female tend to have heart disease compared to Male", fontsize = 10, fontfamily = "sans-serif",
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
labels = ["False", "True"]
colors = ["#CD919E", "#CDC9C9"]
ax = pd.crosstab(df.ca, df.target).plot(kind = "barh", figsize = (8, 5),color = colors, 
        edgecolor = "black", alpha = 0.85)
for rect in ax.patches:
        width, height = rect.get_width(), rect.get_height()
        x, y = rect.get_xy()
        ax.text(x+width/2, y+height/2, "{:.0f}".format(width), horizontalalignment = "center", 
        verticalalignment = "center")
plt.suptitle("Heart Disease Distribution Based On Total Major Vessels", fontweight = "heavy",
        fontsize = 16, fontfamily = "sans-serif", color = "black", x = 0.069, y =0.98, ha = "left")
plt.title("People with 0 and 4 major vessels tend to have heart diseases while People who have a number of vessels between 1 to 3 tend not to have heart diseases", 
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

# Heart Disease Distribution Based On Age
plt.figure(figsize = (10, 8))
plt.suptitle("Heart Disease Distribution Scatter PLot Based On Age", fontweight = "heavy",
        fontsize = 16, fontfamily = "sans-serif", color = "black", x = 0.48, y =0.98)
plt.title("", 
        fontsize = 10, fontfamily = "sans-serif", loc = "left", color = "black")
plt.tight_layout(rect = [0, 0.04, 1, 1.01])
# Scatter Plot
plt.scatter(x = df.age[df.target == 0], y = df.thalach[(df.target == 0)], c = "#483D8B")
plt.scatter(x = df.age[df.target == 1], y = df.thalach[(df.target == 1)], c = "#292421")
plt.legend(["False", "True"], title = "Type", fontsize = 7, title_fontsize = 8, loc = "upper right", 
         frameon = "True")
plt.xlabel("Age", fontweight = "bold", fontsize = 11, fontfamily = "sans-serif", color = "black")
plt.ylabel("Max. Heart Rate", fontweight = "bold", fontsize = 11, fontfamily = "sans-serif", color = "black")
plt.ticklabel_format(style = "plain", axis = "both")
plt.grid(axis = "both", alpha = 0.4, lw = 0.5)
plt.show()
