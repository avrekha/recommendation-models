# Databricks notebook source
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# COMMAND ----------

spark_df = spark.table("default.full_dataset_7_csv")
spark_df

# COMMAND ----------

display(spark_df)

# COMMAND ----------

pandas_df = spark_df.toPandas()
display(pandas_df)
print(pandas_df.head())

# COMMAND ----------

def convert_string_to_float(s):
    return float(s.replace(',', '.'))

columns_to_convert = ['Cotton','Organic_cotton', 'Linen', 'Hemp', 'Jute', 'Other_plant', 'Silk', 'Wool', 'Leather', 'Camel', 'Cashmere', 'Alpaca','Feathers','Other_animal','Polyester','Nylon','Acrylic','Spandex','Elastane','Polyamide','Other_synthetic','Lyocell','Viscose','Acetate','Modal','Rayon','Other_regenerated','Other','Recycled_content','Reused_content']

for column in columns_to_convert:
    pandas_df[column] = pandas_df[column].apply(convert_string_to_float)

pandas_df.head()

# COMMAND ----------

                    '''
                     DATA ANALYSIS
                    '''

#Count the number of items according to their Environmental Impact (EI) value
ei_counts = pandas_df['EI'].value_counts().sort_index()

#Generate the graph for Environmental Impact distribution
plt.figure(figsize=(8, 6))
ei_counts.plot(kind='bar', color=['#045D5D', '#2C3539', '#007C80', '#4E8975'])
plt.xlabel('Environmental Impact (EI) Rating', fontsize=14)
plt.ylabel('Count of Items', fontsize=14)
plt.title('Count of Items by Environmental Impact Rating', fontsize=16)
plt.xticks(rotation=0, ha="center", fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# COMMAND ----------

#To create a graph showing the ratio of eco-friendly fabrics vs. non-eco-friendly fabrics, we need to define which materials are considered eco-friendly.

#Eco-friendly materials
eco_friendly_materials = ['Cotton', 'Organic_cotton', 'Linen', 'Hemp', 'Jute', 'Other_plant', 'Silk', 'Wool', 'Leather', 'Camel', 'Cashmere', 'Alpaca', 'Feathers', 'Other_animal', 'Recycled_content', 'Lyocell', 'Viscose', 'Acetate', 'Modal', 'Rayon', 'Other_regenerated']

#Non-eco-friendly materials
non_eco_friendly_materials = ['Polyester', 'Nylon', 'Acrylic', 'Spandex', 'Elastane', 'Polyamide', 'Other_synthetic']


#Calculate the total eco-friendly and non-eco-friendly content per item
pandas_df['Eco_friendly_total'] = pandas_df[eco_friendly_materials].sum(axis=1)
pandas_df['Non_eco_friendly_total'] = pandas_df[non_eco_friendly_materials].sum(axis=1)

#Calculate the overall totals
total_eco_friendly = pandas_df['Eco_friendly_total'].sum()
total_non_eco_friendly = pandas_df['Non_eco_friendly_total'].sum()

#Labels for the sections
labels = ['Eco-Friendly', 'Non-Eco-Friendly']

#Values for each section
sizes = [total_eco_friendly, total_non_eco_friendly]

#Colors for each section
colors = ['#98AFC7', '#708090']

#Explode the 1st slice (i.e., 'Eco-Friendly')
explode = (0.1, 0)

#Create the pie chart
plt.figure(figsize=(10, 7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Ratio of Eco-Friendly vs Non-Eco-Friendly Fabrics')

#Show the chart
plt.show()

# COMMAND ----------

#To create a graph showing the content of each fabric, we will calculate the total amount of each material across all items in the dataset.

#List of all materials
all_materials = eco_friendly_materials + non_eco_friendly_materials

#Calculate the total content for each material
material_totals = pandas_df[all_materials].sum()

#Sort the materials by their total content for better visualization
material_totals_sorted = material_totals.sort_values(ascending=False)


plt.figure(figsize=(15, 10))
material_totals_sorted.plot(kind='bar', color='skyblue')
plt.title('Content of Each Fabric')
plt.xlabel('Fabric Type')
plt.ylabel('Total Content across all items')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

#Show the chart
plt.show()

# COMMAND ----------

label_encoders = {}
categorical_columns = pandas_df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    le = LabelEncoder()
    pandas_df[column] = le.fit_transform(pandas_df[column])
    label_encoders[column] = le

#splitting the dataset
feature_columns = pandas_df.columns.drop(['ID'])
target_column = 'EI'
X = pandas_df[feature_columns]
y = pandas_df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# COMMAND ----------

#training random forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# COMMAND ----------

#evaluation
predictions = rf_model.predict(X_test)
report = classification_report(y_test, predictions)
print(report)

# COMMAND ----------

#predicting EI for all items
pandas_df['Predicted_EI'] = rf_model.predict(pandas_df[feature_columns])

#extracting items with low environmental impact
recommended_items = pandas_df[(pandas_df['Predicted_EI'] == 1) |  (pandas_df['Predicted_EI'] == 2)]

# COMMAND ----------

#decoding categorical values
for column, le in label_encoders.items():
    recommended_items[column] = le.inverse_transform(recommended_items[column])

print(recommended_items.head())


# COMMAND ----------

recommended_items

# COMMAND ----------

#Predictions
predictions = rf_model.predict(X_test)

#Calculate metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')  #Use 'weighted' for imbalanced classes
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

#Print the metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

#Classification report for a detailed view
print(classification_report(y_test, predictions))



# COMMAND ----------


