# Databricks notebook source
#Import necessary libraries

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

                      '''
                       DATA CONVERSION
                      '''

#Load the dataset from Spark to a pandas DataFrame
spark_df = spark.table("default.full_dataset_8_csv")
display(spark_df) #Displaying the Spark DataFrame

# COMMAND ----------

#Converting Spark DataFrame to pandas DataFrame for processing
pandas_df = spark_df.toPandas()
pandas_df.head()

# COMMAND ----------

                      '''
                       DATA CLEANING
                      '''


#Function to convert strings with commas to float values (e.g., "1,5" to 1.5)
def convert_string_to_float(s):
    return float(s.replace(',', '.'))

#Columns that need to be converted from strings to floats
columns_to_convert = ['Cotton','Organic_cotton', 'Linen', 'Hemp', 'Jute', 'Other_plant', 'Silk', 'Wool', 'Leather', 'Camel', 'Cashmere', 'Alpaca','Feathers','Other_animal','Polyester','Nylon','Acrylic','Spandex','Elastane','Polyamide','Other_synthetic','Lyocell','Viscose','Acetate','Modal','Rayon','Other_regenerated','Other','Recycled_content','Reused_content']

#Applying the conversion to the specified columns
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

                      '''
                       DATA PREPROCESSING
                      '''


#Identifying numerical and categorical columns for preprocessing
numerical_cols = pandas_df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = pandas_df.select_dtypes(include=['object']).columns

print(numerical_cols)
print(categorical_cols)

#Preprocessing: Setting up transformers for numerical and categorical data
numerical_transformer = MinMaxScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

#Bundling transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

#Applying the preprocessing to the DataFrame
data_preprocessed = preprocessor.fit_transform(pandas_df)

#Viewing the shape of the preprocessed data for confirmation
data_preprocessed.shape
pandas_df.info()

# COMMAND ----------

#Converting the preprocessed data back to a DataFrame for further manipulation
columns_transformed = preprocessor.transformers_[0][1].get_feature_names_out(numerical_cols).tolist() + preprocessor.transformers_[1][1].get_feature_names_out(categorical_cols).tolist()
data_preprocessed_df = pd.DataFrame(data_preprocessed, columns=columns_transformed)

#Adjust weights for sustainable materials
sustainable_features = ['Organic_cotton', 'Linen', 'Hemp', 'Recycled_content', 'Reused_content']
weight_multiplier = 2  #example weight multiplier

for feature in sustainable_features:
    if feature in data_preprocessed_df.columns:
        data_preprocessed_df[feature] *= weight_multiplier

#Recompute the cosine similarity matrix with adjusted weights
cosine_sim = cosine_similarity(data_preprocessed_df.values)

display(data_preprocessed_df)


#Calculating cosine similarity among items based on the preprocessed data
cosine_sim = cosine_similarity(data_preprocessed)

# COMMAND ----------

#Function to recommend items based on cosine similarity and a given item ID
def recommend_items(item_id, cosine_sim=cosine_sim, num_items=10):
    """
    Recommend items based on a given item

    Parameters:
    - item_id (int): Item ID for which we want recommendations
    - cosine_sim (function call): Call the cosine similarity function to calculate similarity among items
    - num_items (int): number of items we want to see in the recommendation list

    Returns:
    - Recommended items: Recommends count(num_items) items similar to item_id
    """
    #Get the index of the item that matches the ID
    idx = item_id - 1  #ID starts from 1 for our dataset

    #Computing pairwise similarity scores for all items with the given item
    sim_scores = list(enumerate(cosine_sim[idx]))

    #Sorting the items based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    #Excluding itself
    sim_scores = sim_scores[1:num_items+1]  

    #Extracting the indices of the top similar items (excluding the item itself)
    item_indices = [i[0] for i in sim_scores]

    #Return the top most similar items
    return pandas_df.iloc[item_indices]

#Example: Recommend 10 items similar to item with ID 110
recommended_items = recommend_items(110)
recommended_items

# COMMAND ----------

#Evaluation of the algorithm
def calculate_precision_recall(recommended_items, relevant_items):
    """
    Calculate Precision and Recall for recommendations. Assuming relevant items with EI = 1 due to lack of user-interaction data

    Parameters:
    - recommended_items (list): List of item IDs recommended by the above model.
    - relevant_items (list): List of item IDs that are relevant (e.g., items purchased by the user).

    Returns:
    - Precision (float): The proportion of recommended items that are relevant.
    - Recall (float): The proportion of relevant items that are recommended.
    """
    #Calculate the number of relevant items that are recommended
    true_positives = len(set(recommended_items) & set(relevant_items))
    #Calculate Precision and Recall
    precision = true_positives / len(recommended_items) if recommended_items else 0
    recall = true_positives / len(relevant_items) if relevant_items else 0
    
    return precision, recall


#Assuming that items with an 'EI' value of 1 & 2 are relevant items
relevant_items = pandas_df[pandas_df['EI'].isin([1, 2])]['ID'].tolist()

#Assuming we are recommending items for a user who is looking at item ID 110
recommended_items = recommend_items(110, cosine_sim).index.tolist()

#Calculate Precision and Recall
precision, recall = calculate_precision_recall(recommended_items, relevant_items)

print(f'Precision: {precision}')
print(f'Recall: {recall}')

def calculate_f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

f1_score = calculate_f1_score(precision, recall)
print(f"F1 Score: {f1_score}")

# COMMAND ----------

def suggest_items_by_material(pandas_df, material, minimum_percentage):
    """
    Suggest items based on the material content

    Parameters:
    - pandas_df (DataFrame): The preprocessed pandas DataFrame.
    - material (str): The name of the material (e.g., "Organic_cotton").
    - minimum_percentage (float): The minimum percentage of the material content, from 0 to 1.

    Returns:
    - DataFrame: A DataFrame of items that contain at least the specified minimum percentage of the given material.
    """
    if material not in pandas_df.columns:
        return f"Material '{material}' not found in the dataset."
    else:
        #Filter the DataFrame for items with at least the specified minimum percentage of the given material
        filtered_items = pandas_df[pandas_df[material] >= minimum_percentage]
        return filtered_items

#Example Items should contain atleast 80% Organic Cotton
material_name = "Organic_cotton"
minimum_percentage = 0.8 
suggested_items = suggest_items_by_material(pandas_df, material_name, minimum_percentage)

display(suggested_items)

'''
        Can be customised fucrther to add more types of materials in the filter
'''
