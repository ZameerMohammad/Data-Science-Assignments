

#import the file

import pandas as pd
import numpy as np

df = pd.read_csv('book.csv',encoding='latin')
df
df.shape
df.head()

print(df.columns)

len(df['User.ID'].unique())

df['Book.Title'].unique()

df_no_duplicates = df.drop_duplicates(['User.ID', 'Book.Title'])
user_df = df_no_duplicates.pivot(index='User.ID',
                                  columns='Book.Title',
                                  values='Book.Rating')

user_df

# Create a user-book matrix
user_book_matrix = df.pivot_table(index='User.ID', columns='Book.Title', values='Book.Rating', fill_value=0)

#Impute those NaNs with 0 values
user_df.fillna(0, inplace=True)

user_df

# Calculate cosine similarity between users
from sklearn.metrics import pairwise_distances
user_sim = 1 - pairwise_distances(user_df.values,metric='cosine')

user_sim

type(user_sim)

#Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)

#Set the index and column names to user ids 
user_sim_df.index = df['User.ID'].unique()
user_sim_df.columns = df['User.ID'].unique()

user_sim_df

# Set diagonal elements (self-similarity) to 0 in the user similarity matrix
np.fill_diagonal(user_sim, 0)

# Display a subset of the user similarity matrix
user_sim_df.iloc[0:7, 0:7]

# Find the User ID with the highest similarity for each user
user_sim_df.idxmax(axis=1)[0:2182]
user_sim_df.idxmax(axis=1)[0:2182]

# Filter the original DataFrame for rows where User ID is either 6 or 168
df[(df['User.ID']==276729) | (df['User.ID']==276726)]
df[(df['User.ID']==276736) | (df['User.ID']==276726)]
df[(df['User.ID']==162109) | (df['User.ID']==276726)]
 

