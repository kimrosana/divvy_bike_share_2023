# -*- coding: utf-8 -*-
"""
Bike Share Case Study - 2023 Data from Divvy's public datasets available on their website

by Rosana Kim
"""

#%% importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import prince
pio.renderers.default = 'browser'

#%% Importing data

# List of file paths (adjust as per your directory structure)
file_paths = ["original dataset/202301-divvy-tripdata.csv","original dataset/202302-divvy-tripdata.csv","original dataset/202303-divvy-tripdata.csv", "original dataset/202304-divvy-tripdata.csv","original dataset/202305-divvy-tripdata.csv","original dataset/202306-divvy-tripdata.csv","original dataset/202307-divvy-tripdata.csv","original dataset/202308-divvy-tripdata.csv","original dataset/202309-divvy-tripdata.csv","original dataset/202310-divvy-tripdata.csv","original dataset/202311-divvy-tripdata.csv","original dataset/202312-divvy-tripdata.csv"]

# Initializing an empty list to store DataFrames
dfs = []

# Looping through each file path and reading CSV into a DataFrame
for file in file_paths:
    df = pd.read_csv(file)
    dfs.append(df)

# Concatenating all DataFrames into a single DataFrame
TripData = pd.concat(dfs, ignore_index=True)

#%% Checking data integrity

print(TripData.info)

print(TripData['started_at'].head(10))

print(TripData.shape)

print(TripData.columns)

print(TripData.dtypes)

#%% Converting dtypes

TripData['started_at'] = pd.to_datetime(TripData['started_at'])
TripData['ended_at'] = pd.to_datetime(TripData['ended_at'])
TripData['member_casual'] = TripData['member_casual'].astype(str)

#%% Cleaning dataset

# Maintaining a copy of the original dataset for backup purposes

TripDataorg = TripData ## only 'TripData' will be used

# Checking for duplicates

TripData_noduplicates = TripData.drop_duplicates()

    ## There is no difference in the amount of rows, therefore, the dataset will be deleted
    
del TripData_noduplicates

# As the focus will be on the difference between casual riders and members, rows with NA in 'member_casual' shall be deleted

TripData = TripData.dropna(subset=['member_casual'])

    ## Again, there is no difference in the total amount of rows
#%% Descriptive Analysis

# Total count for each category
member_casual_counts = TripData['member_casual'].value_counts()
rideable_type_counts = TripData['rideable_type'].value_counts()

    ## Graph member_casual
plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
member_casual_counts.plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
plt.title('Counts of Member vs Casual')
plt.xlabel('Member Type')
plt.ylabel('Counts')
plt.tight_layout()
plt.show()

    ## Graph rideable type
plt.figure(figsize=(12,6))
plt.subplot(1, 2, 2)
rideable_type_counts.plot(kind='bar', color=['#2ca02c', '#9467bd', '#d62728'])
plt.title('Counts of Rideable Types')
plt.xlabel('Rideable Types')
plt.ylabel('Counts')
plt.tight_layout()
plt.show()

# Create a ride_length in seconds

TripData['ride_length_seconds'] = TripData['ended_at'] - TripData['started_at'].dt.total_seconds()
print(TripData['ride_length_seconds'])

## Filter rides longer than 30 minutes (long_rides)

long_rides = TripData[TripData['ride_length_seconds'] > 60*30]
print(long_rides)

## Count the number of long_rides

num_long_rides = long_rides.shape[0]
print(num_long_rides)

## Create a table to show the count of long rides dividing into 2 groups: member and casual

ride_counts = long_rides.groupby('member_casual').size().reset_index(name='count')
print(ride_counts)

# Calculate the mean of ride_length

mean_ride_length = TripData['ride_length_seconds'].mean()

# Calculate the max ride_length

max_ride_length = TripData['ride_length_seconds'].max()
print(max_ride_length/60)

# Create a day_of_week column, noting that 1 = Sunday and 7 = Saturday

TripData['day_of_week'] = TripData['started_at'].dt.dayofweek + 1
print(TripData['day_of_week'])

# Calculate the mode of day_of_week

mode_DoF = TripData['day_of_week'].mode()
print(mode_DoF)

# Calculate the mode of ride_length_seconds

mode_ride_length = TripData['ride_length_seconds'].mode()
print(mode_ride_length)

# Calculate the average ride_length for members and casual riders

avg_ride_length_min = TripData.groupby('member_casual')['ride_length_seconds'].mean()/60
print(avg_ride_length_min)

# Calculate the average ride_length for users by day_of_week

avg_RD_DoF_min = TripData.groupby('day_of_week')['ride_length_seconds'].mean()/60
print (avg_RD_DoF_min)

# Calculate the number of rides for users by day_of_week by counting trip_id

NumOfRides_DoF = TripData.groupby('day_of_week')['ride_id'].count().reset_index()
print(NumOfRides_DoF)

# Calculate the number of riders per rideable_type

count_rideable_types = TripData['rideable_type'].value_counts()
print(count_rideable_types)

count_ride_types = count_rideable_types.reset_index()
count_ride_types.columns = ['rideable_type', 'count']

count_ride_types['rideable_type'] = count_ride_types['rideable_type'].astype('category')
count_ride_types['xount'] = count_ride_types['count'].astype('int')
print(count_ride_types['rideable_type'].info())


#%% Correspondence Analysis - finding out relationship between member_casual and rideable type

# Creating a contingency Table

cont_table = pd.crosstab(TripData["member_casual"], TripData["rideable_type"])
print (cont_table)

# Analysing statistical significance (chi square) - is there association between these two variables?

chi_square = chi2_contingency(cont_table)
print(cont_table)

print(f"chi²: {round(chi_square[0], 2)}")
print(f"p-value: {round(chi_square[1],4)}")
print(f"degree of freedom: {chi_square[2]}")

    ## As p-value is smaller than 5%, H0 is rejected and we validate H1, which indicates there is a significative association between these variables

# Residuals - Contingency Table

tab_cont = sm.stats.Table(cont_table)

print(tab_cont.fittedvalues)
print(tab_cont.chi2_contribs)
print(tab_cont.resid_pearson)
print(tab_cont.standardized_resids)

    ## > 1.96 = member X classic_bike, casual X docked_bike and casual X electric_bike
    
TripDataRelevant = TripData.drop(columns = ['ride_id','started_at','ended_at','start_station_name', 'start_station_id', 'end_station_name',
'end_station_id', 'start_lat', 'start_lng', 'end_lat', 'end_lng'])
print(TripDataRelevant.dtypes)

TripDataRelevant_encoded = pd.get_dummies(TripDataRelevant)

mca = prince.CA().fit(TripDataRelevant_encoded)

eigenvalues_table = mca.eigenvalues_summary

print(ca.total_inertia_)

print(ca.svd_.U)
print(ca.svd_.V.T)

print(ca.row_coordinates(cont_table))
print(ca.column_coordinates(cont_table))

print(chart_df_row.columns)
print(chart_df_col.columns)

# Percentual map

chart_df_row = pd.DataFrame({'var_row': cont_table.index,
                             'x_row':ca.row_coordinates(cont_table)[0].values,
                             'y_row':ca.row_coordinates(cont_table)[1].values})

chart_df_col = pd.DataFrame({'var_col': cont_table.columns,
                             'x_col':ca.column_coordinates(cont_table)[0].values,
                             'y_col':ca.column_coordinates(cont_table)[1].values})

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'] + 0.03, point['y'] - 0.02, point['val'], fontsize=6)

label_point(x = chart_df_col['x_col'],
            y = chart_df_col['y_col'],
            val = chart_df_col['var_col'],
            ax = plt.gca())

label_point(x = chart_df_row['x_row'],
            y = chart_df_row['y_row'],
            val = chart_df_row['var_row'],
            ax = plt.gca()) 

sns.scatterplot(data=chart_df_row, x='x_row', y='y_row', s=20)
sns.scatterplot(data=chart_df_col, x='x_col', y='y_col', s=20)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.axhline(y=0, color='lightgrey', ls='--')
plt.axvline(x=0, color='lightgrey', ls='--')
plt.tick_params(size=2, labelsize=6)
plt.title("Mapa Perceptual - Anacor", fontsize=12)
plt.xlabel(f"Dim. 1: {tabela_autovalores.iloc[0,1]} da inércia", fontsize=8)
plt.ylabel(f"Dim. 2: {tabela_autovalores.iloc[1,1]} da inércia", fontsize=8)
plt.show()


#%% How do annual members and casual riders use Cyclistic bikes differently?

# Create a dataframe grouping by 'member_casual'

# variables: min, max, mode, average [ride_length], rideable_type, mode_DoF, min_DoF, max_DoF
# observations: member, casual