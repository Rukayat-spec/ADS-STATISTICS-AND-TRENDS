# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 10:27:54 2023

@author: ibrah
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import stats

# Load climate change data into Pandas DataFrame.
def load_climate_data(filename):

    """The function loads climate change data with its various indicators
    from the World Bank Dataset

    Keyword Argument:
    filename: the name of the world bank data file to be read for the study.

    Returns:
    Dataframe A: a dataframe containing climate change variations for different countries
    over years, with years as the column.
    Dataframe B: a transposed dataframe A, with countries as column.
    """

    # Load the dataset into a Pandas DataFrame, with years as columns.
    climate_df = pd.read_csv(filename, skiprows=4)
    climate_df = climate_df.drop(['Country Code','Indicator Code','Unnamed: 66'], axis=1)

    # Transposing and cleaning dataframe Climate_df, with countries as columns and cleaning.
    df_T = pd.DataFrame.transpose(climate_df)
    climate_T = df_T.set_axis(df_T.iloc[0], axis=1).iloc[1:]
    # returning the climate dataframe and its transpose.
    return climate_df, climate_T

# Passing the data set into the load_climate_data function.
climate_df, climate_T = load_climate_data('Climate_change_data.csv')

# Climate information spanning over a period of 60 years for various countries across the globe.


# Analysing the data set to select different climate change indicators for different
# countries for a period of time.

# Selecting the indicators of choice for the analysis.

climate_ind = ['Urban population', 'Arable land (% of land area)',\
             'CO2 emissions (kt)','Electric power consumption (kWh per capita)'
            ]

indicator = climate_df[climate_df['Indicator Name'].isin (climate_ind)]

# Transposing and cleaning the ind_df dataframe to be able to select countries of choice.
indicator_T =  pd.DataFrame.transpose(indicator)

# Making Country names the columns of the dataset.
indicator_T = indicator_T.set_axis(indicator_T.iloc[0], axis=1).iloc[1:]

# Selecting two countries from different regions of the world
countries = indicator_T.loc[:, ['United Kingdom','Russian Federation','Nigeria','South Africa', \
                          'United States','India', 'China','Brazil']]
ind_coun = countries.iloc[1:,:]


# Dropping missing values from the dataset.
ind_coun.dropna(inplace = True)
data_df = ind_coun.iloc[3:, :]

# Converting dataframe data type to numeric(float64).
# The data frame is the new data frame for this study
study_df = data_df.apply(pd.to_numeric)
print(study_df)

#Visualization 1: Bar Chart
"""Generating a subset of study_df dataframe pertaining to carbon dioxide emissions
 for the selected countries.
 """
co2_emi_df= study_df.iloc[:,[1,5,9,13,17,21,25,29]]
co2_emi_df.index = pd.to_numeric(co2_emi_df.index)
co2_emi_T = co2_emi_df.T
co2_emi_T = co2_emi_T.loc[:, 1995:2014 :3]

# Exploring statistical properties of the C02 emmission for the different countries
co2_emi_T.describe()
print('Mean:', np.mean(co2_emi_T))
print('Skewness:',stats.skew(co2_emi_T))
print('Kurtosis:',stats.kurtosis(co2_emi_T))
('Pearsons:',co2_emi_T.corr())

# plotting  grouped bar chart for c02_emi_T

co2_emi_T.plot(kind='bar',figsize=[10,4])
plt.style.use('ggplot')
plt.title('C02 Emission over the years', fontsize = 12, c='k',  fontname="Times New Roman Bold")
plt.xlabel('Country Name', c ='k')
plt.ylabel('CO2 emissions (kt)', c ='k')
plt.rcParams["figure.dpi"] = 300
plt.legend(loc=2)

#save chart
plt.savefig('bar_chart.png')

#show chart
plt.show()

#Visualization 2: Line Plot for Urban Population

# Generating a subset of study_df dataframe pertaining to urban population for all
# the chosen countries.
pop_urban = study_df.iloc[:,[0,4,8,12,16,20,24,28]]

# Exploring statistical properties of the urban population indicators
pop_urban.describe()
print('Mean:', np.mean(pop_urban))
print('Skewness:',stats.skew(pop_urban))
print('Kurtosis:',stats.kurtosis(pop_urban))
('Pearsons:',pop_urban.corr())

for country in pop_urban:
    pop_urban[country]
    plt.plot(pop_urban.index.astype(int),pop_urban[country], label= country)

# Add title, legend, and labels to plot
plt.title('Trend in Urban Population from 1995 to 2014', fontsize = 11, c = 'k')
plt.legend(bbox_to_anchor=(1.0,1))
plt.xlabel('Year', c ='k')
plt.ylabel('Urban Population', c = 'k')
plt.xticks([1995,1998,2001,2004,2007,2010,2013])

#save plot
plt.savefig('Line_plot.png')

# Show plot
plt.show()

"""Visualization 3: scatter plot showing relationship between Arable land
 and Urban Population of Brazil
"""
# Generating a subset of new_df dataframe pertaining to carbon dioxide emissions for all the chosen countries.
arable_df= study_df.iloc[:,[3,7,11,15,19,23,27,31]]
arable_df.index = pd.to_numeric(arable_df.index)
arable_df

"""
Plotting a scatter plot to show relationship for Co2 emmission and Forest Area for Brazil
"""

plt.scatter(arable_df['Brazil'], pop_urban['Brazil'])
plt.title('Relationship between Forest Area and Co2 emmission in Brazil')
plt.xlabel('Forest area (% of land area)')
plt.ylabel('Nations Co2 Emmision')

plt.savefig('')
plt.show()



china_df = indicator_T.loc[:,'China']
china_df = china_df.set_axis(china_df.iloc[0], axis=1).iloc[1:]
china_df.dropna(inplace= True)
china_df = china_df.astype(float)  #conversion of the dataframe to a float
china_df


china_cor = china_df.corr().round(2)
#plotting the heatmap and specifying the plot parameters
plt.imshow(china_cor, cmap='Accent_r', interpolation='none')
plt.colorbar()
plt.xticks(range(len(china_cor)), china_cor.columns, rotation=90)
plt.yticks(range(len(china_cor)), china_cor.columns)
plt.gcf().set_size_inches(8,5)


#labelling of the little boxes and creation of a legend
labels = china_cor.values
for y in range(labels.shape[0]):
    for x in range(labels.shape[1]):
        plt.text(x,y, '{:.2f}'.format(labels[y,x]), ha='center', va='center',
                  color='black')
plt.title('Correlation Map of  Region')
plt.savefig("Heat Map of China")
















