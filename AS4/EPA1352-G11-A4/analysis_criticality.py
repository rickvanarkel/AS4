import warnings
warnings.filterwarnings('ignore')
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
Er kunnen zowel analyses gedaan worden op de NetworkX als op de dataframe
Voor nu zit er alleen een basisbestand in voor de analyse per road, en per road segment
'''

analysis_link = './data/traffic_network_compact_LB.csv'
analysis_df = pd.read_csv(analysis_link)

criticality_columns = ['road', 'road segment', 'lrp', 'Truck number', 'Truck percentage']
criticality_columns2 = ['road segment', 'Truck number', 'Truck percentage'] # Nu per road segment gekeken, kan op dezelfde manier met road

criticality_df = analysis_df.loc[:, criticality_columns2]

criticality_df_grouped = criticality_df.groupby('road segment').mean()
criticality_df_grouped = criticality_df_grouped.reset_index()
#criticality_df_grouped = criticality_df_grouped.sort_values('Truck number', ascending=False)

print(criticality_df_grouped)
print(criticality_df_grouped.nlargest(10, 'Truck percentage'))
print(criticality_df_grouped.nlargest(10, 'Truck number'))

#sns.histplot(criticality_df_grouped, x='Truck number')
sns.histplot(criticality_df_grouped, x='Truck percentage')
#plt.show()


'''
Er kunnen meer modaliteiten meegenomen worden uit de traffic data, pas dan including_traffic_data aan! 

Wat plaatjes en sorteer dingen doen! 

En sowieso nog wat met networkX!
'''