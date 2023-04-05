import warnings
warnings.filterwarnings('ignore')
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import route_network as rn

'''
Er kunnen zowel analyses gedaan worden op de NetworkX als op de dataframe
Voor nu zit er alleen een basisbestand in voor de analyse per road, en per road segment
'''

analysis_link = './data/traffic_network_LB.csv'
analysis_df = pd.read_csv(analysis_link)

criticality_columns = ['road', 'road segment', 'lrp', 'Truck number', 'Truck percentage']
criticality_columns2 = ['road', 'lrp', 'road segment', 'Truck number', 'Heavy Truck', 'Medium Truck', 'Small Truck', 'Truck percentage'] # Nu per road segment gekeken, kan op dezelfde manier met road

criticality_df = analysis_df.loc[:, criticality_columns2]

criticality_df_grouped = criticality_df.groupby('road segment').mean()
criticality_df_grouped = criticality_df_grouped.reset_index()
total_df_grouped = criticality_df_grouped.copy()

analyze_columns_L = ['road segment', 'Heavy Truck', 'Medium Truck', 'Small Truck']
criticality_df_heavy = criticality_df_grouped.loc[:, analyze_columns_L]

for_loop = ['Heavy Truck', 'Medium Truck', 'Small Truck']
for i in for_loop:
    print(f'This DF shows the most critical segments sorted on the number of {i}: \n {criticality_df_heavy.nlargest(10, i)} \n')

analysis_df['weighted_total'] = analysis_df['Heavy Truck'] + 2*analysis_df['Medium Truck'] + 3*analysis_df['Small Truck']

road_dict = rn.make_points_edges(analysis_df, weight_label="Heavy Truck", id_l='id')
G = rn.make_networkx(road_dict, analysis_df)
rn.create_colored_network(G)

road_dict = rn.make_points_edges(analysis_df, weight_label="Medium Truck", id_l='id')
G = rn.make_networkx(road_dict, analysis_df)
rn.create_colored_network(G)

road_dict = rn.make_points_edges(analysis_df, weight_label="Small Truck", id_l='id')
G = rn.make_networkx(road_dict, analysis_df)
rn.create_colored_network(G)

road_dict = rn.make_points_edges(analysis_df, weight_label="weighted_total", id_l='id')
G = rn.make_networkx(road_dict, analysis_df)
rn.create_colored_network(G)


'''
Er kunnen meer modaliteiten meegenomen worden uit de traffic data, pas dan including_traffic_data aan! 

Wat plaatjes en sorteer dingen doen! 

En sowieso nog wat met networkX!
'''