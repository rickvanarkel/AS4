import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import geopandas as gpd
from scipy.spatial import cKDTree
import requests
from bs4 import BeautifulSoup
import lxml

warnings.filterwarnings('ignore')

network_link = './data/whole_network_compact_LB.csv'
df_network = pd.read_csv(network_link)

traffic_N1_link = './data/test_csv_html_N1.csv'
df_traffic_N1 = pd.read_csv(traffic_N1_link, sep=';')

def process_roads(df_network):
    """

    """

    # Makes a list of unique roads in the dataset
    unique_roads = df_network.road.unique()

    #print(f'In total there are {len(relevant_roads)} relevant roads found, which are: {relevant_roads}.')
    #print(f'The pre-processing of each road is done separately.')

    df_road_N1 = df_network[df_network['road'] == 'N1']
    unique_roads = df_road_N1.road.unique()

    print(unique_roads)

    # Processes all roads individually
    for i in unique_roads:
        print(f'The road that is pre-processed now, is: {i}.')
        df_road_temp = df_network[df_network['road'] == i]
        assign_traffic(df_road_temp, i)

def assign_traffic(df_road, road):
    """

    """
    df_traffic = open_traffic_file(road)
    df_traffic = df_traffic_N1
    df_traffic = clean_traffic(df_traffic)
    match_traffic(df_traffic, df_road)
    concat_roads()
    save_data()

def open_traffic_file(road):
    """

    """
    #print(df_traffic_N1)
    #for i in df_traffic_N1:
    #    print(i)



def clean_traffic(df_traffic):

    # df.columns = df.columns.str.replace('[^a-zA-Z0-9]+', '')

    columns_traffic = ['LRP Start', 'Chainage Start', 'Heavy Truck', 'Medium Truck', 'Small Truck', 'Motorized']
    df_traffic = df_traffic.loc[:, columns_traffic]

    agg_functions = {'LRP Start': 'first', 'Heavy Truck': 'sum', 'Medium Truck': 'sum', \
                     'Small Truck': 'sum', 'Motorized': 'sum'} # , 'Chainage Start': 'first'

    # create new DataFrame by combining rows with same id values
    df_traffic = df_traffic.groupby(df_traffic['Chainage Start']).aggregate(agg_functions)
    df_traffic.reset_index()

    return df_traffic

    #print(df_traffic.head(5))

def match_traffic(df_traffic, df_road):
    # Find match in chainage
    df_test = pd.merge(df_road, df_traffic, left_on='chainage', right_on='Chainage Start' )
    print(df_test.head(5))

    return df_road

def concat_roads():
    pass

def save_data():
    pass

process_roads(df_network)