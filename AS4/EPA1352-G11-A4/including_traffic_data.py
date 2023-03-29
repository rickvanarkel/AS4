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

def process_roads():
    """

    """

    # Makes a list of unique roads in the dataset
    unique_roads = df_network.road.unique()

    print(f'In total there are {len(relevant_roads)} relevant roads found, which are: {relevant_roads}.')
    print(f'The pre-processing of each road is done separately.')

    # Processes all roads individually
    for i in unique_roads:
        print(f'The road that is pre-processed now, is: {i}.')
        df_road_temp = df_network[df_network['road'] == i]
        assign_traffic(df_road_temp, i)

def assign_traffic(df_road, road):
    """

    """
    df_traffic = open_traffic_file(road)
    match_traffic(df_traffic, df_road)
    concat_roads()
    save_data()

def open_traffic_file(road):
    """

    """
    with open('./data/RMMS/N1.traffic.htm', 'r') as file:
        html = file.read()

    # Parse the HTML file using BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    # Find the table element in the HTML
    table = soup.find('table')

    # Extract the data from the table
    data = []
    for row in table.find_all('tr'):
        cells = row.find_all('td')
        if cells:
            data.append([cell.text.strip() for cell in cells])


    # Match file die begint met road nummer, met de extentie
    traffic_link = f'{road}.traffic.htm'
    df_traffic = pd.read_html(traffic_link)

    return df_traffic

def match_traffic(df_traffic, df_road):
    # Find match in chainage
    return df_road



