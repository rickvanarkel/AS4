import pandas as pd
from bs4 import BeautifulSoup
import os

path = './data/RMMS'
files = os.listdir(path)
htm_files = [file for file in files if file.endswith('traffic.htm')]
print(len(htm_files))
df_list = []
error_count = 0
error_list = []

for htm_file in htm_files:
    with open('./data/RMMS/'+htm_file, 'r') as file:
        html = file.read()
    print(htm_file)
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.findAll("table")
    x = 0
    try:
        for table in tables:
            if x == 3:
                data = []
                for row in table.find_all('tr'):
                    cells = row.find_all('td')
                    if cells:
                        data.append([cell.text.strip() for cell in cells])
                df = pd.DataFrame(data)
                df.drop(index=df.index[:4], axis=0, inplace=True)
                df.drop(index=df.index[-2:], axis=0, inplace=True)
                df = df.dropna(axis=1, how="all")
                row_list = df.iloc[0].values.flatten().tolist()
                df = df.set_index(df.iloc[:, 0], drop=True)
                df.columns = row_list
                df = df.iloc[1:]
                #df = df.iloc[:, 2:]
                df_list.append(df)
            x += 1
    except:
        print(f"ERROR: ~~~~~~{htm_file}~~~~~~")
        error_count += 1
        error_list.append(htm_file)

df_all_roads = pd.DataFrame(columns=[])  # initialize empty dataframe
for df in df_list:
    df_all_roads = pd.concat([df_all_roads, df])

df_all_roads.columns = ['Road' ,'Discription' ,'LRP', 'Offset', 'Chainage', 'LRP2' ,'Offset2', 'Chainage2' ,'(Km)' ,
 'Heavy Truck' ,'Medium Truck' ,'Small Truck', 'Large Bus', 'Medium Bus',
 'Micro Bus', 'Utility', 'Car', 'Auto Rickshaw' ,'Motor Cycle', 'Bi-Cycle',
 'Cycle Rickshaw', 'Cart' ,'Motorized', 'Non Motorized', 'Total AADT' ,'(AADT)'] # 'Road Simple'

print(f"During parsing {error_count} roads were empty")

df_all_roads['Simple Road'] = df_all_roads['Road'].str.split('-', expand=True)[0]

df_all_roads.to_csv('./data/all_traffic_info.csv')