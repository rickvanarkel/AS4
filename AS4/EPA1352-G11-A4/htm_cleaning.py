import pandas as pd
from bs4 import BeautifulSoup
import os

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
################################### Functions start here! ############################################
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def get_table_htm(htm_file):
    """
    This first function, get_table_htm(htm_file), takes an argument htm_file, which is a filename in the ./data/RMMS/
    directory. It opens the file, reads its contents, and stores them in the html variable. It then creates a
    BeautifulSoup object from the html string, using the html.parser parser. The function finds all the tables in
    the HTML document and returns them in a list.
    """
    with open('./data/RMMS/' + htm_file, 'r') as file:
        html = file.read()
    print(htm_file)
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.findAll("table")

    return tables

def get_data_htm(tables, df_list):
    """
    This second function, get_data_htm(tables, df_list), takes a list of HTML tables and an empty list of pandas
    DataFrames, df_list, as its arguments. It loops over each table in the tables list, and for the third table only,
    it extracts the data by looping over each row of the table and each cell of the row, appending each cell's text
    content to a nested list. It then creates a pandas DataFrame from the nested list, removing the first four and
    last two rows, dropping columns that are all NaN, setting the first column as the index, and renaming the columns
    using the values from the first row of the DataFrame. Finally, the function appends the resulting DataFrame to the
    df_list. The function returns the modified df_list.
    """
    table_counter = 0

    for table in tables:
        if table_counter == 3:
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
            df_list.append(df)
        table_counter += 1

    return df_list

def display_error(htm_file, error_count, error_list):
    """
    The third function, display_error(htm_file, error_count, error_list), takes an htm_file, an error_count, and an
    error_list as arguments. It prints an error message indicating that the htm_file is empty and increments the
    error_count and error_list accordingly. It then returns the updated error_count and error_list.
    """
    print(f"ERROR: ~~~~~~{htm_file}~~~~~~")
    print(f'Check if this file .htm file contains any data!')
    error_count += 1
    error_list.append(htm_file)

    return error_count, error_list

"""
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 ################################### Code starts here! ################################################
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

path = './data/RMMS'
files = os.listdir(path)
htm_files = [file for file in files if file.endswith('traffic.htm')]
print(len(htm_files))

# create an empty list to store the DataFrames
df_list = []

# initialize error counters
error_count = 0
error_list = []

# loop through each .htm file and extract the tables
for htm_file in htm_files:
    tables = get_table_htm(htm_file)
    # attempt to extract data from the tables, handle errors and record in error_count and error_list
    try:
        df_list = get_data_htm(tables, df_list)
    except:
        error_count, error_list = display_error(htm_file, error_count, error_list)

# concatenate all DataFrames into a single DataFrame
df_all_roads = pd.DataFrame(columns=[])
for df in df_list:
    df_all_roads = pd.concat([df_all_roads, df])

# set column names for the DataFrame
df_all_roads.columns = ['Road', 'Discription', 'LRP', 'Offset', 'Chainage', 'LRP2', 'Offset2', 'Chainage2', '(Km)',
                        'Heavy Truck', 'Medium Truck', 'Small Truck', 'Large Bus', 'Medium Bus',
                        'Micro Bus', 'Utility', 'Car', 'Auto Rickshaw', 'Motor Cycle', 'Bi-Cycle',
                        'Cycle Rickshaw', 'Cart', 'Motorized', 'Non Motorized', 'Total AADT', '(AADT)']

# print a message indicating the number of empty roads found during parsing
print(f"During parsing {error_count} roads were empty")

# extract the first part of the Road column to create a new column called 'Simple Road'
df_all_roads['Simple Road'] = df_all_roads['Road'].str.split('-', expand=True)[0]
df_all_roads.to_csv('./data/all_traffic_info.csv')

