import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

network_link = './data/whole_network_compact_LB.csv'
df_network = pd.read_csv(network_link)

traffic_link = './data/all_traffic_info.csv'
df_traffic = pd.read_csv(traffic_link)
df_traffic = df_traffic.replace('NS', 0) #Check of dit wel goed gaat! Bij np.nan volgens mij niet?

def process_roads(df_network):
    """

    """
    # Makes a list of unique roads in the dataset
    unique_roads = df_network.road.unique()
    #unique_roads = ['N1', 'N2', 'N3', 'N4']

    print(f'In total there are {len(unique_roads)} relevant roads found, which are: {unique_roads}.')
    print(f'The pre-processing of each road is done separately.')

    # Processes all roads individually
    for i in unique_roads:
        print(f'The road that is pre-processed now, is: {i}.')
        df_road_temp = df_network[df_network['road'] == i]
        df_traffic_temp = df_traffic[df_traffic['Simple Road'] == i]
        assign_traffic(df_road_temp, df_traffic_temp)

def assign_traffic(df_road, df_traffic):
    """

    """
    df_traffic = clean_traffic(df_traffic)
    df_matched = match_traffic(df_traffic, df_road)
    calculate_columns(df_matched)
    append_information(df_matched)

def clean_traffic(df_traffic):
    """
    cleans the traffic information of the relevant road,
    generates the columns of interest,
    converts the values to the right data types,
    groups the traffic information per chainage to obtain the right metric for analysis
    """
    columns_traffic = ['LRP', 'Chainage', 'Heavy Truck', 'Medium Truck', 'Small Truck', 'Motorized']
    df_traffic = df_traffic.loc[:, columns_traffic]

    convert_dict = {'LRP': str,
                    'Chainage': float,
                    'Heavy Truck': float,
                    'Medium Truck': float,
                    'Small Truck': float,
                    'Motorized': float
                    }

    df_traffic = df_traffic.astype(convert_dict)

    agg_functions = {'LRP': 'first', 'Heavy Truck': 'sum', 'Medium Truck': 'sum', 'Small Truck': 'sum', 'Motorized': 'sum'}
    df_traffic = df_traffic.groupby(df_traffic['Chainage']).aggregate(agg_functions)
    df_traffic.reset_index()

    return df_traffic

def match_traffic(df_traffic, df_road):
    """
    matches the traffic data to the road data, based on chainage
    generates road segments
    fills in missing values of traffic volume with forward fill
    """
    # Find match in chainage
    df_test = pd.merge(df_road, df_traffic, left_on='chainage', right_on='Chainage')
    df_test2 = pd.concat([df_test, df_road])
    df_test3 = df_test2.drop_duplicates()
    df_test4 = df_test3.sort_values('chainage')
    df_test5 = df_test4.reset_index()
    df_test5 = df_test5.drop('index', axis=1)

    # assign road segments
    df_test6 = generate_segments(df_test5)
    df_test6[['condition', 'bridge_length']] = df_test6[['condition', 'bridge_length']].fillna('NaN')

    # assign traffic volumes between measure points
    df_test7 = df_test6.fillna(method='ffill')
    df_test7['chainage'] = df_test7.chainage.astype(str).str.replace('.', '').astype(float)
    df_test8 = df_test7.drop_duplicates()
    df_test9 = df_test8.replace('NaN', np.nan)

    df_road = df_test9
    return df_road

def generate_segments(df_road):
    """
    recognises different road parts when traffic information is not empty
    """
    road = df_road.road.unique()[0]
    segment = 1
    for index, row in df_road.iterrows():
        if not pd.isna(row['Medium Truck']):
            segment_id = f"{road}_{segment}"
            df_road.at[index, 'road segment'] = segment_id
            segment += 1
        else:
            df_road.at[index, 'road segment'] = np.nan
    return df_road

def calculate_columns(df_matched):
    """
    generates columns with the number of trucks and a truck percentage
    """
    df_matched['Truck number'] = df_matched['Heavy Truck'] + df_matched['Medium Truck'] + df_matched['Small Truck']
    df_matched['Truck percentage'] = df_matched['Truck number'] / df_matched['Motorized']

    return df_matched

traffic_road_list = []
def append_information(df_road):
    """
    appends the dataframe of each processed road to a list
    """
    traffic_road_list.append(df_road)
    return traffic_road_list

def expand_condition(df_road):
    """
    expands the bridge condition into four columns, where the row equals value 1 if there is a bridge with a certain condition
    """
    expanded_df = pd.get_dummies(df_road['condition'], prefix='condition')
    df_expanded = pd.concat([df_road, expanded_df], axis=1)
    return df_expanded

def concat_roads():
    """
    combines the dataframes of all individual roads into one dataframe
    """
    df_all_roads = pd.DataFrame()  # initialize empty dataframe, columns=column_names

    print('All roads are pre-processed. The seperate files are being combined, completed and presented (figure and csv exports).')

    for df in traffic_road_list:
        df_all_roads = pd.concat([df_all_roads, df])  # append to df_all_roads in each iteration

    df_all_roads = df_all_roads.reset_index()
    df_all_roads = expand_condition(df_all_roads)
    save_data(df_all_roads)

    print('The files are ready.')

model_columns = ['road', 'road segment', 'lrp', 'id', 'lat', 'lon', 'condition', 'bridge_length', 'Truck number', 'Truck percentage', \
                 'condition_A', 'condition_B', 'condition_C', 'condition_D']
def save_data(df):
    """
    saves the data frame to csv files
    """
    # Write the dataframe to csv
    df.to_csv('./data/traffic_network_LB.csv')

    # Make compact datafile and export to csv
    df_all_roads_compact = df.loc[:, model_columns]
    df_all_roads_compact.to_csv('./data/traffic_network_compact_LB.csv')

process_roads(df_network)
concat_roads()