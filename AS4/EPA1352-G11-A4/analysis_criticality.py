import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
import route_network as rn

"""Load data and extract criticality columns"""
analysis_link = './data/traffic_network_LB.csv'
analysis_df = pd.read_csv(analysis_link)

criticality_columns = ['road', 'lrp', 'road segment', 'Truck number', 'Heavy Truck', 'Medium Truck', 'Small Truck', 'Truck percentage']
criticality_df = analysis_df.loc[:, criticality_columns]

"""Group criticality data by road segment"""
criticality_df_grouped = criticality_df.groupby('road segment').mean()
criticality_df_grouped = criticality_df_grouped.reset_index()
total_df_grouped = criticality_df_grouped.copy()

"""Prepare data for visualization"""
analyze_columns_L = ['road segment', 'Heavy Truck', 'Medium Truck', 'Small Truck']
criticality_df = criticality_df_grouped.loc[:, analyze_columns_L]
criticality_df['weighted_total'] = criticality_df['Heavy Truck'] + 2*criticality_df['Medium Truck'] + 3*criticality_df['Small Truck']
analyze_columns_L = ['road segment', 'Heavy Truck', 'Medium Truck', 'Small Truck', 'weighted_total']

"""Write data to Excel"""
with pd.ExcelWriter('.\data\criticality_output.xlsx') as writer:
    for i in analyze_columns_L[1:]:
        temp_df = criticality_df.loc[:, ['road segment', i]]
        df_top10 = temp_df.nlargest(10, i)
        sheet_name = f'Top 10 {i}'
        df_top10.to_excel(writer, sheet_name=sheet_name, index=False)
print("Excel file is created!")

"""Create network visualizations"""
def create_visualizations(labels_list, df):
    print(f"{len(labels_list)} images are being created!")
    for label in labels_list:
        road_dict = rn.make_points_edges(df, weight_label=label)
        G = rn.make_networkx(road_dict, df)
        fig = rn.create_colored_network(G)
        # set the resolution to 300 pixels per inch
        dpi = 600
        # specify the output filename based on the weight label
        output_filename = f".\img\{label}.png"

        # save the figure as a PNG file with high resolution
        fig.savefig(output_filename, dpi=dpi)
        plt.show()
        plt.close(fig)
        print(f"{label}.png file was created!")

"""Compute weighted total and create visualizations"""
analysis_df['weighted_total'] = analysis_df['Heavy Truck'] + 2*analysis_df['Medium Truck'] + 3*analysis_df['Small Truck']
weight_labels = ["Heavy Truck", "Medium Truck", "Small Truck" , "weighted_total"]
create_visualizations(weight_labels, analysis_df)
