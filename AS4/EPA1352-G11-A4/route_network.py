import warnings
warnings.filterwarnings('ignore')
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


def make_points_edges(df, weight_label = 'length', id_l = "id"):
    """
    This function takes a pandas DataFrame containing GPS data and converts it into a dictionary of road segments,
    where each road segment is represented by a tuple of points, edges and point IDs. It iterates over each unique
    road type and creates a separate dictionary entry for each one. It then loops over each row in the DataFrame and
    appends the point coordinates, point IDs, and edge information (if applicable) to the appropriate tuple in the dictionary.

    """
    road_dict = {}
    roads = df['road'].unique()

    for road_type in roads:
        temp_df =  df.loc[df['road'] == road_type]
        point_ids = []
        temp_points = []
        temp_edges = []
        temp_df.reset_index(inplace = True)
        temp_id = 0

        for index, row in temp_df.iterrows():
            temp_points.append((row['lat'], row['lon']))
            point_ids.append(row[id_l])
            if index != 0:
                if row[id_l] != temp_id:
                    temp_edges.append((temp_id, row[id_l], row[weight_label]))
            temp_id = row[id_l]
        road_dict[road_type] = (temp_points, temp_edges, point_ids)

    return road_dict



def make_networkx(road_dict, df):
    """
    This function takes the dictionary of road segments and a pandas DataFrame and creates a NetworkX graph object
    representing the road network.  It iterates over each road type, extracts the point coordinates, edge information,
    and point IDs for each segment, and adds nodes and edges to the graph object accordingly.
    """
    G = nx.Graph()
    roads_temp = df['road'].unique()
    for roads in roads_temp:
        temp_list = road_dict[roads]
        points = temp_list[0]
        edges = temp_list[1]
        point_ids = temp_list[2]

        for i in range(len(point_ids)):
            G.add_node(point_ids[i], pos=points[i])
        for i in range(len(edges)):
            G.add_edge(edges[i][0], edges[i][1], weight = edges[i][2])

    return G

def create_graph(G):
    pos = nx.get_node_attributes(G, 'pos')

    nx.draw_networkx(G, pos=pos, node_color='k', with_labels=False)
    nx.draw_networkx(G, pos=pos, node_size=10, with_labels=False)

    nodes_with_more_than_2_edges = [node for node, degree in dict(G.degree()).items() if degree > 2]
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_with_more_than_2_edges, node_color='r', node_size=50)

    plt.axis()
    plt.show()

def create_colored_network(G):
    # Extract node positions from networkx graph
    pos = nx.get_node_attributes(G, 'pos')

    # Get edge weights from networkx graph
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    # Compute mean and standard deviation of edge weights
    mean = np.nanmean(weights)
    std = np.nanstd(weights)

    # Normalize edge weights using z-score
    normalized_weights = [(w - mean) / std for w in weights]

    colormap = plt.cm.copper_r
    # Assign a color to each edge based on its normalized weight
    edge_colors = [colormap(x) for x in normalized_weights]

    fig, ax = plt.subplots()
    nx.draw_networkx(G, pos=pos, edge_color=edge_colors, node_color="none", with_labels=False, ax=ax)

    # Create a colorbar for the edge weights
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=colormap,
                                      values=np.arange(min(weights), max(weights),
                                      ((max(weights)-min(weights))/20)))
    cbar.ax.tick_params(labelsize=10)

    return fig







