import matplotlib.pylab as plt
import matplotlib.colors as colors

import networkx as nx
from suppstats import *
from filefunctions import *
import seaborn as sns
import pandas as pd

import numpy as np

def output_characteristics(network):
    print ("OUTPUT:")
    n_nodes = len(network.nodes)
    print ("n_nodes is " + str(n_nodes))
    n_edges = len(network.edges)
    print ("n_edges is " + str(n_edges))
    avg_degree = get_avg_degree(network)
    print ("avg_degree is " + str(avg_degree))
    # index is the degree, value is the count
    degree_dist = nx.degree_histogram(network)
    #degree_dist = [count/sum(degree_dist) for count in degree_dist]
    print ("degree_dist is " + str(degree_dist))
    avg_clustering = nx.average_clustering(network)
    print ("avg_clustering is " + str(avg_clustering))
    clustering_dist_degree = get_clustering_dist_degree(network)
    print ("clustering_dist_degree is " + str(clustering_dist_degree))
    #max_clique_size = get_max_clique_size(network)
    max_clique_size = nx.graph_clique_number(network)
    print ("max_clique_size is " + str(max_clique_size))
    avg_clique_size = get_avg_clique_size(network)
    print ("avg_clique_size is " + str(avg_clique_size))
    clique_size_dist = get_clique_size_dist(network)
    print ("clique_size_dist is " + str(clique_size_dist))
    diameter = nx.algorithms.distance_measures.diameter(network)
    print ("diameter is " + str(diameter))
    avg_shortest_path_length = nx.algorithms.shortest_paths.generic.average_shortest_path_length(network)
    print ("avg_shortest_path_length is " + str(avg_shortest_path_length))
    hopplot_dist = get_hopplot_dist(network)
    print ("hopplot_dist is " + str(hopplot_dist))
    connected_prob = (len(network.edges) / ((len(network.nodes) * (len(network.nodes) - 1) / 2)))
    print ("Probability any two people are connected: " + str(connected_prob))
    connected_prob_gender = probability_connected(network, "Gender", True)
    print ("Probability any two people are connected given they are the same gender: " + str(connected_prob_gender))
    connected_prob_hall = probability_connected(network, "Hall", True)
    print ("Probability any two people are connected given they live in the same hall: " + str(connected_prob_hall))
    connected_prob_wing = probability_connected(network, "Wing", True)
    print ("Probability any two people are connected given they live in the same wing: " + str(connected_prob_wing))
    degree_by_height = sorted(get_degree_by_height(network).items())
    print ("Degree by height distribution: " + str(degree_by_height))
    degree_by_ec = sorted(get_degree_by_ec(network).items())
    print ("Degree by extracurricular count: " + str(degree_by_ec))
    one_mutual = probability_connected_mutual(network, 1)
    two_mutual = probability_connected_mutual(network, 2)
    three_mutual = probability_connected_mutual(network, 3)
    four_mutual = probability_connected_mutual(network, 4)
    five_mutual = probability_connected_mutual(network, 5)
    six_mutual = probability_connected_mutual(network, 6)
    seven_mutual = probability_connected_mutual(network, 7)
    eight_mutual = probability_connected_mutual(network, 8)
    nine_mutual = probability_connected_mutual(network, 9)
    ten_mutual = probability_connected_mutual(network, 10)
    eleven_mutual = probability_connected_mutual(network, 11)
    #twelve_mutual = probability_connected_mutual(network, 12)
    print (str(one_mutual))
    print (str(two_mutual))
    print (str(three_mutual))
    print (str(four_mutual))
    print (str(five_mutual))
    print (str(six_mutual))
    print (str(seven_mutual))
    print (str(eight_mutual))
    print (str(nine_mutual))
    print (str(ten_mutual))
    print (str(eleven_mutual))
    #print (str(twelve_mutual))
    writer = csv.writer(open("new_file.csv", "w", newline=""))
    writer.writerow(['degree dist'] + degree_dist)
    writer.writerow(['clustering by degree dist'] + clustering_dist_degree)
    writer.writerow(['hopplot dist'] + hopplot_dist)
    writer.writerow(clique_size_dist)
    writer.writerow([i[0] for i in degree_by_ec])
    writer.writerow([i[1] for i in degree_by_ec])
    writer.writerow([i[0] for i in degree_by_height])
    writer.writerow([i[1] for i in degree_by_height])
    #writer.writerow(degree_by_ec)
    #writer.writerow(degree_by_height)

def visualize(network):
    print ("VISUALIZE:")
    #plt.figure(figsize=(10, 7.5), facecolor=(0, 0, 0))
    plt.figure(figsize=(10, 8))

    pos = nx.kamada_kawai_layout(network)
    #pos = nx.spring_layout(network)
    colors = {}
    halls = nx.get_node_attributes(network,'Hall')
    halls_colors = {'1501':'#ffad30', '1502':'#c369ff', '1503':'#2ff59c', '1504':'#b5b5b5', '1505':'#ff4063', '1506':'#4dbeff', '1507':'#ff69b6'}
    genders = nx.get_node_attributes(network, 'Gender')
    genders_colors = {'Male':'#2081C3', 'Female':'#ff6969'}
    for node in network.nodes:
        #colors[node] = nx.degree(network, node)
        #colors[node] = nx.closeness_centrality(network, node)
        #colors[node] = (halls_colors[halls[node]])
        colors[node] = genders_colors[genders[node]]
        
    # sunset colormap - color based on ^^^
    #nx.draw_networkx_nodes(network, pos=pos, edgecolors=(0.1, 0.1, 0.1), node_color=list(colors.values()), node_size=150, cmap=plt.get_cmap("GnBu_r"))
    #nx.draw_networkx_nodes(network, pos=pos, edgecolors=(0.1, 0.1, 0.1), node_color=list(colors.values()), node_size=150, label=halls)
    #nx.draw_networkx_nodes(network, pos=pos, edgecolors=(0.1, 0.1, 0.1), node_color=list(colors.values()), node_size=150, label=genders)
    # normal blue undirected
    nx.draw_networkx_nodes(network, pos=pos, edgecolors=(0.1, 0.1, 0.1), node_color="#34a4eb", node_size=150)
    #nx.draw_networkx_nodes(network, pos=pos, edgecolors=(0.9, 0.9, 0.9), node_color=(253/255, 142 / 255, 171/255), node_size=150)


    # Undirected edges
    #nx.draw_networkx_edges(network, pos=pos, width=1.0, edge_color=(1, 1, 1), alpha=0.6)
    nx.draw_networkx_edges(network, pos=pos, width=1.0, edge_color=(0.1, 0.1, 0.1), alpha=0.4)
    # Directed edges
    #nx.draw_networkx_edges(network, pos=pos, width=1.0, alpha=0.3, arrows=True)
    plt.axis("off")
    #nx.draw(network, pos=nx.kamada_kawai_layout(network))
    #plt.legend(frameon=False)
    #plt.draw()
    plt.show()

def visualize2(network, undirected):
    print ("VISUALIZE:")
    plt.figure(figsize=(10, 8))
    pos = nx.kamada_kawai_layout(undirected)
    #pos = nx.spring_layout(network)
    #pos = nx.kamada_kawai_layout(network)
    colors = {}
    for node in network.nodes:
        #colors[node] = nx.degree(network, node)
        #colors[node] = nx.closeness_centrality(network, node)
        colors[node] = nx.clustering(network, node)
        
    # sunset colormap - color based on ^^^
    #nx.draw_networkx_nodes(network, pos=pos, edgecolors=(0.1, 0.1, 0.1), node_color=list(colors.values()), node_size=150, cmap=plt.get_cmap("plasma"))
    # normal blue undirected
    nx.draw_networkx_nodes(network, pos=pos, edgecolors=(0.1, 0.1, 0.1), node_color="#b5b5b5", node_size=150)

    edge_colors = ["#5ecfff" if network.has_edge(*(edge[1], edge[0])) else "#ff564a" for edge in network.edges()]
    # Undirected edges
    ###nx.draw_networkx_edges(network, pos=pos, width=1.0, alpha=0.3)
    # Directed edges
    nx.draw_networkx_edges(network, pos=pos, width=1.0, edge_color=edge_colors, alpha=0.7, arrows=False)
    plt.axis("off")
    #nx.draw(network, pos=nx.kamada_kawai_layout(network))
    plt.draw()
    plt.show()

def halls_heatmap():
    halls_data = pd.read_csv('halls_heatmap2.csv', index_col='Halls')
    print (halls_data)
    labels = halls_data.round(2)
    #print (halls_data.multiply(100))

    fig, ax = plt.subplots(figsize=(8, 6))

    #ax.xlim((0.01, 1))
    sns.heatmap(data=halls_data, vmin=0.01, vmax=1, annot=labels, cbar=True, cbar_kws={'ticks': [1, 0.1, 0.01]}, norm=colors.LogNorm())
    plt.ylabel('')
    ax.xaxis.tick_top()
    plt.yticks(va='center')
    #plt.colorbar()
    #plt.legend()
    plt.show()

def dist_data(net, node_attributes):
    writer = csv.writer(open('dists.csv', "w", newline=""))
    header = ['Node', 'Gender', 'Degree', 'Clustering', 'Betweenness Centrality']
    writer.writerow(header)

    betweenness_centrality = nx.betweenness_centrality(net)

    for node in net.nodes:
        d = nx.degree(net, node)
        c = nx.clustering(net, node)
        g = node_attributes[node]['Gender']
        cen = betweenness_centrality[node]
        row = [node, g, d, c, cen]
        writer.writerow(row)

def path_dist_data(net):
    writer = csv.writer(open('path_dist.csv', "w", newline=""))
    header = ['Path length']
    writer.writerow(header)
    nodes = list(net.nodes)
    for node1 in range(len(nodes)):
        for node2 in range(node1 + 1, len(nodes)):
            path_length = nx.algorithms.shortest_paths.generic.shortest_path_length(net, nodes[node1], nodes[node2])
            writer.writerow([path_length])


def halls_in_out_barplot():
    halls_data = pd.read_csv('halls_in_out.csv')
    print (halls_data)
    #plt.figure(figsize=(8,5))
    #fig, ax = plt.subplots(figsize=(8, 6))
    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'Bitstream Vera Sans'
    g = sns.catplot(x='Hall', y='Percent2', hue='Type', data=halls_data, kind='bar', palette=sns.color_palette(['#2081C3', '#5BE4FF']), linewidth=0, legend=False, size=6, aspect=1.5)
    plt.ylabel('Average Degree Per Person')
    #sns.barplot(x=halls_data.index, y=halls_data['Out percent'])
    
    plt.legend(frameon=False, bbox_to_anchor=(0.15, 0.975))
    plt.show()

def degree_dist():
    dist_data = pd.read_csv('dists.csv')
    #print (dist_data.loc[dist_data['Gender'] == 'Male', ['Degree']])
    #dist_data.loc[dist_data['Gender'] == 'Female', ['Degree']] = dist_data.loc[dist_data['Gender'] == 'Female', ['Degree']].multiply(10)
    
    plt.figure(figsize=(9, 6))
    min_x = 1
    max_x = 20
    plt.xticks([i for i in range(min_x, max_x)])
    plt.xlim((min_x, max_x))
    kde = True
    norm_hist = True
    bins = (max_x - min_x)

    
    sns.distplot(a=dist_data['Degree'], kde=kde, norm_hist=norm_hist, bins=bins, color='#b082e0', hist_kws={'range':(min_x, max_x)})
    #sns.distplot(a=dist_data.loc[dist_data['Gender'] == 'Male', ['Degree']], kde=kde, bins=bins, norm_hist=norm_hist, hist_kws={'range':(min_x, max_x)}, label='Male')
    #sns.distplot(a=dist_data.loc[dist_data['Gender'] == 'Female', ['Degree']], kde=kde, bins=bins, color='#ff6969', norm_hist=norm_hist, hist_kws={'range':(min_x, max_x)}, label='Female')
    #plt.get_lines()[0].remove()
    plt.xlabel('Degree')
    plt.ylabel('P(x)')
    
    plt.legend(frameon=False)
    plt.show()

def clustering_dist():
    dist_data = pd.read_csv('dists.csv')
    #print (dist_data.loc[dist_data['Gender'] == 'Male', ['Degree']])
    #dist_data.loc[dist_data['Gender'] == 'Female', ['Degree']] = dist_data.loc[dist_data['Gender'] == 'Female', ['Degree']].multiply(10)
    
    plt.figure(figsize=(9, 6))
    min_x = 0
    max_x = 1
    plt.xticks([i * 0.1 for i in range(11)])
    plt.xlim((min_x, max_x))
    kde = True
    norm_hist = True
    bins = 20

    #sns.distplot(a=dist_data['Clustering'], kde=kde, norm_hist=norm_hist, bins=bins, color='#b082e0', hist_kws={'range':(min_x, max_x)})
    sns.distplot(a=dist_data.loc[dist_data['Gender'] == 'Male', ['Clustering']], kde=kde, bins=bins, norm_hist=norm_hist, hist_kws={'range':(min_x, max_x)}, label='Male')
    sns.distplot(a=dist_data.loc[dist_data['Gender'] == 'Female', ['Clustering']], kde=kde, bins=bins, color='#ff6969', norm_hist=norm_hist, hist_kws={'range':(min_x, max_x)}, label='Female')
    #plt.get_lines()[0].remove()
    plt.xlabel('Clustering')
    plt.ylabel('P(x)')
    
    plt.legend(frameon=False)
    plt.show()

def path_dist():
    dist_data = pd.read_csv('path_dist.csv')
    #plt.figure(figsize=(8, 5))
    #sns.distplot(a=dist_data['Path length'], kde=False)
    #plt.show()

    plt.figure(figsize=(9, 6))
    min_x = 1
    max_x = 9
    plt.xticks([i for i in range(min_x, max_x)])
    plt.xlim((min_x, max_x))
    kde = False
    norm_hist = True
    bins = (max_x - min_x)

    sns.distplot(a=dist_data['Path length'], kde=kde, norm_hist=norm_hist, bins=bins, color='#b082e0', hist_kws={'range':(min_x, max_x)})
    plt.xlabel('Shortest Path Length')
    plt.ylabel('P(x)')
    
    plt.legend(frameon=False)
    plt.show()

def betweenness_dist():
    dist_data = pd.read_csv('dists.csv')
    #print (dist_data.loc[dist_data['Gender'] == 'Male', ['Degree']])
    #dist_data.loc[dist_data['Gender'] == 'Female', ['Degree']] = dist_data.loc[dist_data['Gender'] == 'Female', ['Degree']].multiply(10)
    
    plt.figure(figsize=(9, 6))
    min_x = 0
    max_x = 1
    plt.xticks([i * 0.1 for i in range(11)])
    plt.xlim((min_x, max_x))
    kde = True
    norm_hist = True
    bins = 100

    sns.distplot(a=dist_data['Betweenness Centrality'], kde=kde, norm_hist=norm_hist, bins=bins, color='#ff9c40', hist_kws={'range':(min_x, max_x)})
    #sns.distplot(a=dist_data.loc[dist_data['Gender'] == 'Male', ['Clustering']], kde=kde, bins=bins, norm_hist=norm_hist, hist_kws={'range':(min_x, max_x)}, label='Male')
    #sns.distplot(a=dist_data.loc[dist_data['Gender'] == 'Female', ['Clustering']], kde=kde, bins=bins, color='#ff6969', norm_hist=norm_hist, hist_kws={'range':(min_x, max_x)}, label='Female')
    #plt.get_lines()[0].remove()
    plt.xlabel('Betweenness Centrality')
    plt.ylabel('P(x)')
    
    plt.legend(frameon=False)
    plt.show()

def probabilities_barplot():
    prob_data = pd.read_csv('probabilities.csv')
    prob_data = prob_data.sort_values(by=['Probability'])
    #print (prob_data)
    sns.set_style('whitegrid')
    #sns.set_style("whitegrid", {'axes.grid' : False})
    plt.rcParams['font.family'] = 'Bitstream Vera Sans'
    

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.xaxis.grid()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    reader = csv.reader(open('probabilities.csv', "r"))
    next(reader)
    prev_bar = {}
    for values in reader:
        bottom = 0
        if (values[1] in prev_bar.keys()):
            bottom = prev_bar[values[1]]
        plt.bar(x=values[1], height=float(values[2]) - bottom, bottom=bottom, color=values[3], label=values[0], width=1)
        ax.text(values[1], (float(values[2]) - bottom) / 2 + bottom - 0.009, values[0], ha='center')
        prev_bar[values[1]] = float(values[2])
        print (values[2])

    plt.ylabel('Probability of Being Connected')
    plt.yticks([0.1 * i for i in range(11)])
    #plt.xlabel('Characteristics')
    #plt.xticks([0.1, 0.5, 0.9], ['Density', 'Mutual', 'Common'])
    
    #for row in prob_data.iterrows():
    #    plt.bar([row[1]], 5)
    #plt.legend()
    plt.show()

def friends_barhplot():
    fig, ax = plt.subplots(figsize=(8, 1.5))
    reader = csv.reader(open('percents.csv', "r"))
    next(reader)
    prev_bar = 0
    for values in reader:
        bottom = prev_bar
        plt.barh(y='a', width=float(values[1]) - bottom, left=bottom, color=values[2], label=values[0])
        ax.text((float(values[1]) - bottom) / 2 + bottom, -0.53, values[0], ha='center')
        ax.text((float(values[1]) - bottom) / 2 + bottom, -0.04, str(round((float(values[1]) - bottom) * 100, 1)) + '%', ha='center')
        prev_bar = float(values[1])
    
    plt.axis('off')
    plt.show()

def responses_pieplot():
    response_data = pd.read_csv('response_percents.csv')
    #response_data = pd.read_csv('response_percents_reordered.csv')
    #print (response_data)
    fig, ax = plt.subplots(figsize=(7, 7))
    explode = [0, 0,0,0,0,0,0.1]
    labels = response_data['Hall']
    #labels_gender = response_data['Gender']
    colors = ['#ffad30','#c369ff','#2ff59c','#b5b5b5','#ff4063','#4dbeff','#ff69b6']
    colors_gender = ['#2081C3', '#ff6969']
    plt.pie(x=response_data['Percent'], explode=explode, labels=labels, colors=colors, labeldistance=1.05, autopct='%.1f%%', pctdistance=0.64)
    #plt.pie(x=response_data['Percent'], labels=labels_gender, colors=colors_gender, labeldistance=0.5, autopct='%.1f%%', pctdistance=0.6)


    plt.legend(frameon=False)
    plt.show()

def main():
    name_index = 0
    friends_index = 6
    adj_delimiter = ","
    raw_file = 'raw.csv'
    #names = get_unique_names(raw_file, name_index, friends_index, adj_delimiter)
    #cleaned_file = 'cleaned2.csv'
    #clean(raw_file, cleaned_file, name_index, friends_index, adj_delimiter)
    anon_file = 'anon.csv'
    #anonymize(cleaned_file, anon_file, name_index, friends_index, adj_delimiter, start_one=True)
    
    edges_file = 'edges.csv'
    #get_edges(anon_file, edges_file, name_index, friends_index, adj_delimiter)
    #undirected_edges_file = 'undirected_edges.csv'
    #undirected_edges(edges_file, undirected_edges_file)
    
    net = file_to_network(edges_file)
    #net2 = file_to_network(edges_file, directed=True)

    node_attributes = get_node_attributes(anon_file, name_index, friends_index)
    #print (node_attributes)

    nx.set_node_attributes(net, node_attributes)
    #nx.set_node_attributes(net2, node_attributes)

    #halls_heatmap()
    #halls_in_out_barplot()
    #degree_dist()
    clustering_dist()
    #path_dist()
    #betweenness_dist()
    #probabilities_barplot()
    #friends_barhplot()
    #responses_pieplot()

    #plt.figure(figsize=(8, 5))

    #plt.xlim((0, 19))
    #sns.distplot(a=degree_dist_gender['Male'], label="Male", kde=False, bins=19)
    #sns.distplot(a=degree_dist_gender['Female'], label="Female", kde=False, bins=19)
    
    #sns.kdeplot(data=degree_dist_gender['Male'], label="Male", shade=True)
    #sns.kdeplot(data=degree_dist_gender['Female'], label="Female", shade=True)

    #sns.kdeplot(data=clustering_dist_gender['Male'], label="Male", shade=True)
    #ns.kdeplot(data=clustering_dist_gender['Female'], label="Female", shade=True)

    #sns.kdeplot(data=degree_dist_gender['Male'], label="Male", shade=True)
    
    #plt.legend()
    #plt.show()


    #print (len(net.edges))
    #print (len(net2.edges))
    #output_characteristics(net)
    #nx.draw_networkx(net, pos=nx.spring_layout(net))
    #plt.show()
    #visualize(net)
    #visualize(net2)
    #visualize2(net2, net)
    #print ('density', nx.density(net))
    #print ('reciprocity', nx.reciprocity(net2))


main()