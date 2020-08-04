
import networkx as nx

def get_avg_degree(network):
    total = 0
    for node in nx.nodes(network):
        total += nx.degree(network, node)
    n_nodes = nx.number_of_nodes(network)
    avg_degree = total / n_nodes
    return avg_degree

def get_clustering_dist_degree(network):
    clustering_dist_degree = [0] * len(nx.degree_histogram(network))
    for node in nx.nodes(network):
        degree = nx.degree(network, node)
        clustering = nx.clustering(network, node)
        value = clustering / nx.degree_histogram(network)[degree]
        clustering_dist_degree[degree] += value
    return clustering_dist_degree

def get_max_clique_size(network):
    cliques = list(nx.algorithms.clique.find_cliques(network))
    max_size = 0
    for clique in cliques:
        size = len(clique)
        max_size = max(size, max_size)
    return max_size

def get_avg_clique_size(network, threshold=3):
    cliques = list(nx.algorithms.clique.find_cliques(network))
    n_cliques = 0
    total = 0
    for clique in cliques:
        size = len(clique)
        if (size >= threshold):
            total += size
            n_cliques += 1
    avg_clique_size = total / n_cliques
    return avg_clique_size

# returns clique size dist with the index as the clique size and the value as the count - includes only unique cliques
def get_clique_size_dist(network):
    cliques = list(nx.algorithms.clique.find_cliques(network))
    #cliques = list(nx.algorithms.clique.enumerate_all_cliques(network)) --- this would get all possible cliques
    max_clique_size = get_max_clique_size(network)
    clique_size_dist = [0] * (max_clique_size + 1)
    for clique in cliques:
        size = len(clique)
        clique_size_dist[size] += 1
    return clique_size_dist

def get_hopplot_dist(network):
    diameter = nx.algorithms.distance_measures.diameter(network)
    hopplot_dist = [0] * (diameter + 1)
    all_nodes = list(nx.nodes(network))
    n_pairs = nx.number_of_nodes(network) * (nx.number_of_nodes(network) - 1)
    for node1 in range(len(all_nodes)):
        for node2 in range(node1 + 1, len(all_nodes)):
            shortest_path_length = nx.algorithms.shortest_paths.generic.shortest_path_length(network, all_nodes[node1], all_nodes[node2])
            hopplot_dist[shortest_path_length] += 1 / n_pairs
    return hopplot_dist

# mutuality dists for directed networks
def get_mutuality_degree_dist(network):
    dist = [0] * len(nx.degree_histogram(network))
    dist_count = dist.copy()
    n = 0
    for node in nx.nodes(network):
        in_degree = network.in_degree(node)
        out_degree = network.out_degree(node)
        if (out_degree > 0):
            dist[out_degree] += (in_degree / out_degree)
            n += 1
    dist = [dist[i] / max(1, dist_count[i]) for i in range(len(dist))]
    return dist

def get_mutuality_clustering_dist(network):
    dist = [0] * len(nx.degree_histogram(network))
    dist_count = dist.copy()
    n = 0
    for node in nx.nodes(network):
        in_degree = network.in_degree(node)
        out_degree = network.out_degree(node)
        clustering = nx.clustering(network, node)
        if (out_degree > 0):
            dist[out_degree] += (in_degree / out_degree)
            n += 1
    dist = [dist[i] / max(1, dist_count[i]) for i in range(len(dist))]
    return dist

# Demographic stats

# returns a dictionary with height as keys and average degree as values
def get_degree_by_height(network):
    degree_dist = {}
    height_dist = {}
    for node in network.nodes:
        height_str = network.nodes[node]["Height"]
        height_nums = [char for char in height_str if char.isdigit()]
        height = 12 * int(height_nums[0])
        inches = ""
        for i in range(1, len(height_nums)):
            inches += height_nums[i]
        if (inches):
            if (int(inches) < 12):
                height += int(inches)
            else:
                height += round(float(inches[0] + "." + inches[1]))
        if (height in degree_dist.keys()):
            degree_dist[height] += nx.degree(network, node)
        else:
            degree_dist[height] = nx.degree(network, node)
        if (height in height_dist.keys()):
            height_dist[height] += 1
        else:
            height_dist[height] = 1
    degree_dist.update((height, total_degree / height_dist[height]) for height, total_degree in degree_dist.items())
    return degree_dist

# returns a dictionary with extracurricular count (ec) as keys and average degree as values
def get_degree_by_ec(network):
    degree_dist = {}
    ec_dist = {}
    for node in network.nodes:
        ec = network.nodes[node]["Extracurricular Count"]
        if (ec):
            ec = int(ec)
            if (ec in degree_dist.keys()):
                degree_dist[ec] += nx.degree(network, node)
            else:
                degree_dist[ec] = nx.degree(network, node)
            if (ec in ec_dist.keys()):
                ec_dist[ec] += 1
            else:
                ec_dist[ec] = 1
    degree_dist.update((ec, total_degree / ec_dist[ec]) for ec, total_degree in degree_dist.items())
    return degree_dist

# returns the probability that two nodes are connected given their attribute attribute is same(true same false different)
def probability_connected(network, attribute, same):
    total_instances = 0
    total_pairs = len(network.nodes) * (len(network.nodes) - 1) / 2
    for i, node1 in enumerate(list(network.nodes)):
        node1 = node1
        node1_attribute = network.nodes[node1][attribute]
        for j in range(i + 1, len(network.nodes)):
            node2 = list(network.nodes)[j]
            node2_attribute = network.nodes[node2][attribute]
            if ((node1_attribute == node2_attribute) == same):
                total_instances += 1
    total_instances2 = 0
    for edge in network.edges:
        node1_attribute = network.nodes[edge[0]][attribute]
        node2_attribute = network.nodes[edge[1]][attribute]
        if ((node1_attribute == node2_attribute) == same):
                total_instances2 += 1
    prob_same_given_connected = (total_instances2 / (len(network.edges)))
    print ("Prob_same_given_connected: " + str(prob_same_given_connected))
    prob_connected = (len(network.edges) / total_pairs)
    print ("prob_connected: " + str(prob_connected))
    prob_same = (total_instances / total_pairs)
    probability = prob_same_given_connected * prob_connected / prob_same
    return probability

# returns the probability two nodes are connected given they have exactly mutual connections
def probability_connected_mutual(network, n):
    total_instances = 0
    total_pairs = len(network.nodes) * (len(network.nodes) - 1) / 2
    for i, node1 in enumerate(list(network.nodes)):
        node1 = node1
        for j in range(i + 1, len(network.nodes)):
            node2 = list(network.nodes)[j]
            common_neighbors = list(nx.common_neighbors(network, node1, node2))
            n_mutuals = len(common_neighbors)
            if (n_mutuals == n):
                total_instances += 1
    total_instances2 = 0
    for edge in network.edges:
        common_neighbors = list(nx.common_neighbors(network, edge[0], edge[1]))
        n_mutuals = len(common_neighbors)
        if (n_mutuals == n):
                total_instances2 += 1
    prob_same_given_connected = (total_instances2 / (len(network.edges)))
    print ("Prob_same_given_connected: " + str(prob_same_given_connected))
    prob_connected = (len(network.edges) / total_pairs)
    print ("prob_connected: " + str(prob_connected))
    prob_same = (total_instances / total_pairs)
    probability = prob_same_given_connected * prob_connected / prob_same
    return probability


def poop():
    responses = {}
    for node in node_attributes:
        attributes = node_attributes[node]
        for attribute in attributes:
            if (attribute not in responses.keys()):
                responses[attribute] = {}
            characteristic = attributes[attribute]
            if (characteristic not in responses[attribute].keys()):
                responses[attribute][characteristic] = 0
            responses[attribute][characteristic] += 1
    print (responses)
    
    counts = {}
    counts['Hall'] = {}
    for hall in responses['Hall']:
        counts['Hall'][hall] = 0
    counts['Gender'] = {'Female':0, 'Male':0}

    degree_sums = {}
    degree_sums['Hall'] = {}
    for hall in responses['Hall']:
        degree_sums['Hall'][hall] = 0
    degree_sums['Gender'] = {'Female':0, 'Male':0}

    for node in nx.nodes(net):
        d = nx.clustering(net, node)
        hall = node_attributes[node]['Hall']
        gender = node_attributes[node]['Gender']
        degree_sums['Hall'][hall] += d
        degree_sums['Gender'][gender] += d
        counts['Hall'][hall] += 1
        counts['Gender'][gender] += 1
    
    for attribute in degree_sums:
        for characteristic in degree_sums[attribute]:
            ad = degree_sums[attribute][characteristic] / counts[attribute][characteristic]
            print (characteristic, ad)
    
def poop2():
    in_out_degrees_hall = {}
    for i in range(1, 8):
        hall = '150' + str(i)
        in_out_degrees_hall[hall] = {}
        for j in range(1, 8):
            hall2 = '150' + str(j)
            in_out_degrees_hall[hall][hall2] = 0

    for i, node1 in enumerate(list(net.nodes)):
        node1 = node1
        hall1 = node_attributes[node1]['Hall']
        neighbors = net.neighbors(node1)
        for node2 in neighbors:
            hall2 = node_attributes[node2]['Hall']
            in_out_degrees_hall[hall1][hall2] += 1
    print (in_out_degrees_hall)
    writer = csv.writer(open("temp.csv", "w", newline=""))
    for hall in in_out_degrees_hall:
        row = [hall]
        for pair in sorted(list(zip(in_out_degrees_hall[hall].keys(), in_out_degrees_hall[hall].values())), key = lambda x: int(x[0])):
            row.append(pair[1])
        writer.writerow(row)