import csv
import networkx as nx
import difflib

# gets all of the names that have a response in the dataset
def get_responded(file_name, node_index):
    reader = csv.reader(open(file_name, "r", encoding="cp1252"))
    next(reader)
    nodes = set()
    for values in reader:
        node = values[node_index]
        nodes.add(node)
    return nodes

# function I found online that compares similarity of strings
def is_similar(first, second, ratio):
    return difflib.SequenceMatcher(None, first, second).ratio() > ratio   

# compares name against all names and finds the most similar
def get_closest_name(name, all_names):
    closest_name = ""
    if (name.isspace() or not name):
        return ""
    name_thresh = 0.7
    surname_thresh = 0.4
    fname = name.split()[0]
    lname = name.split()[1]
    for other_name in all_names:
        other_fname = other_name.split()[0]
        other_lname = other_name.split()[1]
        if (is_similar(name, other_name, name_thresh) and is_similar(fname, other_fname, surname_thresh) and is_similar(lname, other_lname, surname_thresh)):
            return other_name
    return closest_name

def get_unique_names(file_name, node_index, adj_index, adj_delimiter):
    count = 0
    names = set()
    reader = csv.reader(open(file_name, "r", encoding="cp1252"))
    next(reader)
    for values in reader:
        node = values[node_index]
        adj_nodes = [node] + values[adj_index].split(adj_delimiter)
        for adj_node in adj_nodes:
            adj_node = adj_node.strip(' ')
            closest_name = get_closest_name(adj_node, names)
            if (not closest_name and adj_node):
                names.add(adj_node)
    #print (len(names))
    #print (names)
    return names

# returns true if the row includes one of the two names that are not connected to the total network
def has_only_connected_nodes(row):
    return ("Ian Fowler" not in row and "Tyler Ptak" not in row)

# fixes misspellings, removes nodes not connected to total network (hard coded), removes nodes that did not have their own response in the dataset
def clean(file_name, new_file, node_index, adj_index, adj_delimiter):
    writer = csv.writer(open(new_file, "w", newline=""))
    reader = csv.reader(open(file_name, "r", encoding="cp1252"))
    nodes = get_responded(file_name, node_index)
    #nodes = get_unique_names(file_name, node_index, adj_index, adj_delimiter)
    writer.writerow(next(reader))
    for values in reader:
        #node = values[node_index]
        #node = get_closest_name(node, nodes)

        adj_nodes = values[adj_index].split(adj_delimiter)
        cleaned_adj_nodes = []
        for adj_node in adj_nodes:
            adj_node = get_closest_name(adj_node, nodes)
            if (adj_node):
                cleaned_adj_nodes.append(adj_node)
        cleaned_row = values
        #cleaned_row = [node] + values[1:]
        cleaned_row[adj_index] = ",".join(cleaned_adj_nodes)
        if (has_only_connected_nodes(cleaned_row)):
            writer.writerow(cleaned_row)

# creates the dictionary to store the integer values representing each name
def get_nodes_anon(nodes, start_one=False):
    nodes_anon = {}
    for i, node in enumerate(nodes):
        nodes_anon[node] = i + int(start_one)
    return nodes_anon

# turns all the names in the cleaned raw dataset into numbers - set start one to true if you want the first node to be 1 (handling for weird R conventions)
def anonymize(file_name, new_file, node_index, adj_index, adj_delimiter, start_one=False):
    writer = csv.writer(open(new_file, "w", newline=""))
    reader = csv.reader(open(file_name, "r", encoding="cp1252"))
    nodes = get_responded(file_name, node_index)
    #nodes = get_unique_names(file_name, node_index, adj_index, adj_delimiter)
    nodes_anon = get_nodes_anon(nodes, start_one)
    writer.writerow(next(reader))
    for values in reader:
        node = values[node_index]
        anon_node = nodes_anon[node]
        adj_nodes = values[adj_index].split(adj_delimiter)
        anon_adj_nodes = [str(nodes_anon[adj_node]) for adj_node in adj_nodes if (adj_node)]
        anon_row = values
        anon_row[node_index] = anon_node
        anon_row[adj_index] = ",".join(anon_adj_nodes)
        writer.writerow(anon_row)

# returns a dictionary with nodes as keys, and a dictionary of attributes as values
def get_node_attributes(file_name, node_index, adj_index):
    reader = csv.reader(open(file_name, "r", encoding="cp1252"))
    node_attributes = {}
    headers = next(reader)
    for values in reader:
        node = values[node_index]
        attributes = {}
        for index, value in enumerate(values):
            if (index != node_index and index != adj_index):
                key = headers[index]
                attributes[key] = value
        node_attributes[node] = attributes
    return node_attributes 

# makes a new file with just the edges (directed)
def get_edges(file_name, new_file, node_index, adj_index, adj_delimiter):
    writer = csv.writer(open(new_file, "w", newline=""))
    reader = csv.reader(open(file_name, "r"))
    next(reader)
    for values in reader:
        node = values[node_index]
        adj_nodes = values[adj_index].split(adj_delimiter)
        for adj_node in adj_nodes:
            if (adj_node):
                edge = [node, adj_node]
                writer.writerow(edge)

# from edges file, makes a new file removing any extra edges between nodes, getting only the edges present in an undirected network
def undirected_edges(file_name, new_file):
    writer = csv.writer(open(new_file, "w", newline=""))
    reader = csv.reader(open(file_name, "r"))
    edges = []
    for values in reader:
        edge = values
        if (edge not in edges):
            writer.writerow(edge)
            edges.append(edge)
            edges.append([edge[1], edge[0]])

# creates network from edge file
def file_to_network(file_name, directed=False):
    if (directed):
        network = nx.DiGraph()
    else:
        network = nx.Graph()
    data = csv.reader(open(file_name, "r"))
    for nodes in data:
        edge = tuple(nodes)
        network.add_edge(*edge)
    return network