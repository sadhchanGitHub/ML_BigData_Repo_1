import networkx as nx
import numpy as np
import multiprocessing as mp

# Define a function to calculate betweenness centrality for a subset of nodes
def calculate_betweenness_centrality_chunk(G, nodes):
    # Calculate betweenness centrality only for the given subset of nodes
    return nx.betweenness_centrality_subset(G, sources=nodes, targets=list(G.nodes()), normalized=True)

# Multiprocessing function to parallelize betweenness centrality calculation
def parallel_betweenness_centrality(G, num_partitions=None):
    if num_partitions is None:
        num_partitions = mp.cpu_count()  # Use all available CPU cores

    # Split the list of nodes into chunks
    nodes = list(G.nodes())
    node_chunks = np.array_split(nodes, num_partitions)

    # Create a multiprocessing Pool
    with mp.Pool(num_partitions) as pool:
        # Apply the calculate_betweenness_centrality_chunk function to each chunk in parallel
        results = pool.starmap(calculate_betweenness_centrality_chunk, [(G, chunk) for chunk in node_chunks])

    # Combine results from each chunk into a single dictionary
    betweenness_centrality = {}
    for result in results:
        betweenness_centrality.update(result)

    return betweenness_centrality
