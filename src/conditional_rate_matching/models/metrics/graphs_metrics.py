from pprint import pprint
from pathlib import Path
import concurrent.futures
import os
import sys
import subprocess as sp
from datetime import datetime

from scipy.linalg import eigvalsh
import networkx as nx
import numpy as np
import copy

from conditional_rate_matching.models.metrics.mmd2 import process_tensor, compute_mmd, gaussian, gaussian_emd

PRINT_TIME = True

"""
g++ -O2 -std=c++11 -o orca_berlin orca_berlin.cpp -static-libstdc++ -static-libgcc
"""
from conditional_rate_matching import project_path
project_path = Path(project_path)
ORCA_DIR_BERLIN = project_path / "src" / "conditional_rate_matching" / "models" / "metrics" / "orca_berlin"
ORCA_DIR_NJ = project_path / "src" / "conditional_rate_matching" / "models" / "metrics" / "orca_new_jersey"

def read_orbit_counts(file_path):
    """
    Reads a file where each line corresponds to a node in a graph.
    Each line contains 15 or 73 space-separated orbit counts.

    :param file_path: Path to the file to be read.
    :return: A list of lists, where each sublist contains the orbit counts for a node.
    """
    orbit_counts = []
    with open(file_path, 'r') as file:
        for line in file:
            counts = line.strip().split(' ')

            # Validate the number of orbit counts
            if len(counts) not in [15, 73]:
                raise ValueError(f"Invalid number of orbit counts on line: {line}")

            # Convert counts to integers
            counts = list(map(int, counts))
            orbit_counts.append(counts)

    return np.asarray(orbit_counts)


def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def add_tensor(x, y):
    x, y = process_tensor(x, y)
    return x + y


def degree_stats(graph_ref_list, graph_pred_list, windows=False, is_parallel=True):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)

    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)
    print(len(sample_ref), len(sample_pred))
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


###############################################################################

def spectral_worker(G):
    # eigs = nx.laplacian_spectrum(G)
    eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    # from scipy import stats
    # kernel = stats.gaussian_kde(eigs)
    # positions = np.arange(0.0, 2.0, 0.1)
    # spectral_density = kernel(positions)

    # import pdb; pdb.set_trace()
    return spectral_pmf


def spectral_stats(graph_ref_list, graph_pred_list, is_parallel=True):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_ref_list):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
                sample_pred.append(spectral_density)

        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #   for spectral_density in executor.map(spectral_worker, graph_ref_list):
        #     sample_ref.append(spectral_density)
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #   for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
        #     sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i])
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i])
            sample_pred.append(spectral_temp)
    # print(len(sample_ref), len(sample_pred))

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


###############################################################################

def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist

"""
def clustering_stats(graph_ref_list, graph_pred_list, bins=100, windows=True,is_parallel=True):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
        # check non-zero elements in hist
        # total = 0
        # for i in range(len(sample_pred)):
        #    nz = np.nonzero(sample_pred[i])[0].shape[0]
        #    total += nz
        # print(total)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd,
                           sigma=1.0 / 10, distance_scaling=bins)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    return mmd_dist
"""

def clustering_stats(graph_ref_list, graph_pred_list, KERNEL=gaussian, bins=100, windows=True,is_parallel=True):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)
    try:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd,
                            sigma=1.0 / 10, distance_scaling=bins)
    except:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=KERNEL, sigma=1.0 / 10)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    return mmd_dist

# maps motif/orbit name string to its corresponding list of indices from orca_berlin output
motif_to_indices = {
    '3path': [1, 2],
    '4cycle': [8],
}

def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges

def orca(graph,windows=False):
    if windows: ORCA_DIR = ORCA_DIR_BERLIN
    else: ORCA_DIR = ORCA_DIR_NJ

    tmp_input_path = ORCA_DIR / 'tmp.txt'
    f = open(tmp_input_path, 'w')
    f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()

    if windows:
        command = 'orca.exe  4 ./tmp.txt tmp.out'
        result = sp.run(command, shell=True, cwd=ORCA_DIR, stdout=sp.PIPE, stderr=sp.PIPE)
    else:
        result = sp.check_output([os.path.join(ORCA_DIR, 'orca'), '4', tmp_input_path, os.path.join(ORCA_DIR, 'tmp.out')])

    tmp_output_file = ORCA_DIR / "tmp.out"
    node_orbit_counts = read_orbit_counts(tmp_output_file)

    try:
        os.remove(tmp_input_path)
        os.remove(tmp_output_file)
    except OSError:
        pass

    return node_orbit_counts


def orbit_stats_all(graph_ref_list, graph_pred_list,windows=False):
    total_counts_ref = []
    total_counts_pred = []

    # graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    for G in graph_ref_list:
        try:
            orbit_counts = orca(G, windows)
        except Exception as e:
            print(e)
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        try:
            orbit_counts = orca(G, windows)
        except:
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian,
                           is_hist=False, sigma=30.0)

    #print('-------------------------')
    #print(np.sum(total_counts_ref, axis=0) / len(total_counts_ref))
    #print('...')
    #print(np.sum(total_counts_pred, axis=0) / len(total_counts_pred))
    #print('-------------------------')
    return mmd_dist


def adjs_to_graphs(adjs, node_flags=None):
    graph_list = []
    for adj in adjs:
        G = nx.from_numpy_matrix(adj)
        G.remove_edges_from(G.selfloop_edges())
        G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() < 1:
            G.add_node(1)
        graph_list.append(G)
    return graph_list


def eval_acc_lobster_graph(G_list):
    G_list = [copy.deepcopy(gg) for gg in G_list]

    count = 0
    for gg in G_list:
        if is_lobster_graph(gg):
            count += 1

    return count / float(len(G_list))


def is_lobster_graph(G):
    """
    Check a given graph is a lobster graph or not

    Removing leaf nodes twice:

    lobster -> caterpillar -> path

  """
    ### Check if G is a tree
    if nx.is_tree(G):
        # import pdb; pdb.set_trace()
        ### Check if G is a path after removing leaves twice
        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        num_nodes = len(G.nodes())
        num_degree_one = [d for n, d in G.degree() if d == 1]
        num_degree_two = [d for n, d in G.degree() if d == 2]

        if sum(num_degree_one) == 2 and sum(num_degree_two) == 2 * (num_nodes - 2):
            return True
        elif sum(num_degree_one) == 0 and sum(num_degree_two) == 0:
            return True
        else:
            return False
    else:
        return False


METHOD_NAME_TO_FUNC = {
    'degree': degree_stats,
    'cluster': clustering_stats,
    'orbit': orbit_stats_all,
    #'spectral': spectral_stats
}


def eval_torch_batch(ref_batch, pred_batch, methods=None):
    graph_ref_list = adjs_to_graphs(ref_batch.detach().cpu().numpy())
    grad_pred_list = adjs_to_graphs(pred_batch.detach().cpu().numpy())
    results = eval_graph_list(graph_ref_list, grad_pred_list, methods=methods)
    return results


def eval_graph_list(graph_ref_list, grad_pred_list, methods=None, windows=False):
    if methods is None:
        methods = ['degree', 'cluster', 'orbit']
    results = {}
    for method in methods:
        try:
            results[method] = METHOD_NAME_TO_FUNC[method](graph_ref_list, grad_pred_list,windows)
        except Exception as e:
            print('>>> ', e)
            continue
        print('>>> ', results)
    return results


if __name__=="__main__":
    import subprocess
    from pprint import pprint

    graph = nx.barabasi_albert_graph(100, 3)
    graph_list_1 = [nx.barabasi_albert_graph(100,3) for i in range(20)]
    graph_list_2 = [nx.barabasi_albert_graph(100,3) for i in range(20)]

    node_orbit_counts = orca(graph_list_1[0])
    results_ = eval_graph_list(graph_list_1, graph_list_2,methods=["cluster"],windows=True)
    print(results_)

