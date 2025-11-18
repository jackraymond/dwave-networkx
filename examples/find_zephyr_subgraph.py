# Copyright 2025 D-Wave
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
Embedders for zephyr on zephyr (work in progress, unoptimized)
"""
import itertools
import os

import tqdm

from dwave_networkx import zephyr_sublattice_mappings, zephyr_graph, zephyr_coordinates
from dwave.embedding import is_valid_embedding
import networkx as nx
import minorminer
import pickle


def _decoordinate_best_embedding(
    best_embedding, coordinated, m_target, t_target, m_source, t_source
):
    """Process embedding to standard format"""
    if coordinated:
        return best_embedding
    else:
        c2ls = zephyr_coordinates(m_source, t_source).zephyr_to_linear
        c2lt = zephyr_coordinates(m_target, t_target).zephyr_to_linear
        return {c2ls(k): (c2lt(v[0]),) for k, v in best_embedding.items()}


def zephyr_in_zephyr_utility_function(
    m, t_target=4, t_source=3, node_list=None, edge_list=None
):
    """Define an optimization problem associated to best shores.

    At low edge and node defect rates, a maximum edge-yield zephyr[m,t'] graph within a zephyr[m,t]
    can be found by consideration of only a restricted set of mappings. This can be formalized
    as an optimization problem on categorical variables.

    The zephyr[m,t] target graph can be decomposed into a quotient graph Gq indexed by (w,u,j,z) with (up to)
    t nodes at each site.  The zephyr[m,t'] graph can similarly be decomposed with t' nodes at each
    site. For every (u,w,j,z) selection of a subset t' of nodes from t determines a candidate embedding
    associated to a specific edge yield. The space of subset mappings can be searched by standard methods
    for a maximum edge yield embedding.

    If a fully-yielded subgraph can be found the above mapping is sufficient whenever m>1, i.e.
    if the subgraph isomorphism problem is solvable, the above mapping is sufficient.
    In the case of incomplete edge yield, the mapping is not guaranteed to maximize edge yield,
    although in practice it will do so with high probability if the edge (and node) defect rate is small.

    Each node n (coordinates u,w,j,z) in the quotient is associated an integer indexed categorical variable
    defining one of the {t choose t'} possible mappings of nodes.
    For any edge in the quotient graph (n1,n2), and assignment of categorical variables v_n1, v_n2
    there is an associated edge yield Eyield_{n1,n2}(v_n1, v_n2). The objective is to find an
    assignment such that the total number of edges is maximized:

    Num_edges(v) = sum_{n1 < n2} Eyield_{n1,n2}(v_n1, v_n2)

    Eyield is returned as a sparse dictionary by the function. A best assignment to v in
    |{t choose t'}|^N can be found by heuristic and/or exact optimization methods.
    """
    # For each value of u,w,j,z find the optimal subset of k
    if t_target <= t_source:
        return "Unsuitable format for optimization problem"

    kassignment = {
        idx: v
        for idx, v in enumerate(itertools.combinations(range(t_target), t_source))
    }

    quotient_graph = zephyr_graph(m, t=1, coordinates=True)  # Defect free source graph
    utility = {
        e: [[0] * len(kassignment)] * len(kassignment) for e in quotient_graph.edges()
    }
    target_graph = zephyr_graph(
        m, t=t_target, coordinates=True, node_list=node_list, edge_list=edge_list
    )
    for n1, n2 in quotient_graph.edges():
        for idx1, ks1 in kassignment.items():
            nodes1 = {n1[:2] + (k,) + n1[3:] for k in ks1}
            for idx2, ks2 in kassignment.items():
                nodes = nodes1 | {n2[:2] + (k,) + n2[3:] for k in ks2}
                utility[n1, n2][idx1][idx2] = target_graph.subgraph(
                    nodes
                ).number_of_edges()
    return utility


def embedding_candidate_P1(n, u, w, ks, sublattice_embedding):
    """Permute rail (u,w), k=(0,1,..,t_source) -> ks"""
    e = sublattice_embedding[n]
    if n[0] != u or n[1] != w:
        return e
    else:
        return e[:2] + (ks[n[2]],) + e[3:]  # k mapped.


def embedding_candidate_P2(n, u, w, j, z, ks, sublattice_embedding):
    """Permute rail (u,w), k=(0,1,..,t_source) -> ks"""
    e = sublattice_embedding[n]
    if n[0] != u or n[1] != w or n[3] != j or n[4] != z:
        return e
    else:
        return e[:2] + (ks[n[2]],) + e[3:]


def embed_tprime_in_t(
    m,
    t,
    t_s,
    sublattice_embedding,
    target,
    source=None,
    verbose=True,
    best_num_edges0=-1,
    num_coords=2,
):
    if source is None:
        source = zephyr_graph(m, t_s, coordinates=True)
    max_num_edges = source.number_of_edges()
    subgraph_nodes = set(sublattice_embedding.values())
    best_num_edges = target.subgraph(subgraph_nodes).number_of_edges()
    if best_num_edges == max_num_edges:
        print("Subgraph isomorphism problem solved by displacement")
        return sublattice_embedding, best_num_edges
    if num_coords == 2:
        all_ks = list(itertools.combinations(range(t), t_s))[
            1:
        ]  # First permutation is accounted for by initial condition.

        coords_iterator = itertools.product(range(2), range(2 * m + 1))
        embedding_candidate_P = embedding_candidate_P1
    else:
        all_ks = list(itertools.combinations(range(t), t_s))
        coords_iterator = itertools.product(
            range(2), range(2 * m + 1), range(2), range(m)
        )
        embedding_candidate_P = embedding_candidate_P2

    for coords in coords_iterator:
        bestks = None  # Best rails
        for ks in all_ks:
            subgraph_nodes = {
                embedding_candidate_P(n, *coords, ks, sublattice_embedding)
                for n in source
            }  # TIDY UP: More efficient O(m) to update than recalculate.
            num_edges = target.subgraph(subgraph_nodes).number_of_edges()

            if num_edges > best_num_edges:
                # Record best so far
                bestks = ks
                best_num_edges = num_edges
                if verbose and best_num_edges > best_num_edges0:
                    best_num_edges0 = best_num_edges
                    print(best_num_edges, *coords, ks)
                if num_edges == max_num_edges:
                    break

        if bestks is not None:
            # O(m^2) complexity can be reduced to O(m) if necessary:
            sublattice_embedding = {
                n: embedding_candidate_P(n, *coords, bestks, sublattice_embedding)
                for n in source
            }
        if num_edges == max_num_edges:
            break
    return sublattice_embedding, best_num_edges


def zephyr_in_zephyr_embedding(
    m_target,
    m_source=None,
    t_target=4,
    t_source=4,
    node_list_target=None,
    edge_list_target=None,
    allow_unmapped_edges=True,
    coordinated=False,
):
    """Find a Zephyr[m',t'] 1:1 embedding in a Zephyr[m,t] graph

    This routine either solves the graph isomorphism problem,
    or executes a heuristic to find a high-edge yield 1:1 embedding
    (maximizing edge yield, per zephyr_in_zephyr_utility_function)

    Args:
        m_target: size (num rows) of the target lattice
        m_source: size (num rows) of the source lattice
        t_target: tile parameter of the target lattice
        t_source: tile parameter of the source lattice
        node_list_target: yielded (programmable) nodes on the target
        edge_list_target: yielded (programmable) edges on the target
        allow_unmapped_edges: return incomplete (maximum edge yield)
            lattices.
        coordinates: If linear coordinates are the desired return format set
            to False. The edgelist and nodelist will also assumed to be specified
            with linear coordinates if the value is False.
    """
    best_embedding = {}
    best_num_edges0 = -1
    ideal_num_edges = (
        2 * t_source * ((8 * t_source + 8) * m_source**2 - 2 * m_source - 3)
    )
    print(m_source, t_source, ": ideal number of edges", ideal_num_edges)
    if t_target == t_source:
        target = zephyr_graph(
            m=m_target,
            t=t_target,
            edge_list=edge_list_target,
            node_list=node_list_target,
        )
        source = zephyr_graph(
            m=m_source, t=t_source
        )  # Assumed complete, could generalize

        num_sublattice_candidates = sum(
            1 for _ in zephyr_sublattice_mappings(source=source, target=target)
        )
        sublattice_generator = zephyr_sublattice_mappings(source=source, target=target)
        for _ in tqdm.tqdm(range(num_sublattice_candidates)):
            sublattice_embedding = next(sublattice_generator)

            num_edges = target.subgraph(
                {sublattice_embedding(n) for n in source}
            ).number_of_edges()
            if num_edges == source.number_of_edges():  # Special case of zephyr subgraph
                # Return optimal embedding:
                print("Subgraph isomorphism problem solved")
                best_embedding = {n: (sublattice_embedding(n),) for n in source}
                break
            elif allow_unmapped_edges and num_edges > best_num_edges0:
                best_embedding = {
                    n: (sublattice_embedding(n),)
                    for n in source
                    if target.has_node(sublattice_embedding(n))
                }
                best_num_edges0 = num_edges
                print(num_edges)
        return best_embedding
    else:
        if coordinated is False:
            l2c = zephyr_coordinates(m_target, t_target).linear_to_zephyr
            edge_list_target = [(l2c(n1), l2c(n2)) for n1, n2 in edge_list_target]
            node_list_target = [l2c(n) for n in node_list_target]

        target = zephyr_graph(
            m=m_target,
            t=t_target,
            edge_list=edge_list_target,
            node_list=node_list_target,
            coordinates=True,
        )
        source = zephyr_graph(
            m=m_source, t=t_source, coordinates=True
        )  # Assumed complete, could generalize
        if t_target < t_source:
            raise ValueError("t_target should be greater than or equal to t_source")
        all_ks = list(itertools.combinations(range(t_target), t_source))[
            1:
        ]  # First permutation is accounted for by initial condition.

        # This heuristic could be sped up by ranking the sublattices by target yield at t=t_target,
        # and then executing over most promising sublattices first.
        # Feasibility filters could also be used. There is lots of room for efficiencies.
        print(
            "Progress bar counts sublattices considered, any improvement in edge count "
            "is printed to the screen with the coordinates of the set reassignment."
        )
        num_sublattice_candidates = sum(
            1 for _ in zephyr_sublattice_mappings(source=source, target=target)
        )
        sublattice_generator = zephyr_sublattice_mappings(source=source, target=target)
        for _ in tqdm.tqdm(range(num_sublattice_candidates)):
            def_sublattice_embedding = next(sublattice_generator)
            # For each sublattice mapping, a heuristic is used to maximize the number of edges
            # i.e. the problem zephyr_in_zephyr_utility_function is solved greedily with graph
            # insight
            sublattice_embedding = {n: def_sublattice_embedding(n) for n in source}

            # Greedily select permutations, deterministic and heuristics generalizations are possible:
            # In principle we should iterate more than once to guarantee arrival at a local minima

            sublattice_embedding, best_num_edges = embed_tprime_in_t(
                m_source,
                t_target,
                t_source,
                sublattice_embedding,
                target=target,
                source=source,
                verbose=True,
                best_num_edges0=best_num_edges0,
            )
            if best_num_edges == source.number_of_edges():
                return _decoordinate_best_embedding(
                    sublattice_embedding,
                    coordinated,
                    m_target,
                    t_target,
                    m_source,
                    t_source,
                )
            # If a fully yielded graph is not found by the above heuristic, it cannot be found on this sublattice
            # we can still seek a higher edge yield result by greedy or heuristic search:
            if allow_unmapped_edges:
                sublattice_embedding, best_num_edges = embed_tprime_in_t(
                    m_source,
                    t_target,
                    t_source,
                    sublattice_embedding,
                    target=target,
                    source=source,
                    verbose=True,
                    best_num_edges0=best_num_edges0,
                    num_coords=4,
                )
                if best_num_edges == source.number_of_edges():
                    return _decoordinate_best_embedding(
                        sublattice_embedding,
                        coordinated,
                        m_target,
                        t_target,
                        m_source,
                        t_source,
                    )

            if best_num_edges0 < best_num_edges:
                best_embedding = {
                    n: (sublattice_embedding[n],)
                    for n in source
                    if target.has_node(sublattice_embedding[n])
                }
                best_num_edges0 = best_num_edges
        return _decoordinate_best_embedding(
            best_embedding, coordinated, m_target, t_target, m_source, t_source
        )


def main_example(
    solvers=(
        "Advantage2_system1.7_m484",
    ),  # ("Advantage2_system2_x_internal",), # ("Advantage2_system1.7",), #("Advantage2_system2.1",), # , "Advantage2_system3.1"), ,
    m_source=4,
    t_source=2,
    submit_to_verify=False,
):

    import matplotlib.pyplot as plt

    from dwave.system.testing import MockDWaveSampler
    from dwave.system import DWaveSampler, FixedEmbeddingComposite
    from dwave_networkx import draw_parallel_embeddings
    from dwave.embedding import verify_embedding  # For debugging

    for solver in solvers:
        fn = f"{solver}.pkl"
        if not os.path.isfile(fn):
            if solver == "Advantage2_system2_x_internal":
                qpu = DWaveSampler(
                    solver=solver, profile="benchmarking"
                )  # Could specify zephyr more generally
                with open(fn, "wb") as f:
                    pickle.dump(qpu.properties, f)
            else:
                if solver == "Advantage2_system1.7_m484":
                    badqubits = {484}
                    qpu = DWaveSampler(
                        solver="Advantage2_system1.7"
                    )  # Could specify zephyr more generally
                    nodeset = set(qpu.properties["qubits"]).difference(badqubits)
                    properties = qpu.properties
                    properties["qubits"] = sorted(nodeset)
                    properties["couplers"] = [
                        c
                        for c in properties["couplers"]
                        if c[0] not in badqubits and c[1] not in badqubits
                    ]
                    with open(fn, "wb") as f:
                        pickle.dump(properties, f)

                    qpu = MockDWaveSampler(
                        properties=properties,
                        nodelist=properties["qubits"],
                        edgelist=properties["couplers"],
                    )
                else:
                    qpu = DWaveSampler(
                        solver=solver
                    )  # Could specify zephyr more generally
                    with open(fn, "wb") as f:
                        pickle.dump(qpu.properties, f)
        else:
            with open(fn, "rb") as f:
                properties = pickle.load(f)

            qpu = MockDWaveSampler(
                properties=properties,
                nodelist=properties["qubits"],
                edgelist=properties["couplers"],
            )
        print(solver, qpu.properties["topology"])
        m_target = qpu.properties["topology"]["shape"][0]
        t_target = qpu.properties["topology"]["shape"][1]
        ideal_num_edges = (
            2 * t_target * ((8 * t_target + 8) * m_target**2 - 2 * m_target - 3)
        )
        ideal_num_nodes = 4 * t_target * m_target * (2 * m_target + 1)
        edge_list_target = qpu.edgelist
        node_list_target = qpu.nodelist
        print(
            "Target node and edge yields",
            len(qpu.nodelist) / ideal_num_nodes,
            len(qpu.edgelist) / ideal_num_edges,
        )
        m_source, t_source = 12, 2  # This can be found in both processor graphs
        fn = f"emb{solver}_m{m_source}_t{t_source}.pkl"
        if not os.path.isfile(fn):
            emb = zephyr_in_zephyr_embedding(
                m_target=m_target,
                m_source=m_source,
                t_target=t_target,
                t_source=t_source,
                node_list_target=node_list_target,
                edge_list_target=edge_list_target,
            )
            with open(fn, "wb") as f:
                pickle.dump(emb, f)
        else:
            with open(fn, "rb") as f:
                emb = pickle.load(f)

        plt.figure(solver)
        G = qpu.to_networkx_graph()
        source = zephyr_graph(m_source, t_source)

        Ginduced = G.subgraph({v[0] for v in emb.values()})
        ideal_num_nodes_s = 4 * t_source * m_source * (2 * m_source + 1)
        ideal_num_edges_s = (
            2 * t_source * ((8 * t_source + 8) * m_source**2 - 2 * m_source - 3)
        )
        edge_yield = Ginduced.number_of_edges() / ideal_num_edges_s
        if edge_yield == 1:
            verify_embedding(emb, source, G)

        print(
            "Fraction of possible nodes embedded",
            len(emb) / ideal_num_nodes_s,
            "Fraction of possible edges embedded",
            Ginduced.number_of_edges() / ideal_num_edges_s,
        )
        draw_parallel_embeddings(G=G, embeddings=[emb])
        plt.title(
            f"Nodes {len(emb)}/{ideal_num_nodes_s}, Edges {Ginduced.number_of_edges()}/{ideal_num_edges_s}: Full:{len(qpu.nodelist) / ideal_num_nodes:.3g}, {len(qpu.edgelist) / ideal_num_edges:.3g}"
        )

        plt.savefig(f"{solver}_m{m_source}t{t_source}.png", bbox_inches="tight")
        fn = f"embM{solver}_m{m_source}_t{t_source}.pkl"
        if not os.path.isfile(fn):
            embM, success = minorminer.find_embedding(
                S=source, T=G, initial_chains=emb, verbose=1, return_overlap=True
            )
            ## Look at the shortfall!
            print(embM)
            with open(fn, "wb") as f:
                pickle.dump(embM, f)
        else:

            with open(fn, "rb") as f:
                embM = pickle.load(f)
            success = is_valid_embedding(embM, source, G)
        if True:

            used_nodes = [v for c in emb.values() for v in c]
            used_edges = [
                e for e in G.edges() if e[0] in used_nodes and e[1] in used_nodes
            ]
            Gminor = nx.from_edgelist(used_edges)
            plt.title(
                f"MM{sum(len(e) for e in embM.values())}{success}, Nodes {len(emb)}/{ideal_num_nodes_s}, Edges {Ginduced.number_of_edges()}/{ideal_num_edges_s}: Full:{len(qpu.nodelist) / ideal_num_nodes:.3g}, {len(qpu.edgelist) / ideal_num_edges:.3g}"
            )
        plt.savefig(f"{solver}_m{m_source}t{t_source}.png", bbox_inches="tight")

        # E.g. submit a ferromagnetic problem using this embedding
        if submit_to_verify:
            if edge_yield == 1:
                ss = FixedEmbeddingComposite(qpu, embedding=emb).sample_ising(
                    {n: 0 for n in source.nodes}, {e: -1 for e in source.edges}
                )
                ss.resolve()  # Executed successfully

                print(
                    f"Zephyr[m={m_source},t={t_source}] over {solver} successfully embedded and sampled."
                )
            else:
                print(
                    f"Zephyr[m={m_source},t={t_source}] over {solver} has defects. "
                    "Some heuristic required to deal with defects."
                )
    plt.show()


if __name__ == "__main__":
    # embed Z[m=4, t=2] (as best as possible) in the default zephyr solver:
    print(
        "Zephyr[m,t] graphs allow for many interesting regular sublattices "
        "including bicliques, cubic lattices and chimera graphs "
        "Amongst the most complicated allowing efficient embedding and with "
        "robustness to moderate defect rates are "
        "Zephyr[mp, tp] graphs with tp<=t and mp<=m. "
        "Spin glasses defined over these graphs are relatively challenging "
        "owing to a combination of size, high degree and graph complexity (such as high tree-width). "
        "This module contains a few specific heuristics for this problem. "
        "Routines are experimental and unoptimized."
    )

    # E.g. utility function
    print("Example of a utility function to be minimized (uses coordinates)")
    objective = zephyr_in_zephyr_utility_function(m=4, t_target=4, t_source=3)
    print("Solve by greedy permutation search: requires a client connection")
    main_example()
