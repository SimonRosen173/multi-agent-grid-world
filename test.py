import networkx as nx


def main():
    # Create graph
    prev_joint_pos = [(0, 0), (0, 1), (1, 1), (1, 0), (1, 2)]
    cand_joint_pos = [(0, 0), (0, 0), (0, 1), (0, 0), (2, 2)]
    goals_set = {(0, 1)}

    n_agents = len(prev_joint_pos)
    nodes_set = set(prev_joint_pos + cand_joint_pos)

    dg = nx.DiGraph()
    dg.add_nodes_from(nodes_set)
    edges_list = list(zip(prev_joint_pos, cand_joint_pos))
    dg.add_edges_from(edges_list)

    # Set attributes
    # node attributes
    node_attributes = {node: {"is_goal": node in goals_set} for node in dg.nodes()}
    nx.set_node_attributes(dg, node_attributes)

    # edge attributes
    edge_agent_id = [i for i in range(n_agents)]
    edge_is_stationary = [(True if x[0]==x[1] else False) for x in edges_list]
    edge_attributes = {edges_list[i]: {"agent_id":i, "is_stationary": edge_is_stationary[i]}
                       for i in range(len(edges_list))}
    nx.set_edge_attributes(dg, edge_attributes)

    #
    in_edges_nodes = [(node, list(dg.in_edges(node))) for node in dg.nodes()]
    problem_nodes_edges = list(filter(lambda x: True if len(x[1])>1 else False, in_edges_nodes))
    problem_nodes = [el[0] for el in problem_nodes_edges]

    n_removed_edges = 0
    problem_agents = set()

    while len(problem_nodes) > 0:
        problem_node = problem_nodes.pop(0)
        if problem_node not in goals_set:
            problem_node_edges = list(dg.in_edges(problem_node))
            non_stationary_edges = list(filter(lambda x: False if x[0]==x[1] else True,
                                               problem_node_edges))
            for edge in non_stationary_edges:
                print(edge)
                n_removed_edges += 1
                edge_attr = dg[edge[0]][edge[1]]
                problem_agents.add(edge_attr["agent_id"])

                new_edge = (edge[0], edge[0])  # Stationary edge
                dg.add_edge(*new_edge, **edge_attr)
                dg.remove_edge(*edge)
                if len(dg.in_edges(new_edge[0])) > 1:
                    problem_nodes.append(new_edge[0])

    print("No of edges removed: ", n_removed_edges)
    print("Problem agents: ", list(problem_agents))

    # Compute next_joint_pos
    next_joint_pos = [None for _ in range(n_agents)]
    next_pos_agent = [(dg[edge[0]][edge[1]]["agent_id"], edge[1]) for edge in dg.edges()]

    for agent_id, pos in next_pos_agent:
        next_joint_pos[agent_id] = pos

    print("next_joint_pos:", next_joint_pos)


if __name__ == "__main__":
    main()
