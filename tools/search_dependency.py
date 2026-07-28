def get_node_dependencies(edges):
    dependencies = {}
    nodes = set()
    for edge in edges:
        nodes.add(edge["from_node"])
        nodes.add(edge["to_node"])
    
    for node in sorted(nodes):
        dependencies[node] = set()
    
    adjacency = {node: [] for node in nodes}
    reverse_adjacency = {node: [] for node in nodes}
    
    for edge in edges:
        from_node = edge["from_node"]
        to_node = edge["to_node"]
        adjacency[from_node].append(to_node)
        reverse_adjacency[to_node].append(from_node)
    
    def dfs(node, visited, current_dependencies):
        for neighbor in reverse_adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                current_dependencies.add(neighbor)
                dfs(neighbor, visited, current_dependencies)
    
    for node in sorted(nodes):
        visited = set()
        dfs(node, visited, dependencies[node])
    
    result = {}
    for node in sorted(nodes):
        result[node] = sorted(list(dependencies[node]))
    
    return result
