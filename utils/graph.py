# Python Program to detect cycle in an undirected graph
from collections import defaultdict


# This class represents a undirected
# graph using adjacency list representation


class Graph:

    def __init__(self, vertices):

        # No. of vertices
        self.V = vertices  # No. of vertices

        # Default dictionary to store graph
        self.graph = defaultdict(list)

    # Function to add an edge to graph
    def addEdge(self, v, w):

        # Add w to v_s list
        self.graph[v].append(w)

        # Add v to w_s list
        self.graph[w].append(v)



# Create a graph given in the above diagram
g = Graph(5)
g.addEdge(1, 0)
g.addEdge(1, 2)
g.addEdge(2, 0)
g.addEdge(0, 3)
g.addEdge(3, 4)

if g.isCyclic():
    print("Graph contains cycle")
else:
    print("Graph doesn't contain cycle ")
g1 = Graph(3)
g1.addEdge(0, 1)
g1.addEdge(1, 2)

if g1.isCyclic():
    print("Graph contains cycle")
else:
    print("Graph doesn't contain cycle ")

# This code is contributed by Neelam Yadav
