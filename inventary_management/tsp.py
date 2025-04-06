import math
import numpy as np

INF = float('inf') 

def tsp(pos, mask, dp, distMatrix, parent, n):
    if mask == (1 << n) - 1:  
        if distMatrix[pos][0] == INF:
            return INF
        return distMatrix[pos][0]
    
    if dp[pos][mask] != -1:
        return dp[pos][mask]

    ans = INF
    nextCity = -1

    for city in range(n):
        if not (mask & (1 << city)) and distMatrix[pos][city] != INF:  # Check if the city is not yet visited and is reachable
            newCost = distMatrix[pos][city] + tsp(city, mask | (1 << city), dp, distMatrix, parent, n)
            if newCost < ans:
                ans = newCost
                nextCity = city  # Track the next city in the optimal path

    parent[pos][mask] = nextCity  # Store the city that gives the minimum cost
    dp[pos][mask] = ans
    return ans

# Function to read the adjacency matrix from a file
def readAdjacencyMatrixFromFile(filename):
    with open(filename, 'r') as file:
        # First, read the number of nodes
        n = int(file.readline().strip())

        # Initialize the distance matrix
        distMatrix = []
        for _ in range(n):
            row = list(map(float, file.readline().strip().split()))
            # Replace -1 (no direct edge) with INF
            distMatrix.append([INF if x == -1 else x for x in row])

    return n, distMatrix

# Function to reconstruct the TSP path from the parent array
def reconstructPath(start, parent, n):
    path = []
    mask = 1
    pos = start

    path.append(pos)  # Start from the first node

    # Reconstruct the path based on the parent table
    for i in range(1, n):
        nextPos = parent[pos][mask]
        path.append(nextPos)
        mask |= (1 << nextPos)  # Mark the next city as visited
        pos = nextPos

    path.append(start)
    return path

def solve_tsp(filename):
    n, distMatrix = readAdjacencyMatrixFromFile(filename)

    dp = [[-1 for _ in range(1 << n)] for _ in range(n)]

    # Parent table to reconstruct the path
    parent = [[-1 for _ in range(1 << n)] for _ in range(n)]

    # Start the TSP from position 0 with only node 0 visited (mask = 1)
    result = tsp(0, 1, dp, distMatrix, parent, n)

    if result == INF:
        print("There is no valid tour that visits all cities and returns to the start.")
    else:
        print(f"The minimum cost to visit all nodes is: {result}")
        
        path = reconstructPath(0, parent, n)
        #print("The optimal path is: ", ' -> '.join(map(str, path)))
        return path
    
filename = 'adjacency_matrix.txt'
path = solve_tsp(filename)
print(path)