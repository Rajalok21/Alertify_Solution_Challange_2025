import math
import numpy as np
import folium
import webbrowser
from folium.plugins import AntPath

INF = float('inf')

description = []

# TSP Solver with Dynamic Programming
def tsp(pos, mask, dp, distMatrix, parent, n):
    if mask == (1 << n) - 1:  
        if distMatrix[pos][0] == INF:  # No path back to start
            return INF
        return distMatrix[pos][0]  # Return to start
    
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

# Read adjacency matrix from file
def readAdjacencyMatrixFromFile(filename):
    with open(filename, 'r') as file:
        n = int(file.readline().strip())
        distMatrix = []
        for _ in range(n):
            row = list(map(float, file.readline().strip().split()))
            # Replace -1 (no direct edge) with INF
            distMatrix.append([INF if x == -1 else x for x in row])

    return n, distMatrix

# Reconstruct the path from parent array
def reconstructPath(start, parent, n):
    path = []
    mask = 1
    pos = start
    path.append(pos)

    for i in range(1, n):
        nextPos = parent[pos][mask]
        path.append(nextPos)
        mask |= (1 << nextPos)  # Mark the next city as visited
        pos = nextPos

    path.append(start)
    return path

# Solve the TSP problem
def solve_tsp(distMatrix):
    n = len(distMatrix)
    dp = [[-1 for _ in range(1 << n)] for _ in range(n)]
    parent = [[-1 for _ in range(1 << n)] for _ in range(n)]
    result = tsp(0, 1, dp, distMatrix, parent, n)

    if result == INF:
        print("There is no valid tour that visits all cities and returns to the start.")
    else:
        path = reconstructPath(0, parent, n)
        return path

# Drone delivery simulation using TSP
def drone_delivery(filename, demand_list, drone_capacity, ans):
    n, distMatrix = readAdjacencyMatrixFromFile(filename)
    unvisited_nodes = [i for i in range(len(demand_list)) if demand_list[i] > 0]  # Include base node in TSP calculation

    while len(unvisited_nodes) > 1:  # More than just base node (index 0)
        # Create a new distance matrix considering only unvisited nodes (including base node 0)
        new_distMatrix = [[distMatrix[i][j] for j in unvisited_nodes] for i in unvisited_nodes]
        node_mapping = {new_idx: original_idx for new_idx, original_idx in enumerate(unvisited_nodes)}

        # Solve TSP for the remaining unvisited nodes including the base (node 0)
        path = solve_tsp(new_distMatrix)

        # Convert path indices back to original node indices
        path = [node_mapping[idx] for idx in path]

        print(f"Current TSP path: {path}")
        current_capacity = drone_capacity

        for i in path[1:]:  # Start from the first unvisited node (ignore the base node 0 in the loop)
            if current_capacity >= demand_list[i]:
                current_capacity -= demand_list[i]
                ans.append(i)
                text = f'Supplies delivered to zone {i}. Needed {demand_list[i]}, remaining capacity {current_capacity}'
                description.append(text)
                print(text)
                demand_list[i] = 0  # Mark this zone as fully supplied
            else:
                # Partial delivery, drone returns to base
                # ans.append(i)
                # print(f'Partial supplies delivered to zone {i}. Needed {demand_list[i]}, delivered {current_capacity}')
                # demand_list[i] -= current_capacity
                ans.append(0)  # Returning to base node

                break

        # Update the list of unvisited nodes (recalculate based on remaining demand)
        unvisited_nodes = [i for i in range(len(demand_list)) if demand_list[i] > 0]  # Include base node 0

        if len(unvisited_nodes) > 1:  # More than just the base node
            text = "Returning to base for refilling and recalculating TSP." 
            description.append(text)
            print(text)
        else:
            print("All zones have received their required supplies.")
    
    return ans, description

# Function to convert degrees to radians
def degtorad(degrees):
    return degrees * (np.pi / 180)

# Function to calculate distance between two points on the Earth
def distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in km
    d_lat = degtorad(lat2 - lat1)
    d_lon = degtorad(lat2 - lat1)
    a = np.sin(d_lat / 2) ** 2 + np.cos(degtorad(lat1)) * np.cos(degtorad(lat2)) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Function to map points and simulate an object moving along the TSP path
def mapping(L1, tsp_path):
    # Initialize the map with the first location in L1
    lat1, long1 = L1[0][0], L1[0][1]
    mapp = folium.Map(location=[lat1, long1], zoom_start=15)
    
    # Add markers for all locations in L1
    for i in L1:
        folium.Marker([i[0], i[1]], popup=i[2], icon=folium.Icon(color=i[3], icon_color="white", prefix="fa", icon="location-dot")).add_to(mapp)

    # Convert the TSP path into pairs of edges and simulate the path
    tsp_edges = [(tsp_path[i], tsp_path[i + 1]) for i in range(len(tsp_path) - 1)]
    tsp_edges.append((tsp_path[-1], tsp_path[0]))  # Complete the loop by connecting the last to the first

    # Collect the coordinates along the TSP path for the moving object
    tsp_coords = [[L1[vertex][0], L1[vertex][1]] for vertex in tsp_path]

    # Add the TSP path to the map
    folium.PolyLine(tsp_coords, color="green", weight=5, opacity=0.85).add_to(mapp)

    # Simulate the object moving along the path using AntPath
    AntPath(tsp_coords, color="blue", weight=5, delay=1000).add_to(mapp)

    # Save and open the map
    mapp.save("mymap2.html")
    #webbrowser.open("mymap2.html", new=2)

# Read coordinates and names from file and create markers
def read_coordinates_from_file(filename, lat1, long1):
    L1 = []
    with open(filename, 'r') as file:
        for line in file:
            lat, lon, name = line.split()
            L1.append([float(lat), float(lon), name, "red"])
    L1.insert(0, [lat1, long1, "base", "blue"])
    return L1

# Base location
def memain2():
    lat1 = 22.488540958444293
    long1 = 88.36633469004033

    # Sample TSP path
    filename = 'adjacency_matrix.txt'
    demand_list = [150, 40, 55, 20, 60, 30]  # Example demands for zones
    drone_capacity = demand_list[0]  # Assuming drone starts with full capacity of the first zone's demand
    ans = []
    # filename = 'adjacency_matrix.txt'
    # path = solve_tsp(filename)
    # print(path)
    # Solve the drone delivery problem and get the path
    drone_delivery(filename, demand_list, drone_capacity, ans)
    print(ans)
    # Add base (starting point) to the final path for round trip
    ans = [0] + ans
    # Read the coordinates and create the map
    L1 = read_coordinates_from_file('ndrf_small_units.txt', lat1, long1)
    mapping(L1, ans)

print(description)