import folium
import numpy as np
import webbrowser
from folium.plugins import AntPath

# Function to convert degrees to radians
def degtorad(degrees):
    return degrees * (np.pi / 180)

# Function to calculate distance between two points on the Earth
def distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in km
    d_lat = degtorad(lat2 - lat1)
    d_lon = degtorad(lon2 - lon1)
    a = np.sin(d_lat / 2) ** 2 + np.cos(degtorad(lat1)) * np.cos(degtorad(lat2)) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Function to map points and simulate an object moving along the TSP path
def mapping(L1, tsp_path):
    mapp = folium.Map(location=[lat1, long1], zoom_start=15)
    
    # Add markers for all locations
    for i in L1:
        folium.Marker([i[0], i[1]], popup=i[2], icon=folium.Icon(color=i[3], icon_color="white", prefix="fa", icon="location-dot")).add_to(mapp)

    # Convert the TSP path into pairs of edges and simulate the path
    tsp_edges = [(tsp_path[i], tsp_path[i+1]) for i in range(len(tsp_path)-1)]
    tsp_edges.append((tsp_path[-1], tsp_path[0]))  # Complete the loop by connecting last to first

    # Collect the coordinates along the TSP path for the moving object
    tsp_coords = [[L1[vertex][0], L1[vertex][1]] for vertex in tsp_path]

    # Add the TSP path to the map
    folium.PolyLine(tsp_coords, color="green", weight=5, opacity=0.85).add_to(mapp)

    # Simulate the object moving along the path using AntPath
    AntPath(tsp_coords, color="blue", weight=5, delay=1000).add_to(mapp)

    # Save and open the map
    mapp.save("mymap.html")
    #webbrowser.open("mymap.html", new=2)

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
lat1 = 22.488540958444293
long1 = 88.36633469004033

# Sample TSP path (modify this to get the real tsp_path from your TSP implementation)
tsp_path = [0, 1, 2, 3, 4, 5]  # Example: Adjust this based on your TSP solution

# Read the coordinates and create the map
L1 = read_coordinates_from_file('ndrf_small_units.txt', lat1, long1)
mapping(L1, tsp_path)
