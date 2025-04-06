import dash
from dash import dcc, html, Input, Output, State
import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import dash_leaflet as dl
import pandas as pd
from dash.exceptions import PreventUpdate
from geopy.distance import geodesic
import base64
import io
import threading
from datetime import timedelta
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import geopandas as gpd
import cv2
import numpy as np
import plotly.graph_objects as go
import base64
import tempfile
import os
from ultralytics import YOLO
from flask import send_from_directory
from threading import Thread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dash_bootstrap_templates import load_figure_template
import dash_leaflet as dl
from geopy.distance import great_circle
import json
from twilio.rest import Client
import requests
from geopy.distance import geodesic
import extract_sound_from_videio
import mp3towav
import speech
import transcription
import tensorflow as tf
import pymongo
import gridfs
import base64
import io
import subprocess
from PIL import Image


import dash
from dash import dcc, html
from app import memain
from routing import memain2





class JatayuVideoProcessor:
    def __init__(self, jatayu_video_path):
        self.jatayu_video_path = jatayu_video_path
        self.jatayu_cap = cv2.VideoCapture(jatayu_video_path)
        self.jatayu_running = False
        self.jatayu_model = YOLO("yolov5m6u.pt")
        self.jatayu_current_frame = None
        self.jatayu_detected_frame = None
        self.jatayu_thermal_frame = None
        self.jatayu_magma_frame = None
    
    def jatayu_start_processing(self):
        self.jatayu_running = True
        threading.Thread(target=self.jatayu_process_frames, daemon=True).start()
    
    def jatayu_process_frames(self):
        while self.jatayu_cap.isOpened() and self.jatayu_running:
            ret, frame = self.jatayu_cap.read()
            if not ret:
                break
            
            results = self.jatayu_model.predict(frame)
            self.jatayu_detected_frame = results[0].plot()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.jatayu_thermal_frame = cv2.applyColorMap(gray_frame, cv2.COLORMAP_JET)
            self.jatayu_magma_frame = cv2.applyColorMap(gray_frame, cv2.COLORMAP_MAGMA)
            self.jatayu_current_frame = frame
    
    def jatayu_stop_processing(self):
        self.jatayu_running = False
        self.jatayu_cap.release()

jatayu_video_path = r"C:\Users\mohit\Desktop\FINAL_RUN\FINAL_testcase3.mp4"  # Update this with the actual video path
jatayu_video_processor = JatayuVideoProcessor(jatayu_video_path)




#########################DRONE JATAYU
# Global variables
drones_data = pd.DataFrame()
locations_data = pd.DataFrame()
assigned_drones = {}
is_taking_off = False
static_markers = []  # Store static markers

# Parse uploaded data
def parse_data(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode('utf-8')))
#############################
#database connection
# Connect to MongoDB
client = pymongo.MongoClient('mongodb+srv://jenamohit24:mohit%40123@mohitjena.puhcv.mongodb.net')  # Replace with your MongoDB URI
db = client['disaster_reporting_system']  # Database name
disaster_collection = db['disaster_reports']  # Collection for disaster reports
missing_person_collection = db['missing_person_reports']  # Collection for missing person reports

# Initialize GridFS
fs = gridfs.GridFS(db)
###################################



# Run your external functions (memain and memain2)
memain()
memain2()

# Initialize the Dash app
app = dash.Dash(__name__)

# Reading the contents of the HTML files
with open('mymap.html', 'r') as f1:
    html_content1 = f1.read()

with open('mymap2.html', 'r') as f2:
    html_content2 = f2.read()


#database connection
# Connect to MongoDB
client = pymongo.MongoClient('mongodb+srv://jenamohit24:mohit%40123@mohitjena.puhcv.mongodb.net')  # Replace with your MongoDB URI
db = client['disaster_reporting_system']  # Database name
disaster_collection = db['disaster_reports']  # Collection for disaster reports
missing_person_collection = db['missing_person_reports']  # Collection for missing person reports

# Initialize GridFS
fs = gridfs.GridFS(db)
###################################



##################################################################################################################
'''SABKA BAAAP MODEL KA KAAMM YAHA HAI'''
import pandas as pd
import pickle
import re
from dash.dependencies import Input, Output
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import translate, location_fetch
from news_fetch import all_news_fetch


# Load pre-trained vectorizer and model
cv = pickle.load(open(r'C:\Users\mohit\Desktop\now_me\FINAL_RUN\vectorizer.pkl', 'rb'))
# Neural network model loading using keras
model = tf.keras.models.load_model(r'C:\Users\mohit\Desktop\now_me\FINAL_RUN\model.keras')
loaded_pipeline = pickle.load(open(r'C:\Users\mohit\Desktop\now_me\FINAL_RUN\disaster_classifier.pkl','rb'))

# Initialize stemmer and stop words
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

#FUNCTIONs

def transform_text(text):
    """Preprocess text by lowercasing, removing punctuation, and stemming."""
    text = text.lower()
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove punctuation
    words = text.split()
    filtered_words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(filtered_words)

def classify_with_threshold(text, threshold=0.6):
    """Classify text using a given threshold and return classification results."""
    translated_text = translate.translate_to_english(text)
    transformed_sms = transform_text(translated_text)
    vector_input = cv.transform([transformed_sms])
    probas = model.predict_proba(vector_input)
    
    is_disaster = 1 if probas[0][1] >= threshold else 0
    disaster_class = loaded_pipeline.predict([translated_text])[0] if is_disaster == 1 else None
    location = location_fetch.loc_fetch2(text) if is_disaster == 1 else None
    
    return {
        'Is_Disaster': 'Yes' if is_disaster == 1 else 'No',
        'Disaster_Class': disaster_class,
        'Location': location
    }

def fetch_news_and_create_csv():
    """Fetch news from social media and save to a CSV file."""
    usernames = ["priyanshu_pilaniwala"]
    twitter_url = "https://x.com/IndiaToday"
    count = 5
    csv_filename = "fetched.csv"

    all_news_fetch.fetch(twitter_url, csv_filename)
    all_news_fetch.scrape_instagram_data(usernames, count, csv_filename)

    return csv_filename

def run_classification(csv_filename):
    """Load CSV file, classify news, and save results to a new CSV file."""
    df = pd.read_csv(csv_filename)
    df['Is_Disaster'] = ''
    df['Disaster_Class'] = ''
    df['Location'] = ''

    for index, row in df.iterrows():
        input_text = row['news']
        result = classify_with_threshold(input_text, threshold=0.3)
        df.at[index, 'Is_Disaster'] = result['Is_Disaster']
        df.at[index, 'Disaster_Class'] = result['Disaster_Class']
        df.at[index, 'Location'] = result['Location']

    output_filename = 'classified.csv'
    df.to_csv(output_filename, index=False)
    
    return df
##################################################################################################################

# Initialize the Dash app with Bootstrap and Font Awesome for icons
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME])
app.title = "Disaster Monitoring Dashboard"
load_figure_template("cyborg")

# Load earthquake data for today
url = 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson'
gdf = gpd.read_file(url)

# Filter out negative magnitudes and sort by magnitude
gdf = gdf[gdf['mag'] > 0]  # Keep only positive magnitudes
top_5_earthquakes = gdf.nlargest(5, 'mag')[['place', 'mag', 'time', 'geometry']]

# Convert time to a readable format
top_5_earthquakes['time'] = pd.to_datetime(top_5_earthquakes['time'], unit='ms')
top_5_earthquakes['time'] = top_5_earthquakes['time'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Global variables for video capture, YOLO model, and processed frames
video_capture = None
yolo_model = YOLO("yolov5m6u.pt")
latest_detection_frame = None
latest_thermal_frame = None
latest_magma_frame = None
person_count_data = []
chunked_person_count_data_10 = []
object_detection_counts = {}
processing_thread = None
video_file_path = None
processing_started = False
processing_completed = False




##for the cyclone relief 
# Load the data from the Excel file for cyclone relief centers

DEFAULT_ICON = "https://raw.githubusercontent.com/mjsonu/TicTacToe_Game/ad355f7171cb41ee687f910aa9065caa7e1e63d7/Pin.svg"
SELECTED_ICON = "https://raw.githubusercontent.com/mjsonu/TicTacToe_Game/9d957d56cecd1529f5d2fc6156114e5fd645ff83/Pin_red.svg"
EFFECTED_ICON = "https://raw.githubusercontent.com/mjsonu/TicTacToe_Game/9d957d56cecd1529f5d2fc6156114e5fd645ff83/Pin_green.svg"
# Load the data from the Excel file for the cyclone shelters
shelter_df = pd.read_excel("Alertify shelter details.xlsx")  # Replace with your file path

# Ensure latitude and longitude columns are numeric and within valid ranges
shelter_df['Latitude '] = pd.to_numeric(shelter_df['Latitude '], errors='coerce')
shelter_df['Longitude'] = pd.to_numeric(shelter_df['Longitude'], errors='coerce')
shelter_df = shelter_df[(shelter_df['Latitude '].between(-90, 90)) & (shelter_df['Longitude'].between(-180, 180))]

# Define the affected location (example coordinates)
affected_location = [22.488769165624642, 88.3658908363286]  # Replace with your specific coordinates ##IMPIMPIMPIMPIMPIMPIMPIMPIMPIMPIMPIMPIMPIP

# Create a list to hold the marker data
markers_cc = [
    {
        "position": [row['Latitude '], row['Longitude']],
        "id": {'type': 'shelter-marker', 'index': i},
        "name": row['Name']
    }
    for i, row in shelter_df.iterrows()
]

#########################################


########for population map 
# Load the population data
df_population = pd.read_csv(r"state.csv")

# Load GeoJSON data with utf-8 encoding
with open(r"state.geojson", encoding='utf-8') as f:
    geojson_data = json.load(f)

# Load the new input CSV file with locations to mark on the map
df_locations = pd.read_csv(r"lat_long.csv")  # Make sure this CSV contains 'name', 'latitude', 'longitude' columns

# Define specific population ranges and corresponding colors
range_colors = {
    (0, 10_000_000): "rgba(255, 245, 240, 0.8)",
    (10_000_000, 50_000_000): "rgba(255, 200, 200, 0.8)",
    (50_000_000, 100_000_000): "rgba(255, 100, 100, 0.8)",
    (100_000_000, 150_000_000): "rgba(200, 0, 0, 0.8)",
    (150_000_000, 200_000_000): "rgba(140, 0, 0, 0.8)"
}

# Create a list of tuples for color scale
color_scale = [(0, "rgb(255, 245, 240)"), 
               (0.2, "rgb(255, 200, 200)"), 
               (0.5, "rgb(255, 100, 100)"), 
               (0.8, "rgb(200, 0, 0)"), 
               (1, "rgb(140, 0, 0)")]

# Create choropleth map using graph_objects
fig = go.Figure()

fig.add_trace(go.Choroplethmapbox(
    geojson=geojson_data,
    locations=df_population['state_name'],
    featureidkey="properties.NAME_1",
    z=df_population['population'],
    colorscale=color_scale,
    colorbar=dict(
        title="Population",
        tickvals=[0, 10_000_000, 50_000_000, 100_000_000, 150_000_000, 200_000_000],
        ticktext=["0-10M", "10-50M", "50-100M", "100-150M", "150-200M", "200M+"],
        thickness=20,
        len=0.8,
        xanchor='left',
        yanchor='top',  
        y=0.9,  
        x=1.02,  
        xpad=10,  # Reduce padding on the x-axis
        ypad=10
    ),
    marker_line_width=0.7,
    marker_line_color="black",
))

# Add locations from the input CSV file
fig.add_trace(go.Scattermapbox(
    lat=df_locations['lat'],
    lon=df_locations['long'],
    mode='markers',
    marker=dict(
        size=10,
        color='#0C79FE',  
        opacity=0.8
    ),
    text=df_locations['name'],
    name='Specific Locations'
))

# for adding the source location
fig.add_trace(go.Scattermapbox(
    lat=[str(affected_location[0])],
    lon=[str(affected_location[1])],
    mode='markers',
    marker=dict(
        size=12,
        color='#61ff00',  
        opacity=0.8
    ),
    text=df_locations['name'],
    name='Affected Location'
))

fig.update_layout(
    plot_bgcolor='#2c2c2c',  # Background color of the plot area
    paper_bgcolor='#2c2c2c',
    mapbox_style="open-street-map",
    mapbox=dict(
        center=dict(lat=affected_location[0], lon=affected_location[1]),
        zoom=5,  
    ),
    geo=dict(
        visible=False  
    ),
    margin={"r":20,"t":50,"l":20,"b":20},
    font=dict(family='Arial, sans-serif', color='#ffffff'),
)

#########################################
#SOSOSOSOSOSOSOSOSOSOSOSOSSSSSOOOOOOOOOSOSOSOSOSOSO
# List of phone numbers to send SMS
phone_numbers_police = ["+918585050721","+917764039797"]  # Add the phone numbers here
phone_numbers_hospital = ["+918961891811"]  # Add the phone numbers here
phone_numbers_costal = ["+917667813756"]  # Add the phone numbers here



# Twilio credentials
account_sid = 'AC1f6fb2e97a56a44cdfea48d067c9c744'
auth_token = '9b35acc7ecb3bf8940e593a9fb50221c'
twilio_number = '+17604291475'  # Replace with your Twilio number

# Initialize the Twilio client
client = Client(account_sid, auth_token)


# data for police stations and hospitals
def query_osm_nearest(lat, lon, amenity, count):
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    overpass_query = f"""
    [out:json];
    node
      ["amenity"="{amenity}"]
      (around:5000,{lat},{lon});
    out body;
    """
    
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    
    places = []
    
    if 'elements' in data and len(data['elements']) > 0:
        for element in data['elements']:
            place_lat = element['lat']
            place_lon = element['lon']
            place_name = element['tags'].get('name', 'Unnamed')
            contact_number = element['tags'].get('contact:phone', 'No contact info')
            distance = geodesic((lat, lon), (place_lat, place_lon)).kilometers
            
            places.append({
                'name': place_name,
                'lat': place_lat,
                'lon': place_lon,
                'distance_km': distance,
                'contact_number': contact_number
            })
        
        places.sort(key=lambda x: x['distance_km'])
        return places[:count]
    
    return []

# Function to find the nearest police stations and hospitals
def find_nearest_services(lat, lon):
    nearest_police = query_osm_nearest(lat, lon, 'police', 2)
    nearest_hospitals = query_osm_nearest(lat, lon, 'hospital', 4)
    
    return nearest_police, nearest_hospitals

# Example affected location
#affected_location = [22.48881661932125, 88.36586488081795]
nearest_police_stations, nearest_hospitals = find_nearest_services(affected_location[0], affected_location[1])

# Prepare the markers for the map
markers = [dl.Marker(position=affected_location, children=dl.Tooltip("Affected Location"),icon={
                "iconUrl": EFFECTED_ICON,
                "iconSize": [30, 30],
                "iconAnchor": [12, 24]})]

for place in nearest_police_stations:
    markers.append(dl.Marker(position=[place['lat'], place['lon']], children=dl.Tooltip(place['name']), icon={
                "iconUrl": SELECTED_ICON,
                "iconSize": [30, 30],
                "iconAnchor": [12, 24],
            }))

for place in nearest_hospitals:
    markers.append(dl.Marker(position=[place['lat'], place['lon']], children=dl.Tooltip(place['name']),icon={
                "iconUrl": DEFAULT_ICON,
                "iconSize": [30, 30],
                "iconAnchor": [12, 24]}))


# Define the SMS sending function
def send_sms(phone_numbers, message_body):
    for number in phone_numbers:
        message = client.messages.create(
            body=message_body,
            from_=twilio_number,
            to=number
        )
        print(f"Message sent to {number}: {message.sid}")
        
########################################



# Function to generate the top 5 earthquakes table with clickable rows
def generate_earthquake_table(dataframe):
    return dbc.Table([
        html.Thead(html.Tr([html.Th(col) for col in dataframe.columns if col != 'geometry'])),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns if col != 'geometry'
            ], id=f"earthquake-row-{i}", style={'cursor': 'pointer'}) for i in range(len(dataframe))
        ])
    ], bordered=True, hover=True, responsive=True, className="table table-hover", style={
                                'font-size': '12px','font-weight': '500','width': '100%'})
    
    
    
    
##############################
import csv
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager as CM
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# Function to run the web scraping script and fetch news
def fetch_sachet_news():
  

    chrome_options = Options()
    chrome_options.add_argument("--headless")  
    chrome_options.add_argument("--disable-gpu")  
    chrome_options.add_argument("--no-sandbox")  
    chrome_options.add_argument("--window-size=1920x1080") 

    service = Service(executable_path=CM().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.get("https://sachet.ndma.gov.in/")

    wait = WebDriverWait(driver, 10)
    all_india_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="alertView_AlertBannerForMobile__iKSIs"]/div/div/div/div/button[2]')))

    all_india_button.click()

    sachet = []
    headings = driver.find_elements(By.XPATH, '//*[@id="style-1"]/div/div[1]')
    elements = driver.find_elements(By.XPATH, '//*[@id="style-1"]/div/div[2]')

    min_length = min(len(headings), len(elements))
    for i in range(min_length):
        h = headings[i].text
        e = elements[i].text
        sachet.append([h, e])

    driver.quit()

    top_sachet = sachet[:10]

    with open('sachet_data_top10.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Heading', 'Element'])
        writer.writerows(top_sachet)

    print("Fetched and saved to sachet_data_top10.csv!")

# Read the CSV file and return the news data
def read_sachet_csv(file_path='sachet_data_top10.csv'):
    try:
        df = pd.read_csv(file_path)
        news = [f"{row['Heading']}: {row['Element']}" for _, row in df.iterrows()]
        return " | ".join(news)  # Join all news into a single string for the ticker
    except FileNotFoundError:
        return "No news data available."

# CSS for the ticker animation (inline style)
ticker_container_style = {
    'width': '100%',             # Full width of the container
    'overflow': 'hidden',        # Hide overflowed content  # Optional: Add border around the ticker
    'backgroundColor': '#C6F806',  # Background color
    'padding': '10px',           # Padding for better visual spacing
    'color': '#000000',            # Text color
    'fontSize': '18px',          # Font size for the ticker text
    'fontWeight': 'bold',        # Make text bold
    'position': 'relative',
    'border-radius': '5px'# Positioning for animation
    
}

# Keyframes style for scrolling effect
ticker_animation = {
    'whiteSpace': 'nowrap',      # Prevent text wrapping
    'animation': 'scroll 50s linear infinite',  # Animation duration and infinite loop
    'display': 'inline-block',   # Ensure text stays in a single line
}

    


# Define the layout for each tab
def create_tab_layout(disaster_type):
    windy_url = "https://embed.windy.com/embed.html?type=map&location=coordinates&metricRain=mm&metricTemp=default&metricWind=km/h&zoom=5&overlay=wind&product=ecmwf&level=surface&lat=18.771&lon=88.374&detailLat=22.479&detailLon=88.365&detail=true&pressure=true&message=true"

    if disaster_type == 'Flood':
        return html.Div([
            dbc.Container([
                html.Div(
                    html.Div(
                        id='news-ticker',  # Div to hold the dynamically updated news ticker
                        style=ticker_animation
                    ),
                    style=ticker_container_style  # Outer container to control scrolling
                ),
                
                # Interval component to trigger the news fetching every hour (3600 seconds)
                dcc.Interval(
                    id='interval-component-ticker',
                    interval=3600*1000,  # in milliseconds (3600 seconds = 1 hour)
                    n_intervals=0  # Start with 0 intervals
                )
            ], fluid=True,style={'margin-bottom':'5px','margin-top':'10px'}),
            html.Div([
                html.Iframe(
                    src=windy_url,
                    className="iframe-container",
                    style={
                        'width': '100%',
                        'height': '540px',
                        'border-radius': '8px'  # Apply border-radius here
                    }
                ),
            ], style={'padding': '0.5rem','borderRadius': '5px','margin-bottom':'10px','margin-top':'20px'}),
            
            html.Div([
                html.Div([
                    # Left side with the Graph
                    dcc.Graph(
                        id='earthquake-map',
                        style={
                            'border-radius': '8px',
                            'overflow': 'hidden',
                            'width': '100%',
                            'height': '500px'  # You can adjust the height as needed
                        }
                    )
                ], style={'flex': '1.2', 'padding-right': '0.5rem'}),  # Adjusting the left column width and spacing

                html.Div([
                    # Right side with the table and selected earthquake info
                    
                   html.Div([
                        html.H6(
                            "Top 5 Highest Magnitude Earthquakes Today", 
                            className='text-center mt-4 medium', 
                            style={
                                'font-size': '18px',
                                'font-weight': '700',
                                'color': 'white'  # Optional text color to contrast background
                            }
                        ),
                        generate_earthquake_table(top_5_earthquakes)
                    ], style={'backgroundColor': '#2c2c2c','border-radius': '8px','width': '100%',"padding-right":'10px',"padding-left":'10px','margin-bottom':'8px'}),
                    
                    html.Div(id='selected-earthquake-info',style={'font-size': '18px','font-weight': '500', 'backgroundColor': '#2c2c2c','border-radius': '8px'})
                ], style={'flex': '0.8', 'display': 'flex', 'flex-direction': 'column','border-radius': '8px'})  # Adjusting the right column
                ], className='d-flex', style={'padding': '0.5rem', 'border-radius': '8px', 'display': 'flex', 'flex-wrap': 'wrap','border-radius': '8px'}
                ),
            
            
            html.Div([
                dl.Map(
                    center=affected_location,  # Centered on the affected location
                    zoom=8,
                    children=[
                        dl.TileLayer(),  # Base layer
                        dl.LayerGroup(id="shelter-marker-layer"),  # Marker layer
                        dl.LayerGroup(id="shelter-line-layer")  # Line layer
                    ],
                id="shelter-map",
                style={'width': '100%', 'height': '400px','backgroundColor': '#2c2c2c',
                'borderRadius': '8px',}
                ),
            ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px',}),
            
            
            

            html.Div([
                html.P("Information of Cyclone Relief Center", style={'padding': '10px', 'text-align': 'center','font-size': '24px','font-weight': '700','color':'white','backgroundColor': '#2c2c2c',
                'borderRadius': '10px'}),
                html.Div(id='shelter-info', style={'padding': '20px', 'fontSize': '16px','font-weight': '500','backgroundColor': '#2c2c2c',
                'borderRadius': '10px',})
            ], style={'width': '50%','height': '400px', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'}),
            
            html.Div(children=[
                dcc.Graph(
                    id='population-map',
                    figure=fig,
                    style={'width': '100%', 'height': '600px','border-radius': '10px',  # Rounded corners
                    'overflow': 'hidden'}
                )
            ],style={'border-radius': '10px','display': 'flex','width': '100%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px'}),
        ])
        
    elif disaster_type == 'Video':
        return html.Div([
            #VIDEO PROCESSING
            html.Div([
                # First row with the upload, video display, and button
                html.Div([
                    dcc.Upload(id='upload-video', children=html.Div(['Drag and Drop or ', html.A('Select a Video')]), style={
                        'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                        'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '2px'
                    }),
                    html.Video(id='video-display', controls=True, style={'width': '100%','height':'500px','borderRadius': '5px','margin': '2px','margin-top': '10px','objectFit': 'contain'}),
                    dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
                    html.Div(id='dummy-div', style={'display': 'none'})
                ], style={'width': '65%','padding': '10px','backgroundColor': '323332','float': 'left','border-radius': '8px'}),
                
                html.Div([
                        html.Button('Start Processing', id='start-processing', n_clicks=0,style={'width': '100%', 'height': '60px', 'backgroundColor': '#03DAC6',
                                'border-radius': '8px', 'border': 'none', 'font-size': '20px',
                                'font-weight': '700'}),
                ], style={'width': '34%','padding': '10px','margin': '2px','backgroundColor': '323332','float': 'right','border-radius': '8px'}),
                
                html.Div([
                        html.Div(dcc.Graph(id='person-count-graph',style={'width': '99%','height':'350px','marginTop': '10px'}),style={'width': '100%','height':'350px','border-radius': '8px','display': 'flex', 'justify-content': 'center','align-items': 'center',})
                ], style={'display': 'flex','width': '100%','height':'400px','marginTop': '10px','border-radius': '8px'}),

                # Second row with the live detection, thermal mapping, and magma mapping images
                html.Div([
                    html.Div([
                        html.Div(html.Img(id='live-detection', style={'width': '100%','height': '100%','objectFit': 'contain','border-radius': '8px'}), style={'padding': '10px','width': '100%','height':'100%','border-radius': '8px'})
                    ],style={'width': '55%','height':'600px','padding': '10px','margin-right': '5px','float': 'left','border-radius': '8px'}),
                    html.Div([
                        html.Div(
                            html.Img(id='live-thermal-mapping', style={
                                'width': '99%',            # Set the image width to 100% of the container
                                'height': '99%',           # Set the image height to 100% of the container
                                'border-radius': '8px',
                                'objectFit': 'contain'      # Maintain aspect ratio within the given dimensions
                            }),
                            style={
                                #'background-color': '#323332',
                                'padding': '10px',
                                'margin-bottom': '10px',
                                'width': '100%',
                                'height': '49%',            # Height of the container
                                'border-radius': '8px'
                            }
                        ),

                        html.Div(
                            html.Img(id='live-magma-mapping', style={
                                'width': '99%',            # Set the image width to 100% of the container
                                'height': '99%',           # Set the image height to 100% of the container
                                'border-radius': '8px',
                                'objectFit': 'contain'      # Maintain aspect ratio within the given dimensions
                            }),
                            style={
                                #'background-color': '#323332',
                                'padding': '10px',
                                'width': '100%',
                                'height': '49%',            # Height of the container
                                'border-radius': '8px'
                            }
                        )
                    ], style={'width': '45%','height':'600px','padding': '10px','float': 'right','border-radius': '8px'})
                ], style={'display': 'flex', 'width': '100%'}),

                

                # Fourth row with the chunked person count heatmap and object detection pie chart
                html.Div([
                    html.Div(dcc.Graph(id='chunked-person-count-heatmap'), style={'flex': '1', 'padding-right': '10px'}),
                    html.Div(dcc.Graph(id='object-detection-pie-chart'), style={'flex': '1', 'padding-left': '10px'}),
                ], style={'display': 'flex', 'width': '100%', 'marginTop': '20px'})
            ])

        ])
        
    elif disaster_type == 'audio':
        return html.Div([
            # Container for audio analysis
            
                html.Div([
                    # Full-width div
                    html.Div([    
                        dcc.Upload(
                            id='upload-audio-video',
                            children=html.Div(['Drag and Drop or ', html.A('Select a Video File')]),
                            style={
                                'width': '72%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'margin-right':'5px',
                                'float': 'left'
                                
                            },
                            multiple=False
                        ),
                        
                        # Button to trigger audio analysis
                        html.Button('Process Audio', id='process-audio-button', n_clicks=0, style={
                            'display': 'block', 'backgroundColor': '#03DAC6', 
                            'color': 'black', 'borderRadius': '5px', 'padding': '10px 20px', 
                            'border': 'none', 'width': '25%', 'height': '60px','font-size': '20px','font-weight': '700','margin-right':'5px',
                            'float': 'right'
                        })                     
                    ], style={
                        'width': '100%',              # Ensures the div takes up the full width
                        'padding': '10px',            # Adds padding around the content
                        'height': '70px',             # Sets the height of the div
                        'textAlign': 'center'         # Aligns content inside the div to the center (if needed)
                    }),

                    

                    # Prediction and Transcript Section
                    html.Div([
                        html.Div([   
                            html.Audio(id='audio-player', controls=True, style={'width': '100%','margin-top': '20px'})
                        ],style={'textAlign': 'center', 'margin-right': '5px', 'width': '48%','height': '100px','float': 'left','padding':'10px'}
                        ),
                        
                        # Div to display transcript and translated text
                        html.Div(id='transcript', style={
                            'font-size': '18px',
                            'padding':'10px', 
                            'font-weight': '700',
                            'width': '48%', 
                            'marginTop': '20px',
                            'height': '150px', 
                            'backgroundColor': '#323332', 
                            'color': 'white', 
                            'border-radius': '8px',
                            'float': 'right'       # Keeps it inline with the previous div
                        })
                    ], style={'width': '100%','height': '200px', 'textAlign': 'center'}),

                    # Image to display audio plot
                    html.Img(id='audio-plot', style={'width': '100%', 'marginTop': '20px'})
                ], style={'width': '100%', 'height': 'auto'})  # Set height to 'auto' to accommodate content
            ], style={'width': '100%', 'height': '100vh', 'display': 'flex'})  # Full-width container

    elif disaster_type == 'sos':
        
       return html.Div([
            # First section: SOS information and map
            html.Div([
                # SOS information box and buttons
                html.Div([
                    html.Div(id='info-box', children=[
                        html.H5("Nearest Police Stations and Hospitals", style={'textAlign': 'left','font-weight': '700'}),
                        html.P("Nearest Police Stations:", style={'font-weight': '700','font-size': '18px'}),
                        html.Ul([html.Li(f"{p['name']} - {p['distance_km']:.2f} km - {p['contact_number']}") for p in nearest_police_stations]),
                
                        html.P("Nearest Hospitals:", style={'font-weight': '700','font-size': '18px'}),
                        html.Ul([html.Li(f"{h['name']} - {h['distance_km']:.2f} km - {h['contact_number']}") for h in nearest_hospitals]),
                    ], style={
                        'backgroundColor': '#323332', 
                        'color': 'white', 
                        'border-radius': '8px',
                        'width': '100%',
                        'padding': '20px'
                    }),
                    html.Br(),
                    html.Button("SOS Police", id="sos-btn-1", n_clicks=0, style={
                        'width': '32%', 'height': '50px', 'background-color': '#dc3545', 'color': 'white', 

                        'border-radius': '8px', 'border': '2px solid #a71d2a', 'font-weight': '700'
                    }),
                    html.Button("SOS Hospital", id="sos-btn-2", n_clicks=0, style={
                        'width': '32%', 'height': '50px', 'margin-right': '10px', 'margin-left': '10px', 
                        'background-color': '#ffc107', 'color': 'black', 'border-radius': '8px', 
                        'border': '2px solid #d39e00', 'margin-top': '10px', 'font-weight': '700'
                    }),
                    html.Button("SOS Coastal Guard", id="sos-btn-3", n_clicks=0, style={
                        'width': '32%', 'height': '50px', 'background-color': '#0d6efd', 'color': 'white', 
                        'border-radius': '8px', 'border': '2px solid #0a58ca', 'margin-top': '10px', 'font-weight': '700'
                    }),
                ], style={'width': '42%', 'padding': '5px', 'float': 'left','height': '400px','margin-right':'10px'}),

                # Map display
                html.Div([
                    dl.Map(center=affected_location, zoom=15, children=[
                        dl.TileLayer(),
                        dl.LayerGroup(markers),
                    ], style={'width': '100%', 'height': '420px', 'border-radius': '10px'})
                ], style={'width': '58%', 'float': 'right', 'padding': '5px'}),
            ], style={'width': '100%', 'display': 'flex','backgroundColor': '#121212'}),  # Adjust width and layout

            # Spacer to add some margin before the next section
            html.Br(),
            
            html.Div([
                html.H4("Missing Person Reports", style={'textAlign': 'center', 'color': '#03DAC6', 'font-weight': '700','margin-top': '20px','margin-bottom': '10px'}),
                html.Div(id='missing_person_reports_container', style={'margin-left': '10px','width':"100%"})
            ], style={'width': '100%','backgroundColor': '#121212'}),
            
            html.Br(),

            # Second section: Relief funds distribution system
            html.Div([
                html.H4("Relief Funds Distribution System", style={'textAlign': 'center', 'color': '#03DAC6', 'font-weight': '700'}),
                html.Div([
                    # First HTML file (left side)
                    html.Iframe(srcDoc=html_content1, style={
                        "width": "48%", "height": "500px", "border": "none", "display": "inline-block", 'border-radius': '8px','margin-right': '20px'
                    }),
                    
                    # Second HTML file (right side)
                    html.Iframe(srcDoc=html_content2, style={
                        "width": "48%", "height": "500px", "border": "none", "display": "inline-block", 'border-radius': '8px'
                    }),
                ], style={
                    "width": "100%", "textAlign": "center", "justify-content": "space-between", 'margin-top': '20px'
                })
            ],style={'margin-top': '20px','width': '100%','backgroundColor': '#121212'}),
            
            
        ], style={'width': '100%', 'height': '100vh', 'backgroundColor': '#121212','display': 'block', 'margin-top': '20px'})
    
    elif disaster_type == 'Drone': 
        return html.Div([
            
            html.Div([
                html.Button('Jatayu System', id='jatayu_start-button', n_clicks=0, style={'margin': '10px',
                    'width': '300px',
                    'height': '50px',# Adjusted button width
                    'border-radius': '8px',
                    'font-weight': '600',
                    'font-size': '18px',
                    'display': 'inline-block','justify-content': 'center', 'align-items': 'center'}),
                dcc.Interval(id='jatayu_frame-update', interval=1000, n_intervals=0),  # Reduced interval for faster updates
                html.Div([
                    html.Div([html.Img(id='jatayu_original-video', style={'width': '90%','borderRadius': '10px'})], style={'width': '48%', 'display': 'inline-block', 'padding': '10px','marginLeft': '40px'}),
                    html.Div([html.Img(id='jatayu_object-detection', style={'width': '90%','borderRadius': '10px'})], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
                ], style={'display': 'flex', 'justify-content': 'center','backgroundColor': '#1e1e1e'}),
                html.Div([
                    html.Div([html.Img(id='jatayu_thermal-mapping', style={'width': '90%','borderRadius': '10px'})], style={'width': '48%', 'display': 'inline-block', 'padding': '10px','marginLeft': '40px'}),
                    html.Div([html.Img(id='jatayu_magma-mapping', style={'width': '90%','borderRadius': '10px'})], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
                ], style={'display': 'flex', 'justify-content': 'center','backgroundColor': '#1e1e1e'})
            ],style={'backgroundColor': '#1e1e1e','padding': '10px','borderRadius': '10px'}),
            
            html.H4("DRONE CONTROL SYSTEM", style={'textAlign': 'center', 'marginBottom': '20px','marginTop': '60px',
                                                   'color': '#03DAC6', 'font-weight': '700',}),
    
            html.Div([
            dcc.Upload(
                id='upload-drones-data',
                children=html.Button('Upload Drone Data CSV', style={
                    'margin': '10px',
                    'width': '300px',
                    'height': '50px',# Adjusted button width
                    'border-radius': '8px',
                    'font-weight': '600',
                    'font-size': '18px',
                    'display': 'inline-block'
                }),
                multiple=False
            ),
            dcc.Upload(
                id='upload-locations-data',
                children=html.Button('Upload Locations Data CSV', style={
                    'margin': '10px',
                    'width': '300px',
                    'height': '50px',# Adjusted button width
                    'border-radius': '8px',
                    'font-weight': '600',
                    'font-size': '18px',
                    'display': 'inline-block'
                }),
                multiple=False
            )
            ], style={
                'textAlign': 'center',
                'marginBottom': '20px',
                'display': 'flex',            # Use flex layout for alignment
                'justifyContent': 'center',    # Center the buttons
                'gap': '10px'                  # Space between buttons
            }),
            
            html.Div(id='file-upload-info', style={'textAlign': 'center', 'marginBottom': '20px',
                                                   'font-weight': '500','font-size': '16px',}),
            
            html.Div([
                dl.Map(id="map", center=[22.5726, 88.3639], zoom=15, children=[
                    dl.TileLayer(),
                    dl.LayerGroup(id="marker-layer"),  # Static base and affected location markers
                    dl.LayerGroup(id="drone-layer"),   # Drone positions
                    dl.LayerGroup(id="path-layer")     # Drone paths
                ], style={'width': '100%', 'height': '100%','padding': '10px'}),
            ], style={'width': '100%', 'height': '500px','border': '2px solid #ccc', 'borderRadius': '10px', 'marginBottom': '20px','overflow': 'hidden'}),
            
            html.Div(id="control-panel", children=[
                html.H3("Control Panel", style={'textAlign': 'center', 'marginBottom': '10px','font-weight': '600'}),
                
                html.Div([
                    html.Button("Assign Drones", id="assign-drones-btn", style={'margin': '10px','width': '300px',
                    'height': '50px',# Adjusted button width
                    'border-radius': '8px',
                    'font-weight': '600',
                    'font-size': '18px',
                    'display': 'inline-block'}),
                    html.Button("Take Off", id="take-off-btn", disabled=True, style={'margin': '10px','width': '300px',
                    'height': '50px',# Adjusted button width
                    'border-radius': '8px',
                    'font-weight': '600',
                    'font-size': '18px',
                    'display': 'inline-block'})
                ], style={'textAlign': 'center', 'marginBottom': '20px'}),
                
            dash_table.DataTable(
                id="drone-info-table", 
                columns=[
                    {"name": "Drone_ID", "id": "Drone_ID"},
                    {"name": "Location", "id": "Location"},
                    {"name": "Remaining_Distance", "id": "Remaining_Distance"},
                    {"name": "Estimated_Time", "id": "Estimated_Time"}
                ],
                style_cell={
                    'textAlign': 'center',
                    'padding': '10px',
                    'backgroundColor': '#1e1e1e',  # Dark background for cells
                    'color': '#ffffff',  # White text color
                    'font-weight': '500',
                    'font-size': '16px',
                    'fontFamily': 'Arial, sans-serif',
                    'border': '1px solid #444444'  # Dark border
                },
                style_header={
                    'backgroundColor': '#2a2a2a',  # Slightly darker header background
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    'font-weight': '600',
                    'font-size': '18px',
                    'fontFamily': 'Arial, sans-serif',
                    'color': '#f2f2f2',  # Light text color for headers
                    'border': '1px solid #444444'
                },
                style_table={
                    'width': '80%', 
                    'margin': 'auto',
                    'font-weight': '500',
                    'font-size': '16px',
                    'fontFamily': 'Arial, sans-serif',
                    'backgroundColor': '#1e1e1e',  # Table background to match dark mode
                },
                style_data_conditional=[
                    {
                        'if': {'state': 'active'},  # Highlight on hover or active cell
                        'backgroundColor': '#444444',
                        'font-weight': '500',
                        'font-size': '16px',
                        'fontFamily': 'Arial, sans-serif',
                        'border': '1px solid #555555'
                    }
                ]
            )
            ], style={'backgroundColor': '#121212','borderRadius': '10px', 
                      'padding': '20px', 'width': '100%', 'margin': 'auto'}),
            
            dcc.Interval(id="update-interval", interval=1000, n_intervals=0)
        ], style={'width': '100%', 'height': '100vh', 'backgroundColor': '#121212','display': 'block', 'margin-top': '20px'})

        
        
    
   
# Define the app layout with tabs
app.layout = dbc.Container([
    
    html.Div(style={'fontFamily': 'Orbitron, system-ui'}, children=[
        html.Div([
            html.Img(
            src="https://raw.githubusercontent.com/mjsonu/captcha_gen/main/ALERTIFY.png",  # Replace with your image URL
            style={
                'float': 'left',  # Align image to the left
                'margin-right': '10px',  # Add some space between the image and text or other content
                'margin-bottom': '20px',
                'margin-top': '20px'
            }
        )],
            style={
                'color': '#ffffff', 
                'font-size': '2rem',
                'font-weight': '700',  # Specify font weight here
                'padding': '1rem',
                'margin-bottom': '5rem',
            }
        ),
    ]),
    
    dcc.Tabs([
        dcc.Tab(
            label='DISASTER ANALYSIS', 
            children=[
                #################
                dbc.Container([
                    # Twitter Analysis Section
                    html.Div([
                        html.Div([
                            html.Button("Start Twitter Analysis", id="start-twitter", n_clicks=0, style={
                                'width': '20%', 'height': '50px', 'backgroundColor': '#03DAC6',
                                'border-radius': '8px', 'border': 'none', 'font-size': '18px',
                                'font-weight': '700',
                            })
                        ], style={'border-radius': '10px', 'width': '100%','margin-bottom': '10px'}),
                        html.Div([
                            html.Div(
                                children=[
                                    html.Video(
                                        controls=True,
                                        autoPlay=True,
                                        muted=True,
                                        loop=True,
                                        children=[
                                            html.Source(src='assets/twitter_recording.mp4', type='video/mp4')
                                        ],
                                        style={'width': '100%', 'height': '100%','objectFit': 'contain','border-radius': '8px'}
                                    )
                                ]
                            )
                        ], style={'padding': '10px', 'margin-right': '5px', 'margin-bottom': '10px',
                                'backgroundColor': '#2f2f2f', 'border-radius': '8px', 'width': '43%', 'float': 'left','height': '400px'}),
                        html.Div([
                            html.H6("Top 5 News From NDTV Twitter Channel",
                                    style={'font-size': '18px', 'font-weight': '700', 'color': 'white', 'padding': '10px'}
                            ),
                            html.Div(id="twitter-output", style={'backgroundColor': '#A9A9A9', 'height': 'auto', 'margin-top': '5px',
                                                                'border-radius': '8px','margin-bottom': '5px'})
                        ], style={'padding': '10px', 'margin-bottom': '10px', 'backgroundColor': '#2f2f2f',
                                'border-radius': '8px', 'width': '56%', 'float': 'right','height': '400px','overflow-y': 'scroll'}),
                    ], style={'width': '100%', 'height': '500px', 'float': 'center', 'margin-bottom': '10px'}),  # Left side

                    # Facebook Analysis Section
                    html.Div([
                        html.Div([
                            html.Button("Start Facebook Analysis", id="start-facebook", n_clicks=0, style={
                                'width': '20%', 'height': '50px', 'backgroundColor': '#BB86FC',
                                'border-radius': '8px', 'border': 'none', 'font-size': '18px',
                                'font-weight': '700',
                            })
                        ], style={'padding': '5px', 'border-radius': '10px', 'width': '100%','margin-bottom': '5px'}),
                       html.Div([
                            html.Div(
                                children=[
                                    html.Video(
                                        controls=True,
                                        autoPlay=True,
                                        muted=True,
                                        loop=True,
                                        children=[
                                            html.Source(src='assets/fb.mp4', type='video/mp4')
                                        ],
                                        style={'width': '100%', 'height': '100%','objectFit': 'contain','border-radius': '8px'}
                                    )
                                ]
                            )
                        ], style={'padding': '10px', 'margin-right': '5px', 'margin-bottom': '10px',
                                'backgroundColor': '#2f2f2f', 'border-radius': '8px', 'width': '43%', 'float': 'left','height': '400px'}),
                        html.Div([
                            html.H6("Facebook Top 10 News in India",
                                    style={'font-size': '18px', 'font-weight': '700', 'color': 'white', 'padding': '10px'}
                            ),
                            html.Div(id="facebook-output", style={'backgroundColor': '#A9A9A9', 'height': 'auto', 'margin-top': '5px',
                                                                'border-radius': '8px','margin-bottom': '5px'})
                        ], style={'padding': '10px', 'margin-bottom': '10px', 'backgroundColor': '#2f2f2f',
                                'border-radius': '8px', 'width': '56%', 'float': 'right','height': '400px','overflow-y': 'scroll'}),
                    ], style={'width': '100%', 'height': '500px', 'float': 'center', 'margin-bottom': '10px'}),
                    
                    # Instagram Analysis Section
                    html.Div([
                        html.Div([
                            html.Button("Start Instagram Analysis", id="start-instagram", n_clicks=0, style={
                                'width': '20%', 'height': '50px', 'backgroundColor': '#03DAC6',
                                'border-radius': '8px', 'border': 'none', 'font-size': '18px',
                                'font-weight': '700',
                            })
                        ], style={'border-radius': '10px', 'width': '100%','margin-bottom': '8px'}),
                        html.Div([
                            html.H6("Instagram Top 5 Posts from Team Mates Account",
                                    style={'font-size': '18px', 'font-weight': '700', 'color': 'white', 'padding': '10px'}
                            ),
                            html.Div(id="instagram-output", style={'backgroundColor': '#A9A9A9', 'height': 'auto', 'margin-top': '5px',
                                                                'border-radius': '8px','margin-bottom': '5px'})
                        ], style={'padding': '5px', 'margin-left': '10px', 'margin-bottom': '10px', 'backgroundColor': '#2f2f2f',
                                'border-radius': '8px', 'width': '100%', 'float': 'right','height': '446px','overflow-y': 'scroll'}),
                    ], style={'width': '100%', 'height': '500px', 'float': 'center', 'margin-bottom': '10px'}),
                    
                ], fluid=True, style={'backgroundColor': '#121212','margin-top':'20px'}) 
             
            ],
            style={
                'height': '50px', 'width': '300px', 'padding': '10px',
                'font-size': '18px',
                'font-weight': '700',
                'backgroundColor': '#323332', 
                'color': '#FFFFFF', 'border': 'none', 
                'borderRadius': '5px',
                'marginLeft': '5px', 'align-items': 'center',  # Vertical center
                        'justify-content': 'center', 
                'marginRight': '5px'
            },
            selected_style={
                'height': '50px', 'width': '300px', 'padding': '10px',
              'font-size': '18px',
                'font-weight': '700',
                'backgroundColor': '#FFFFFF', 
                'color': '#000000',  'border': 'none', 
                'borderRadius': '5px',
                'marginLeft': '5px', 'align-items': 'center',  # Vertical center
                        'justify-content': 'center', 
                'marginRight': '5px'
            }
        ),
        dcc.Tab(
            label='VIDEO ANALYSIS', 
            children=create_tab_layout('Video'),
         style={
              'height': '50px', 'width': '250px', 'padding': '10px',
                'font-size': '18px',
                'font-weight': '700',
                'backgroundColor': '#323332', 'border': 'none', 
                'color': '#FFFFFF',  'align-items': 'center',  # Vertical center
                        'justify-content': 'center', 
                'borderRadius': '5px',
               'marginLeft': '5px',
                'marginRight': '5px'
            },
            selected_style={
                 'height': '50px', 'width': '250px', 'padding': '10px',
               'font-size': '18px',
                'font-weight': '700',
                'backgroundColor': '#FFFFFF', 'border': 'none', 
                'color': '#000000',  'align-items': 'center',  # Vertical center
                        'justify-content': 'center', 
                'borderRadius': '5px',
                'marginLeft': '5px',
                'marginRight': '5px'
            }
        ),
        
        dcc.Tab(
            label='AUDIO ANALYSIS', 
            children=create_tab_layout('audio'),
         style={
              'height': '50px', 'width': '250px', 'padding': '10px',
                'font-size': '18px',
                'font-weight': '700',
                'backgroundColor': '#323332', 'border': 'none', 
                'color': '#FFFFFF',  'align-items': 'center',  # Vertical center
                        'justify-content': 'center', 
                'borderRadius': '5px',
                'marginLeft': '5px',
                'marginRight': '5px'
            },
            selected_style={
                 'height': '50px', 'width': '250px', 'padding': '10px',
               'font-size': '18px',
                'font-weight': '700',
                'backgroundColor': '#FFFFFF', 'border': 'none', 
                'color': '#000000',  'align-items': 'center',  # Vertical center
                        'justify-content': 'center', 
                'borderRadius': '5px',
               'marginLeft': '5px',
                'marginRight': '5px'
            }
        ),
        dcc.Tab(
            label='GEOGRAPHICAL INFORMATION', 
            children=create_tab_layout('Flood'),
         style={
              'height': '50px', 'width': '300px', 'padding': '10px',
               'font-size': '18px',
                'font-weight': '700',
                'backgroundColor': '#323332', 'border': 'none', 
                'color': '#FFFFFF',  'align-items': 'center',  # Vertical center
                        'justify-content': 'center', 
                'borderRadius': '5px',
                'marginLeft': '5px',
                'marginRight': '5px'
            },
            selected_style={
                 'height': '50px', 'width': '300px', 'padding': '10px',
              'font-size': '18px',
                'font-weight': '700',
                'backgroundColor': '#FFFFFF', 'border': 'none', 
                'color': '#000000',  'align-items': 'center',  # Vertical center
                        'justify-content': 'center', 
                'borderRadius': '5px',
                'marginLeft': '5px',
                'marginRight': '5px'
            }
        ),
        dcc.Tab(
            label='SOS', 
            children=create_tab_layout('sos'),
         style={
              'height': '50px', 'width': '150px', 'padding': '10px',
               'font-size': '18px',
                'font-weight': '700',
                'backgroundColor': '#323332', 'border': 'none', 
                'color': '#FFFFFF',  'align-items': 'center',  # Vertical center
                        'justify-content': 'center', 
                'borderRadius': '5px',
                'marginLeft': '5px',
                'marginRight': '5px'
            },
            selected_style={
                 'height': '50px', 'width': '150px', 'padding': '10px',
                'font-size': '18px',
                'font-weight': '700',
                'backgroundColor': '#FFFFFF', 'border': 'none', 
                'color': '#000000',  'align-items': 'center',  # Vertical center
                        'justify-content': 'center', 
                'borderRadius': '5px',
                'marginLeft': '5px',
                'marginRight': '5px'
            }
        ),
        dcc.Tab(
            label='JATAYU', 
            children=create_tab_layout('Drone'),
         style={
              'height': '50px', 'width': '250px', 'padding': '10px',
                'font-size': '18px',
                'font-weight': '700',
                'backgroundColor': '#323332', 'border': 'none', 
                'color': '#FFFFFF',  'align-items': 'center',  # Vertical center
                        'justify-content': 'center', 
                'borderRadius': '5px',
               'marginLeft': '5px',
                'marginRight': '5px'
            },
            selected_style={
                 'height': '50px', 'width': '250px', 'padding': '10px',
               'font-size': '18px',
                'font-weight': '700',
                'backgroundColor': '#FFFFFF', 'border': 'none', 
                'color': '#000000',  'align-items': 'center',  # Vertical center
                        'justify-content': 'center', 
                'borderRadius': '5px',
                'marginLeft': '5px',
                'marginRight': '5px'
            }
        )
    ], style={
        'height': '50px',  # Height for the entire tab bar
        'width': '100%',
        'font-size': '18px',
        'font-weight': '700',
        'marginBottom': '10px','align-items': 'center',  # Vertical center
        'justify-content': 'center'
    })
],style={'backgroundColor': '#121212','height': '100vh',  # Ensure it covers the full viewport height
        'width': '100%'}, fluid=True)

# Callback to update the Earthquake tab graphs and info based on table row selection
@app.callback(
    Output('earthquake-map', 'figure'),
    Output('selected-earthquake-info', 'children'),
    [Input(f'earthquake-row-{i}', 'n_clicks') for i in range(len(top_5_earthquakes))]
)
def update_earthquake_tab(*args):
    template = "cyborg"
    ctx = dash.callback_context

    fig = px.scatter_mapbox(
        gdf, lat=gdf.geometry.y, lon=gdf.geometry.x,
        hover_name="place", hover_data=["mag", "time"],
        color="mag", size="mag",
        color_continuous_scale=px.colors.cyclical.IceFire,
        size_max=15, zoom=1,
        title="Earthquake Magnitude",  # Title as a string
    )
    fig.update_layout(
       title={
            'text': "USGS Global Earthquake Tracker",  # Title text
            'x': 0.5,  # Center the title
            'xanchor': 'center',  # Align the title center horizontally
            'yanchor': 'top',  # Align the title to the top vertically
            'font': {
                'size': 20,  # Font size
                'weight': 700,  # Font weight
                'color': 'white',  # Font color
                'family': 'Arial, sans-serif'  # Font family
            },
            'pad': {
                't': 5,  # Padding at the top
                'b': 10,  # Padding at the bottom
                'l': 10,  # Padding at the left
                'r': 10   # Padding at the right
            }
        },
        mapbox_style="carto-positron",
        template=template,
        coloraxis_colorbar=dict(
            title="Magnitude",
            tickvals=[0, 1, 2, 3, 4, 5],
            ticktext=["0", "1", "2", "3", "4", "5"]
        ),
        margin=dict(l=20, r=20, t=80, b=20)
    )

    selected_info = ""
    if ctx.triggered:
        for i in range(len(top_5_earthquakes)):
            if ctx.triggered[0]['prop_id'] == f"earthquake-row-{i}.n_clicks":
                selected_location = top_5_earthquakes.iloc[i]
                selected_lat = selected_location.geometry.y
                selected_lon = selected_location.geometry.x

                # Highlight the selected earthquake on the map
                fig.add_scattermapbox(
                    lat=[selected_lat], lon=[selected_lon],
                    mode='markers',
                    marker=dict(size=25, color='#09FF08'),
                    name="Selected"
                )

                # Display the selected location info including lat and lon
                selected_info = dbc.Alert([
                    html.H5(f"Location: {selected_location['place']}", 
                            style={'font-size': '18px', 'font-weight': '700', 'color': 'white'}),
                    dcc.Markdown(f"""
                        **Latitude**: {selected_lat:.4f}  
                        **Longitude**: {selected_lon:.4f}  
                        **Magnitude**: {selected_location['mag']}  
                        **Time**: {selected_location['time']}
                        """, style={'font-size': '16px', 'color': 'white'}),
                ], style={'backgroundColor': '#2c2c2c', 'border-radius': '8px','margin-top':'8px','padding-bottom':'0.5px','height':'190px'})



                break

    return fig, selected_info

# Function to save uploaded video file to a temporary file
def save_uploaded_file(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    with tempfile.NamedTemporaryFile(delete=False, suffix=filename) as temp_file:
        temp_file.write(decoded)
        return temp_file.name

# Function to process the video and apply YOLO model
def process_video(video_path):
    print("PROCESSING.....")
    global video_capture, latest_detection_frame, latest_thermal_frame, latest_magma_frame
    global person_count_data, chunked_person_count_data_10, object_detection_counts
    global processing_completed

    video_capture = cv2.VideoCapture(video_path)
    person_count_data = []
    chunked_person_count_data_10 = []
    object_detection_counts = {}

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        results = yolo_model.predict(frame, conf=0.4)

        annotated_frame = results[0].plot()
        latest_detection_frame = annotated_frame
        latest_thermal_frame = cv2.applyColorMap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
        latest_magma_frame = cv2.applyColorMap(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_HOT)

        detections = results[0].boxes.data.cpu().numpy()
        person_count = sum(1 for det in detections if det[-1] == 0)
        person_count_data.append(person_count)

        for detection in detections:
            class_id = int(detection[-1])
            class_name = yolo_model.names[class_id]
            object_detection_counts[class_name] = object_detection_counts.get(class_name, 0) + 1

    processing_completed = True
    video_capture.release()

# Callback to handle video upload, processing, and display
@app.callback(
    Output('video-display', 'src'),
    Output('dummy-div', 'children'),
    Input('upload-video', 'contents'),
    State('upload-video', 'filename')
)
def upload_video(contents, filename):
    if contents is not None:
        global video_file_path
        video_file_path = save_uploaded_file(contents, filename)
        return f'/videos/{os.path.basename(video_file_path)}', ''
    return '', ''

# Serve video files for display
@app.server.route('/videos/<path:path>')
def serve_video(path):
    
    return send_from_directory(os.path.dirname(video_file_path), path)

# Callback to start video processing thread
@app.callback(
    Output('start-processing', 'disabled'),
    Input('start-processing', 'n_clicks')
)
def start_video_processing(n_clicks):
    global processing_thread, processing_started
    
    if n_clicks > 0 and not processing_started:
        processing_thread = Thread(target=process_video, args=(video_file_path,))
        processing_thread.start()
        processing_started = True
        return True
    return False


# Callback to update live object detection frames and graphs
mohit_flag=0
chunked_person_count_fig = go.Figure()
object_detection_pie_chart = go.Figure()
@app.callback(
    Output('live-detection', 'src'),
    Output('live-thermal-mapping', 'src'),
    Output('live-magma-mapping', 'src'),
    Output('person-count-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)

def update_live_frames(n):
    global latest_detection_frame, latest_thermal_frame, latest_magma_frame
    global person_count_data, chunked_person_count_data_10, object_detection_counts

    detection_img_src = ''
    thermal_img_src = ''
    magma_img_src = ''
    person_count_fig = go.Figure()
    global chunked_person_count_fig 
    global object_detection_pie_chart

    if latest_detection_frame is not None:
        _, buffer = cv2.imencode('.jpg', latest_detection_frame)
        detection_img_src = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')

    if latest_thermal_frame is not None:
        _, buffer = cv2.imencode('.jpg', latest_thermal_frame)
        thermal_img_src = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')

    if latest_magma_frame is not None:
        _, buffer = cv2.imencode('.jpg', latest_magma_frame)
        magma_img_src = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')

    if person_count_data:
        # Create the scatter plot with custom line and marker styles
        person_count_fig = go.Figure(
            data=[go.Scatter(
                y=person_count_data, 
                mode='lines+markers', 
                name='Person Count',
                line=dict(color='#05FF00', width=1, dash='dash'),  # Customize line color, width, and style
                marker=dict(size=5, color='red', symbol='circle')   # Customize marker size, color, and shape
            )],
        )

        # Update layout with styling options
        person_count_fig.update_layout(
            title=dict(
                text="Person Count Over Time",
                font=dict(size=24, color='#03DAC6',weight=700),
                x=0.5,  # Center the title
                xanchor='center'
            ),
            xaxis_title="Frame",
            yaxis_title="Person Count",
            plot_bgcolor='#121212',  # Set background color of the plot area
            paper_bgcolor='#323332 ',    # Set background color outside the plot area
            font=dict(size=14, color='white',weight=500),        # Set global font size
            xaxis=dict(
                showgrid=True,         # Enable gridlines
                gridcolor='#1D1E1E',      # Set gridline color
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#1D1E1E'
            ),
           
        )

        # Chunk the person count data for further use
        chunk_size = 10
        chunked_data = [sum(person_count_data[i:i + chunk_size]) for i in range(0, len(person_count_data), chunk_size)]
        chunked_person_count_data_10 = chunked_data
    

    return detection_img_src, thermal_img_src, magma_img_src, person_count_fig

@app.callback(
    Output('chunked-person-count-heatmap', 'figure'),
    Output('object-detection-pie-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_end_of_analysis(n):
    global processing_completed, chunked_person_count_data_10, object_detection_counts

    global chunked_person_count_fig
    global object_detection_pie_chart

    if processing_completed:
        if chunked_person_count_data_10:
            chunked_person_count_fig = go.Figure(data=[go.Heatmap(z=[chunked_person_count_data_10], colorscale='Viridis')])
            chunked_person_count_fig.update_layout(title="Person Count Heatmap (10 Frame Chunks)", xaxis_title="Chunk Index", yaxis_title="Person Count")

        if object_detection_counts:
            object_detection_pie_chart = go.Figure(data=[go.Pie(labels=list(object_detection_counts.keys()), values=list(object_detection_counts.values()))])
            object_detection_pie_chart.update_layout(title="Object Detection Counts")

    return chunked_person_count_fig, object_detection_pie_chart



# New callback to update the markers_cc, lines, and information based on the selected marker
# Callback to update the markers_cc, lines, and information based on the selected marker
@app.callback(
    [Output('shelter-info', 'children'),
     Output('shelter-marker-layer', 'children'),
     Output('shelter-line-layer', 'children')],
    [Input({'type': 'shelter-marker', 'index': dash.dependencies.ALL}, 'n_clicks')],
    [dash.dependencies.State({'type': 'shelter-marker', 'index': dash.dependencies.ALL}, 'id')]
)
def display_shelter_info(n_clicks, ids):
    if not n_clicks or all(click is None for click in n_clicks):
        markers_list = [
            dl.Marker(
                position=marker["position"],
                id=marker["id"],
                icon={
                    "iconUrl": DEFAULT_ICON,  # Default marker icon
                    "iconSize": [30, 30],
                    "iconAnchor": [12, 24],
                },
                children=[dl.Tooltip(marker["name"])]
            ) for marker in markers_cc
        ]
        
        affected_marker = dl.Marker(
            position=affected_location,
            id="affected-location",
            icon={
                "iconUrl": EFFECTED_ICON,
                "iconSize": [30, 30],
                "iconAnchor": [12, 24],
            },
            children=[dl.Tooltip("Affected Location")]
        )
        
        return "Click on a marker to see the shelter information.", markers_list + [affected_marker], []

    idx = next((i for i, click in enumerate(n_clicks) if click), None)
    if idx is not None:
        shelter = shelter_df.iloc[ids[idx]['index']]
        distances = [(i, great_circle(affected_location, marker["position"]).kilometers) for i, marker in enumerate(markers_cc)]
        nearest_shelters = sorted(distances, key=lambda x: x[1])[:5]
        
        
        
        lines = [
            dl.Polyline(
                positions=[affected_location, markers_cc[i]["position"]],
                dashArray="5, 10",  # 5px dash, 10px gap
                color="green",
                weight=2
            )
            for i, _ in nearest_shelters
        ]

        markers_list = []
        for i, marker in enumerate(markers_cc):
            icon_url = SELECTED_ICON if i == idx else DEFAULT_ICON
            markers_list.append(
                dl.Marker(
                    position=marker["position"],
                    id=marker["id"],
                    icon={
                        "iconUrl": icon_url,
                        "iconSize": [30, 30],
                        "iconAnchor": [12, 24],
                    },
                    children=[dl.Tooltip(marker["name"])]
                )
            )
        
        markers_list.append(
            dl.Marker(
                position=affected_location,
                id="affected-location",
                icon={
                    "iconUrl": EFFECTED_ICON,
                    "iconSize": [30, 30],
                    "iconAnchor": [12, 24],
                },
                children=[dl.Tooltip("Affected Location")]
            )
        )

        shelter_info = [
            html.B(shelter['Name']),
            html.Br(),
            f"Location: {shelter['Location']}",
            html.Br(),
            f"Coordinates: ({shelter['Latitude ']}, {shelter['Longitude']})",
            html.Br(),
            html.Hr(),
            html.P("Nearest 5 Relief Centers:",style={'font-size': '18px','font-weight': '700','color':'white'}),
            html.Ul([html.Li(f"{shelter_df.iloc[i]['Name']} - {round(dist, 2)} km") for i, dist in nearest_shelters])
        ]

        return shelter_info, markers_list, lines

    return "No shelter selected.", markers_cc, []



@app.callback(
    dash.dependencies.Output('sos-btn-1', 'children'),
    [dash.dependencies.Input('sos-btn-1', 'n_clicks')]
)
def sos_button_1(n_clicks):
    if n_clicks > 0:
        send_sms(phone_numbers_police, "SOS Alert: Emergency To Police Station")
    return 'SOS Police Station'

@app.callback(
    dash.dependencies.Output('sos-btn-2', 'children'),
    [dash.dependencies.Input('sos-btn-2', 'n_clicks')]
)
def sos_button_2(n_clicks):
    if n_clicks > 0:
        send_sms(phone_numbers_hospital, "SOS Alert: Emergency To Hospital")
    return 'SOS Hospital'

@app.callback(
    dash.dependencies.Output('sos-btn-3', 'children'),
    [dash.dependencies.Input('sos-btn-3', 'n_clicks')]
)
def sos_button_3(n_clicks):
    if n_clicks > 0:
        send_sms(phone_numbers_costal, "SOS Alert: Emergency To Coastal Guard")
    return 'SOS Costal Gaurd'

####################################################################
#BAAP KO CALL KIYA GAYA HAI
'''
@app.callback(
    [Output('output-div', 'children'),
     Output('table-div', 'children'),
     Output('details-div', 'children')],
    Input('run-button', 'n_clicks')
)
def update_output(n_clicks):
    if n_clicks > 0:
        # Fetch news and create CSV
        fetched_csv = fetch_news_and_create_csv()
        
        # Run classification process
        df = run_classification(fetched_csv)
        
        # Generate a table to display classified data
        rows = []
        for i in range(len(df)):
            row_style = {'backgroundColor': '#f8d7da'} if df.loc[i, 'Is_Disaster'] == 'Yes' else {'backgroundColor': '#ffffff'}
            rows.append(
                html.Tr([
                    html.Td(df.iloc[i]['news'], style={'padding': '10px', 'border': '1px solid #ddd'})  # Only show 'news' column
                ], style=row_style, id=f"row-{i}")
            )
        
        table = html.Table([
            html.Thead(html.Tr([html.Th('News', style={
                'padding': '10px', 
                'border': '1px solid #ddd', 
                'backgroundColor': '#007bff', 
                'color': '#ffffff',
                'fontSize': '16px',
                'textAlign': 'left'
            })])),
            html.Tbody(rows)
        ], style={
            'width': '100%', 
            'margin-left': '10px',
            'margin-rigth': '10px',
            'borderCollapse': 'collapse', 
            'border': '1px solid #ddd',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '14px',
            'border-radius': '8px',
            'display': 'inline-block'
        })
        
        # Generate details section
        details = []
        for i in range(len(df)):
            if df.loc[i, 'Is_Disaster'] == 'Yes':
                details.append(
                    html.Div([
                        html.H2(f"Details for News ID {i + 1}", style={'marginTop': '20px','color': '#FF0000'}),
                        html.H5(f"News: {df.loc[i, 'news']}"),
                        html.H5(f"Disaster Class: {df.loc[i, 'Disaster_Class']}"),
                        html.H5(f"Location: {df.loc[i, 'Location']}")
                    ], style={'padding': '20px', 'border': '1px solid #ddd', 'marginTop': '10px', 'backgroundColor': '#FF9898','border-radius': '8px','margin-left': '20px',
                    'margin-rigth': '20px',})
                )
        
        return (
            'Classification Complete!',
            table,
            html.Div(details)
        )
    else:
        return 'Click the button to start classification', None, None
'''
    
#for audio 
@app.callback(
    [
     Output('transcript', 'children'),
     Output('audio-plot', 'src'),
     Output('audio-player', 'src')],
    [Input('process-audio-button', 'n_clicks')],
    [State('upload-audio-video', 'contents'),
     State('upload-audio-video', 'filename')]
)
def process_audio(n_clicks, contents, filename):
    if n_clicks > 0 and contents:
        # Decode the uploaded file and process it as in your second code
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Save the uploaded file to a temporary location
        video_path = os.path.join(filename)  
        print(filename)
        print(video_path)
        with open(video_path, 'wb') as f:
            f.write(decoded)
        
        # Extract audio from video, process it, and return the results
        audio_path = 'output_audio.mp3'
        extract_sound_from_videio.extract_audio(video_path, audio_path)
        audio_path = mp3towav.convert_to_wav(audio_path)
        
        category, image_base64 = speech.predict_and_plot(audio_path)
        language, transcript, translated_text = transcription.process_audio(audio_path)

        predicted_sound =  html.Div([html.P(f"Predicted sound: {category}")])
        transcript_text = html.Div([
            html.P(f"Predicted sound: {category}",style={'marginTop': '5px'}),
            html.P(f"Transcript: {transcript}"),
            html.P(f"Translated Text: {translated_text}")
        ])
        image_src = f"data:image/png;base64,{image_base64}"

        # Convert the MP3 file to base64 so it can be used in the audio player
        with open(audio_path, 'rb') as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode()

        audio_src = f"data:audio/mp3;base64,{audio_base64}"

        return transcript_text, image_src, audio_src
    
    return "", "", ""



########### NEW UI
# Function to run Python scripts and return CSV content
def run_script(script_name, output_csv):
    try:
        subprocess.run(['python', script_name], check=True)
        df = pd.read_csv(output_csv)
        return df.to_dict('records'), df.columns
    except Exception as e:
        return f"Error: {str(e)}", []

# Helper function to create table with disaster data
def create_table(data):
    # Define rows to hold table data
    rows = []

    # Iterate over each row in the data
    for row in data:
        if row['Is_Disaster'] == "Yes":
            # Apply inline styles for disaster rows
            disaster_info = f"Class: {row['Disaster_Class']}, Location: {row['Location']}"
            rows.append(html.Tr([
                html.Td(row['news'], style={'padding': '12px', 'border': '1px solid #ddd','font-size': '18px',
                'font-weight': '500'}),
                html.Td(disaster_info, style={'padding': '12px', 'border': '1px solid #ddd','font-size': '18px',
                'font-weight': '500'})
            ], style={'backgroundColor': 'red', 'color': 'white'}))  # Inline CSS for red background
        else:
            # No highlighting for non-disaster news, just regular styling
            rows.append(html.Tr([
                html.Td(row['news'], style={'padding': '12px', 'border': '1px solid #ddd','font-size': '18px',
                'font-weight': '500', 'color': 'white','backgroundColor': '#232323'}),
                html.Td('', style={'padding': '12px', 'border': '1px solid #ddd', 'color': 'white','backgroundColor': '#232323'})
            ]))

    # Build the table structure with two columns (news, disaster_info)
    return html.Table(
        # Add a header row with inline CSS
        [html.Tr([
            html.Th('News Headlines', style={'backgroundColor': '#121212', 'padding': '12px', 'border-bottom': '2px solid #000','font-size': '20px',
                'font-weight': '500','width': '70%','color':'white'}),
            html.Th('Disaster Info', style={'backgroundColor': '#121212', 'padding': '12px', 'border-bottom': '2px solid #000','font-size': '20px',
                'font-weight': '500','color':'white'})
        ])] + rows,
        style={'width': '100%', 'border-collapse': 'collapse', 'margin': '20px 0', 'font-size': '18px', 'text-align': 'left','border-radius': '8px',  # Rounded corners
        'overflow': 'hidden',  # Ensures the rounded corners are visible
        'border': '1px solid #ddd'}
    )


# Callbacks for each button
@app.callback(
    [Output('twitter-output', 'children')],
    [Input('start-twitter', 'n_clicks')],
    prevent_initial_call=True
)
def start_twitter_analysis(n_clicks):
    data, columns = run_script('twitter.py', 'twitter.csv')
    if isinstance(data, str):  # Error occurred
        return [data], None
    return [create_table(data)]

@app.callback(
    [Output('facebook-output', 'children')],
    [Input('start-facebook', 'n_clicks')],
    prevent_initial_call=True
)
def start_facebook_analysis(n_clicks):
    data, columns = run_script('fetchfb.py', 'fb.csv')
    if isinstance(data, str):  # Error occurred
        return [data]
    return [create_table(data)]

@app.callback(
    [Output('instagram-output', 'children')],
    [Input('start-instagram', 'n_clicks')],
    prevent_initial_call=True
)
def start_instagram_analysis(n_clicks):
    data, columns = run_script('fetch_insta.py', 'insta.csv')
    if isinstance(data, str):  # Error occurred
        return [data]
    return [create_table(data)]





# Inline CSS for animation (define keyframes)
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @keyframes scroll {
                0% { transform: translateX(100%); }  /* Start from right */
                100% { transform: translateX(-100%); }  /* End at left */
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Callback to fetch the latest news and update the ticker every hour
@app.callback(
    Output('news-ticker', 'children'),
    [Input('interval-component-ticker', 'n_intervals')]
)
def update_news_ticker(n):
    # Run the scraping function to fetch the latest news
    print("Fetch started")
    fetch_sachet_news()
    # Read the news from the CSV and display it in the ticker
    news = read_sachet_csv()
    print("Fetch Completed")
    return news





# missing person
@app.callback(
    Output('missing_person_reports_container', 'children'),
    Input('missing_person_reports_container', 'id')
)
def update_missing_person_reports(_):
    # Fetch missing person reports from the database
    reports = missing_person_collection.find()
    
    # Create a list of report entries
    report_entries = []
    for report in reports:
        image_url = None
        if 'image_id' in report:
            image_file = fs.get(report['image_id'])
            image = Image.open(io.BytesIO(image_file.read()))
            # Convert image to RGB if it's in RGBA mode
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            image_url = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
        
        report_entries.append(html.Div([
            html.Table([  # Creating the table structure
                html.Tr([  # First row with text on the left and image on the right
                    html.Td([  # Left side: Text details
                        html.H3(f"Name: {report.get('name', 'N/A')}", style={'font-size': '20px', 'font-weight': '700'}),
                        html.P(f"Age: {report.get('age', 'N/A')}", style={'font-size': '16px', 'font-weight': '500'}),
                        html.P(f"Last Seen: {report.get('last_seen', 'N/A')}", style={'font-size': '16px', 'font-weight': '500'}),
                        html.P(f"Birthmark: {report.get('birthmark', 'N/A')}", style={'font-size': '16px', 'font-weight': '500'}),
                        html.P(f"Contact: {report.get('contact', 'N/A')}", style={'font-size': '16px', 'font-weight': '500'})
                    ], style={'verticalAlign': 'top', 'paddingRight': '20px','paddingLeft': '20px','width':"60%"}),
                    
                    html.Td([  # Right side: Image
                        html.Img(src=image_url, style={'maxWidth': '200px', 'marginTop': '10px'}) if image_url else html.P("No image available.")
                    ], style={'verticalAlign': 'top','width':"40%",'textAlign': 'center'})
                ])
            ],style={'width':"100%"}),  # Bootstrap classes for styling the table
            
            html.Hr()
        ], style={'marginBottom': '20px'}))


    
    return report_entries


##########################DRONE JATAYU

@app.callback(
    Output('file-upload-info', 'children'),
    Input('upload-drones-data', 'contents'),
    Input('upload-locations-data', 'contents'),
    State('upload-drones-data', 'filename'),
    State('upload-locations-data', 'filename')
)
def handle_file_upload(drones_content, locations_content, drones_filename, locations_filename):
    global drones_data, locations_data

    if drones_content and drones_filename:
        drones_data = parse_data(drones_content, drones_filename)
    if locations_content and locations_filename:
        locations_data = parse_data(locations_content, locations_filename)

    return f"Uploaded files: {drones_filename} and {locations_filename}"

@app.callback(
    Output("marker-layer", "children"),
    Output("drone-layer", "children"),
    Output("path-layer", "children"),
    Output("drone-info-table", "data"),
    Output("take-off-btn", "disabled"),
    Input("assign-drones-btn", "n_clicks"),
    Input("take-off-btn", "n_clicks"),
    Input("update-interval", "n_intervals"),
    State("take-off-btn", "disabled"),
    State("assign-drones-btn", "n_clicks")
)
def manage_drones(assign_clicks, takeoff_clicks, n_intervals, takeoff_disabled, prev_assign_clicks):
    global drones_data, locations_data, assigned_drones, is_taking_off, static_markers

    ctx = callback_context

    # Check which button triggered the callback
    if not ctx.triggered:
        raise PreventUpdate

    drone_markers = []
    updated_paths = []
    drone_info = []

    # Handle assigning drones only if "Assign Drones" was clicked
    if ctx.triggered[0]['prop_id'] == 'assign-drones-btn.n_clicks':
        if drones_data.empty or locations_data.empty:
            raise PreventUpdate

        available_drones = drones_data[(drones_data["Status"] == "fit") & (drones_data["Charge"] >= 50)]
        base_location = locations_data[locations_data["Location"] == "Base_Station"].iloc[0]
        base_lat, base_lon = base_location["Latitude"], base_location["Longitude"]

        assigned_drones.clear()
        static_markers.clear()  # Clear any previous markers

        # Add base station marker
        base_marker = dl.Marker(
            position=[base_lat, base_lon],
            children=dl.Tooltip("Base Station"),
            icon=dict(iconUrl="https://raw.githubusercontent.com/mjsonu/TicTacToe_Game/662b6053ee4127e443a568ec1cf14bf12a7212e5/Pin_red.svg", iconSize=[25, 41])
        )
        static_markers.append(base_marker)

        for _, loc in locations_data[locations_data["Location"] != "Base_Station"].iterrows():
            if available_drones.empty:
                break

            drone = available_drones.iloc[0]
            available_drones = available_drones.iloc[1:]
            dest_lat, dest_lon = loc["Latitude"], loc["Longitude"]
            drone_distance = geodesic((base_lat, base_lon), (dest_lat, dest_lon)).kilometers

            assigned_drones[drone["Drone_ID"]] = {
                "Location": loc["Location"],
                "Base_Coords": (base_lat, base_lon),
                "Destination_Coords": (dest_lat, dest_lon),
                "Current_Coords": (base_lat, base_lon),
                "Remaining_Distance": drone_distance,
                "Speed": 100
            }

            # Add affected location marker
            destination_marker = dl.Marker(
                position=[dest_lat, dest_lon],
                children=dl.Tooltip(f"{loc['Location']}"),
                icon=dict(iconUrl="https://raw.githubusercontent.com/mjsonu/TicTacToe_Game/662b6053ee4127e443a568ec1cf14bf12a7212e5/Pin_green.svg", iconSize=[25, 41])
            )
            static_markers.append(destination_marker)

            # Calculate estimated time in hr:min:sec format
            estimated_time_hours = drone_distance / 100
            estimated_timedelta = timedelta(hours=estimated_time_hours)
            estimated_time_str = str(estimated_timedelta)

            # Add initial drone info
            drone_info.append({
                "Drone_ID": drone["Drone_ID"],
                "Location": loc["Location"],
                "Remaining_Distance": f"{drone_distance:.2f} km",
                "Estimated_Time": estimated_time_str
            })

            # Place drone at base
            drone_markers.append(dl.Marker(
                position=[base_lat, base_lon],
                children=dl.Tooltip(f"Drone {drone['Drone_ID']}"),
                icon=dict(iconUrl="https://raw.githubusercontent.com/mjsonu/TicTacToe_Game/4fb2afec418a7603d8e345868591af30f7120ca5/drone.svg", iconSize=[25, 41])
            ))

        # Enable "Take Off" button after assignment
        return static_markers, drone_markers, updated_paths, drone_info, False

    # Handle take off logic
    if ctx.triggered[0]['prop_id'] == 'take-off-btn.n_clicks':
        is_taking_off = True

    if is_taking_off and assigned_drones:
        for drone_id, drone_data in assigned_drones.items():
            start_coords = drone_data["Base_Coords"]
            end_coords = drone_data["Destination_Coords"]
            current_coords = drone_data["Current_Coords"]
            distance_covered = drone_data["Speed"] / 3600
            remaining_distance = geodesic(current_coords, end_coords).kilometers

            if remaining_distance <= distance_covered:
                new_coords = end_coords
                remaining_distance = 0
                estimated_time_str = "00:00:00"
            else:
                lat_diff = end_coords[0] - current_coords[0]
                lon_diff = end_coords[1] - current_coords[1]
                factor = distance_covered / remaining_distance
                new_coords = (current_coords[0] + lat_diff * factor, current_coords[1] + lon_diff * factor)
                estimated_time_hours = remaining_distance / 100
                estimated_timedelta = timedelta(hours=estimated_time_hours)
                estimated_time_str = str(estimated_timedelta)

            assigned_drones[drone_id]["Current_Coords"] = new_coords
            assigned_drones[drone_id]["Remaining_Distance"] = remaining_distance

            drone_markers.append(dl.Marker(
                position=new_coords,
                children=dl.Tooltip(f"Drone {drone_id}"),
                icon=dict(iconUrl="https://raw.githubusercontent.com/mjsonu/TicTacToe_Game/4fb2afec418a7603d8e345868591af30f7120ca5/drone.svg", iconSize=[25, 41])
            ))

            updated_paths.append(dl.Polyline(positions=[start_coords, new_coords, end_coords],
                                             color="black",  # Set path color to black
                                             dashArray="5, 5"))

            drone_info.append({
                "Drone_ID": drone_id,
                "Location": drone_data["Location"],
                "Remaining_Distance": f"{remaining_distance:.2f} km",
                "Estimated_Time": estimated_time_str
            })

        return static_markers, drone_markers, updated_paths, drone_info, takeoff_disabled

    raise PreventUpdate



@app.callback(
    Output('jatayu_frame-update', 'disabled'),
    Input('jatayu_start-button', 'n_clicks')
)
def jatayu_start_processing(n_clicks):
    if n_clicks > 0:
        jatayu_video_processor.jatayu_start_processing()
        return False
    return True

@app.callback(
    [Output('jatayu_original-video', 'src'),
     Output('jatayu_object-detection', 'src'),
     Output('jatayu_thermal-mapping', 'src'),
     Output('jatayu_magma-mapping', 'src')],
    Input('jatayu_frame-update', 'n_intervals')
)
def jatayu_update_frames(n_intervals):
    if jatayu_video_processor.jatayu_current_frame is None:
        return ['', '', '', '']
    
    def jatayu_encode_frame(frame):
        _, buffer = cv2.imencode('.jpg', frame)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"
    
    return [jatayu_encode_frame(jatayu_video_processor.jatayu_current_frame),
            jatayu_encode_frame(jatayu_video_processor.jatayu_detected_frame),
            jatayu_encode_frame(jatayu_video_processor.jatayu_thermal_frame),
            jatayu_encode_frame(jatayu_video_processor.jatayu_magma_frame)]



if __name__ == '__main__':
    app.run_server(debug=True)
