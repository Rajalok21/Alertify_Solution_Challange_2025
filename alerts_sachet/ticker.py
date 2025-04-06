import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import csv
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager as CM
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

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

    sachet = []
    headings = driver.find_elements(By.XPATH, '//*[@id="style-1"]/div/div[1]')  
    elements = driver.find_elements(By.XPATH, '//*[@id="style-1"]/div/div[2]')

    min_length = min(len(headings), len(elements))
    for i in range(min_length):
        h = headings[i].text
        e = elements[i].text
        sachet.append([h, e])

    driver.quit()

    # Save the top 10 news items to a CSV file
    top_sachet = sachet[:10]
    with open('sachet_data_top10.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Heading', 'Element'])  
        writer.writerows(top_sachet)

# Read the CSV file and return the news data
def read_sachet_csv(file_path='sachet_data_top10.csv'):
    try:
        df = pd.read_csv(file_path)
        news = [f"{row['Heading']}: {row['Element']}" for _, row in df.iterrows()]
        return " | ".join(news)  # Join all news into a single string for the ticker
    except FileNotFoundError:
        return "No news data available."

# Create a Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
    'border-radius': '8px'# Positioning for animation
    
}

# Keyframes style for scrolling effect
ticker_animation = {
    'whiteSpace': 'nowrap',      # Prevent text wrapping
    'animation': 'scroll 100s linear infinite',  # Animation duration and infinite loop
    'display': 'inline-block',   # Ensure text stays in a single line
}

# Layout of the app
app.layout = dbc.Container([
    html.Div(
        html.Div(
            id='news-ticker',  # Div to hold the dynamically updated news ticker
            style=ticker_animation
        ),
        style=ticker_container_style  # Outer container to control scrolling
    ),
    
    # Interval component to trigger the news fetching every hour (3600 seconds)
    dcc.Interval(
        id='interval-component',
        interval=3600*1000,  # in milliseconds (3600 seconds = 1 hour)
        n_intervals=0  # Start with 0 intervals
    )
], fluid=True)

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
    [Input('interval-component', 'n_intervals')]
)
def update_news_ticker(n):
    # Run the scraping function to fetch the latest news
    fetch_sachet_news()

    # Read the news from the CSV and display it in the ticker
    news = read_sachet_csv()
    return news

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
