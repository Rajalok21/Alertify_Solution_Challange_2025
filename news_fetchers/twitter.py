from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager as CM
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import pandas as pd
import time
from datetime import datetime
import tensorflow as tf
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import translate
import os
import location_fetch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  
    words = text.split()
    filtered_words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(filtered_words)

cv = pickle.load(open(r'ml_models\vectorizer.pkl', 'rb'))
model = tf.keras.models.load_model(r'ml_models\model.keras')
loaded_pipeline = pickle.load(open(r'ml_models/disaster_classifier.pkl', 'rb'))
def classify_with_threshold(text, threshold=0.85):
    # Translate to English
    translated_text = translate.translate_to_english(text)
    
    # Preprocess
    transformed_sms = transform_text(translated_text)
    
    # Vectorize
    vector_input = cv.transform([transformed_sms])
    
    # Get prediction probabilities
    probas = model.predict(vector_input.toarray()) 
    
    # Apply the threshold
    is_disaster = 1 if probas[0] >= threshold else 0
    
    # Optionally use the loaded pipeline for additional classification if necessary
    disaster_class = loaded_pipeline.predict([translated_text])[0] if is_disaster == 1 else None
    
    # Example location fetch (assuming a function exists)
    location = location_fetch.loc_fetch2(text) if is_disaster == 1 else None
    
    return {
        'Is_Disaster': 'Yes' if is_disaster == 1 else 'No',
        'Disaster_Class': disaster_class,
        'Location': location
    }

def fetch_tweets(url, csv_filename):
    print("Fetching tweets from Twitter")
    chrome_options = Options()
    #chrome_options.add_argument("--headless")  # Run Chrome in headless mode
    #chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration
    #chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
    #chrome_options.add_argument("--window-size=1920x1080")  # Set the window size if needed

    service = Service(executable_path=CM().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)  # Pass the options here

    driver.get(url)
    time.sleep(5)
    
    for i in range(1):  # Adjust range for more scrolling if needed
        driver.execute_script("window.scrollBy(0,2000)")
        time.sleep(2)

    tweets_data = []
    tweet_elements = driver.find_elements(By.XPATH, '//article')

    max_tweets = 5
    tweet_count = 0

    for tweet in tweet_elements:
        if tweet_count >= max_tweets:
            break
        try:
            # Extract tweet text
            tweet_text = tweet.find_element(By.XPATH, './/div[@lang]').text

            # Extract tweet URL
            tweet_url = tweet.find_element(By.XPATH, './/a[contains(@href, "/status/")]').get_attribute('href')

            # Extract tweet time
            tweet_time_iso = tweet.find_element(By.XPATH, './/time').get_attribute('datetime')
            tweet_time = datetime.strptime(tweet_time_iso, "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d %H:%M:%S")

            # Try to find image links
            images = tweet.find_elements(By.XPATH, './/img')

            if len(images) >= 2:
                img_link = images[1].get_attribute('src')  # Get the second image link
            else:
                img_link = 'No image found'

            # Classify the tweet
            result = classify_with_threshold(tweet_text, threshold=0.8)

            # Store tweet data along with classification
            tweets_data.append({
                'news': tweet_text,
                'media_link': img_link,
                'timespan': tweet_time,
                'for twitter video': tweet_url,
                'Is_Disaster': result['Is_Disaster'],
                'Disaster_Class': result['Disaster_Class'],
                'Location': result['Location']
            })
            tweet_count += 1
        except Exception as e:
            print(f"Error extracting tweet: {e}")

    # Save to a CSV file with specified column names
    df = pd.DataFrame(tweets_data)
    df.to_csv(csv_filename, index=False, columns=['news', 'media_link', 'timespan', 'for twitter video', 'Is_Disaster', 'Disaster_Class', 'Location'])

    # Clean up
    driver.quit()

if __name__ == "__main__":
    url = "https://x.com/IndiaToday"
    csv_filename = "twitter.csv"
    fetch_tweets(url, csv_filename)
