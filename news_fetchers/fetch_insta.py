import instaloader
import csv
import pandas as pd
import tensorflow as tf
import os
import time
import pickle
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import news_fetchers.translate as translate
import news_fetchers.location_fetch as location_fetch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Instaloader
L = instaloader.Instaloader()

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  
    words = text.split()
    filtered_words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(filtered_words)

# Load the pre-trained vectorizer and model
cv = pickle.load(open(r'ml_models\vectorizer.pkl', 'rb'))
model = tf.keras.models.load_model(r'ml_models\model.keras')
loaded_pipeline = pickle.load(open(r'ml_models/disaster_classifier.pkl', 'rb'))


def classify_with_threshold(text, threshold):
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

def get_posts_data(profile, count):
    posts_data = []
    for i, post in enumerate(profile.get_posts()):
        if i >= count:
            break
        # Determine if the post is a video or image
        if post.is_video:
            media_url = post.video_url
        else:
            media_url = post.url
        
        post_info = {
            "news": post.caption,
            "media_url": media_url,  # No angle brackets here
            "timestamp": post.date_utc.strftime("%Y-%m-%d %H:%M:%S")
        }
        posts_data.append(post_info)
    return posts_data

def save_data_to_csv(posts_data, csv_filename):
    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['news', 'media_url', 'timestamp', 'Is_Disaster', 'Disaster_Class', 'Location']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if the file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        for post in posts_data:
            # Classify the post's caption (news)
            result = classify_with_threshold(post["news"], threshold=0.8)
            
            # Add classification results to the post data
            post.update({
                'Is_Disaster': result['Is_Disaster'],
                'Disaster_Class': result['Disaster_Class'],
                'Location': result['Location']
            })
            
            writer.writerow(post)

    print(f"Data appended to {csv_filename}")

def scrape_instagram_data(usernames, count, csv_filename):
    for username in usernames:
        print(f"Fetching data for profile: {username}")
        
        # Load the profile
        profile = instaloader.Profile.from_username(L.context, username)

        # Fetch posts data
        posts_data = get_posts_data(profile, count)
        print(f"Fetched {len(posts_data)} posts for {username}")

        save_data_to_csv(posts_data, csv_filename)
        
        
if __name__ == "__main__":
    usernames = ["priyanshu_pilaniwala"] 
    count = 5
    csv_filename = "insta.csv"
    scrape_instagram_data(usernames, count, csv_filename)
