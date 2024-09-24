import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server
from urllib.parse import parse_qs

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

#reviews = pd.read_csv('data/reviews.csv').to_dict('records')
reviews = pd.read_csv('data/reviews.csv')

QUERYABLE_LOCATIONS = [
    'Albuquerque, New Mexico',
    'Carlsbad, California',
    'Chula Vista, California',
    'Colorado Springs, Colorado',
    'Denver, Colorado',
    'El Cajon, California',
    'El Paso, Texas',
    'Escondido, California',
    'Fresno, California',
    'La Mesa, California',
    'Las Vegas, Nevada',
    'Los Angeles, California',
    'Oceanside, California',
    'Phoenix, Arizona',
    'Sacramento, California',
    'Salt Lake City, Utah',
    'Salt Lake City, Utah',
    'San Diego, California',
    'Tucson, Arizona',
]

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":

            df = reviews.copy()
            df['Timestamp'] = pd.to_datetime(df.Timestamp)

            params = parse_qs(environ["QUERY_STRING"])
            if 'location' in params:
                locations = [loc for loc in params['location'] if loc in QUERYABLE_LOCATIONS]
                if len(locations) > 0:
                    df = df.loc[df.Location.isin(locations), :]    
            
            if 'start_date' in params:
                start_date = datetime.strptime(params['start_date'][0], '%Y-%M-%d')
                df = df.loc[df.Timestamp >= start_date, :]
            
            if 'end_date' in params:
                end_date = datetime.strptime(params['end_date'][0], '%Y-%M-%d')
                df = df.loc[df.Timestamp <= end_date, :]

            df['Timestamp'] = df.Timestamp.astype(str)

            records = df.to_dict('records')
            for record in records:
                record['sentiment'] = self.analyze_sentiment(record['ReviewBody'])

            records = sorted(records, key=lambda r: r['sentiment']['compound'], reverse=True)

            # Create the response body from the reviews and convert to a JSON byte string
            response_body = json.dumps(records, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":

            post_data = environ['wsgi.input'].read().decode()
            params = parse_qs(post_data)

            if 'Location' not in params:
                http_code = "400 BadRequest"
                record = {}
            elif params["Location"][0] not in QUERYABLE_LOCATIONS:
                http_code = "400 BadRequest"
                record = {}
            elif 'ReviewBody' not in params:
                http_code = "400 BadRequest"
                record = {}
            else:
                http_code = "201 Created"
                record = {
                    "ReviewId": str(uuid.uuid4()),
                    "Timestamp": str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                    "Location": params["Location"][0],
                    "ReviewBody": params["ReviewBody"][0],
                }

            response_body = json.dumps(record, indent=2).encode("utf-8")
            start_response(http_code, [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])

            return [response_body]


if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()