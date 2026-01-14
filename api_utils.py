######################################################################################
#                        Python api_utils.py - Backend Utilities                     #
######################################################################################
import requests
import json                  
import os
from dotenv import load_dotenv
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from functools import wraps
from time import time,sleep
# --- FIRESTORE / GOOGLE CLOUD IMPORTS ---
from google.cloud.exceptions import NotFound # For robust error checking
from google.api_core import exceptions as gcp_exceptions
# --- IMAGE PROCESSING ---
from PIL import Image
import io   
import base64
# --- TENSORFLOW  ---    Check if TensorFlow is available (Local Dev vs Docker/GCP)
try:                                            
    import tensorflow as tf
    from tensorflow import keras
    from keras import models, layers
    HAS_TF = True
    print("Using Local TensorFlow Model...")
except ImportError:
    HAS_TF = False
    print("TensorFlow not found. Use Google Gemini API...")
mnist_model1 = None

# Variables initialized from environment (defaulting to PRODUCTION for GCP)
if os.path.exists('.env'):                            # Only load .env if it exists (local dev)
    load_dotenv()
#FLASK_ENV - UPDATE_LOCAL_FILE_AND_EXIT - load from API/FIRESTORE + write to file
#          - USE_LOCAL_FILE_FOR_TESTING - load from file + do not write file, 
#          - DEVELOPMENT                - load from API/FIRESTORE + write to FIRESTORE
#          - PRODUCTION - Google Cloud  - load from API/FIRESTORE + write to FIRESTORE
FLASK_ENV =  os.getenv('FLASK_ENV','PRODUCTION') # PRODUCTION / DEVELOPMENT / USE_LOCAL_FILE_FOR_TESTING / UPDATE_LOCAL_FILE_AND_EXIT
   
# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "DEFAULT_KEY_FALLBACK") 
GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

# --- TMDB & FIRESTORE API Configuration ---
API_KEY = os.getenv('API_KEY', "DEFAULT_KEY_FALLBACK")
API_TOKEN = os.getenv('API_TOKEN', "DEFAULT_TOKEN_FALLBACK")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', "DEFAULT_CREDENTIALS_FALLBACK")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'serviceAccountKey.json')

BASE_URL = "https://api.themoviedb.org/3/movie/top_rated?language=en-US&page="  
FIRESTORE_COLLECTION = "portfolio_cache"
TMDB_DATA_FILE = 'tmdb_top_movies.json'
FIRESTORE_DOCUMENT = "tmdb_top_movies"
CACHE_TTL_MINUTES = 1440 # Time-To-Live for the cached data (60 * 24 = 1 day)
TMDB_QUANTILE = 0.75          # Requires movies to have more votes than (100 * TMDB_QUANTILE) e.g.75% of the dataset
TMDB_MIN_VOTES = 50
if "DEFAULT_KEY_FALLBACK" in API_KEY:
    print("WARNING: API_KEY not found in environment, using fallback.")
if "DEFAULT_TOKEN_FALLBACK" in API_TOKEN:
    print("WARNING: API_TOKEN not found in environment, using fallback.")
if "DEFAULT_CREDENTIALS_FALLBACK" in GOOGLE_APPLICATION_CREDENTIALS:
    print("WARNING: GOOGLE_APPLICATION_CREDENTIALS not found in environment, using fallback.")
if "DEFAULT_KEY_FALLBACK" in GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment, using fallback.")
headers = {"accept": "application/json", "Authorization": "Bearer " + API_TOKEN}
mnist_model1 = None
    
# ---------------------------------------------------------------------------------- #
# ---                            FIRESTORE LOAD DATA                             --- #
# --- Reads the cached movie data and timestamp from the Firestore document.     --- #
# --- for testing "USE LOCAL FILE FOR TESTING" Reads the movie data from local file. #
# ---------------------------------------------------------------------------------- #
def load_TMDB_data(db_client):
    if FLASK_ENV == "USE_LOCAL_FILE_FOR_TESTING":                         # testing - Fetch from Local File
        if os.path.exists(TMDB_DATA_FILE):
            try:
                with open(TMDB_DATA_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    status_msg = f"{FLASK_ENV}: Fetched from local file TMDB_DATA_FILE: {TMDB_DATA_FILE}"
                    print(status_msg)
                    return data.get('results', []), status_msg
            except Exception as e:
                status_msg = f"{FLASK_ENV}: Failed to fetch data from local file TMDB_DATA_FILE: {TMDB_DATA_FILE}"
                print(status_msg)
        else:
            status_msg = f"{FLASK_ENV}: Failed to fetch data. local file not exists TMDB_DATA_FILE: {TMDB_DATA_FILE}"
            print(status_msg)
        return None, status_msg
        
    if db_client is None:                                                        # Fetch from Firestore
        print("Cloud read skipped: Firestore client is None.")
        return None, "Firestore not initialized."
    try:
        doc_ref = db_client.collection(FIRESTORE_COLLECTION).document(FIRESTORE_DOCUMENT)
        doc = doc_ref.get()
        if not doc.exists:
            print("Cloud cache document not found.")
            return None, "Cloud cache empty."
        data_entry = doc.to_dict()
        last_fetch = data_entry.get("timestamp")
        now_utc = datetime.now(timezone.utc)
        if last_fetch:
            # Firestore timestamps are usually timezone-aware datetime objects
            if not last_fetch.tzinfo:
                # If timezone is missing, assume UTC for local comparison
                last_fetch = last_fetch.replace(tzinfo=timezone.utc)
            # Check if the cache is expired
            if last_fetch + timedelta(minutes=CACHE_TTL_MINUTES) > now_utc:
                status_msg = f"SUCCESS: Data loaded from REAL Firestore Cache (Last updated: {last_fetch.strftime('%Y-%m-%d %H:%M:%S UTC')})."
                print(status_msg)
                return data_entry.get("data"), status_msg   
            else:
                status_msg = f"REAL Firestore Cache expired. Last fetch: {last_fetch.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                print(status_msg)
                return None, status_msg
        else:
             status_msg = "Firestore document found, but timestamp field is missing or invalid."
             print(status_msg)
             return None, status_msg
    # Use specific GCP exceptions for better clarity
    except gcp_exceptions.PermissionDenied as e:
        status_msg = f"Error reading from Firestore (Permission Denied): {e}"
        print(status_msg)
        return None, status_msg
    except Exception as e:
        status_msg = f"Generic Error reading from Firestore: {e}"
        print(status_msg)
        return None, status_msg
    
######################################################################################
#                      Python Performance Test (3 x Fibonacci)                       #
######################################################################################
N_VALUE = 13 # target Fibonacci number Fn where n = 33

# --- Helper Function ---
def duration(dur): 
    """Formats time duration into [s ms µs] string."""
    s  = int(dur)
    ms = int((dur-s) * 1000)
    mu = int((dur-s-ms/1000) * 1000 * 1000)
    return f"<span style='color: #00ffff;'>{s:2}s {ms:3}ms {mu:3}μs</span>"

# --- Decorators --- 
def accelerator(func):
    """Decorator that caches a function's results."""
    cache = {}
    @wraps(func)
    def wrapper(*args):
        # 1. Check if the result is already known
        n_arg = args[0]
        if n_arg in cache:
            return cache[n_arg], None, None # Return cached result instantly
        # 2. Calculate the result if not cached
        result, run_time, out_str = func(*args)
        # 3. Store the result before returning
        cache[n_arg] = result
        return result, run_time, out_str
    # 4. EXPOSE THE CACHE CLEAR FUNCTION
    def clear_cache():
        """Clears the cache dictionary."""
        nonlocal cache
        cache = {}
    wrapper.clear_cache = clear_cache
    return wrapper

def timer(func):
    """Decorator that measures the duration of a function."""
    # Initialize a flag on the function object to track if we are in the timing context
    func._timing_context = {'in_progress': False}     
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if timing is already in progress for this function
        if func._timing_context['in_progress']:
            return func(*args, **kwargs)
        else:
            func._timing_context['in_progress'] = True
            start = time()
            try:
                # Execute the function (this will trigger all the recursive calls)
                result_tuple = func(*args, **kwargs)
            finally:
                # Ensure the flag is reset even if the function raises an error
                func._timing_context['in_progress'] = False
            # Print the total time only once
            result = result_tuple[0]
            run_time = time() - start
            # comparison_time_d1 = args[1]
            str1= " -> " + "<span style='color: #00ff00;'>   base time for speed comparison</span>"
            if args[1]>0:
                str1= f" -> <span style='color: #00ff00;'>{round(args[1]/run_time):8,} times faster calculation</span>"
            out_str = f"<b>Finished: <span style='color: #ffff00;'>{func.__name__}</span></b>({args[0]}) = {result}, run_time: {duration(run_time)}{str1}"
            return result, run_time, out_str
    return wrapper

@timer 
def slow_fibonacci(n, d1): # This simulates a slow, expensive calculation without accelerator
    if n < 0:
        return 0, None, None
    if n <= 1:
        return n, None, None
    return slow_fibonacci(n - 1, d1)[0] + slow_fibonacci(n - 2, d1)[0], None, None

@accelerator
@timer 
def fast_fibonacci(n, d1): # This simulates a slow, expensive calculation with accelerator
    if n < 0:
        return 0, None, None
    if n <= 1:
        return n, None, None
    return fast_fibonacci(n - 1, d1)[0] + fast_fibonacci(n - 2, d1)[0], None, None

@timer    
def loop_fibonacci(n,d1): # This simulates a loop calculation without accelerator
    f2=0
    f1=1
    if n<1:
        return 0, None, None
    for _ in range(n-1):
        f1 = f2 + f1
        f2 = f1 - f2
    return f1, None, None

# ####################################################### TESTING AREA ####################################################### # 
# import re
# N_VALUE = 1
# fib_result1, total_time1, str1 = slow_fibonacci(N_VALUE, 0)           # slow recursive fibonacci calculation without accelerator
# fib_result2, total_time2, str2 = fast_fibonacci(N_VALUE, total_time1) # slow recursive fibonacci calculation with accelerator
# fib_result3, total_time2, str3 = loop_fibonacci(N_VALUE, total_time1) # loop fibonacci calculation 
# print(f"{re.sub(r"<.*?>", "", str1)}\n{re.sub(r"<.*?>", "", str2)}\n{re.sub(r"<.*?>", "", str3)}")


# ---------------------------------------------------------------------------------- #
# ---                          UTILITY TO RUN PYTEST                             --- #
# --- Runs pytest using subprocess and captures stdout/stderr.                   --- #
# --- Returns the raw terminal output as a string.                               --- #
# --- Use subprocess.run to execute the command                                  --- #
# --- 'python -m pytest' to ensure the correct Python env pytest is used.        --- #
# --- '-s' to show print statements (though usually pytest captures stdout       --- #
# --- '-v' for verbose output.                                                   --- #
# ---------------------------------------------------------------------------------- #
def run_pytest(test_file='api_test.py'):
    try:
        # Determine the full path to the test file relative to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_path = os.path.join(script_dir, test_file)

        # Check if the test file exists before trying to run it
        if not os.path.exists(test_path):
             return f"ERROR: Test file not found at path: {test_path}"

        result = subprocess.run(
            ['python', '-m', 'pytest', test_path, '-v'],
            capture_output=True,
            text=True,
            check=False # Do not raise an exception for non-zero exit codes (i.e., test failures)
        )
        # Combine stdout and stderr for the terminal output
        output = result.stdout + result.stderr
        
        # Clean up the output to make it more readable on the web
        # Pytest output can be long; we grab the summary section
        output_lines = output.splitlines()
        summary_start_index = -1
        # Look for the line starting with '=====' to find the summary
        for i, line in enumerate(output_lines):
            if line.startswith('====='):
                summary_start_index = i
                break
        
        if summary_start_index != -1:
            # Return all lines from the summary start to the end
            return "\n".join(output_lines[summary_start_index:])
        else:
            # If summary not found, return the full output (e.g., if there's an error before collection)
            return output

    except FileNotFoundError:
        # Handles cases where 'python' or 'pytest' command isn't found
        return "ERROR: Python or pytest command not found. Ensure pytest is installed."
    except Exception as e:
        return f"ERROR: Failed to run pytest: {str(e)}"

# ---------------------------------------------------------------------------------- #
# ---                         UTILITY TO RUN 3 x FIBONACCI                       --- #
# ---------------------------------------------------------------------------------- #
def generate_fibonacci_results(n):
    yield json.dumps({"step": 0, "message": f"<b>TERMINAL OUTPUT:</b> Starting tests fibonacci calculation for N={n}...\n"}) + "\n"
    # --- 1. Slow Recursive Test ---------------------- recursive fibonacci calculation without accelerator
    sleep(0.5)
    yield json.dumps({"step": 1, "message": f"slow recursive fibonacci calculation without accelerator O(2^n) ...\n"}) + "\n"
    fib_result1, total_time1, str1 = slow_fibonacci(n, 0) 
    yield json.dumps({"step": 2, "message": f"{str1}\n"}) + "\n"               
    sleep(0.5)
    # --- 2. Memoized Recursive Test (Fast) ----------- recursive fibonacci calculation with accelerator
    yield json.dumps({"step": 3, "message": f"slow recursive fibonacci calculation with accelerator O(n) ... \n"}) + "\n"
    fast_fibonacci.clear_cache()
    fib_result2, total_time2, str2 = fast_fibonacci(n, total_time1)  
    yield json.dumps({"step": 4, "message": f"{str2}\n"}) + "\n"        
    sleep(0.5)
    # --- 3. Iterative Loop Test (Fastest Baseline) --- loop fibonacci calculation 
    yield json.dumps({"step": 5, "message": f"loop fibonacci calculation O(n) ... \n"}) + "\n"
    fib_result3, total_time3, str3 = loop_fibonacci(n, total_time1) 
    yield json.dumps({"step": 6, "message": f"{str3}\n"}) + "\n"      
    sleep(0.5)

# ---------------------------------------------------------------------------------- #
# ---                     SAVE TMDB DATA TO FIRESTORE CLOUD                      --- #
# --- Writes the movie data and the current timestamp to the Firestore document.---- #
# ---------------------------------------------------------------------------------- #
def save_TMDB_data_to_cloud(data, db_client):
    if db_client is None:
        status_msg = f"--- Cloud write skipped: Firestore client is None/not initialized."
        print(status_msg)
        return status_msg
    print("Writing data to REAL Firestore Persistence layer...")
    data_to_store = {
        # IMPORTANT: Use timezone.utc for consistency
        "timestamp": datetime.now(timezone.utc), 
        "data": data 
    }
    try:
        doc_ref = db_client.collection(FIRESTORE_COLLECTION).document(FIRESTORE_DOCUMENT)
        doc_ref.set(data_to_store) 
        status_msg = "SUCCESS: Data written to Firestore."
        print(status_msg)
        return status_msg
    # Use specific GCP exceptions for definitive error reporting
    except gcp_exceptions.PermissionDenied as e:
        status_msg = f"--- FIRESTORE WRITE FAILED (Permission Denied): Check Rules --- Error: {e}"
        print(status_msg)
        return status_msg
    except Exception as e:
        status_msg = f"--- FIRESTORE WRITE FAILED (Unknown Error) --- Error: {e}"
        print(status_msg)
        return status_msg

# ---------------------------------------------------------------------------------- #      
# ---                   TMDB Data Loading (Cache-or-Fetch Logic)                ---- #
# --- The main data fetching function:                                          ---- #
# --- Fetch tmdb_data from FIRESTORE or TMDB API or Local File                  ---- #
# ---------------------------------------------------------------------------------- #
def load_movie_data(num_pages=1, db_client=None):
    tmdb_data, status_message = load_TMDB_data(db_client)                # Fetch from FIRESTORE or Local File
    if tmdb_data:
        return tmdb_data, status_message
    if FLASK_ENV == "USE_LOCAL_FILE_FOR_TESTING":               # testing - Fetch from Local File
        return None, status_message

    print(f"No valid cache found. Fetching from TMDB API...")   # Fetch from external API (TMDB)
    all_results = []
    for page in range(1, num_pages + 1):    # Fetch movie data (1 page = 20 movies)
        url = f"{BASE_URL}{page}"
        try:
            response = requests.get(url, headers=headers) 
            response.raise_for_status()     # Raises an HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            all_results.extend(data.get('results', []))
            print(f"Fetched page {page}. Total movies: {len(all_results)}")
            if len(all_results) >= 120:     # Stop after 120 movies (6 page)
                break
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to fetch data from TMDB: {e}"
            print(error_msg)                # Return empty data and the error message
            return {"results": [], "error": error_msg}, error_msg 

    if all_results:                         # Save tmdb_data, return fetched data and status
        data_to_store = {"results": all_results}
        write_status = save_TMDB_data_to_cloud(data_to_store, db_client) 
        return data_to_store, f"TMDB API Fetched and cache written: {write_status}"
    else:
        print(f"Error: Save to Cloud Cache [all_results] is empty")   
                      
    return {"results": []}, status_message  # Return empty data and the last status message 
 
# ---------------------------------------------------------------------------------- #      
# ---                        PANDAS & SCIKIT-LEARN                               --- #
# ---            Formula: W = (v / (v + m)) * R + (m / (v + m)) * C              --- #
# ---------------------------------------------------------------------------------- #
def weighted_rating(row, m, C):                   # m = minimum votes required (75th percentile of vote_count)
    v = row['vote_count']                         # v = number of votes for the movie (vote_count)
    R = row['vote_average']                       # R = average for the movie (vote_average)
    return (v / (v + m)) * R + (m / (v + m)) * C  # C = mean vote across the whole report (mean of vote_average)
    
# ---------------------------------------------------------------------------------- #      
# ---                     PANDAS/SKLEARN UTILITY FUNCTION                       ---- #
# --- Processes the raw TMDB movie data (list of dicts) into a Pandas DataFrame, --- #
# --- cleans it, calculates the weighted score formula constants (C and m),      --- #
# --- and adds the final weighted_score column.                                  --- #
# --- Returns tuple: (pd.DataFrame, float C, int m)                              --- #
# ---------------------------------------------------------------------------------- #
def process_data_for_analysis(raw_data):
    if not raw_data or not raw_data.get('results'):
        return pd.DataFrame(), 0.0, 0
    df = pd.DataFrame(raw_data['results'])                              # Convert list of dicts to DataFrame  
    
    # Convert necessary columns to appropriate types and handle NaNs    # Cleaning and Standardization
    df = df.dropna(subset=['vote_average', 'vote_count', 'popularity', 'release_date'])
    df['vote_count'] = df['vote_count'].astype(int)                     # Convert types
    df['vote_average'] = df['vote_average'].astype(float)
    
    # Extract year
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df.dropna(subset=['release_year'], inplace=True)
    df['release_year'] = df['release_year'].astype(int)

    # Use drop_duplicates with subset='id' to prevent the 'unhashable type: list' error    
    df.drop_duplicates(subset=['id'], keep='first', inplace=True)
    
    # Remove rows where key metrics are missing or zero (not a valid movie rating)         
    df = df[
        (df['vote_count'] > 0) & 
        (df['vote_average'] > 0) & 
        (df['popularity'].notna()) &
        (df['vote_average'].notna())
    ].copy()
    if len(df) == 0:                                                     # IMDB Weighted Rating Formula Implementation
        return df, 0.0, 0                                                # Formula: W = (v / (v + m)) * R + (m / (v + m)) * C
    C = df['vote_average'].mean()                                        # C = mean vote across the whole report (mean of vote_average)
    m = df['vote_count'].quantile(TMDB_QUANTILE)                         # m = minimum votes required (75th percentile of vote_count)
    m = max(TMDB_MIN_VOTES, int(m))   # Ensure m is an integer and at least TMDB_MIN_VOTES = 50     
    # Apply the function to create the new column                        # R = average for the movie (vote_average)
    df['weighted_score'] = df.apply(weighted_rating, axis=1, m=m, C=C)   # v = number of votes for the movie (vote_count)
    
    # Select final columns for clarity and relevance
    df = df[['title', 'vote_average', 'vote_count', 'popularity', 'release_date', 'release_year', 'weighted_score']].copy()
    return df, C, m

# ---------------------------------------------------------------------------------- #       
# ---                            TensorFlow with Keras                           --- # 
# ---                            train a model on MNIST                          --- #
# ---                 This will only train on first app start                    --- #
# ---------------------------------------------------------------------------------- #
def get_or_train_model1():
    global mnist_model1
    if mnist_model1 is not None:
        return mnist_model1
    basedir = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(basedir, "mnist_model1.h5")
    try:
        mnist_model1 = tf.keras.models.load_model(model_path)         # Load the model if it exists
        print(f"MNIST {model_path} loaded from disk.")
    except Exception as e:
        print(f"Training a new MNIST model... Reason: {e}")
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        # x_train, x_test = x_train / 255.0, x_test / 255.0           # Normalize pixel values to be between 0 and 1
        x_train = x_train / 255.0                                     # Normalize pixel values to be between 0 and 1
        mnist_model1 = models.Sequential([                            # Build a simple Sequential model
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
            # layers.Dropout(0.2)                                     # Add dropout for better generalization
        ])
        mnist_model1.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        mnist_model1.fit(x_train, y_train, epochs=5, batch_size=32, verbose=2)    # Train for 5 epochs
        # mnist_model1.evaluate(x_train, y_train, verbose=0)          # initialize metrics (optional, not needed for most)
        mnist_model1.save(model_path)                                 # Save for future use     
    return mnist_model1

# ---------------------------------------------------------------------------------- #       
# ---                            TensorFlow with Keras                           --- # 
# ---                             preprocess_image 2D                            --- #
# ---             Returns a 2D numpy array (28, 28) normalized to 0-1.           --- #
# ---------------------------------------------------------------------------------- #
def preprocess_image(image_file):
    img = Image.open(image_file).convert("L").resize((28, 28)) # Open and convert to grayscale
    # Invert if necessary (MNIST is white on black) If users draw white on black in HTML, we stay as is.
    inverted_img = img                                         # Assuming white on black
    bbox = inverted_img.getbbox()                              # Find bounding box of the drawing to center it
    if bbox:                                                   # This helps a lot with "squiggles" or off-center drawings
        img = inverted_img.crop(bbox)                          # Crop to the digit and add padding to make it 20x20 (MNIST style)
        img.thumbnail((20, 20), Image.Resampling.LANCZOS)      # Resize the actual drawing to 20x20
        new_img = Image.new("L", (28, 28), 0)                  # Create a new black 28x28 canvas 
        offset = ((28 - img.width) // 2, (28 - img.height) // 2)
        new_img.paste(img, offset)                             # paste the 20x20 digit in the center
        img = new_img
    else:
        img = img.resize((28, 28), Image.Resampling.LANCZOS)   # Empty canvas
    arr = np.array(img)                                        # Convert to array 
    if np.mean(arr) > 127: 
        arr = 255 - arr 
    return arr / 255.0                                         # normalize

# ---------------------------------------------------------------------------------- #       
# ---                            TensorFlow with Keras                           --- # 
# ---              preprocess_image base64 PNG for Google Gemini API             --- #
# ---                RConverts a 2D numpy array (0-1) to base64 PNG.             --- #
# ---------------------------------------------------------------------------------- #
def get_image_b64(arr):
    buf = io.BytesIO()
    # Ensure to use the 2D array here
    pil_img = Image.fromarray((arr * 255).astype("uint8"))
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ---------------------------------------------------------------------------------- #       
# ---                     Fallback prediction using Gemini API                   --- # 
# ---              If TensorFlow not found. Use Google Gemini API                --- #
# ---------------------------------------------------------------------------------- # 
def call_gemini_vision(img_b64):
    prompt = "Identify the handwritten digit (0-9). Return ONLY a JSON: {\"digit\": number, \"confidence\": float}"
    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/png", "data": img_b64}}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    for i in range(2):
        try:
            res = requests.post(GEMINI_URL, json=payload, timeout=10)
            if res.status_code == 200:
                return json.loads(res.json()['candidates'][0]['content']['parts'][0]['text'])
            error_info = res.json().get('error', {})
            error_msg = error_info.get('message', 'Unknown API Error')
            error_code = error_info.get('code', res.status_code)
            print(f"Gemini API Error ({error_code}): {error_msg}")
            if res.status_code != 429:
                return {"digit": "gemini_API", "confidence": 0.0}  # gemini_API
            sleep(2**i)
        except:
            sleep(2**i)
    return {"digit": "gemini_ERR", "confidence": 0.0}


