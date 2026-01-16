######################################################################################
#              Python api.py - server-side Python environment (Flask)               #
######################################################################################
import json                  
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from jinja2 import TemplateNotFound
from datetime import datetime
# --- FIRESTORE / GOOGLE CLOUD IMPORTS ---
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.exceptions import NotFound # For robust error checking
from google.api_core import exceptions as gcp_exceptions
# --- MACHINE LEARNING ---
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor 
# --- IMAGE PROCESSING ---
from PIL import Image
import io
import base64
import api_utils
      
# --- CONSTANTS ONLY HERE IN api.py ---
PORTFOLIO_VERSION = "V3.3"
CONTENT_SECTIONS = [
    'python',
    'api',
    'pandas', 
    'numpy',
    'scikitlearn',
    # 'pytorch',
    'tensorflow',
    'cpp',
    'docker',
    'linux',
    'git',
    'bootstrap' 
]
NUMPY_DISPLAY_LIMIT = 120

# ---------------------------------------------------------------------------------- #  
# ---------------------------- FIRESTORE INITIALIZATION ---------------------------- #
# ---------------------------------------------------------------------------------- # 
db_firestore_client = None 
if api_utils.FLASK_ENV != "USE_LOCAL_FILE_FOR_TESTING":
    if not firebase_admin._apps:                             # Prevent "App already exists" error
        try:
            if api_utils.FLASK_ENV == "PRODUCTION":
                print("Firestore: Initializing in PRODUCTION mode (No JSON key needed).")
                firebase_admin.initialize_app()              # GCP Cloud Run Path: Uses Service Identities automatically
            else:
                # Local/Development Path: Uses the serviceAccountKey.json
                print(f"Firestore: Initializing in DEVELOPMENT mode using: {api_utils.GOOGLE_APPLICATION_CREDENTIALS}")
                cred = credentials.Certificate(api_utils.GOOGLE_APPLICATION_CREDENTIALS)
                firebase_admin.initialize_app(cred)          # Initialize the app 
            print("SUCCESS: Firestore client initialized.")
            db_firestore_client = firestore.client()         # Get the Firestore client instance
        except Exception as e:
            print(f"FATAL ERROR during Firestore initialization. Firestore will be disabled: {e}")
# ---------------------------------------------------------------------------------- #

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_session_management' # secret key is required to use Flask's session object (for storing movie data temporarily)

# ---------------------------------------------------------------------------------- #
# ---        JINJA Utility function to safely check/load template content        --- #
# ---    Attempts to load the template source content using the Jinja loader.    --- #
# ---------------------------------------------------------------------------------- #
def load_from_include(template_name):
    try:
        # This line expects the template_name to be simple, like "python.html"
        source, _, _ = app.jinja_env.loader.get_source(app.jinja_env, template_name)
        return source
    except TemplateNotFound:
        # Safely returns None if the file is missing
        return None
def now():   # Returns the current datetime object for use in Jinja templates.
    return datetime.now()
# CRITICAL STEP: Add the function as a global so it can be called inside index.html
app.jinja_env.globals['load_from_include'] = load_from_include
app.jinja_env.globals['now'] = now 

# ---------------------------------------------------------------------------------- #      
# ---                          ONLY FOR TESTING                                  --- #
# ---                     UPDATE_LOCAL_FILE_AND_EXIT                             --- #
# ---------------------------------------------------------------------------------- #
if api_utils.FLASK_ENV == "UPDATE_LOCAL_FILE_AND_EXIT":
    tmdb_data, status_message = api_utils.load_movie_data(6, db_firestore_client)
    print(status_message)
    if tmdb_data:
        try:
            print(f"{api_utils.FLASK_ENV}: save {len(tmdb_data.get('results', []))} movies to {api_utils.TMDB_DATA_FILE}")
            with open(api_utils.TMDB_DATA_FILE, "w", encoding='utf-8') as f:
                json.dump({"results": tmdb_data}, f, indent=4)
                #json.dump(tmdb_data, f, indent=4)
        except Exception as e:
            print(f"{api_utils.FLASK_ENV}: Failed to save data to local file TMDB_DATA_FILE: {api_utils.TMDB_DATA_FILE}\nError: {e}")
    print("EOE - EXIT")
    exit()
            
######################################################################################
###                                   Flask Routes                                ####
###       Renders the main portfolio page with all dynamic content included       ####
######################################################################################

# 1. Health check for the Splash Screen
# Endpoint for the Splash Screen to verify the container is awake.
@app.route('/api/health')
def health_check():
    return jsonify({"status": "awake", "mode": api_utils.FLASK_ENV}), 200

# 2. Dedicated Portfolio route
# The actual portfolio page, loaded after the Splash Screen is done.
@app.route('/portfolio')
def portfolio():
    return render_template("index.html", content_sections=CONTENT_SECTIONS, index_html_info=api_utils.FLASK_ENV, version = PORTFOLIO_VERSION)

# 3. Original root as backup / Original Endpoint
@app.route('/')
def home():
    return render_template("index.html", content_sections=CONTENT_SECTIONS, index_html_info = api_utils.FLASK_ENV, version = PORTFOLIO_VERSION)

######################################################################################
###                     3 x fibonacci test AJAX endpoint                           ###
###            run the performance test requested by the user.                     ###
###        It takes 'n' from the request body, and runs the three functions        ###
######################################################################################
@app.route('/api/run_fibonacci_test', methods=['POST'])
def run_fibonacci_test():
    try:
        n = int(request.json.get('n', 35))    # Default to 35 if not provided for safety
        if n > 40:                            # Limit N for the slow test to prevent timeouts
            return jsonify({"error": "Input N is too high. Please use N <= 40 for a reasonable test duration."}), 400
        return Response(api_utils.generate_fibonacci_results(n), content_type='application/json')
    except ValueError:
        return jsonify({"error": "Invalid input: N must be an integer."}), 400
    except Exception as e:
        print(f"Fibonacci test failed: {e}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

######################################################################################
###                      fibonacci pytest AJAX endpoint                            ###
###            run the automated testing using the pytest framework                ###
######################################################################################
@app.route('/api/run_fibonacci_pytest', methods=['POST'])
def run_fibonacci_pytest():
    try:
        pytest_output = api_utils.run_pytest('api_test.py')  # Run the tests using the utility function
        # Return the raw output as a JSON response. The client-side will handle displaying this output.
        return jsonify({"results": ["PYTEST TERMINAL OUTPUT:"], "pytest_output": pytest_output})
    except Exception as e:
        print(f"Pytest API call failed: {e}")
        return jsonify({"error": f"Internal Server Error during pytest execution: {str(e)}"}), 500
 
######################################################################################
###                    REST API section AJAX endpoint                              ###
###            Exposes a RESTful endpoint to the frontend                          ###
###   It calls the core data loading function and returns a clean subset of data   ###
######################################################################################
@app.route('/api/fetch_movies', methods=['POST'])
def fetch_movies_endpoint():

    # Load 1 page (20 movies) for the cache, but only return top 5
    data, cache_status = api_utils.load_movie_data(6, db_firestore_client)   
    if data.get('error'):
        return jsonify({"movies": [], "cache_status": cache_status, "error": data.get('error')}), 500

    # Process movie list (Limit to top 5)
    movie_list = []
    for i, result in enumerate(data.get('results', [])):
        if i >= 5: # Limit to top 5 for display
            break
        movie_list.append({
            "title": result.get('title'),
            "score": f"{result.get('vote_average'):.1f}", # Format score to 1 decimal
            "votes": result.get('vote_count'),
            "release_date": result.get('release_date'),
            "year": int(result.get('release_date', '0000').split('-')[0]) # Correctly extracts the year
        })
    # Return the full JSON response, including the new cache_status
    return jsonify({
        "movies": movie_list, 
        "cache_status": cache_status, 
        "total_cached": len(data.get('results', [])),
        "expiration": api_utils.CACHE_TTL_MINUTES/60
    })
    
######################################################################################
###                              PANDAS endpoint                                   ###
###   Retrieves TMDB data, processes it using Pandas, and returns summary stats.   ###
###      Demonstrates data loading, cleaning, transformation, and aggregation.     ###
###################################################################################### 
@app.route('/api/get_pandas_data', methods=['GET'])
def get_pandas_analysis():
    raw_data, status_message = api_utils.load_movie_data(6, db_firestore_client)  # Load the raw data (Cache-or-Fetch logic)
    df, C, m = api_utils.process_data_for_analysis(raw_data) # cleans, calculates C, m and adds the  weighted_score column
    if df.empty:
        return jsonify({"error": "Failed to load movie data for Pandas analysis.", "source_status": status_message}), 500
    qualified_movies = df[df['vote_count'] >= m].copy() 
    qualified_movies['weighted_score'] = qualified_movies.apply(lambda row: api_utils.weighted_rating(row, m, C), axis=1)
    
    # Top 5 Best Rated Qualified Movies (by Weighted Score)
    top_movies_weighted = qualified_movies.sort_values(by='weighted_score', ascending=False).head(5)
    
    # Simple Summary Statistics
    stats = {
        'mean_score': f"{df['vote_average'].mean():.2f}",
        'median_votes': f"{df['vote_count'].median():,.0f}",
        'total_movies_analyzed': len(df),
        'min_votes_for_qualification': f"{m:,.0f}"
    }

    # Aggregation by Year (Top 3 Recent Years with Highest Average Score)
    yearly_avg = df.groupby('release_year')['vote_average'].mean().reset_index()
    yearly_avg = yearly_avg.sort_values(by=['release_year', 'vote_average'], ascending=[False, False]).head(3)

    # Convert DataFrames to JSON structures - Prepare Final JSON Response
    top_movies_json = top_movies_weighted[[
        'title', 'vote_average', 'vote_count', 'release_year', 'weighted_score'
    ]].to_dict(orient='records')
    
    yearly_json = yearly_avg.to_dict(orient='records')
    return jsonify({
        "status": "SUCCESS",
        "source_status": status_message,
        "summary_stats": stats,
        "top_movies_weighted": top_movies_json,
        "top_years": yearly_json
    })
    
######################################################################################
###                       SCIKIT-LEARN ANALYSIS ENDPOINT 1                         ###
###                 Performs Linear Regression using Scikit-learn                  ###
###                to predict Weighted Score from other TMDB features.             ###
######################################################################################
@app.route('/api/run_sklearn_analysis', methods=['POST'])
def run_sklearn_analysis():
    try:
        raw_data, status_message = api_utils.load_movie_data(6, db_firestore_client)  # Load data (this calls the TMDB API if cache is old)
        df, C, m = api_utils.process_data_for_analysis(raw_data) # cleans, calculates C, m and adds the  weighted_score column
        if df.empty:
            return jsonify({'error': 'DataFrame is empty. Cannot run ML model.', 'source_status': status_message}), 500
        
        features = ['vote_average', 'vote_count', 'popularity'] # Define Features (X) and Target (y)
        X = df[features]
        y = df['weighted_score']

        # Split Data into training and testing sets (80/20 split)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # ------------- Linear Regression (keep for comparison) -----------------
        lin_model = LinearRegression()                                   # Train Model
        lin_model.fit(X_train, y_train)
        y_pred = lin_model.predict(X_test)                               # Evaluate
        r2_lin = r2_score(y_test, y_pred)
        mse_lin = mean_squared_error(y_test, y_pred)
        cv_scores_lin = cross_val_score(lin_model, X, y, cv=5, scoring='r2') # 5-fold Cross-Validation
        # -----------------------------------------------------------------------
    
        # ------------- Random Forest Regressor (tree-based) --------------------- 
        tree_model = RandomForestRegressor(n_estimators=100, random_state=42)     
        tree_model.fit(X_train, y_train)                                          
        y_pred_tree = tree_model.predict(X_test)                                  
        r2_tree = r2_score(y_test, y_pred_tree)                                   
        mse_tree = mean_squared_error(y_test, y_pred_tree)                        
        cv_scores_tree = cross_val_score(tree_model, X, y, cv=5, scoring='r2')    
        # -----------------------------------------------------------------------

        # Prepare feature importances for tree model  
        tree_importances = {k: float(v) for k, v in zip(features, tree_model.feature_importances_)}  

        # Example Prediction
        mean_votes = df['vote_count'].mean()
        mean_popularity = df['popularity'].mean()
        example_input = pd.DataFrame([
            {'vote_average': 7.0, 'vote_count': mean_votes, 'popularity': mean_popularity}
        ])
        # Example predictions # CHANGED: now for both models
        example_pred_lin = lin_model.predict(example_input)[0]
        example_pred_tree = tree_model.predict(example_input)[0]  

        # Prepare and return results as JSON (BOTH MODELS!)  
        return jsonify({
            'model_types': ['Linear Regression', 'Random Forest Regressor'],                
            'source_status': status_message,
            'linear_metrics': {                                                            
                'r2_score': f"{r2_lin:.4f}",
                'mean_squared_error': f"{mse_lin:.4f}",
                'cross_validation_mean_r2': f"{cv_scores_lin.mean():.4f}"
            },
            'tree_metrics': {                                                               
                'r2_score': f"{r2_tree:.4f}",
                'mean_squared_error': f"{mse_tree:.4f}",
                'cross_validation_mean_r2': f"{cv_scores_tree.mean():.4f}"
            },
            'lin_cv_scores': cv_scores_lin.tolist(),                                       
            'tree_cv_scores': cv_scores_tree.tolist(),                                      
            'feature_coefficients': {k: f"{v:.6f}" for k, v in zip(features, lin_model.coef_)},
            'tree_feature_importances': tree_importances,                                   
            'linear_model_info': {                                                          
                'intercept': lin_model.intercept_,
                'coefficients': {k: v for k, v in zip(features, lin_model.coef_)},
            },
            'tree_model_info': {},                                                        
            'example_prediction': {                                                       
                'input': f"Raw Score: 7.0, Votes: {mean_votes:.0f}, Popularity: {mean_popularity:.2f}",
                'predicted_weighted_score_linear': f"{example_pred_lin:.3f}",              
                'predicted_weighted_score_tree': f"{example_pred_tree:.3f}"                
            }
        })
    except Exception as e:
        print(f"Scikit-learn Analysis Error: {e}")
        return jsonify({'error': f"Internal Server Error during ML analysis: {str(e)}"}), 500

######################################################################################
###                       SCIKIT-LEARN ANALYSIS ENDPOINT 2                         ###
###      Generates data for a 3D surface plot to visualize the Weighted Score      ###
###                   function based on Raw Score and Vote Count.                  ###
######################################################################################
@app.route('/api/get_sklearn_plot_data', methods=['GET'])
def get_sklearn_plot_data():
    raw_data, status_message = api_utils.load_movie_data(6, db_firestore_client)
    
    if not raw_data or not raw_data.get('results'):
        return jsonify({"error": f"Failed to load data for Plotly. Status: {status_message}"}), 500
    
    try:
        df = pd.DataFrame(raw_data['results'])
        
        # Calculate C and m from the current dataset (as in the analysis)
        df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')
        df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
        df.dropna(subset=['vote_count', 'vote_average'], inplace=True)
        df = df[df['vote_count'] > 0]
        
        C = df['vote_average'].mean()
        m = df['vote_count'].quantile(api_utils.TMDB_QUANTILE) 
        
        # 1. Define the range for the two input axes (X and Y)
        # X-axis: Votes (v). Range from 0 to 5 times the min qualified votes (m).
        v_min = 0
        v_max = int(m * 5) + 1 
        votes = np.linspace(v_min, v_max, 50) # 50 points for smoothness
        
        # Y-axis: Raw Score (R). Range from 5.0 to 10.0 (TMDB range).
        R_min = 5.0
        R_max = 10.0
        raw_scores = np.linspace(R_min, R_max, 20) # 20 points
        
        # 2. Create the meshgrid for all combinations
        V, R = np.meshgrid(votes, raw_scores)
        
        # 3. Calculate the Z-axis (Weighted Score) for every point in the grid
        # The formula: (v / (v + m)) * R + (m / (v + m)) * C
        Z = (V / (V + m)) * R + (m / (V + m)) * C

        # 4. Prepare the data for Plotly (must be converted to lists)
        plot_data = {
            "x": V.tolist(),        # Votes (x-axis)
            "y": R.tolist(),        # Raw Score (y-axis)
            "z": Z.tolist(),        # Weighted Score (z-axis)
            "C_constant": f"{C:.3f}",
            "m_constant": f"{m:.0f}"
        }
        return jsonify(plot_data)
    except Exception as e:
        print(f"Plotly Data Generation Failed: {e}")
        return jsonify({"error": f"Internal Server Error during Plotly data generation: {str(e)}"}), 500

######################################################################################
###                                  NUMPY ENDPOINT                                ###
###           Demonstrates fundamental data preprocessing using only NumPy:        ###
### Standardization (Z-score) and Normalization (Min-Max Scaling) on movie scores. ###
######################################################################################
@app.route('/api/run_numpy_analysis', methods=['GET'])
def run_numpy_analysis():
    raw_data, status_message = api_utils.load_movie_data(6, db_firestore_client)
    if not raw_data or not raw_data.get('results'):
        return jsonify({"error": f"Failed to load data for Plotly. Status: {status_message}"}), 500
    try:
        # 1. Prepare Data
        df = pd.DataFrame(raw_data['results'])

        # Filter for numerical stability
        df = df[df['vote_average'].notna() & (df['vote_count'] > 0)].copy()
        
        # Extract the relevant series (Score) into a NumPy array
        scores = df['vote_average'].to_numpy()
        movie_titles = df['title'].tolist()

        if len(scores) < 5:
            return jsonify({"error": "Not enough data points for NumPy analysis."}), 500

        # 2. Standardization (Z-Score): z_scores = (scores - mean) / std_dev
        mean = np.mean(scores)
        std_dev = np.std(scores)
        z_scores = (scores - mean) / std_dev

        # 3. Normalization (Min-Max Scaling): (X - min) / (max - min)
        min_score = np.min(scores)
        max_score = np.max(scores)
        min_max_scores = (scores - min_score) / (max_score - min_score)
        
        # 4. Compile results for the top movies for display
        results = []
        for i in range(NUMPY_DISPLAY_LIMIT):
            if i >= len(scores):
                break
            results.append({
                "title": movie_titles[i],
                "raw": f"{scores[i]:.2f}",
                "z_score": f"{z_scores[i]:.4f}",
                "min_max": f"{min_max_scores[i]:.4f}"
            })
            
        analysis_stats = {
            "mean": f"{mean:.2f}",
            "std": f"{std_dev:.2f}",
            "min": f"{min_score:.2f}",
            "max": f"{max_score:.2f}"
        }

        return jsonify({
            "results": results,
            "stats": analysis_stats
        })
    except Exception as e:
        print(f"Numpy Data Generation Failed: {e}")
        return jsonify({"error": f"Internal Server Error during Numpy data generation: {str(e)}"}), 500

######################################################################################
###                            TensorFlow ENDPOINT                                 ###
###           Digit recognition (MNIST) - The "hello world" of ML/DL               ###
###         build+train a neural net to recognize hand-written digits              ###
######################################################################################  
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"result": "Program error - No image uploaded"}), 400
    try:
        file = request.files["image"]
        arr_2d = api_utils.preprocess_image(file)           # Preprocess to 2D array (28, 28) convert to grayscale, resize, convert to numpy array, normalize ...
        img_b64 = api_utils.get_image_b64(arr_2d)           # Generate Base64 from 2D array for Gemini/Response
        digit = None
        confidence = 0.0
        margin = 0.0
        engine = "TensorFlow (Local)"
        if np.sum(arr_2d > 0.2) < 10:                           # Almost nothing drawn
            digit = "EMPTY"
            result = "Drawing/Canvas is empty or too small."
            confidence = 0.0
        else:
            if api_utils.HAS_TF:                                # Using Local TensorFlow Model...
                model = api_utils.get_or_train_model1()
                input_arr = arr_2d.reshape(1, 28, 28)           # Reshape ONLY for the prediction
                probs = model.predict(input_arr, verbose=0)[0]  # predict Using Local TensorFlow Model
                
                digit = int(np.argmax(probs))
                confidence = float(np.max(probs))
                sorted_probs = np.sort(probs)                   # Additional TF Heuristics
                margin = float(sorted_probs[-1] - sorted_probs[-2])
            else:                                               # TensorFlow not found. Use Google Gemini API...
                gemini_res = api_utils.call_gemini_vision(img_b64)
                digit = gemini_res.get("digit")
                confidence = float(gemini_res.get("confidence", 0.0))
                margin = 0.5 
                engine = "Gemini (Cloud API)"
            # Heuristics for "No Digit"
            if str(digit) == "gemini_ERR":
                result = "Google Gemini API AI Service temporarily unavailable."
            elif str(digit) == "gemini_API":
                result = "Google Gemini API key not valid."
            elif confidence < 0.80 or margin < 0.4:             # Model is guessing/unsure
                digit = "UNSURE"
                source = "TF Model" if api_utils.HAS_TF else "Gemini API"
                result = f"Input is unclear. {source} is guessing/unsure."
            else:
                result = f"{digit} (confidence={confidence:.2f})"
        return jsonify({"result": result,
                        "digit": str(digit),
                        "confidence": str(round(confidence, 4)), 
                        "mnist_img": img_b64,
                        "engine": engine}) 
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"result": f"Server Error: {str(e)}"}), 500

######################################################################################
###       C / Python Integration and Performance Benchmark Test AJAX endpoint      ###
###    run the Performance Benchmark C / Python - tests the sum of 1 million int   ###
######################################################################################
from api_c_bridge import c_bridge
@app.route('/api/runCbridge', methods=['POST'])
def runCbridge():
    try:
        benchmark_output,time_c,time_py = c_bridge(10_000_000) # Create a large dataset (10 Million items)
        return jsonify({
            "status": "success",
            "result": benchmark_output,
            "time_c": time_c,
            "time_py": time_py
        })
    except Exception as e:
        print(f"Performance Benchmark test failed: {e}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
    
######################################################################################
###                             Flask application start                            ###
###################################################################################### 
if __name__ == '__main__':
    if api_utils.FLASK_ENV == 'PRODUCTION':
        print("Flask application starting... Running in PRODUCTION mode")
        app.run(port=8080)
    else:
        print("Flask application starting... Running in DEVELOPMENT mode")
        app.run(port=5000, debug=True)