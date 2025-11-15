# Import Flask components for web framework, routing, and session management
from flask import Flask, render_template, request, jsonify, session
# Import pandas for data manipulation and analysis
import pandas as pd
# Import plotly express for quick and easy chart creation
import plotly.express as px
# Import plotly graph objects for advanced chart customization
import plotly.graph_objects as go
# Import json for handling JSON data structures
import json
# Import os for file system operations
import os
# Import secure_filename to safely handle uploaded filenames
from werkzeug.utils import secure_filename
# Import warnings module to suppress non-critical warnings
import warnings

# Suppress all warnings to keep the output clean
warnings.filterwarnings('ignore')

# Create a Flask application instance
app = Flask(__name__)
# Set the secret key for session management and secure cookies (should be changed in production)
app.secret_key = 'your-secret-key-here-change-this'  # Change this in production
# Configure the folder where uploaded files will be stored
app.config['UPLOAD_FOLDER'] = 'uploads'
# Set the maximum file upload size to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
# Define the file extensions that are allowed to be uploaded
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

# Create the uploads folder if it doesn't exist (exist_ok=True prevents error if folder already exists)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define a function to check if a file has an allowed extension
def allowed_file(filename):
    # Check if filename contains a dot and extract the extension after the last dot
    # Then verify if the extension (in lowercase) is in the ALLOWED_EXTENSIONS set
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define a function to load data from Excel or CSV files
def load_data(filepath):
    # Try to load the file as either CSV or Excel depending on file extension
    """Load Excel or CSV file"""
    try:
        # Check if the file ends with .csv
        if filepath.endswith('.csv'):
            # If CSV, read using pandas read_csv function
            df = pd.read_csv(filepath)
        else:
            # Otherwise, read as Excel file using pandas read_excel function
            df = pd.read_excel(filepath)
        # Return the dataframe and None for error
        return df, None
    except Exception as e:
        # If any error occurs, return None for dataframe and the error message as a string
        return None, str(e)

# Define a function to analyze and categorize columns as dimensions (categorical) or measures (numerical)
def analyze_columns(df):
    # Start with basic type detection by selecting columns with object or category data types
    """Analyze and categorize columns"""
    # Start with basic type detection
    dimensions = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Select columns with numeric data types (float64, int64, float32, int32)
    measures = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()
    
    # Try to convert object columns to numeric and identify actual measures
    # Loop through a copy of dimensions list to safely remove items during iteration
    for col in dimensions[:]:  # Create a copy to iterate over
        try:
            # Attempt to convert column values to numeric, replacing non-numeric with NaN
            numeric_vals = pd.to_numeric(df[col], errors='coerce')
            # If at least 80% of values can be converted to numeric, treat column as a measure
            if numeric_vals.notna().sum() / len(df) > 0.8:  # At least 80% numeric
                # Remove column from dimensions list
                dimensions.remove(col)
                # Add column to measures list
                measures.append(col)
        except:
            # If conversion fails, keep column in dimensions
            pass
    
    # Remove ID-like columns from measures to exclude identifier columns
    # Define keywords that typically indicate ID columns
    id_keywords = ['id', 'ID', 'Id', 'code', 'Code', 'number', 'Number', 'Code']
    # Filter out measures that contain any ID keywords in their name
    measures = [col for col in measures if not any(keyword in col for keyword in id_keywords)]
    
    # Remove columns with too many unique values (likely IDs or primary keys)
    # Keep only columns where unique value count is less than 95% of total rows
    measures = [col for col in measures if df[col].nunique() < len(df) * 0.95]
    
    # Remove measures with no variation (all same value or all unique single values)
    # Keep only columns with more than 1 unique value
    measures = [col for col in measures if df[col].nunique() > 1]
    
    # Remove columns that are entirely empty (all NaN values)
    # Keep only columns that have at least some non-null values
    measures = [col for col in measures if df[col].notna().sum() > 0]
    
    # Return the lists of dimensions and measures
    return dimensions, measures

# Define route handler for the home page (GET request to '/')
@app.route('/')
# Define the index function that handles the home page
def index():
    # Return the home page by rendering the index.html template
    """Home page with file upload"""
    return render_template('index.html')

# Define route handler for file upload (POST request to '/upload')
@app.route('/upload', methods=['POST'])
# Define the upload_file function that processes uploaded files
def upload_file():
    # Check if 'file' key exists in the request.files dictionary
    """Handle file upload"""
    if 'file' not in request.files:
        # If no file was provided in the upload, return an error response
        return jsonify({'error': 'No file provided'}), 400
    
    # Extract the uploaded file object from request.files
    file = request.files['file']
    
    # Check if a filename was actually provided (filename is empty string if not selected)
    if file.filename == '':
        # If filename is empty, return an error response
        return jsonify({'error': 'No file selected'}), 400
    
    # Verify that the uploaded file has an allowed extension
    if not allowed_file(file.filename):
        # If file extension is not allowed, return an error response
        return jsonify({'error': 'Invalid file type. Please upload Excel (.xlsx, .xls) or CSV files'}), 400
    
    try:
        # Save the uploaded file securely using secure_filename to prevent path traversal attacks
        filename = secure_filename(file.filename)
        # Create the full file path by joining the upload folder and filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # Save the file to the uploads folder
        file.save(filepath)
        
        # Load and analyze the uploaded data
        # Call load_data function to read the file into a dataframe
        df, error = load_data(filepath)
        # Check if there was an error loading the file
        if error:
            # If error occurred, return error message with 400 status code
            return jsonify({'error': f'Error loading file: {error}'}), 400
        
        # Analyze the dataframe to categorize columns as dimensions or measures
        dimensions, measures = analyze_columns(df)
        
        # Store the file path in the session so it can be accessed in other routes
        session['filepath'] = filepath
        # Store the original filename in the session for display purposes
        session['filename'] = filename
        
        # Return a successful response with data information
        return jsonify({
            # Indicate success
            'success': True,
            # Include the filename
            'filename': filename,
            # Include the number of rows in the dataframe
            'rows': len(df),
            # Include the number of columns in the dataframe
            'columns': len(df.columns),
            # Include the list of dimension columns
            'dimensions': dimensions,
            # Include the list of measure columns
            'measures': measures,
            # Include the URL to redirect to after upload
            'redirect': '/dashboard'
        })
        
    except Exception as e:
        # If any unexpected error occurs, return error message with 500 status code
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

# Define route handler for the dashboard page (GET request to '/dashboard')
@app.route('/dashboard')
# Define the dashboard function that displays visualizations
def dashboard():
    # Retrieve the file path from the session
    """Dashboard page with visualizations"""
    filepath = session.get('filepath')
    # Retrieve the filename from the session
    filename = session.get('filename')
    
    # Check if filepath exists in session and if the file still exists on disk
    if not filepath or not os.path.exists(filepath):
        # If no file path or file doesn't exist, return to home page with error message
        return render_template('index.html', error='Please upload a file first')
    
    # Load the data from the saved file path
    df, error = load_data(filepath)
    # Check if there was an error loading the file
    if error:
        # If error occurred, return to home page with error message
        return render_template('index.html', error=f'Error loading data: {error}')
    
    # Analyze the dataframe to categorize columns
    dimensions, measures = analyze_columns(df)
    
    # Return the dashboard page with data information
    return render_template('dashboard.html',
                         # Pass the filename to the template
                         filename=filename,
                         # Pass the number of rows to the template
                         rows=len(df),
                         # Pass the number of columns to the template
                         columns=len(df.columns),
                         # Pass the dimension columns to the template
                         dimensions=dimensions,
                         # Pass the measure columns to the template
                         measures=measures)

# Define API endpoint to create a bar chart (POST request to '/api/chart/bar')
@app.route('/api/chart/bar', methods=['POST'])
# Define the create_bar_chart function that generates bar chart visualizations
def create_bar_chart():
    # Add try-except block to handle errors gracefully
    """API endpoint to create bar chart with optional second dimension and grouping"""
    try:
        # Import traceback module to print detailed error stack traces
        import traceback
        # Print debug header to console
        print("\n=== BAR CHART REQUEST RECEIVED ===")
        
        # Extract JSON data from the request body
        data = request.json
        # Print the request data to console for debugging
        print(f"Request data: {data}")
        
        # Extract the first dimension from the request data
        dimension1 = data.get('dimension1')
        # Extract the second dimension from the request data (optional)
        dimension2 = data.get('dimension2')
        # Extract the measure (metric) to be visualized from the request data
        measure = data.get('measure')
        # Extract the facet flag to determine if a faceted chart should be used
        use_facet = data.get('use_facet', False)
        
        # Print extracted parameters to console for debugging
        print(f"D1: {dimension1}, D2: {dimension2}, M: {measure}, Facet: {use_facet}")
        
        # Validate that required inputs (dimension1 and measure) are provided
        # Validate inputs
        if not dimension1 or not measure:
            # If required inputs are missing, return error response
            return jsonify({'error': 'dimension1 and measure are required'}), 400
        
        # Retrieve the file path from the session
        filepath = session.get('filepath')
        # Check if filepath exists and the file actually exists on disk
        if not filepath or not os.path.exists(filepath):
            # If no data is loaded, return error response
            return jsonify({'error': 'No data loaded'}), 400
        
        # Load the data from the file
        # Load data
        df, error = load_data(filepath)
        # Check if there was an error loading the file
        if error:
            # If error occurred, return error response
            return jsonify({'error': f'Error loading data: {error}'}), 400
        
        # Print the dataframe shape (rows, columns) to console for debugging
        print(f"Data loaded. Shape: {df.shape}")
        # Print the list of column names to console for debugging
        print(f"Columns: {df.columns.tolist()}")
        
        # Initialize list with columns needed for the visualization
        # Prepare data columns
        cols_to_use = [dimension1, measure]
        # Check if a second dimension was provided and is not empty or 'None'
        if dimension2 and dimension2 != '' and dimension2 != 'None':
            # Add the second dimension to the columns list
            cols_to_use.append(dimension2)
            # Store the second dimension for later use
            dimension2_use = dimension2
        else:
            # If no second dimension provided, set it to None
            dimension2_use = None
        
        # Verify that all required columns exist in the dataframe
        # Verify columns exist
        for col in cols_to_use:
            # Check if the column name exists in the dataframe
            if col not in df.columns:
                # If column doesn't exist, return error response
                return jsonify({'error': f'Column "{col}" not found in data'}), 400
        
        # Create a working copy of only the needed columns from the dataframe
        # Select and clean data
        df_work = df[cols_to_use].copy()
        # Print the shape before removing null values
        print(f"Before dropna: {df_work.shape}")
        # Remove rows with any null values in the selected columns
        df_work = df_work.dropna()
        # Print the shape after removing null values
        print(f"After dropna: {df_work.shape}")
        
        # Check if the dataframe is empty after removing null values
        if df_work.empty:
            # If no data remains, return error response
            return jsonify({'error': 'No valid data after removing nulls'}), 400
        
        # Convert the measure column values to numeric type
        # Convert measure to numeric
        # Print the data type of the measure column before conversion
        print(f"Measure column dtype before: {df_work[measure].dtype}")
        # Print sample values from the measure column before conversion
        print(f"Sample values: {df_work[measure].head(10).tolist()}")
        
        # Convert measure column to numeric, replacing non-numeric values with NaN
        df_work[measure] = pd.to_numeric(df_work[measure], errors='coerce')
        # Remove rows where the measure column is null after conversion
        df_work = df_work.dropna(subset=[measure])
        
        # Print the data type of the measure column after conversion
        print(f"Measure column dtype after: {df_work[measure].dtype}")
        # Print sample values from the measure column after conversion
        print(f"Sample values after conversion: {df_work[measure].head(10).tolist()}")
        
        # Check if the dataframe is empty after numeric conversion
        if df_work.empty:
            # If no numeric values exist, return error response
            return jsonify({'error': 'No numeric values in measure column'}), 400
        
        # Aggregate the data by grouping and summing the measure values
        # Aggregate data
        # Check if a second dimension exists
        if dimension2_use:
            # Group by both dimensions and sum the measure values
            grouped = df_work.groupby([dimension1, dimension2_use], as_index=False)[measure].sum()
        else:
            # Group by first dimension only and sum the measure values
            grouped = df_work.groupby(dimension1, as_index=False)[measure].sum()
        
        # Sort the grouped data by measure values in descending order
        grouped = grouped.sort_values(measure, ascending=False)
        
        # Print the shape of the grouped data
        print(f"Grouped data shape: {grouped.shape}")
        # Print the grouped data to console for debugging
        print(f"Grouped data:\n{grouped}")
        
        # Create the visualization based on the parameters
        # Create visualization
        # Check if a second dimension exists and facet mode is enabled
        if dimension2_use and use_facet:
            # Print debug message indicating faceted chart creation
            print("Creating faceted chart...")
            # Create a faceted bar chart with one facet per category of the second dimension
            fig = px.bar(grouped,
                        x=dimension1,
                        y=measure,
                        color=dimension2_use,
                        facet_col=dimension2_use,
                        barmode='group',
                        title=f'{measure} by {dimension1} and {dimension2_use}')
        # Check if a second dimension exists but facet mode is not enabled
        elif dimension2_use:
            # Print debug message indicating grouped chart creation
            print("Creating grouped chart...")
            # Create a grouped bar chart with bars side-by-side for each category
            fig = px.bar(grouped,
                        x=dimension1,
                        y=measure,
                        color=dimension2_use,
                        barmode='group',
                        title=f'{measure} by {dimension1} (Grouped by {dimension2_use})')
        else:
            # Print debug message indicating simple chart creation
            print("Creating simple bar chart...")
            # Create a simple bar chart with one bar per category
            fig = px.bar(grouped,
                        x=dimension1,
                        y=measure,
                        title=f'{measure} by {dimension1}')
        
        # Customize the chart appearance and layout
        # Update layout
        # Add value labels on top of each bar showing the exact value
        fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
        # Update the overall chart layout with formatting options
        fig.update_layout(
            # Set the height of the chart in pixels
            height=500,
            # Rotate x-axis labels by 45 degrees for better readability
            xaxis_tickangle=-45,
            # Use the plotly white template for a clean appearance
            template='plotly_white',
            # Set hover mode to show values for all series on x-axis
            hovermode='x unified'
        )
        
        # Print success message to console
        print("Chart created successfully")
        # Return the chart as JSON in the response
        return jsonify({'chart': fig.to_json()})
        
    except Exception as e:
        # Capture the error message as a string
        error_msg = str(e)
        # Print error message to console
        print(f"\n!!! ERROR in bar chart: {error_msg}")
        # Print the full error traceback to console
        print(traceback.format_exc())
        # Print error end marker to console
        print("=== END ERROR ===\n")
        # Return error response with 500 status code
        return jsonify({'error': error_msg}), 500

# Define API endpoint to create a pie chart (POST request to '/api/chart/pie')
@app.route('/api/chart/pie', methods=['POST'])
# Define the create_pie_chart function that generates pie chart visualizations
def create_pie_chart():
    # Add try-except block to handle errors gracefully
    """API endpoint to create pie chart"""
    try:
        # Extract JSON data from the request body
        data = request.json
        # Extract the dimension (category) for the pie chart
        dimension = data.get('dimension')
        # Extract the measure (value) to be visualized
        measure = data.get('measure')
        
        # Retrieve the file path from the session
        filepath = session.get('filepath')
        # Check if a file path exists in the session
        if not filepath:
            # If no data is loaded, return error response
            return jsonify({'error': 'No data loaded'}), 400
        
        # Load the data from the file
        df, error = load_data(filepath)
        # Check if there was an error loading the file
        if error:
            # If error occurred, return error response
            return jsonify({'error': error}), 400
        
        # Create a clean copy of data with only the needed columns
        # Clean data
        df_clean = df[[dimension, measure]].dropna()
        
        # Check if the dataframe is empty after removing null values
        if df_clean.empty:
            # If no data remains, return error response
            return jsonify({'error': 'No valid data'}), 400
        
        # Check if the measure column contains numeric data
        if not pd.api.types.is_numeric_dtype(df_clean[measure]):
            # If measure is not numeric, return error response
            return jsonify({'error': f'{measure} is not numeric'}), 400
        
        # Group the data by dimension and sum the measure values
        # Group and filter
        grouped = df_clean.groupby(dimension)[measure].sum().reset_index()
        # Filter out rows with zero or negative values
        grouped = grouped[grouped[measure] > 0]
        
        # Limit the pie chart to the top 10 categories for readability
        # Limit to top 10
        if len(grouped) > 10:
            # Keep only the 10 largest values
            grouped = grouped.nlargest(10, measure)
        
        # Create the pie chart visualization
        # Create chart
        fig = px.pie(grouped, names=dimension, values=measure,
                    title=f'{measure} Distribution by {dimension}')
        
        # Customize the pie chart appearance
        # Add percentage and label information inside pie slices
        fig.update_traces(textposition='inside', textinfo='percent+label')
        # Set the height of the chart
        fig.update_layout(height=500)
        
        # Return the chart as JSON in the response
        return jsonify({'chart': fig.to_json()})
        
    except Exception as e:
        # If any error occurs, return error response with error message
        return jsonify({'error': str(e)}), 500

# Define API endpoint to create a sunburst chart (POST request to '/api/chart/sunburst')
@app.route('/api/chart/sunburst', methods=['POST'])
# Define the create_sunburst_chart function that generates hierarchical sunburst visualizations
def create_sunburst_chart():
    # Add try-except block to handle errors gracefully
    """API endpoint to create sunburst chart"""
    try:
        # Extract JSON data from the request body
        data = request.json
        # Extract the first dimension (highest level hierarchy)
        dim1 = data.get('dim1')
        # Extract the second dimension (middle level hierarchy)
        dim2 = data.get('dim2')
        # Extract the third dimension (lowest level hierarchy)
        dim3 = data.get('dim3')
        # Extract the measure (value) to be visualized
        measure = data.get('measure')
        
        # Combine all three dimensions into a list
        dims = [dim1, dim2, dim3]
        
        # Validate that all three dimensions are different (unique)
        # Validate unique dimensions
        if len(set(dims)) < len(dims):
            # If dimensions are not unique, return error response
            return jsonify({'error': 'All three dimensions must be different'}), 400
        
        # Retrieve the file path from the session
        filepath = session.get('filepath')
        # Check if a file path exists in the session
        if not filepath:
            # If no data is loaded, return error response
            return jsonify({'error': 'No data loaded'}), 400
        
        # Load the data from the file
        df, error = load_data(filepath)
        # Check if there was an error loading the file
        if error:
            # If error occurred, return error response
            return jsonify({'error': error}), 400
        
        # Create a clean copy of data with only the needed columns
        # Clean data
        df_clean = df[dims + [measure]].dropna()
        
        # Check if the dataframe is empty after removing null values
        if df_clean.empty:
            # If no data remains, return error response
            return jsonify({'error': 'No complete data available'}), 400
        
        # Check if the measure column contains numeric data
        if not pd.api.types.is_numeric_dtype(df_clean[measure]):
            # If measure is not numeric, return error response
            return jsonify({'error': f'{measure} is not numeric'}), 400
        
        # Convert all dimension columns to string type for consistency
        # Convert to string
        for dim in dims:
            # Convert each dimension column to string
            df_clean[dim] = df_clean[dim].astype(str)
        
        # Filter out rows with zero or negative measure values
        # Filter positive values
        df_clean = df_clean[df_clean[measure] > 0]
        
        # Check if any data remains after filtering
        if df_clean.empty:
            # If no positive values found, return error response
            return jsonify({'error': 'No positive values found'}), 400
        
        # Limit the data to the top 1000 records to avoid performance issues
        # Limit data
        if len(df_clean) > 1000:
            # Keep only the 1000 largest values
            df_clean = df_clean.nlargest(1000, measure)
        
        # Create the sunburst hierarchical visualization
        # Create chart
        fig = px.sunburst(df_clean, path=dims, values=measure,
                         title=f'{measure}: {dim1} → {dim2} → {dim3}')
        
        # Customize the sunburst chart appearance
        # Add label and percentage information to each segment
        fig.update_traces(textinfo='label+percent entry')
        # Set the height of the chart (larger for better visibility)
        fig.update_layout(height=650)
        
        # Return the chart as JSON in the response
        return jsonify({'chart': fig.to_json()})
        
    except Exception as e:
        # If any error occurs, return error response with error message
        return jsonify({'error': str(e)}), 500

# Define API endpoint to get information about loaded data (GET request to '/api/data-info')
@app.route('/api/data-info')
# Define the get_data_info function that returns metadata about the loaded dataset
def get_data_info():
    # Retrieve the file path from the session
    """Get information about loaded data"""
    filepath = session.get('filepath')
    # Check if a file path exists in the session
    if not filepath:
        # If no data is loaded, return error response
        return jsonify({'error': 'No data loaded'}), 400
    
    # Load the data from the file
    df, error = load_data(filepath)
    # Check if there was an error loading the file
    if error:
        # If error occurred, return error response
        return jsonify({'error': error}), 400
    
    # Analyze the dataframe to categorize columns
    dimensions, measures = analyze_columns(df)
    
    # Return metadata about the loaded data
    return jsonify({
        # Include the number of rows in the dataframe
        'rows': len(df),
        # Include the number of columns in the dataframe
        'columns': len(df.columns),
        # Include the list of dimension columns
        'dimensions': dimensions,
        # Include the list of measure columns
        'measures': measures,
        # Include the first 5 rows as sample data (converted to list of dictionaries)
        'sample': df.head(5).to_dict('records')
    })

# Entry point for running the Flask application
if __name__ == '__main__':
    # Run the Flask application in debug mode on port 5000
    app.run(debug=True, port=5000)