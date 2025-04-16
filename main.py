import requests
import pandas as pd
pd.set_option('future.no_silent_downcasting', True) # Address FutureWarning
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
# Import request object to access query parameters
from flask import Flask, render_template_string, abort, request
import traceback # Import traceback for detailed error logging

# --- Flask App Setup ---
app = Flask(__name__)
# --- HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    {# Updated title to reflect two lines more clearly if needed, keeping original structure #}
    <title>Data Quality Analysis for Measurement Site {{ station_name | default('', true) }} Site ID{{ site_id }}</title>
    <script src='https://cdn.plot.ly/plotly-latest.min.js'></script> {# Load Plotly.js from CDN #}
    <style>
        body { font-family: sans-serif; margin: 20px; }
        /* Keep h1 centered */
        h1 {
           text-align: center;
           /* Remove default browser margins for h1 if needed for tighter spacing */
           margin-block-start: 0.67em;
           margin-block-end: 0.67em;
        }
        .error { color: red; text-align: center; font-weight: bold; }
        #plot { margin-top: 20px; }
    </style>
</head>
<body>
    {# CHANGE 1 (Header): Applied multi-line styling with different font sizes/color #}
    <h1>
        <span style="font-size: 18px;">Data Quality Analysis for Measurement Site</span><br>
        <span style="font-size: 14px; color: darkblue;">{{ station_name | default('Unknown Station', true) }} (Site ID={{ site_id | default('N/A', true) }})</span>
    </h1>
    {% if error %}
        <p class="error">Error: {{ error }}</p>
    {% elif plot_div %} {# Check if plot_div exists #}
        {# Embed the Plotly plot #}
        <div id='plot'>{{ plot_div | safe }}</div>
    {% else %}
        {# Optional: Message if no plot and no error (e.g., missing ID) #}
         <p style="text-align: center;">Please provide a valid site ID in the URL (e.g., /plot?id=10987).</p>
    {% endif %}
</body>
</html>
"""

# --- Core Data Processing and Plotting Function ---
def generate_plot_for_site(site_id):
    """
    Fetches data for a given site_id, performs anomaly detection,
    and returns a Plotly figure, an error message, and the station name.
    """
    station_name = None # Initialize station_name
    try:
        # Define parameters
        # Use current date based on system clock when the function is called
        # Note: Using today's date for the end_date
        end_date = datetime.today().strftime("%Y-%m-%d")
        print(f"Using end_date: {end_date}") # Log the date being used

        # Construct API URL
        api_url = f"https://www.waterrights.utah.gov/dvrtdb/daily-chart.asp?station_id={site_id}&end_date={end_date}&f=json"
        print(f"Fetching data from: {api_url}") # Keep for debugging on server console

        # Fetch data from the API
        response = requests.get(api_url, timeout=30) # Added timeout

        # Check if request was successful
        if response.status_code != 200:
            return None, f"Error fetching data from API (Status code: {response.status_code}) for site {site_id}", None

        data = response.json()

        # Extract time series data
        if "data" not in data or not data["data"]: # Check if 'data' exists and is not empty
             return None, f"No time series 'data' found in API response for site {site_id}.", None

        # --- Start of data processing logic ---
        df = pd.DataFrame(data["data"], columns=["date", "value"])
        df.rename(columns={"date": "Date", "value": "DISCHARGE"}, inplace=True)

        metadata_fields = ["station_id", "station_name", "system_name", "units"]
        metadata = {field: data.get(field, "N/A") for field in metadata_fields}
        station_name = metadata.get('station_name', 'N/A') # Extract station name

        for key, value in metadata.items():
            df[key] = value

        column_order = ["Date", "DISCHARGE"]
        for field in metadata_fields:
            if field in df.columns:
                column_order.append(field)
        df = df[column_order]

        # Convert 'Date' to datetime objects for proper sorting and plotting
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date') # Ensure data is sorted by date

        df['DISCHARGE'] = pd.to_numeric(df['DISCHARGE'], errors='coerce')

        # === FLAGGING CRITERIA ===
        df['FLAG_NEGATIVE'] = (df['DISCHARGE'] < 0) & (df['DISCHARGE'] != 0)
        df['FLAG_ZERO'] = (df['DISCHARGE'] == 0)

        df_nonzero = df[df["DISCHARGE"] > 0].copy()

        # Initialize flags and metrics with defaults in the main df
        df["RATE_OF_CHANGE"] = np.nan
        df["OUTLIER_IF"] = pd.NA # Use pandas NA for nullable boolean
        df["FLAG_REPEATED"] = pd.NA
        df["PERCENT_DEV"] = np.nan
        # Initialize boolean flags to False
        df["FLAG_RSD"] = False
        df["FLAG_Discharge"] = False
        df["FLAG_IQR"] = False
        df["FLAG_RoC"] = False # Initialize RoC flag
        df["FLAG_ABOVE_MAX_OVERLAP"] = False
        df["FLAG_LARGE_SPIKES"] = False

        # Variables needed outside the 'if not df_nonzero.empty' block
        discharge_90th_percentile = 0
        discharge_95th_percentile = 0
        Q1, Q3, IQR = 0, 0, 0

        if not df_nonzero.empty:
            Q1, Q3 = df_nonzero["DISCHARGE"].quantile([0.25, 0.75])
            IQR = Q3 - Q1 if pd.notna(Q1) and pd.notna(Q3) else 0 # Calculate IQR safely
            # Ensure dropna() before percentile calculation
            discharge_clean = df_nonzero["DISCHARGE"].dropna()
            if not discharge_clean.empty:
                discharge_90th_percentile = np.percentile(discharge_clean, 90)
                discharge_95th_percentile = np.percentile(discharge_clean, 95)

            # Compute Rate of Change only on non-zero, then merge
            df_nonzero["RATE_OF_CHANGE"] = df_nonzero["DISCHARGE"].diff().abs()
            # Ensure RATE_OF_CHANGE exists before merging, even if all NaNs
            if "RATE_OF_CHANGE" in df_nonzero.columns:
                df = df.merge(df_nonzero[["Date", "RATE_OF_CHANGE"]], on="Date", how="left", suffixes=('', '_nonzero'))
                # If merge created duplicate RoC columns, prioritise the one from df_nonzero
                if 'RATE_OF_CHANGE_nonzero' in df.columns:
                    df['RATE_OF_CHANGE'] = df['RATE_OF_CHANGE_nonzero'].combine_first(df['RATE_OF_CHANGE'])
                    df = df.drop(columns=['RATE_OF_CHANGE_nonzero'])
            else:
                 # If RoC wasn't calculated (e.g., single non-zero point), ensure column still exists from init
                 if "RATE_OF_CHANGE" not in df.columns:
                     df["RATE_OF_CHANGE"] = np.nan


            # Flag Repeated Values (4 or more in a row) on non-zero
            if len(df_nonzero) >= 4: # Need at least 4 rows for this to be possible
                is_different = (df_nonzero["DISCHARGE"] != df_nonzero["DISCHARGE"].shift())
                df_nonzero["FLAG_REPEATED"] = df_nonzero["DISCHARGE"].groupby(
                    is_different.cumsum()
                ).transform("count") >= 4
            else:
                df_nonzero["FLAG_REPEATED"] = False # Not possible with < 4 rows

            # Run Isolation Forest
            df_iforest = df_nonzero[["DISCHARGE"]].dropna()
            if not df_iforest.empty:
                # Adjust contamination if dataset is small
                contamination_value = 'auto' if len(df_iforest) > 10 else 0.01
                model = IsolationForest(contamination=contamination_value, random_state=42)
                try:
                    # Assign prediction back using index
                    df_nonzero.loc[df_iforest.index, "OUTLIER_IF_PREDICT"] = model.fit_predict(df_iforest)
                    df_nonzero["OUTLIER_IF"] = df_nonzero["OUTLIER_IF_PREDICT"] == -1
                    df_nonzero.drop(columns=["OUTLIER_IF_PREDICT"], inplace=True)
                except ValueError as if_error:
                    print(f"Isolation Forest failed: {if_error}. Assigning False to OUTLIER_IF.")
                    df_nonzero["OUTLIER_IF"] = False # Handle potential fit errors
            else:
                df_nonzero["OUTLIER_IF"] = False

            # Merge IF and Repeated flags back
            flags_to_merge = df_nonzero[["Date", "OUTLIER_IF", "FLAG_REPEATED"]].copy()
            flags_to_merge['OUTLIER_IF'] = flags_to_merge['OUTLIER_IF'].astype('boolean')
            flags_to_merge['FLAG_REPEATED'] = flags_to_merge['FLAG_REPEATED'].astype('boolean')
            # Merge back to the main df using suffixes to avoid column name conflicts
            df = df.merge(flags_to_merge, on="Date", how="left", suffixes=('', '_nonzero'))
            # Combine the flags, giving priority to the merged values
            if 'OUTLIER_IF_nonzero' in df.columns:
                 df['OUTLIER_IF'] = df['OUTLIER_IF_nonzero'].combine_first(df['OUTLIER_IF'])
                 df = df.drop(columns=['OUTLIER_IF_nonzero'])
            if 'FLAG_REPEATED_nonzero' in df.columns:
                 df['FLAG_REPEATED'] = df['FLAG_REPEATED_nonzero'].combine_first(df['FLAG_REPEATED'])
                 df = df.drop(columns=['FLAG_REPEATED_nonzero'])


            # Compute Relative Standard Deviation on non-zero data
            mean_discharge = discharge_clean.mean()
            if mean_discharge != 0:
                # Calculate on non-zero, then map back based on index/date might be safer
                df_nonzero["PERCENT_DEV"] = ((df_nonzero["DISCHARGE"] - mean_discharge).abs() / mean_discharge) * 100
                # Merge PERCENT_DEV back
                df = df.merge(df_nonzero[["Date", "PERCENT_DEV"]], on="Date", how="left", suffixes=('', '_nonzero'))
                if 'PERCENT_DEV_nonzero' in df.columns:
                    df['PERCENT_DEV'] = df['PERCENT_DEV_nonzero'].combine_first(df['PERCENT_DEV'])
                    df = df.drop(columns=['PERCENT_DEV_nonzero'])
            else:
                 df["PERCENT_DEV"] = np.nan # Already initialized, but good practice

            threshold = 1000
            # Calculate FLAG_RSD based on potentially updated PERCENT_DEV
            # Ensure PERCENT_DEV is numeric before comparison
            percent_dev_numeric = pd.to_numeric(df["PERCENT_DEV"], errors='coerce')
            df["FLAG_RSD"] = ((percent_dev_numeric > threshold) & (df["DISCHARGE"] != 0)).fillna(False)


        # === Final Flag Calculations (on the main df) ===
        # Ensure boolean flags are standard boolean type, filling NA with False
        # Apply conversion here, *after* all merging/calculation is done for these flags
        df['OUTLIER_IF'] = df['OUTLIER_IF'].fillna(False).astype(bool)
        df['FLAG_REPEATED'] = df['FLAG_REPEATED'].fillna(False).astype(bool)
        df['FLAG_NEGATIVE'] = df['FLAG_NEGATIVE'].fillna(False).astype(bool)
        df['FLAG_ZERO'] = df['FLAG_ZERO'].fillna(False).astype(bool)


        # === Robust Flag Calculations (using variables from df_nonzero analysis) ===
        if pd.api.types.is_number(discharge_95th_percentile):
            df["FLAG_Discharge"] = (df["DISCHARGE"] > discharge_95th_percentile).fillna(False)
        else:
            df["FLAG_Discharge"] = False

        if pd.api.types.is_number(Q1) and pd.api.types.is_number(Q3) and pd.api.types.is_number(IQR) and IQR > 0: # IQR must be positive
            discharge_numeric = pd.to_numeric(df["DISCHARGE"], errors='coerce')
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df["FLAG_IQR"] = ((discharge_numeric < lower_bound) | (discharge_numeric > upper_bound)).fillna(False)
        else:
            df["FLAG_IQR"] = False

        # --- MODIFIED RoC FLAG CALCULATION ---
        if "RATE_OF_CHANGE" in df.columns and pd.api.types.is_number(discharge_90th_percentile) and discharge_90th_percentile >= 0:
            roc_numeric = pd.to_numeric(df["RATE_OF_CHANGE"], errors='coerce')
            df["FLAG_RoC"] = (roc_numeric > discharge_90th_percentile).fillna(False)
        else:
            df["FLAG_RoC"] = False
        # --- END MODIFIED RoC FLAG CALCULATION ---

        # Ensure final boolean flags are actual booleans (not objects/nullable)
        df['FLAG_RSD'] = df['FLAG_RSD'].fillna(False).astype(bool)
        df['FLAG_IQR'] = df['FLAG_IQR'].fillna(False).astype(bool)
        df['FLAG_Discharge'] = df['FLAG_Discharge'].fillna(False).astype(bool)
        df['FLAG_RoC'] = df['FLAG_RoC'].fillna(False).astype(bool)


        # Above Max Overlap (using final boolean flags)
        df["FLAG_ABOVE_MAX_OVERLAP"] = (df["FLAG_IQR"] & df["FLAG_Discharge"] & df["OUTLIER_IF"])

        # Large Spikes (using final boolean flags)
        df["FLAG_LARGE_SPIKES"] = df["FLAG_RSD"] & df["FLAG_RoC"]

        # Overall Flagged
        df["FLAGGED"] = df[
            ["FLAG_NEGATIVE", "FLAG_ZERO", "FLAG_REPEATED", "FLAG_ABOVE_MAX_OVERLAP", "FLAG_LARGE_SPIKES"]
        ].any(axis=1)

        # --- Plotting ---
        # Plot title uses station name from metadata
        plot_title = f""
        flag_colors = {
            'FLAG_NEGATIVE': ('red', 'Below Capacity (-)'),
            'FLAG_ZERO': ('blue', 'Value = 0'),
            'FLAG_REPEATED': ('green', 'Repeated (≥4 days)'),
            'FLAG_ABOVE_MAX_OVERLAP': ('brown', 'Over Capacity'),
            'FLAG_LARGE_SPIKES': ('orange', 'Large Spikes'),
        }

        fig = go.Figure()

        # Background Line
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['DISCHARGE'],
            mode='lines',
            line=dict(color='lightgray', width=1.5),
            name='Mean Daily Discharge',
            connectgaps=False
        ))

        # Add flagged points
        for flag, (color, legend_name) in flag_colors.items():
             # Ensure flag column exists and is boolean
             if flag in df.columns and df[flag].dtype == 'bool':
                subset = df[df[flag]] # Select rows where the flag is True
                # Calculate count and modify legend name
                count = len(subset) # Or df[flag].sum()
                legend_entry = f"{legend_name} [{count}]"
                if not subset.empty:
                    fig.add_trace(go.Scatter(
                        x=subset['Date'], y=subset['DISCHARGE'],
                        mode='markers',
                        marker=dict(color=color, size=7),
                        name=legend_entry # Use modified name with count
                    ))

        # Add horizontal line for min value of Above Max
        above_max_subset = df[df["FLAG_ABOVE_MAX_OVERLAP"]]
        if not above_max_subset.empty:
            min_above_max = above_max_subset["DISCHARGE"].min()
            if pd.notna(min_above_max) and np.isfinite(min_above_max):
                fig.add_trace(go.Scatter(
                    x=[df["Date"].min(), df["Date"].max()],
                    y=[min_above_max, min_above_max],
                    mode="lines",
                    line=dict(color="brown", width=2, dash="dash"),
                    name="Estimated Max Capacity" # Consider adding count here too if desired
                ))

        # Update layout with new settings
        fig.update_layout(
            title=dict(text=plot_title, x=0.5, font=dict(size=20)),
            xaxis=dict(title="Date", title_font=dict(size=18), tickfont=dict(size=14)),
            yaxis=dict(
                title=f"Mean Daily Discharge ({metadata.get('units', 'CFS')})",
                title_font=dict(size=18),
                tickfont=dict(size=14),
                # type='log' # Uncomment this if log scale is often needed
            ),
            legend=dict(
                orientation="v",
                x=1.02,
                y=1,
                xanchor="left",
                yanchor="top",
                title=dict(text="Suspected Data Quality Issues:", font=dict(size=16)),
                font=dict(size=14)
            ),
            # ***** INCORPORATED SETTINGS *****
            template="plotly_white",
            margin=dict(t=100, r=250),  # Added margin (top=100, right=250)
            width=1500,               # Added fixed width
            height=800                # Added fixed height
            # *********************************
        )
        # --- End of plotting logic ---

        return fig, None, station_name # Return figure, no error, and station_name

    except requests.exceptions.RequestException as e:
        print(f"API Request failed for site {site_id}: {e}")
        return None, f"Could not connect to the data API: {e}", None # Return None for station_name on error
    except (KeyError, IndexError, ValueError, TypeError) as e:
         # Catch potential errors during data processing/parsing
        print(f"Data processing error for site {site_id}: {e}")
        print(traceback.format_exc()) # Print detailed traceback to console
        # Try to return station_name if it was fetched before the error
        fetched_station_name = station_name if 'station_name' in locals() and station_name else None
        return None, f"Error processing data for site {site_id}: {e}. Check API response format or data values.", fetched_station_name
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred for site {site_id}: {e}")
        print(traceback.format_exc()) # Print detailed traceback to console
        # Try to return station_name if it was fetched before the error
        fetched_station_name = station_name if 'station_name' in locals() and station_name else None
        return None, f"An unexpected error occurred while processing data for site {site_id}.", fetched_station_name


# --- Flask Route ---
@app.route('/plot') # Route path is /plot
def show_plot():
    """
    Flask route to display the data plot for a given site ID from query parameter.
    e.g., /plot?id=10987
    """
    # Get 'id' from query parameters (e.g., ?id=10987)
    site_id = request.args.get('id')

    # --- Input Validation ---
    if not site_id:
        # If no 'id' provided in the URL
        return render_template_string(HTML_TEMPLATE, site_id=None, station_name=None, error="Missing 'id' parameter in URL. Please use format /plot?id=SITE_ID", plot_div=None), 400

    if not site_id.isdigit():
        # If 'id' is not numeric
         return render_template_string(HTML_TEMPLATE, site_id=site_id, station_name=None, error="Invalid Site ID format. Must be numeric.", plot_div=None), 400

    # --- Processing ---
    print(f"--- Processing request for site_id: {site_id} from query parameter ---") # Log start of request
    # Capture all three return values
    fig, error_message, station_name = generate_plot_for_site(site_id)
    print(f"--- Finished processing for site_id: {site_id} ---") # Log end of request

    # --- Response Handling ---
    plot_div = None
    status_code = 200 # Default status code

    if error_message:
        # Determine status code based on error type
        status_code = 404 if "fetching data" in error_message or "No time series" in error_message else 500
    elif fig:
        # Convert plot to HTML only if figure exists
        plot_div = fig.to_html(full_html=False, include_plotlyjs=False)
    else:
        # Handle case where fig is None but no specific error message was returned
        error_message = "Failed to generate plot (unknown reason)."
        status_code = 500

    # Render the template, passing all necessary context including station_name
    return render_template_string(
        HTML_TEMPLATE,
        site_id=site_id,
        station_name=station_name, # Pass station_name to the template
        error=error_message,
        plot_div=plot_div
    ), status_code


# --- Root Route ---
@app.route('/')
def index():
    """
    Provides basic instructions and links using the new URL format.
    """
    example_ids = ["10987", "3133", "10543"] # Add some known working IDs if possible
    links_html = "<ul>"
    for ex_id in example_ids:
        # Link format uses query parameter /plot?id=...
        links_html += f'<li><a href="/plot?id={ex_id}">View data for site {ex_id}</a></li>'
    links_html += "</ul>"
    return f"""
    <h1>Water Data Viewer</h1>
    <p>Enter a site ID using the format /plot?id=SITE_ID</p>
    <h2>Examples:</h2>
    {links_html}
    """

# --- Run the App ---
if __name__ == '__main__':
    # Runs the Flask development server.
    # Important: debug=True is for development only. Set to False in production.
    # host='0.0.0.0' makes it accessible on your network.
    app.run(debug=True, host='0.0.0.0', port=5000)
