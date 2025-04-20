# --- Imports ---
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from flask import Flask, render_template_string, request, redirect, url_for
import traceback
import logging
import json
from typing import Dict, Any, Tuple, Optional, Sequence # Ensure Sequence and Tuple are here
import sys
import os
from pathlib import Path

# --- Pandas Option ---
pd.set_option('future.no_silent_downcasting', True)

# --- Flask App Setup ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# --- Constants ---
REQUIRED_THRESHOLD_COLS = ["Over_Capacity", "Unusual_Spike"] # Corrected spelling
STATIC_MIN_THRESHOLD = 0
BUFFER_PERCENTAGE = 0.10 # Define buffer width as 10% of max capacity
BUFFER_NUM_BANDS = 15 # Number of bands for the gradient buffer
BUFFER_START_COLOR_RGBA = (128, 0, 128, 0.2) # Semi-transparent purple near the line
BUFFER_END_COLOR_RGBA = (128, 0, 128, 0.0)   # Fully transparent purple at the edge


# --- Path Definition (Using robust method with pathlib) ---
try:
    # Use __file__ if running as a script
    script_dir = Path(__file__).resolve().parent
    csv_filename = "thresholds.csv"
    THRESHOLDS_CSV_PATH = script_dir / csv_filename
    if not THRESHOLDS_CSV_PATH.is_file():
        app.logger.warning(f"Threshold file not found at '{THRESHOLDS_CSV_PATH}'. Falling back to relative path '{csv_filename}'.")
        THRESHOLDS_CSV_PATH = csv_filename # Fallback to relative path
    else:
        app.logger.info(f"Using threshold file path: {THRESHOLDS_CSV_PATH}")
except NameError:
    # Fallback if __file__ is not defined (e.g., interactive session)
    csv_filename = "thresholds.csv"
    THRESHOLDS_CSV_PATH = csv_filename
    app.logger.warning(f"Could not determine script directory (__file__ not defined). Using relative path: {THRESHOLDS_CSV_PATH}")


# --- Threshold Loading ---
thresholds_df_global = None

# --- !!! ENHANCED load_thresholds Function !!! ---
def load_thresholds(file_path) -> Optional[pd.DataFrame]: # file_path can be string or Path object
    """Loads the thresholds CSV file into a DataFrame. Returns None on error."""
    global thresholds_df_global # To modify the global variable
    file_path_str = str(file_path) # Ensure it's a string for printing/pandas
    app.logger.info(f"Attempting to load thresholds from: {file_path_str}")
    try:
        # Attempt to read the CSV
        thresholds_df = pd.read_csv(file_path)
        app.logger.info(f"Successfully read CSV file into DataFrame. Checking columns...") # Log success before checks

        # --- Check for Required Columns ---
        missing_cols = [col for col in REQUIRED_THRESHOLD_COLS if col not in thresholds_df.columns]
        if missing_cols:
            # Log the columns *found* in the CSV to help debugging
            found_cols = thresholds_df.columns.tolist()
            app.logger.error(f"Missing required threshold columns in '{file_path_str}': {', '.join(missing_cols)}")
            app.logger.error(f"Columns *found* in CSV header: {found_cols}")
            app.logger.error("Please ensure the column names in the CSV exactly match the REQUIRED_THRESHOLD_COLS list (including capitalization).")
            return None # Return None if required columns are missing
        # --- End Column Check ---

        if "SiteID" not in thresholds_df.columns:
            found_cols = thresholds_df.columns.tolist()
            app.logger.error(f"'SiteID' column not found in thresholds file: '{file_path_str}'.")
            app.logger.error(f"Columns *found* in CSV header: {found_cols}")
            return None # Return None if SiteID column is missing

        # Ensure SiteID is string for reliable comparison
        thresholds_df["SiteID_str"] = thresholds_df["SiteID"].astype(str)

        app.logger.info(f"Thresholds loaded and columns validated successfully from '{file_path_str}'.")
        thresholds_df_global = thresholds_df # Store loaded data globally
        return thresholds_df # Return the DataFrame on success

    except FileNotFoundError:
        app.logger.error(f"!!! FileNotFoundError: Thresholds file not found at '{file_path_str}'. Please verify the path and file existence.")
        return None # Return None specifically for FileNotFoundError
    except pd.errors.EmptyDataError:
         app.logger.error(f"!!! pandas EmptyDataError: The file at '{file_path_str}' seems to be empty.")
         return None # Return None if file is empty
    except pd.errors.ParserError as pe:
         app.logger.error(f"!!! pandas ParserError: Could not parse the CSV file at '{file_path_str}'. Check CSV format (delimiters, quoting, etc.). Error: {pe}", exc_info=False) # exc_info=False as error message is usually clear
         return None # Return None for parsing errors
    except PermissionError:
         app.logger.error(f"!!! PermissionError: Do not have permission to read the file at '{file_path_str}'. Check file/folder permissions.")
         return None # Return None for permission errors
    except Exception as e:
        # Catch any other unexpected exceptions during loading/processing
        app.logger.error(f"!!! Unexpected Error during threshold loading/validation for '{file_path_str}'. Error Type: {type(e).__name__}, Message: {e}", exc_info=True) # Log full traceback for unexpected errors
        return None # Return None for any other exception
# --- !!! END ENHANCED load_thresholds Function !!! ---

# Load thresholds when the application starts
load_thresholds(THRESHOLDS_CSV_PATH)


# --- Helper Functions (Unchanged from previous version, except for added buffer functions below) ---

def get_site_thresholds(thresholds_df: pd.DataFrame, site_id: str) -> Optional[Dict[str, float]]:
    """Gets and validates thresholds for a specific site."""
    app.logger.info(f"Finding and validating thresholds for SiteID {site_id}...")
    if thresholds_df is None or thresholds_df.empty:
        app.logger.error("Thresholds DataFrame is not loaded or empty.")
        return None
    if "SiteID_str" not in thresholds_df.columns:
        app.logger.error("'SiteID_str' column missing from thresholds DataFrame.")
        return None

    site_thresholds_row = thresholds_df[thresholds_df["SiteID_str"] == site_id]

    if site_thresholds_row.empty:
        app.logger.warning(f"SiteID {site_id} not found in thresholds data.")
        return None

    try:
        threshold_row = site_thresholds_row.iloc[0]
        validated_thresholds = {"min_val": float(STATIC_MIN_THRESHOLD)}
        missing_details = []

        for col_name in REQUIRED_THRESHOLD_COLS:
            raw_value = threshold_row[col_name]
            numeric_value = pd.to_numeric(raw_value, errors='coerce')

            if pd.isna(numeric_value):
                missing_details.append(f"'{col_name}' (value found: '{raw_value}')")
            else:
                # Map column names to keys in validated_thresholds
                if col_name == "Over_Capacity":
                    validated_thresholds["max_val"] = float(numeric_value)
                elif col_name == "Unusual_Spike":
                    validated_thresholds["spike_unusual"] = float(numeric_value)
                # Add other mappings here if REQUIRED_THRESHOLD_COLS expands

        if missing_details:
            error_message = (f"Missing or invalid required threshold value(s) for SiteID {site_id} "
                             f"in thresholds data: {', '.join(missing_details)}. Please check the source CSV.")
            app.logger.error(error_message)
            return None # Return None if any required value is missing/invalid

        # Final check to ensure all required keys were populated
        if "max_val" not in validated_thresholds or "spike_unusual" not in validated_thresholds:
            app.logger.error(f"Internal logic error: Did not populate 'max_val' or 'spike_unusual' for SiteID {site_id}")
            return None

        app.logger.info(f"Required threshold values successfully validated for SiteID {site_id}.")
        return validated_thresholds

    except Exception as e:
        app.logger.error(f"Unexpected error extracting/validating thresholds for SiteID {site_id}: {e}", exc_info=True)
        return None


def apply_flagging(df: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    """Applies data quality flags based on provided thresholds."""
    app.logger.info("Applying flagging logic...")
    if not thresholds or 'min_val' not in thresholds or 'max_val' not in thresholds or 'spike_unusual' not in thresholds:
        app.logger.error("Cannot apply flagging: Invalid or incomplete thresholds provided.")
        # Ensure expected columns exist even if flagging fails
        df['FLAGGED'] = False
        flag_cols_expected = ['FLAG_LESS_THAN_Min._Value', 'FLAG_ZERO', 'FLAG_REPEATED',
                              'FLAG_GREATER_THAN_MaxValue', 'UNUSUAL_SPIKE', 'FLAG_BELOW_CAPACITY']
        for col in flag_cols_expected:
            if col not in df.columns: df[col] = False
        return df

    min_val = thresholds["min_val"]
    max_val = thresholds["max_val"]
    spike_unusual = thresholds["spike_unusual"]
    app.logger.info(f"Using thresholds: Min={min_val}, Max={max_val}, Spike={spike_unusual}")

    # Ensure DISCHARGE column exists and is numeric
    if 'DISCHARGE' in df.columns:
        df['DISCHARGE'] = pd.to_numeric(df['DISCHARGE'], errors='coerce')

        # Apply flags (handle NaNs introduced by coerce or already present)
        df['FLAG_LESS_THAN_Min._Value'] = (df['DISCHARGE'] < min_val) & (df['DISCHARGE'].notna()) & (df['DISCHARGE'] != 0) # Exclude zero from this flag
        df['FLAG_ZERO'] = df['DISCHARGE'] == 0
        df['FLAG_BELOW_CAPACITY'] = (df['DISCHARGE'] < 0) & (df['DISCHARGE'].notna()) # Specifically for negative values
        df['FLAG_GREATER_THAN_MaxValue'] = (df['DISCHARGE'] > max_val) & (df['DISCHARGE'].notna())

        # Rate of change calculation (ensure it handles NaNs gracefully)
        df['RATE_OF_CHANGE'] = df['DISCHARGE'].diff().abs()
        df['UNUSUAL_SPIKE'] = (df['RATE_OF_CHANGE'] > spike_unusual) & (df['RATE_OF_CHANGE'].notna())
        app.logger.info(f"Applied Unusual Spike Threshold: {spike_unusual}. Found {df['UNUSUAL_SPIKE'].sum()} spikes.")


        # Flag repeated non-zero values (more robust check)
        df['FLAG_REPEATED'] = False # Initialize
        non_zero_discharge = df['DISCHARGE'].where(df['DISCHARGE'] != 0)
        if not non_zero_discharge.isna().all(): # Check if there are any non-zero values
            group_ids = (non_zero_discharge != non_zero_discharge.shift()).cumsum()
            repeat_counts = non_zero_discharge.groupby(group_ids).transform('size')
            df.loc[non_zero_discharge.notna(), 'FLAG_REPEATED'] = repeat_counts >= 4

        app.logger.info(f"Found {df['FLAG_REPEATED'].sum()} instances of repeated non-zero values (>=4 days).")

        # Combine flags
        df['FLAG_DISCHARGE_CHANGE'] = False # Placeholder if needed later
        flag_columns_list = ['FLAG_LESS_THAN_Min._Value', 'FLAG_ZERO', 'FLAG_REPEATED',
                             'FLAG_GREATER_THAN_MaxValue', 'UNUSUAL_SPIKE', 'FLAG_BELOW_CAPACITY']
        existing_flag_columns = [col for col in flag_columns_list if col in df.columns]
        if existing_flag_columns:
            df['FLAGGED'] = df[existing_flag_columns].any(axis=1)
        else:
            df['FLAGGED'] = False # Should not happen if columns are created above

        app.logger.info(f"Total flagged points: {df['FLAGGED'].sum()}")

    else:
        app.logger.warning("Cannot apply flagging: 'DISCHARGE' column not found in DataFrame.")
        df['FLAGGED'] = False
        # Ensure expected columns exist even if flagging fails
        flag_cols_expected = ['FLAG_LESS_THAN_Min._Value', 'FLAG_ZERO', 'FLAG_REPEATED',
                              'FLAG_GREATER_THAN_MaxValue', 'UNUSUAL_SPIKE', 'FLAG_BELOW_CAPACITY']
        for col in flag_cols_expected:
            if col not in df.columns: df[col] = False

    return df


def validate_date(date_str):
    """Validate date string format YYYY-MM-DD."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return None

# --- NEW HELPER FUNCTIONS FOR GRADIENT BUFFER ---

def interpolate_color(color1_rgba: Tuple[int, int, int, float],
                      color2_rgba: Tuple[int, int, int, float],
                      fraction: float) -> str:
    """ Interpolates between two RGBA colors. """
    r1, g1, b1, a1 = color1_rgba
    r2, g2, b2, a2 = color2_rgba
    # Clamp fraction between 0 and 1
    fraction = max(0.0, min(1.0, fraction))
    # Interpolate RGBA components
    r = int(r1 + (r2 - r1) * fraction)
    g = int(g1 + (g2 - g1) * fraction)
    b = int(b1 + (b2 - b1) * fraction)
    a = a1 + (a2 - a1) * fraction
    # Clamp RGB values to 0-255 range
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    # Return formatted RGBA string
    return f'rgba({r},{g},{b},{a:.4f})'


def add_gradient_buffer(fig: go.Figure,
                        dates: Sequence,
                        mean_value: float,
                        buffer: float,
                        start_color_rgba: Tuple[int, int, int, float],
                        end_color_rgba: Tuple[int, int, int, float],
                        num_bands: int = 15):
    """ Adds gradient buffer bands around a central line (mean_value). """
    # Basic validation
    if buffer <= 0 or num_bands <= 0:
        app.logger.warning("Buffer width or number of bands is non-positive, skipping gradient.")
        return
    n_points = len(dates)
    if n_points < 2:
         app.logger.warning("Not enough date points to draw gradient buffer.")
         return
    # Ensure dates are usable for polygon shape
    x_coords_polygon = list(dates) + list(dates)[::-1] # x-coordinates for fill: forward then backward

    # Draw bands from outermost to innermost
    for i in range(num_bands - 1, -1, -1):
        outer_fraction = (i + 1) / num_bands # Fraction for the outer edge of this band
        inner_fraction = i / num_bands       # Fraction for the inner edge of this band

        # Calculate the color for this specific band based on its outer edge position
        band_color = interpolate_color(start_color_rgba, end_color_rgba, outer_fraction)

        # Upper Band (Above mean_value)
        band_lower_y_upper = mean_value + inner_fraction * buffer # Inner edge y-value
        band_upper_y_upper = mean_value + outer_fraction * buffer # Outer edge y-value

        # Check for NaN/inf values before plotting
        if not (np.isfinite(band_lower_y_upper) and np.isfinite(band_upper_y_upper)):
            app.logger.debug(f"Skipping upper band {i} due to non-finite y-values.")
            continue

        y_coords_upper = [band_lower_y_upper] * n_points + [band_upper_y_upper] * n_points # y-coordinates for fill
        fig.add_trace(go.Scatter(
            x=x_coords_polygon, y=y_coords_upper, fill='toself', fillcolor=band_color,
            line=dict(color='rgba(0,0,0,0)', width=0), # No border line for the band itself
            hoverinfo="skip", showlegend=False,
            mode='lines' # mode='lines' needed for fill='toself' to work correctly
        ))

        # Lower Band (Below mean_value)
        band_upper_y_lower = mean_value - inner_fraction * buffer # Inner edge y-value (closer to mean)
        band_lower_y_lower = mean_value - outer_fraction * buffer # Outer edge y-value (further from mean)

        # Check for NaN/inf values before plotting
        if not (np.isfinite(band_upper_y_lower) and np.isfinite(band_lower_y_lower)):
            app.logger.debug(f"Skipping lower band {i} due to non-finite y-values.")
            continue

        y_coords_lower = [band_lower_y_lower] * n_points + [band_upper_y_lower] * n_points # y-coordinates for fill
        fig.add_trace(go.Scatter(
            x=x_coords_polygon, y=y_coords_lower, fill='toself', fillcolor=band_color,
            line=dict(color='rgba(0,0,0,0)', width=0), # No border line for the band itself
            hoverinfo="skip", showlegend=False,
            mode='lines' # mode='lines' needed for fill='toself' to work correctly
        ))

# --- HTML Template (Unchanged) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{{ station_name | default('Data Quality Analysis', true) }} (Site ID {{ site_id | default('N/A', true) }}) {{ start_date }} to {{ end_date }}</title>
    <style>
        body { font-family: sans-serif; margin: 40px; } h1 { text-align: center; margin-block-start: 0.67em; margin-block-end: 0.67em; line-height: 1.3; } .error { color: red; text-align: center; font-weight: bold; margin-top: 15px; } #plot_container { margin-top: 20px; min-height: 100px; background-color: #f0f0f0; } .controls { text-align: center; margin-bottom: 20px; padding: 15px; border: 1px solid #ccc; border-radius: 5px; background-color: #f9f9f9; } .controls label, .controls input, .controls button { margin: 0 5px; vertical-align: middle; } .controls input[type="submit"], .controls button { padding: 5px 15px; cursor: pointer; font-size: 1em; } .plot-title-info { text-align: center; font-size: 30px; margin-bottom: 10px; } .header-link { font-size: 30px; color: darkblue; text-decoration: none; } .header-link:hover { text-decoration: underline; } .header-text-no-link { font-size: 14px; color: darkblue; } .modal { display: none; position: fixed; z-index: 1000; left: 50%; top: 50%; transform: translate(-50%, -50%); width: 300px; max-width: 90%; padding: 20px; background-color: #fefefe; border: 3px solid red; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); border-radius: 5px; text-align: center; } .modal-content { position: relative; } .modal-content h4 { margin-top: 0; } .modal-content p { margin: 10px 0; font-size: 14px; word-wrap: break-word; } .modal-close { color: #aaa; position: absolute; top: 5px; right: 10px; font-size: 24px; font-weight: bold; line-height: 1; cursor: pointer; padding: 0 5px; } .modal-close:hover, .modal-close:focus { color: black; text-decoration: none; } .modal-button { display: inline-block; padding: 5px 10px; font-size: 12px; font-weight: bold; font-family: sans-serif; margin: 5px 3px; cursor: pointer; background-color: #e7e7e7; color: #333; border: 1px solid #adadad; border-radius: 4px; text-decoration: none; text-align: center; line-height: 1.4; white-space: nowrap; box-shadow: 0 1px 1px rgba(0,0,0,0.1); -webkit-appearance: button; -moz-appearance: button; appearance: button; } .modal-button:hover { background-color: #dcdcdc; border-color: #999999; text-decoration: none; color: #000; box-shadow: 0 1px 1px rgba(0,0,0,0.2); } .modal-button:active { background-color: #cccccc; box-shadow: inset 0 1px 2px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
    <h1><span style="font-size: 18px;">Data Quality Analysis for Measurement Site</span><br>{% if site_id and site_id != 'N/A' and site_id is not none %}<a href="https://waterrights.utah.gov/cgi-bin/dvrtview.exe?Modinfo=StationView&STATION_ID={{ site_id }}" target="_blank" rel="noopener noreferrer" class="header-link">{% if units and units != 'Unknown Units' %}{{ units }} - {% endif %}{{ station_name | default('Unknown Station', true) }} (Site ID={{ site_id }})</a>{% else %}<span class="header-text-no-link">{% if units and units != 'Unknown Units' %}{{ units }} - {% endif %}{{ station_name | default('Unknown Station', true) }} (Site ID={{ site_id | default('N/A', true) }})</span>{% endif %}</h1>
    <div class="controls"><form method="GET" action="/plot"><label for="id">Site ID:</label><input type="text" id="id" name="id" value="{{ site_id | default('', true) }}" required><label for="start_date">Start Date:</label><input type="date" id="start_date" name="start_date" value="{{ start_date | default('', true) }}" required><label for="end_date">End Date:</label><input type="date" id="end_date" name="end_date" value="{{ end_date | default('', true) }}" required><input type="submit" value="Update Plot"><button type="button" onclick="resetDates()">Reset Dates</button></form></div>
    {% if error %} <p class="error">Error: {{ error }}</p> {% endif %}
    {% if plot_div %}<div id='plot_container'>{{ plot_div | safe }}</div><script>console.log("[HTML] Plot div content should be inside plot_container above this line.");</script><div id="pointActionModal" class="modal"><div class="modal-content"><span id="closeModal" class="modal-close" title="Close">&times;</span><h4>Quality Control Decision</h4><p id="modalPointInfo">Point details will appear here.</p><div id="modalActions"></div></div></div>
    <script>function resetDates(){var e=document.getElementById("id");if(!e){console.error("Site ID input field not found.");alert("Internal error: Cannot find Site ID field.");return}var t=e.value;if(!t||t.trim()===""){alert("Please enter a Site ID first before resetting dates.");return}window.location.href="/plot?id="+encodeURIComponent(t)+"&reset=true"}function pointAction(e,t,o,n){console.log("Action:",e,"Index:",t,"Date:",o,"Value:",n);let l=typeof n=="number"?n.toFixed(2):n;alert(e+" clicked for point:\\nDate: "+o+"\\nValue: "+l+"\\n(Point Index: "+t+")\\n\\nAction not yet implemented.");var a=document.getElementById("pointActionModal");a&&(a.style.display="none")}window.plotInteractionRetry=!1;function initializePlotInteraction(){console.log("[Init] Starting interaction setup (include_plotlyjs='cdn' mode)...");var e=document.getElementById("plot_container"),t=null;if(!e){console.error("[Init] Plot container (#plot_container) not found! Cannot find plot div.");return}setTimeout(function(){console.log("[Init] Attempting to find plot div inside #plot_container after delay...");t=e.querySelector("div.js-plotly-plot");t||(t=e.querySelector("div.plotly-graph-div"));if(!t){var o=e.getElementsByTagName("div");o.length>0&&o[0].id&&o[0].id.startsWith("plotly-")&&(t=o[0],console.warn("[Init] Couldn't find plot div by class, using first child div with Plotly ID:",t))}if(!t){console.error("[Init] Plot div element NOT FOUND within #plot_container even after delay. Cannot attach listener.");typeof Plotly!="undefined"?console.warn("[Init] Plotly library IS loaded, but the plot div element wasn't found by selectors."):console.error("[Init] Plotly library is ALSO not loaded (unexpected in this mode).");return}console.log("[Init] Found plotDiv element:",t);attachPlotlyListeners(t)},200)}function attachPlotlyListeners(e){console.log("[Attach] Attempting to attach listeners to:",e);var t=document.getElementById("pointActionModal"),o=document.getElementById("closeModal"),n=document.getElementById("modalPointInfo"),l=document.getElementById("modalActions");if(!t||!o||!n||!l)console.error("[Attach] Modal elements missing.");else if(!window.modalHandlersAttached){o.onclick=function(){t&&(t.style.display="none")};window.addEventListener("click",function(e){t&&t.style.display==="block"&&e.target===t&&!t.querySelector(".modal-content").contains(e.target)&&(t.style.display="none")});console.log("[Attach] Modal close handlers attached.");window.modalHandlersAttached=!0}try{if(typeof e.on!="function"){e=document.getElementById(e.id);if(typeof e.on!="function")throw new Error("plotDiv.on is still not a function after re-fetch.");console.warn("[Attach] Re-fetched plotDiv to find .on method.")}console.log("[Attach] Attaching 'plotly_click' listener...");e.on("plotly_click",function(e){console.log("==== Plotly CLICK Event Fired ====");console.log("[plotly_click] Received data:",e);if(!t||!n||!l){console.error("[plotly_click] Modal elements missing inside handler.");return}if(!e||!e.points||e.points.length===0){console.log("[plotly_click] Click was not on a data point.");return}var o=e.points[0];if(o.curveNumber>0&&o.fullData&&o.fullData.mode&&o.fullData.mode.includes("markers")){console.log("[plotly_click] Clicked on a flagged point (marker trace, curveNumber > 0).");var a=o.pointNumber,r=o.x,i=o.y,d=o.fullData?o.fullData.name:"Unknown Trace",s=r,c=typeof i=="number"?i.toFixed(2):String(i),u=String(d).split("[")[0].trim();n.innerHTML=`<b>Date:</b> ${s}<br><b>Value:</b> ${c}<br><b>Flag:</b> ${u}`;l.innerHTML="";var m=document.createElement("button");m.className="modal-button";m.innerText="Approve - Correct Value";m.onclick=()=>pointAction("Approve",a,r,i);var p=document.createElement("button");p.className="modal-button";p.innerText="Interpolate - Estimate";p.onclick=()=>pointAction("Interpolate",a,r,i);var _=document.createElement("button");_.className="modal-button";_.innerText="Delete: enter manual measurement";_.onclick=()=>pointAction("DeleteManual",a,r,i);l.appendChild(m);l.appendChild(p);l.appendChild(_);t.style.display="block";console.log("[plotly_click] Modal displayed with point info and actions.")}else{console.log("[plotly_click] Clicked on the base line or a non-marker trace. Modal not shown.");t&&(t.style.display="none")}});console.log("[Attach] 'plotly_click' listener attached successfully.");e.on("plotly_afterplot",function(){console.log("---- Plotly AFTERPLOT Event Fired ----")});console.log("[Attach] 'plotly_afterplot' listener attached successfully.")}catch(a){console.error("[Attach] FAILED to attach listener:",a);console.log("[Attach] PlotDiv object during failure:",e)}}document.readyState==="loading"?document.addEventListener("DOMContentLoaded",initializePlotInteraction):initializePlotInteraction();</script>
    {% elif not error and not site_id %}<p style="text-align: center;">Please enter a Site ID and select a date range above.</p>{% elif not error and site_id and not plot_div %}<p style="text-align: center;">No plot generated. Check if data exists for the selected Site ID and date range, or if there was an error fetching/processing data or loading thresholds.</p>{% endif %}
</body>
</html>"""


# --- Core Data Processing and Plotting Function (MODIFIED TO INCLUDE BUFFER) ---
def generate_plot_for_site(site_id, start_date_str_requested, end_date_str_requested, is_reset=False):
    """Fetches data, applies flags, and generates the Plotly figure for a given site."""
    station_name = None
    actual_start_date_str = start_date_str_requested
    actual_end_date_str = end_date_str_requested
    df = pd.DataFrame()
    metadata = {}
    units = 'Unknown Units'
    site_thresholds = None

    app.logger.info(f"Generating plot - Input: Site ID: {site_id}, Start Req: {start_date_str_requested}, End Req: {end_date_str_requested}, Reset Flag: {is_reset}")

    # --- 1. Load/Verify Thresholds ---
    global thresholds_df_global
    if thresholds_df_global is None or thresholds_df_global.empty:
        app.logger.warning("Thresholds not loaded globally. Attempting to load now...")
        load_thresholds(THRESHOLDS_CSV_PATH)
        if thresholds_df_global is None or thresholds_df_global.empty:
            err_msg = f"Threshold data could not be loaded. Cannot process site {site_id}. Check if '{THRESHOLDS_CSV_PATH}' exists and is valid."
            app.logger.error(err_msg)
            return None, err_msg, "Data Quality Analysis", start_date_str_requested, end_date_str_requested, units

    site_thresholds = get_site_thresholds(thresholds_df_global, site_id)
    if site_thresholds is None:
        err_msg = f"Could not find or validate required thresholds for SiteID {site_id} in '{THRESHOLDS_CSV_PATH}'. Plot generation aborted."
        app.logger.error(err_msg) # Already logged in get_site_thresholds, but good to log here too
        return None, err_msg, "Data Quality Analysis", start_date_str_requested, end_date_str_requested, units

    # --- 2. Fetch Data from API ---
    # Determine API call dates (use full range if reset or invalid dates given)
    api_end_date_call = datetime.now().strftime('%Y-%m-%d')
    if not is_reset and validate_date(end_date_str_requested):
        api_end_date_call = end_date_str_requested
    elif not is_reset:
        app.logger.warning(f"Invalid/missing end date ('{end_date_str_requested}'). Using today '{api_end_date_call}' for API call.")

    api_start_date_call = "1900-01-01" # API default seems to be roughly this
    if not is_reset and validate_date(start_date_str_requested):
        api_start_date_call = start_date_str_requested
    elif not is_reset:
        app.logger.warning(f"Invalid/missing start date ('{start_date_str_requested}'). Using default '{api_start_date_call}' for API call.")

    app.logger.info(f"API call parameters - Start: {api_start_date_call}, End: {api_end_date_call}")

    try:
        api_url = f"https://www.waterrights.utah.gov/dvrtdb/daily-chart.asp?station_id={site_id}&start_date={api_start_date_call}&end_date={api_end_date_call}&f=json"
        app.logger.info(f"Fetching data from: {api_url}")
        response = requests.get(api_url, timeout=45) # Increased timeout slightly

        if response.status_code != 200:
            err_msg = f"API Error (Status {response.status_code}) for site {site_id}"
            app.logger.error(err_msg + f". URL: {api_url}. Response: {response.text[:200]}...")
            # Try to get station name even on error if possible
            name_from_thresh = thresholds_df_global.loc[thresholds_df_global['SiteID_str'] == site_id, 'station_name'].iloc[0] if 'station_name' in thresholds_df_global.columns and not thresholds_df_global[thresholds_df_global['SiteID_str'] == site_id].empty else 'N/A'
            return None, err_msg, name_from_thresh, start_date_str_requested, end_date_str_requested, units

        # Attempt to parse JSON
        try:
            data = response.json()
            if not isinstance(data, dict):
                 raise ValueError("API response was not a JSON object (dictionary).")
        except (requests.exceptions.JSONDecodeError, ValueError) as json_err:
            snippet = response.text[:200] if hasattr(response, 'text') else '(No text)'
            err_msg = f"JSON Decode Error for site {site_id}. Error: {json_err}. Response snippet: {snippet}..."
            app.logger.error(err_msg + f" URL: {api_url}")
            name_from_thresh = thresholds_df_global.loc[thresholds_df_global['SiteID_str'] == site_id, 'station_name'].iloc[0] if 'station_name' in thresholds_df_global.columns and not thresholds_df_global[thresholds_df_global['SiteID_str'] == site_id].empty else 'N/A'
            return None, err_msg, name_from_thresh, start_date_str_requested, end_date_str_requested, units

        # Extract metadata
        metadata_fields = ["station_id", "station_name", "system_name", "units"]
        metadata = {f: data.get(f, "N/A") for f in metadata_fields}
        station_name = metadata.get('station_name', 'N/A') # Use name from API if available
        units = metadata.get('units')
        if not units or units == 'N/A':
            units = 'Unknown Units'
            app.logger.warning(f"Units missing or N/A in API response for site {site_id}. Using '{units}'.")
        else:
            app.logger.info(f"Units found in API response: {units}")


        # --- 3. Process Data into DataFrame ---
        if "data" not in data or not isinstance(data["data"], list) or not data["data"]:
            err_msg = f"No 'data' array found or empty in API response for site {site_id}. Station: {station_name}"
            app.logger.warning(err_msg + f" URL: {api_url}")
            # Return successfully processed metadata even if no data points
            return None, err_msg, station_name, start_date_str_requested, end_date_str_requested, units

        # Create DataFrame
        try:
            df = pd.DataFrame(data["data"], columns=["date", "value"]) # Explicitly name columns
        except Exception as df_err:
            err_msg = f"DataFrame creation error for site {site_id} from API data structure. Error: {df_err}"
            app.logger.error(err_msg, exc_info=True)
            # Log the first few data items to see structure
            app.logger.error(f"First few items in data['data']: {data.get('data', [])[:3]}")
            return None, err_msg, station_name, start_date_str_requested, end_date_str_requested, units

        if df.empty:
            err_msg = f"DataFrame created but is empty for site {site_id} despite non-empty API data list. Station: {station_name}"
            app.logger.warning(err_msg + f" URL: {api_url}")
            return None, err_msg, station_name, start_date_str_requested, end_date_str_requested, units

        # Rename and clean columns
        if "date" in df.columns and "value" in df.columns:
            df.rename(columns={"date": "Date", "value": "DISCHARGE"}, inplace=True)
        else:
            # This case should be caught by the column naming above, but as a failsafe:
            err_msg = f"Critical error: DataFrame created but columns 'date' or 'value' are missing. Site {site_id}."
            app.logger.error(err_msg + f" Actual Columns Found: {df.columns.tolist()}")
            return None, err_msg, station_name, start_date_str_requested, end_date_str_requested, units

        # Convert types and sort
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True) # Remove rows where date conversion failed
        df = df.sort_values('Date').reset_index(drop=True)

        if df.empty:
            err_msg = f"No valid dates found in the data after conversion for site {site_id}."
            app.logger.warning(err_msg)
            return None, err_msg, station_name, start_date_str_requested, end_date_str_requested, units

        df['DISCHARGE'] = pd.to_numeric(df['DISCHARGE'], errors='coerce') # Coerce errors to NaN

        # --- 4. Filter Data by Date Range ---
        min_data_dt = df['Date'].min()
        max_data_dt = df['Date'].max()
        app.logger.info(f"Full data range available from API: {min_data_dt:%Y-%m-%d} to {max_data_dt:%Y-%m-%d}")

        if is_reset:
            start_dt_final = min_data_dt
            end_dt_final = max_data_dt
            app.logger.info(f"Reset requested. Using full data range for plot: {start_dt_final:%Y-%m-%d} to {end_dt_final:%Y-%m-%d}")
        else:
            # Use requested dates if valid, otherwise clamp to available data range
            start_req_dt = validate_date(start_date_str_requested)
            end_req_dt = validate_date(end_date_str_requested)

            # Default to available range if requested dates are invalid
            if not start_req_dt: start_req_dt = min_data_dt
            if not end_req_dt: end_req_dt = max_data_dt

            # Clamp the valid/defaulted requested dates to the actual data range
            start_dt_final = max(start_req_dt, min_data_dt)
            end_dt_final = min(end_req_dt, max_data_dt)

            # Handle case where requested range is entirely outside data range
            if start_dt_final > end_dt_final:
                 app.logger.warning(f"Requested range [{start_req_dt:%Y-%m-%d} - {end_req_dt:%Y-%m-%d}] "
                                    f"is outside or incompatible with available data range [{min_data_dt:%Y-%m-%d} - {max_data_dt:%Y-%m-%d}]. "
                                    f"Plotting full available range instead.")
                 start_dt_final = min_data_dt
                 end_dt_final = max_data_dt
            else:
                 app.logger.info(f"Using date range for plot: {start_dt_final:%Y-%m-%d} to {end_dt_final:%Y-%m-%d}")

        # Store the actual dates used for the plot
        actual_start_date_str = start_dt_final.strftime('%Y-%m-%d')
        actual_end_date_str = end_dt_final.strftime('%Y-%m-%d')

        # Filter the DataFrame
        df_filtered = df.loc[(df['Date'] >= start_dt_final) & (df['Date'] <= end_dt_final)].copy().reset_index(drop=True)

        if df_filtered.empty:
            err_msg = f"No data available after filtering for site {site_id} in range [{actual_start_date_str} to {actual_end_date_str}]."
            app.logger.warning(err_msg)
            # Return metadata even if no data in range
            return None, err_msg, station_name, actual_start_date_str, actual_end_date_str, units

        # Use the filtered dataframe from now on
        df = df_filtered
        app.logger.info(f"Processing {len(df)} data points for flagging and plotting.")

        # --- 5. Apply Flagging ---
        df = apply_flagging(df, site_thresholds) # apply_flagging handles missing DISCHARGE internally

        # --- 6. Create Plot ---
        plot_title = f"Data from {actual_start_date_str} to {actual_end_date_str}"
        fig = go.Figure()

        # Add Base Discharge Line (handle potential NaNs with connectgaps=False)
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['DISCHARGE'],
            mode='lines',
            line=dict(color='lightgray', width=1.5),
            name='Mean Daily Discharge',
            connectgaps=False, # Don't connect across NaN gaps
            hoverinfo='skip' # Base line doesn't need hover usually
        ))

        # Get Threshold Values for Plotting
        min_val_thresh = site_thresholds.get("min_val", float('nan'))
        max_val_thresh = site_thresholds.get("max_val", float('nan'))
        spike_unusual_thresh = site_thresholds.get("spike_unusual", float('nan'))

        # Format thresholds for labels/legends
        formatted_spike_threshold = f"{spike_unusual_thresh:.2f}" if pd.notna(spike_unusual_thresh) else "N/A"
        formatted_max_threshold = f"{max_val_thresh:.2f}" if pd.notna(max_val_thresh) else "N/A"
        formatted_min_threshold = f"{min_val_thresh:.2f}" if pd.notna(min_val_thresh) else "N/A"

        # Add Flagged Points as Markers
        flag_plot_info = {
            # Order matters for legend display
            'FLAG_BELOW_CAPACITY': ('red', 'Below Measuring Capacity (Negative) [{}]'),
            'FLAG_ZERO': ('blue', 'Zero Discharge [{}]'),
            'FLAG_REPEATED': ('green', 'Repeated Value (>=4 days, non-zero) [{}]'),
            'FLAG_GREATER_THAN_MaxValue': ('purple', f'Over Max Capacity ({formatted_max_threshold})' + ' [{}]'),
            'UNUSUAL_SPIKE': ('orange', f"Unusual Spike (RoC > {formatted_spike_threshold})" + " [{}]")
            # Add FLAG_LESS_THAN_Min._Value if desired - maybe dark grey?
             #'FLAG_LESS_THAN_Min._Value': ('darkgrey', f'Below Min Value ({formatted_min_threshold})' + ' [{}]'),
        }

        hover_tmpl = (f'<b>Date:</b> %{{x|%Y-%m-%d}}<br>'
                      f'<b>Value:</b> %{{y:.2f}} {units}<br>'
                      f'<b>Flag Type:</b> %{{meta}}' # Use meta for flag type
                      f'<extra></extra>') # Hides extra hover info

        for flag_col_name, (color, legend_format) in flag_plot_info.items():
             if flag_col_name in df.columns and df[flag_col_name].any():
                subset = df.loc[df[flag_col_name]]
                count = len(subset)
                flag_label_only = legend_format.split('[')[0].strip() # Get text part for hover
                fig.add_trace(go.Scatter(
                    x=subset['Date'],
                    y=subset['DISCHARGE'],
                    mode='markers',
                    marker=dict(color=color, size=7, symbol='circle'),
                    name=legend_format.format(count), # Legend entry with count
                    meta=flag_label_only,             # Store flag type description in meta
                    hovertemplate=hover_tmpl,
                    showlegend=True
                ))

        # Get actual min/max dates from the *filtered* data for threshold lines
        min_plot_dt, max_plot_dt = df["Date"].min(), df["Date"].max()

        if pd.notna(min_plot_dt) and pd.notna(max_plot_dt):
             # Add Min Value Threshold Line (if not zero)
             if pd.notna(min_val_thresh) and min_val_thresh != 0:
                 fig.add_trace(go.Scatter(
                     x=[min_plot_dt, max_plot_dt], y=[min_val_thresh, min_val_thresh],
                     mode='lines', line=dict(color="gray", dash="dash", width=1),
                     name=f"Min Value Threshold ({formatted_min_threshold})",
                     hoverinfo='skip'
                 ))

             # Add Max Capacity Threshold Line
             if pd.notna(max_val_thresh):
                 fig.add_trace(go.Scatter(
                     x=[min_plot_dt, max_plot_dt], y=[max_val_thresh, max_val_thresh],
                     mode='lines', line=dict(color="purple", dash="dash", width=1),
                     name=f"Max Capacity Threshold ({formatted_max_threshold})",
                     hoverinfo='skip'
                 ))

                 # --- *** ADD GRADIENT BUFFER AROUND MAX THRESHOLD HERE *** ---
                 if max_val_thresh > 0: # Only add buffer if threshold is positive
                     buffer_width = max_val_thresh * BUFFER_PERCENTAGE # Calculate buffer size
                     dates_for_buffer = df['Date'].tolist() # Get dates as a list
                     if len(dates_for_buffer) >= 2:
                         app.logger.info(f"Adding gradient buffer around max capacity ({max_val_thresh:.2f}) with width {buffer_width:.2f}")
                         add_gradient_buffer(
                             fig=fig,
                             dates=dates_for_buffer,
                             mean_value=max_val_thresh,
                             buffer=buffer_width,
                             start_color_rgba=BUFFER_START_COLOR_RGBA,
                             end_color_rgba=BUFFER_END_COLOR_RGBA,
                             num_bands=BUFFER_NUM_BANDS
                         )
                     else:
                         app.logger.warning("Not enough data points in the filtered range to draw gradient buffer.")
                 else:
                     app.logger.info(f"Max capacity threshold is {max_val_thresh:.2f}, skipping gradient buffer.")
                 # --- *** END GRADIENT BUFFER ADDITION *** ---

        else:
             app.logger.warning("Could not determine plot date range for threshold lines because min/max plot dates are invalid.")


        # --- 7. Finalize Plot Layout ---
        fig.update_layout(
            title=dict(text=plot_title, x=0.5, y=0.95, font_size=24),
            xaxis=dict(
                title_text="Date",
                title_font_size=18,
                tickfont_size=14,
                showline=False, # Show axis line? Personal preference
                zeroline=True, zerolinewidth=2, zerolinecolor='black'
            ),
            yaxis=dict(
                title_text=f"Mean Daily Discharge ({units})",
                title_font_size=18,
                tickfont_size=14,
                showline=False,
                zeroline=True, zerolinewidth=2, zerolinecolor='black'
            ),
            legend=dict(
                orientation="v", # Vertical legend
                x=1.02, y=1,    # Position outside plot area top-right
                xanchor="left", yanchor="top",
                title=dict(text="Data Flagging Criteria:", font=dict(size=14)),
                font=dict(size=12)
            ),
            template="plotly_white", # Clean background
            margin=dict(t=80, r=300, b=80, l=80), # Adjust right margin for legend
            height=700, # Adjust plot height
            hovermode='closest' # Show hover for nearest point
        )

        app.logger.info(f"Plot generated successfully for {site_id} [{actual_start_date_str} to {actual_end_date_str}]")
        return fig, None, station_name, actual_start_date_str, actual_end_date_str, units

    # --- Error Handling for API Request/Data Processing ---
    except requests.exceptions.RequestException as e:
        err = f"Network error fetching data: {e}"
        app.logger.error(f"API Request failed for site {site_id}: {e}", exc_info=True)
        # Try to get station name if metadata was partially fetched or from thresholds
        name = metadata.get('station_name', 'N/A') if 'metadata' in locals() and metadata else \
               (thresholds_df_global.loc[thresholds_df_global['SiteID_str'] == site_id, 'station_name'].iloc[0]
                if 'thresholds_df_global' in globals() and thresholds_df_global is not None and 'station_name' in thresholds_df_global.columns and not thresholds_df_global[thresholds_df_global['SiteID_str'] == site_id].empty else 'N/A')
        return None, err, name, start_date_str_requested, end_date_str_requested, units
    except Exception as e:
        err = f"Unexpected error during plot generation process."
        # Try to get station name if metadata was partially fetched or from thresholds
        name = metadata.get('station_name', 'N/A') if 'metadata' in locals() and metadata else \
               (thresholds_df_global.loc[thresholds_df_global['SiteID_str'] == site_id, 'station_name'].iloc[0]
                if 'thresholds_df_global' in globals() and thresholds_df_global is not None and 'station_name' in thresholds_df_global.columns and not thresholds_df_global[thresholds_df_global['SiteID_str'] == site_id].empty else 'N/A')

        app.logger.error(f"Plot generation internal error for site {site_id}: {e}", exc_info=True)
        # Use the actual dates calculated if possible, otherwise fall back to requested
        final_start = actual_start_date_str if 'actual_start_date_str' in locals() and actual_start_date_str else start_date_str_requested
        final_end = actual_end_date_str if 'actual_end_date_str' in locals() and actual_end_date_str else end_date_str_requested
        final_units = units if 'units' in locals() and units != 'Unknown Units' else 'Unknown Units'
        return None, err, name, final_start, final_end, final_units



# --- Flask Route: /plot (Unchanged logic, relies on modified generate_plot_for_site) ---
@app.route('/plot')
def show_plot():
    site_id = request.args.get('id')
    start_date_req = request.args.get('start_date')
    end_date_req = request.args.get('end_date')
    is_reset = request.args.get('reset') == 'true' # Check if reset parameter is 'true'

    app.logger.info(f"Request received: id={site_id}, start={start_date_req}, end={end_date_req}, reset={is_reset}")

    # --- Initial Page Load or Missing ID ---
    if not site_id:
        app.logger.info("No Site ID provided. Rendering initial form.")
        # Provide default date range for the form (e.g., last year)
        today = datetime.now()
        one_yr_ago = today - timedelta(days=365)
        def_start = one_yr_ago.strftime('%Y-%m-%d')
        def_end = today.strftime('%Y-%m-%d')
        return render_template_string(HTML_TEMPLATE,
                                      site_id=None,
                                      station_name="Data Quality Analysis",
                                      start_date=def_start,
                                      end_date=def_end,
                                      error=None, plot_div=None, units='Unknown Units')

    # --- Handle Empty Site ID String ---
    if not site_id.strip():
         app.logger.warning("Empty Site ID provided.")
         err_msg = "Site ID cannot be empty."
         today = datetime.now(); one_yr_ago = today - timedelta(days=365); def_start = one_yr_ago.strftime('%Y-%m-%d'); def_end = today.strftime('%Y-%m-%d')
         start_render = start_date_req if start_date_req else def_start
         end_render = end_date_req if end_date_req else def_end
         return render_template_string(HTML_TEMPLATE, site_id=site_id, station_name="Data Quality Analysis", start_date=start_render, end_date=end_render, error=err_msg, plot_div=None, units='Unknown Units'), 400 # Bad request


    # --- Determine if Reset is Needed (Implicitly if dates are missing) ---
    if not is_reset and (not start_date_req or not end_date_req):
        app.logger.info(f"Site ID '{site_id}' provided without full date range. Treating as reset to get full range.")
        is_reset = True
        start_date_req = None # Ensure these are None if reset is triggered here
        end_date_req = None

    # --- Validate Dates if Not Resetting ---
    err_msg = None
    fig = None
    plot_div = None
    st_name = "Data Quality Analysis" # Default station name
    status = 200 # Default HTTP status
    start_proc = None # Date to pass to processing function
    end_proc = None   # Date to pass to processing function
    start_render = start_date_req # Date to display in the form initially
    end_render = end_date_req   # Date to display in the form initially
    units_val = 'Unknown Units' # Default units

    if not is_reset:
        start_dt = validate_date(start_date_req)
        end_dt = validate_date(end_date_req)
        if not start_dt or not end_dt:
            err_msg = "Valid Start and End Dates required (YYYY-MM-DD format)."
            status = 400 # Bad request
            app.logger.warning(f"Invalid/incomplete dates provided: Start='{start_date_req}', End='{end_date_req}'")
            start_render = start_date_req or "" # Keep invalid input in form
            end_render = end_date_req or ""
        elif start_dt > end_dt:
            err_msg = "Start date cannot be after end date."
            status = 400 # Bad request
            app.logger.warning(f"Date range error: Start={start_dt:%Y-%m-%d}, End={end_dt:%Y-%m-%d}")
            start_render = start_date_req # Keep input in form
            end_render = end_date_req
        else:
            # Dates are valid and in correct order
            start_proc = start_dt.strftime('%Y-%m-%d')
            end_proc = end_dt.strftime('%Y-%m-%d')
            start_render = start_proc # Update render dates to validated format
            end_render = end_proc
            app.logger.info(f"Using specific date range for processing: {start_proc} to {end_proc}")
    else:
        app.logger.info("Reset requested or dates missing. Will use full data range determined by generate_plot_for_site.")
        # start_proc and end_proc remain None, generate_plot handles it

    # --- Generate Plot if No Date Errors ---
    if not err_msg:
        app.logger.info(f"Calling generate_plot_for_site: id={site_id}, start={start_proc}, end={end_proc}, is_reset={is_reset}")
        try:
            # Call the core function
            fig, err_func, name_func, final_start, final_end, units_val = generate_plot_for_site(
                site_id, start_proc, end_proc, is_reset=is_reset
            )

            # Update render dates based on what generate_plot actually used
            start_render = final_start
            end_render = final_end

            # Update station name if returned
            if name_func and name_func != 'N/A':
                st_name = name_func

            # Handle errors returned from the function
            if err_func:
                err_msg = err_func
                # Set appropriate HTTP status based on error type
                if "API Error" in err_msg or "Network error" in err_msg: status = 502 # Bad Gateway
                elif "JSON Decode Error" in err_msg: status = 500 # Internal Server Error (bad data format)
                elif "Threshold data could not be loaded" in err_msg: status = 500 # Server config issue
                elif "Could not find or validate" in err_msg: status = 404 # Not Found (bad SiteID or config)
                elif "Unexpected" in err_msg: status = 500 # Generic server error
                elif "No data" in err_msg or "empty" in err_msg: status = 200 # OK, just no data to plot
                elif "Critical error" in err_msg: status = 500 # Internal Server Error
                else: status = 400 # Default to Bad Request for other validation errors

            app.logger.info(f"Processing complete. Render dates (actual range used) set to: Start={start_render}, End={end_render}, Units: {units_val}")

        except Exception as e:
            # Catch unexpected errors during the call itself
            app.logger.error(f"Unhandled exception during plot generation call for {site_id}: {e}", exc_info=True)
            err_msg = "Unexpected server error during plot generation."
            status = 500 # Internal Server Error
            # Try to preserve dates if possible
            start_render = final_start if 'final_start' in locals() and final_start else (start_date_req or "")
            end_render = final_end if 'final_end' in locals() and final_end else (end_date_req or "")
            units_val = units_val if units_val != 'Unknown Units' else 'Unknown Units'


    # --- Convert Plot to HTML if Successful ---
    if fig:
        try:
            plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': True, 'scrollZoom': True})
            app.logger.info(f"Plot converted to HTML div (using CDN) for {site_id}.")
        except Exception as plot_e:
            app.logger.error(f"Error converting plot to HTML for {site_id}: {plot_e}", exc_info=True)
            err_msg = "Error preparing plot for display."
            status = 500 # Internal Server Error
            plot_div = None # Ensure no plot is shown
            fig = None      # Ensure fig object is cleared

    # --- Handle Cases Where No Plot Was Generated but No Error Was Set Explicitly ---
    # (e.g., API returned data, but filtering removed everything)
    if not fig and not err_msg:
        # Only add this generic message if it wasn't a specific known non-plot outcome
        # (like threshold loading failure or invalid site ID)
        range_str = f"[{start_render} to {end_render}]" if start_render and end_render else "[evaluated range]"
        if not (status == 500 and "Threshold data" in (err_msg or "")) and \
           not (status == 404 and "Could not find or validate" in (err_msg or "")):
            err_msg = f"No plot generated. Check data existence/range for site {site_id} or threshold validity ({range_str})."
            app.logger.info(f"Final state: No plot generated, adding generic 'no plot' message for {site_id}.")
            # Keep status 200 if it was previously set (e.g., 'No data available'), otherwise use 404
            if status != 200:
                status = 404 # Treat as "not found" if no specific reason given


    # --- Render Final Template ---
    # Ensure dates passed to template are strings, handle None cases
    start_final = start_render if start_render is not None else ""
    end_final = end_render if end_render is not None else ""

    return render_template_string(HTML_TEMPLATE,
                                  site_id=site_id,
                                  station_name=st_name,
                                  start_date=start_final,
                                  end_date=end_final,
                                  error=err_msg,
                                  plot_div=plot_div,
                                  units=units_val), status


# --- Flask Route: / (Index - Unchanged) ---
@app.route('/')
def index():
    """Renders the initial page with default date range."""
    today_dt = datetime.now()
    today_str = today_dt.strftime('%Y-%m-%d')
    one_year_ago_str = (today_dt - timedelta(days=365)).strftime('%Y-%m-%d')
    app.logger.info("Rendering index page with default date range (last year).")
    return render_template_string(HTML_TEMPLATE,
                                  site_id=None,
                                  station_name="Data Quality Analysis",
                                  start_date=one_year_ago_str,
                                  end_date=today_str,
                                  error=None, plot_div=None, units='Unknown Units')

# --- Run the App ---
if __name__ == '__main__':
    # Explicitly check if thresholds were loaded successfully before starting server
    if thresholds_df_global is None:
        app.logger.warning("Thresholds were not loaded at import time. Attempting load again before starting server.")
        load_thresholds(THRESHOLDS_CSV_PATH) # Attempt load again
        if thresholds_df_global is None:
            # Print fatal error to console and exit if essential thresholds missing
            print(f"\nFATAL ERROR: Could not load required thresholds from '{THRESHOLDS_CSV_PATH}'. The application cannot run without them.", file=sys.stderr)
            print("Please ensure the file exists at the expected location and is a valid CSV.", file=sys.stderr)
            print(f"Expected columns include: SiteID, {', '.join(REQUIRED_THRESHOLD_COLS)}", file=sys.stderr)
            print("Check the logs above for specific errors (e.g., FileNotFoundError, ParserError, missing columns, permission issues).", file=sys.stderr)
            sys.exit(1) # Exit with a non-zero code indicates an error

    app.logger.info(f"Starting Flask server...")
    # Consider turning debug=False for production
    app.run(debug=True, host='127.0.0.1', port=5000)
