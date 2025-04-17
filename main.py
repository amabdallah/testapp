# --- Imports ---
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
from flask import Flask, render_template_string, request, redirect, url_for
import traceback
import logging
import json # Used for safe JS data embedding if needed
from typing import Tuple, Sequence # Added for type hinting in new functions

# --- Pandas Option ---
pd.set_option('future.no_silent_downcasting', True) # Address FutureWarning

# --- Flask App Setup ---
app = Flask(__name__)
# Configure logging
logging.basicConfig(level=logging.INFO) # Use INFO level for general flow, DEBUG for more detail

# --- Helper Function for Date Validation ---
def validate_date(date_str):
    """Attempts to parse a date string (YYYY-MM-DD) and returns a datetime object or None."""
    if not date_str:
        return None
    try:
        # Ensure format includes century for robustness
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return None

# --- Helper function to interpolate RGBA colors ---
def interpolate_color(color1_rgba: Tuple[int, int, int, float],
                      color2_rgba: Tuple[int, int, int, float],
                      fraction: float) -> str:
    """ Interpolates between two RGBA colors. """
    r1, g1, b1, a1 = color1_rgba
    r2, g2, b2, a2 = color2_rgba
    fraction = max(0.0, min(1.0, fraction))
    r = int(r1 + (r2 - r1) * fraction)
    g = int(g1 + (g2 - g1) * fraction)
    b = int(b1 + (b2 - b1) * fraction)
    a = a1 + (a2 - a1) * fraction
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return f'rgba({r},{g},{b},{a:.4f})'

# --- Function to add gradient buffer bands ---
def add_gradient_buffer(fig: go.Figure,
                        dates: Sequence,
                        mean_value: float,
                        buffer: float,
                        start_color_rgba: Tuple[int, int, int, float],
                        end_color_rgba: Tuple[int, int, int, float],
                        num_bands: int = 15):
    """ Adds gradient buffer bands around a central line. """
    if buffer <= 0 or num_bands <= 0:
        app.logger.warning("Buffer width or number of bands is non-positive, skipping gradient.")
        return
    n_points = len(dates)
    if n_points < 2:
         app.logger.warning("Not enough date points to draw gradient buffer.")
         return
    x_coords_polygon = list(dates) + list(dates)[::-1]
    for i in range(num_bands - 1, -1, -1):
        outer_fraction = (i + 1) / num_bands
        inner_fraction = i / num_bands
        band_color = interpolate_color(start_color_rgba, end_color_rgba, outer_fraction)
        # Upper Band
        band_lower_y_upper = mean_value + inner_fraction * buffer
        band_upper_y_upper = mean_value + outer_fraction * buffer
        if not (np.isfinite(band_lower_y_upper) and np.isfinite(band_upper_y_upper)): continue
        y_coords_upper = [band_lower_y_upper] * n_points + [band_upper_y_upper] * n_points
        fig.add_trace(go.Scatter(
            x=x_coords_polygon, y=y_coords_upper, fill='toself', fillcolor=band_color,
            line=dict(color='rgba(0,0,0,0)', width=0), hoverinfo="skip", showlegend=False,
        ))
        # Lower Band
        band_upper_y_lower = mean_value - inner_fraction * buffer
        band_lower_y_lower = mean_value - outer_fraction * buffer
        if not (np.isfinite(band_upper_y_lower) and np.isfinite(band_lower_y_lower)): continue
        y_coords_lower = [band_lower_y_lower] * n_points + [band_upper_y_lower] * n_points
        fig.add_trace(go.Scatter(
            x=x_coords_polygon, y=y_coords_lower, fill='toself', fillcolor=band_color,
            line=dict(color='rgba(0,0,0,0)', width=0), hoverinfo="skip", showlegend=False,
        ))

# --- HTML Template (Adding Units to Header) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{{ station_name | default('Data Quality Analysis', true) }} (Site ID {{ site_id | default('N/A', true) }}) {{ start_date }} to {{ end_date }}</title>
    <style>
        /* --- CSS --- */
        body { font-family: sans-serif; margin: 40px; }
        h1 { text-align: center; margin-block-start: 0.67em; margin-block-end: 0.67em; line-height: 1.3; }
        .error { color: red; text-align: center; font-weight: bold; margin-top: 15px; }
        #plot_container { margin-top: 20px; min-height: 100px; background-color: #f0f0f0; }
        .controls { text-align: center; margin-bottom: 20px; padding: 15px; border: 1px solid #ccc; border-radius: 5px; background-color: #f9f9f9; }
        .controls label, .controls input, .controls button { margin: 0 5px; vertical-align: middle; }
        .controls input[type="submit"], .controls button { padding: 5px 15px; cursor: pointer; font-size: 1em; }
        .plot-title-info { text-align: center; font-size: 30px; margin-bottom: 10px; }
        .header-link { font-size: 30px; color: darkblue; text-decoration: none; }
        .header-link:hover { text-decoration: underline; }
        .header-text-no-link { font-size: 14px; color: darkblue; }
        .modal { display: none; position: fixed; z-index: 1000; left: 50%; top: 50%; transform: translate(-50%, -50%); width: 300px; max-width: 90%; padding: 20px; background-color: #fefefe; border: 3px solid red; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); border-radius: 5px; text-align: center; }
        .modal-content { position: relative; }
        .modal-content h4 { margin-top: 0; }
        .modal-content p { margin: 10px 0; font-size: 14px; word-wrap: break-word; }
        .modal-close { color: #aaa; position: absolute; top: 5px; right: 10px; font-size: 24px; font-weight: bold; line-height: 1; cursor: pointer; padding: 0 5px; }
        .modal-close:hover, .modal-close:focus { color: black; text-decoration: none; }
        .modal-button { display: inline-block; padding: 5px 10px; font-size: 12px; font-weight: bold; font-family: sans-serif; margin: 5px 3px; cursor: pointer; background-color: #e7e7e7; color: #333; border: 1px solid #adadad; border-radius: 4px; text-decoration: none; text-align: center; line-height: 1.4; white-space: nowrap; box-shadow: 0 1px 1px rgba(0,0,0,0.1); -webkit-appearance: button; -moz-appearance: button; appearance: button; }
        .modal-button:hover { background-color: #dcdcdc; border-color: #999999; text-decoration: none; color: #000; box-shadow: 0 1px 1px rgba(0,0,0,0.2); }
        .modal-button:active { background-color: #cccccc; box-shadow: inset 0 1px 2px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
    <h1>
        <span style="font-size: 18px;">Data Quality Analysis for Measurement Site</span><br>
        {# **** MODIFICATION: Add units before station name **** #}
        {% if site_id and site_id != 'N/A' and site_id is not none %}
            <a href="https://waterrights.utah.gov/cgi-bin/dvrtview.exe?Modinfo=StationView&STATION_ID={{ site_id }}" target="_blank" rel="noopener noreferrer" class="header-link">
                {% if units and units != 'Unknown Units' %}{{ units }} - {% endif %}{{ station_name | default('Unknown Station', true) }} (Site ID={{ site_id }})
            </a>
        {% else %}
            <span class="header-text-no-link">
                 {% if units and units != 'Unknown Units' %}{{ units }} - {% endif %}{{ station_name | default('Unknown Station', true) }} (Site ID={{ site_id | default('N/A', true) }})
            </span>
        {% endif %}
        {# **** END MODIFICATION **** #}
    </h1>
    <div class="controls">
        <form method="GET" action="/plot">
            <label for="id">Site ID:</label>
            <input type="text" id="id" name="id" value="{{ site_id | default('', true) }}" required>
            <label for="start_date">Start Date:</label>
            <input type="date" id="start_date" name="start_date" value="{{ start_date | default('', true) }}" required>
            <label for="end_date">End Date:</label>
            <input type="date" id="end_date" name="end_date" value="{{ end_date | default('', true) }}" required>
            <input type="submit" value="Update Plot">
            <button type="button" onclick="resetDates()">Reset Dates</button>
        </form>
    </div>
    {% if error %} <p class="error">Error: {{ error }}</p> {% endif %}


    {# --- Plot and Modal Section --- #}
    {% if plot_div %}
        <div id='plot_container'>
             {{ plot_div | safe }}
        </div>
        <script>console.log("[HTML] Plot div content should be inside plot_container above this line.");</script>

        <div id="pointActionModal" class="modal">
             <div class="modal-content">
                 <span id="closeModal" class="modal-close" title="Close">&times;</span>
                 <h4>Quality Control Decision</h4>
                 <p id="modalPointInfo">Point details will appear here.</p>
                 <div id="modalActions"></div>
             </div>
        </div>

        <script>
            // --- Util Functions ---
             function resetDates() {
                 var siteIdInput = document.getElementById('id');
                 if (!siteIdInput) { console.error("Site ID input field not found."); alert("Internal error: Cannot find Site ID field."); return; }
                 var siteId = siteIdInput.value;
                 if (!siteId || siteId.trim() === "") { alert("Please enter a Site ID first before resetting dates."); return; }
                 window.location.href = '/plot?id=' + encodeURIComponent(siteId) + '&reset=true';
             }

             function pointAction(action, pointIndex, xValue, yValue) {
                 console.log("Action:", action, "Index:", pointIndex, "Date:", xValue, "Value:", yValue);
                 let displayValue = typeof yValue === 'number' ? yValue.toFixed(2) : yValue;
                 alert(action + " clicked for point:\\nDate: " + xValue + "\\nValue: " + displayValue + "\\n(Point Index: " + pointIndex + ")\\n\\nAction not yet implemented.");
                 var modal = document.getElementById('pointActionModal');
                 if (modal) { modal.style.display = "none"; }
             }

             // Flag to prevent infinite retries
             window.plotInteractionRetry = false;

             function initializePlotInteraction() {
                 console.log("[Init] Starting interaction setup (include_plotlyjs='cdn' mode)...");
                 var plotContainer = document.getElementById('plot_container');
                 var plotDiv = null;

                 if (!plotContainer) {
                      console.error("[Init] Plot container (#plot_container) not found! Cannot find plot div.");
                      return;
                 }

                 setTimeout(function() {
                     console.log("[Init] Attempting to find plot div inside #plot_container after short delay...");
                     plotDiv = plotContainer.querySelector('div.js-plotly-plot');
                     if (!plotDiv) { plotDiv = plotContainer.querySelector('div.plotly-graph-div'); }
                     if (!plotDiv) {
                         var childDivs = plotContainer.getElementsByTagName('div');
                         if (childDivs.length > 0) {
                             plotDiv = childDivs[0];
                             console.warn("[Init] Couldn't find plot div by class, using first child div:", plotDiv);
                         }
                     }

                     if (!plotDiv) {
                         console.error("[Init] Plot div element NOT FOUND within #plot_container even after delay. Cannot attach listener.");
                         if(typeof Plotly !== 'undefined') {
                              console.warn("[Init] Plotly library IS loaded, but the plot div element wasn't found by selectors.");
                         } else {
                              console.error("[Init] Plotly library is ALSO not loaded (unexpected in this mode).");
                         }
                         return;
                     } else {
                         console.log("[Init] Found plotDiv element:", plotDiv);
                         attachPlotlyListeners(plotDiv);
                     }
                 }, 100);

             } // End of initializePlotInteraction

             function attachPlotlyListeners(plotDiv) {
                  console.log("[Attach] Attempting to attach listeners to:", plotDiv);

                  var modal = document.getElementById('pointActionModal');
                  var closeModalBtn = document.getElementById('closeModal');
                  var modalPointInfo = document.getElementById('modalPointInfo');
                  var modalActionsDiv = document.getElementById('modalActions');

                  if (!modal || !closeModalBtn || !modalPointInfo || !modalActionsDiv) {
                      console.error("[Attach] Modal elements missing.");
                  } else if (!window.modalHandlersAttached) {
                      closeModalBtn.onclick = function() { if(modal) modal.style.display = "none"; };
                      window.addEventListener('click', function(event) {
                           if (modal && modal.style.display === "block" && event.target === modal) {
                                if (!plotDiv.contains(event.target) || event.target === plotDiv ) {
                                      modal.style.display = "none";
                                }
                           }
                      });
                      console.log("[Attach] Modal close handlers attached.");
                      window.modalHandlersAttached = true;
                  }


                  // --- Attach Plotly Event Listener ---
                  try {
                      if (typeof plotDiv.on !== 'function') {
                           throw new Error("plotDiv.on is still not a function. Plotly failed to initialize this div correctly.");
                      }

                      console.log("[Attach] Attaching 'plotly_click' listener...");
                      plotDiv.on('plotly_click', function(data) {
                          console.log("==== Plotly CLICK Event Fired ====");
                          console.log("[plotly_click] Received data:", data);

                          if (!modal || !modalPointInfo || !modalActionsDiv) {
                              console.error("[plotly_click] Modal elements missing inside handler."); return;
                          }
                          if (!data || !data.points || data.points.length === 0) {
                              console.log("[plotly_click] Click was not on a data point."); return;
                          }

                          var pointData = data.points[0];
                          if (pointData.curveNumber > 0) { // Only act on flagged points (traces > 0)
                              console.log("[plotly_click] Clicked on a flagged point (curveNumber > 0).");
                              var pIndex = pointData.pointNumber;
                              var pX = pointData.x; var pY = pointData.y;
                              var pTraceName = pointData.fullData ? pointData.fullData.name : 'Unknown Trace';
                              var displayX = pX;
                              var displayY = typeof pY === 'number' ? pY.toFixed(2) : String(pY);
                              var flagType = String(pTraceName).split('[')[0].trim();

                              modalPointInfo.innerHTML = `<b>Date:</b> ${displayX}<br><b>Value:</b> ${displayY}<br><b>Flag:</b> ${flagType}`;
                              modalActionsDiv.innerHTML = ''; // Clear previous buttons

                              // ---- CREATE BUTTONS ----
                              var btnApprove = document.createElement('button');
                              btnApprove.className = 'modal-button';
                              btnApprove.innerText = 'Approve - Correct Value';
                              btnApprove.onclick = () => pointAction('Approve', pIndex, pX, pY);

                              var btnInterpolate = document.createElement('button');
                              btnInterpolate.className = 'modal-button';
                              btnInterpolate.innerText = 'Interpolate - Estimate';
                              btnInterpolate.onclick = () => pointAction('Interpolate', pIndex, pX, pY);

                              var btnDeleteManual = document.createElement('button');
                              btnDeleteManual.className = 'modal-button';
                              btnDeleteManual.innerText = 'Delete: enter manual measurement';
                              btnDeleteManual.onclick = () => pointAction('DeleteManual', pIndex, pX, pY);

                              // ---- APPEND BUTTONS ----
                              modalActionsDiv.appendChild(btnApprove);
                              modalActionsDiv.appendChild(btnInterpolate);
                              modalActionsDiv.appendChild(btnDeleteManual);

                              modal.style.display = 'block'; // Show the modal
                              console.log("[plotly_click] Modal displayed with point info and actions.");
                          } else {
                              console.log("[plotly_click] Clicked on the base line (curveNumber 0). Modal not shown.");
                          }
                      });
                      console.log("[Attach] 'plotly_click' listener attached successfully.");

                      plotDiv.on('plotly_afterplot', function() {
                           console.log("---- Plotly AFTERPLOT Event Fired ----");
                      });
                      console.log("[Attach] 'plotly_afterplot' listener attached successfully.");

                  } catch(err) {
                      console.error("[Attach] FAILED to attach listener:", err);
                      console.log("[Attach] PlotDiv object during failure:", plotDiv);
                      alert("Error attaching plot listener. Check console.");
                  }
             } // End of attachPlotlyListeners


            // --- Run Initialization ---
             if (document.readyState === 'loading') {
                  console.log("DOM not ready yet, adding listener for DOMContentLoaded.");
                 document.addEventListener('DOMContentLoaded', initializePlotInteraction);
             } else {
                  console.log("DOM already ready, running initializePlotInteraction directly.");
                 initializePlotInteraction();
             }
        </script>
        {# --- Fallback messages etc --- #}
    {% elif not error and not site_id %}
        <p style="text-align: center;">Please enter a Site ID and select a date range above.</p>
    {% elif not error and site_id and not plot_div %}
        <p style="text-align: center;">No plot generated. Check if data exists for the selected Site ID and date range, or if there was an error fetching/processing data.</p>
    {% endif %}

</body>
</html>
"""
# **** END OF HTML TEMPLATE VARIABLE ****


# --- Core Data Processing and Plotting Function (Zero line changes, return units) ---
def generate_plot_for_site(site_id, start_date_str_requested, end_date_str_requested, is_reset=False):
    station_name = None
    actual_start_date_str = start_date_str_requested
    actual_end_date_str = end_date_str_requested
    df = pd.DataFrame()
    metadata = {}
    units = 'Unknown Units' # Default units if not found

    app.logger.info(f"Generating plot - Input: Site ID: {site_id}, Start Req: {start_date_str_requested}, End Req: {end_date_str_requested}, Reset Flag: {is_reset}")

    api_end_date_call = datetime.now().strftime('%Y-%m-%d')
    if not is_reset and validate_date(end_date_str_requested): api_end_date_call = end_date_str_requested
    elif not is_reset: app.logger.warning(f"Invalid/missing end date ('{end_date_str_requested}'). Using today '{api_end_date_call}'.")

    api_start_date_call = "1900-01-01"
    if not is_reset and validate_date(start_date_str_requested): api_start_date_call = start_date_str_requested
    elif not is_reset: app.logger.warning(f"Invalid/missing start date ('{start_date_str_requested}'). Using default '{api_start_date_call}'.")

    app.logger.info(f"API call parameters - Start: {api_start_date_call}, End: {api_end_date_call}, Reset Flag Effect: {is_reset}")

    try:
        api_url = f"https://www.waterrights.utah.gov/dvrtdb/daily-chart.asp?station_id={site_id}&start_date={api_start_date_call}&end_date={api_end_date_call}&f=json"
        app.logger.info(f"Fetching data from: {api_url}")
        response = requests.get(api_url, timeout=30)

        if response.status_code != 200:
            err_msg = f"API Error (Status {response.status_code}) for site {site_id}"
            app.logger.error(err_msg + f" URL: {api_url}")
            # **** Return default units on error ****
            return None, err_msg, None, start_date_str_requested, end_date_str_requested, units

        try:
            data = response.json()
            if not isinstance(data, dict): raise ValueError("API response not JSON object.")
        except (requests.exceptions.JSONDecodeError, ValueError) as json_err:
            snippet = response.text[:200] if hasattr(response, 'text') else '(No text)'
            err_msg = f"JSON Decode Error for site {site_id}. Error: {json_err}. Snippet: {snippet}..."
            app.logger.error(err_msg + f" URL: {api_url}")
            return None, err_msg, None, start_date_str_requested, end_date_str_requested, units

        metadata_fields = ["station_id", "station_name", "system_name", "units"]
        metadata = {f: data.get(f, "N/A") for f in metadata_fields}
        station_name = metadata.get('station_name', 'N/A')
        units = metadata.get('units')
        if not units or units == 'N/A':
            units = 'Unknown Units'
            app.logger.warning(f"Units missing from API metadata for site {site_id}. Using '{units}'.")
        else:
            app.logger.info(f"Units found in metadata: {units}")


        if "data" not in data or not isinstance(data["data"], list) or not data["data"]:
            err_msg = f"No 'data' array found or empty in API response for site {site_id}. Station: {station_name}"
            app.logger.warning(err_msg + f" URL: {api_url}")
            return None, err_msg, station_name, start_date_str_requested, end_date_str_requested, units

        try:
            df = pd.DataFrame(data["data"], columns=["date", "value"])
        except Exception as df_err:
            err_msg = f"DataFrame creation error for site {site_id}. Error: {df_err}"
            app.logger.error(err_msg, exc_info=True)
            return None, err_msg, station_name, start_date_str_requested, end_date_str_requested, units

        if df.empty:
            err_msg = f"DataFrame created but is empty for site {site_id}. Station: {station_name}"
            app.logger.warning(err_msg + f" URL: {api_url}")
            return None, err_msg, station_name, start_date_str_requested, end_date_str_requested, units

        if "date" in df.columns and "value" in df.columns:
             df.rename(columns={"date": "Date", "value": "DISCHARGE"}, inplace=True)
        else:
             err_msg = f"Critical error: API response structure changed - missing 'date' or 'value' column for site {site_id}."
             app.logger.error(err_msg + f" Actual Columns: {df.columns.tolist()}")
             return None, err_msg, station_name, start_date_str_requested, end_date_str_requested, units

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df = df.sort_values('Date').reset_index(drop=True)

        if df.empty:
            err_msg = f"No valid dates found after conversion for site {site_id}."
            app.logger.warning(err_msg)
            return None, err_msg, station_name, start_date_str_requested, end_date_str_requested, units

        min_data_dt = df['Date'].min(); max_data_dt = df['Date'].max()
        app.logger.info(f"Full data range available: {min_data_dt:%Y-%m-%d} to {max_data_dt:%Y-%m-%d}")

        if is_reset:
            start_dt_final = min_data_dt; end_dt_final = max_data_dt
            app.logger.info(f"Reset requested or dates missing. Using full data range for plot: {start_dt_final:%Y-%m-%d} to {end_dt_final:%Y-%m-%d}")
        else:
            start_req_dt = validate_date(start_date_str_requested)
            end_req_dt = validate_date(end_date_str_requested)
            if not start_req_dt or not end_req_dt or start_req_dt > end_req_dt:
                 app.logger.warning(f"Invalid requested dates reached generate_plot. Reverting to full data range.")
                 start_dt_final = min_data_dt; end_dt_final = max_data_dt
            else:
                start_dt_final = max(start_req_dt, min_data_dt)
                end_dt_final = min(end_req_dt, max_data_dt)
                if start_dt_final > end_dt_final:
                    app.logger.warning(f"Requested range outside available data. Plotting full available range.")
                    start_dt_final = min_data_dt; end_dt_final = max_data_dt
                else:
                    app.logger.info(f"Using requested (and clamped) date range for plot: {start_dt_final:%Y-%m-%d} to {end_dt_final:%Y-%m-%d}")

        actual_start_date_str = start_dt_final.strftime('%Y-%m-%d')
        actual_end_date_str = end_dt_final.strftime('%Y-%m-%d')

        df_filtered = df.loc[(df['Date'] >= start_dt_final) & (df['Date'] <= end_dt_final)].copy().reset_index(drop=True)

        if df_filtered.empty:
            err_msg = f"No data available after filtering for site {site_id} in range [{actual_start_date_str} to {actual_end_date_str}]."
            app.logger.warning(err_msg)
            return None, err_msg, station_name, actual_start_date_str, actual_end_date_str, units

        df = df_filtered
        df['DISCHARGE'] = pd.to_numeric(df['DISCHARGE'], errors='coerce')
        app.logger.info(f"Processing {len(df)} data points for flagging and plotting.")

        # === FLAGGING CRITERIA === (Same logic)
        flag_cols = ["FLAG_NEGATIVE", "FLAG_ZERO", "FLAG_REPEATED", "OUTLIER_IF", "FLAG_RSD", "FLAG_Discharge", "FLAG_IQR", "FLAG_RoC", "FLAG_ABOVE_MAX_OVERLAP", "FLAG_LARGE_SPIKES", "FLAGGED"]
        for col in flag_cols: df[col] = False
        df["RATE_OF_CHANGE"] = df['DISCHARGE'].diff().abs()
        df["PERCENT_DEV"] = np.nan
        df['FLAG_NEGATIVE'] = (df['DISCHARGE'].notna()) & (df['DISCHARGE'] < 0)
        df['FLAG_ZERO'] = (df['DISCHARGE'].notna()) & (df['DISCHARGE'] == 0)
        df_nz = df[df['DISCHARGE'].notna() & (df['DISCHARGE'] > 0)].copy()
        df_calc = df_nz.dropna(subset=['DISCHARGE'])
        q90=np.nan; q95=np.nan; Q1=np.nan; Q3=np.nan; IQR=np.nan; mean_val=np.nan
        if not df_calc.empty:
            try:
                Q1, Q3 = df_calc["DISCHARGE"].quantile([0.25, 0.75])
                if pd.notna(Q1) and pd.notna(Q3): IQR = Q3 - Q1
                else: IQR = np.nan
                q90 = df_calc["DISCHARGE"].quantile(0.90)
                q95 = df_calc["DISCHARGE"].quantile(0.95)
                mean_val = df_calc["DISCHARGE"].mean()
                if pd.notna(mean_val) and mean_val != 0:
                     df_nz['PERCENT_DEV'] = ((df_nz["DISCHARGE"] - mean_val).abs() / mean_val * 100)
                     df = df.merge(df_nz[['PERCENT_DEV']], left_index=True, right_index=True, how='left')
                else: app.logger.warning("Mean value is NaN or zero, cannot calculate PERCENT_DEV.")
            except Exception as e: app.logger.warning(f"Statistical calculation error: {e}")
        else: app.logger.warning("No positive, non-null discharge values for statistical calculations.")
        if len(df) >= 4:
             is_diff = (df["DISCHARGE"] != df["DISCHARGE"].shift()) | df["DISCHARGE"].isna() | df["DISCHARGE"].shift().isna()
             grp_ids = is_diff.cumsum(); grp_counts = df.groupby(grp_ids)['DISCHARGE'].transform('count')
             df['FLAG_REPEATED'] = (grp_counts >= 4) & df["DISCHARGE"].notna() & (df["DISCHARGE"] != 0)
        else: df['FLAG_REPEATED'] = False
        df_if = df_calc[["DISCHARGE"]].copy()
        if not df_if.empty:
             cont = 'auto' if len(df_if) >= 20 else max(0.001, min(0.5, 0.01))
             model = IsolationForest(contamination=cont, random_state=42)
             try:
                 preds = model.fit_predict(df_if)
                 df_calc['OUTLIER_IF_NZ'] = (preds == -1)
                 df = df.merge(df_calc[['OUTLIER_IF_NZ']], left_index=True, right_index=True, how='left')
                 df['OUTLIER_IF'] = df['OUTLIER_IF_NZ'].fillna(False).astype(bool)
                 df.drop(columns=['OUTLIER_IF_NZ'], inplace=True, errors='ignore')
             except ValueError as ife: app.logger.warning(f"IForest error: {ife}"); df['OUTLIER_IF'] = False
        else: df['OUTLIER_IF'] = False
        for col in ["FLAG_Discharge", "FLAG_IQR", "FLAG_RoC", "FLAG_RSD", "OUTLIER_IF", "FLAG_REPEATED"]:
            if col not in df: df[col] = False
            else: df[col] = df[col].fillna(False).astype(bool)
        if pd.notna(q95): df["FLAG_Discharge"] = (df["DISCHARGE"].notna()) & (df["DISCHARGE"] > q95)
        if pd.notna(IQR) and pd.notna(Q1) and pd.notna(Q3):
             if IQR == 0 and Q1 != 0: df["FLAG_IQR"] = (df["DISCHARGE"].notna()) & (df["DISCHARGE"] != Q1)
             elif IQR > 0: low = Q1 - 1.5 * IQR; high = Q3 + 1.5 * IQR; df["FLAG_IQR"] = (df["DISCHARGE"].notna()) & ((df["DISCHARGE"] < low) | (df["DISCHARGE"] > high))
        if pd.notna(q90): df["FLAG_RoC"] = (df["RATE_OF_CHANGE"].notna()) & (df["RATE_OF_CHANGE"] > q90)
        if pd.notna(mean_val) and mean_val != 0 and "PERCENT_DEV" in df.columns:
             df["FLAG_RSD"] = (df["PERCENT_DEV"].notna()) & (df["PERCENT_DEV"] > 1000)
        else: df["FLAG_RSD"] = False
        df["FLAG_ABOVE_MAX_OVERLAP"] = (df["FLAG_IQR"] & df["FLAG_Discharge"] & df["OUTLIER_IF"])
        df["FLAG_LARGE_SPIKES"] = df["FLAG_RSD"] & df["FLAG_RoC"]
        primary = ["FLAG_NEGATIVE", "FLAG_ZERO", "FLAG_REPEATED", "FLAG_ABOVE_MAX_OVERLAP", "FLAG_LARGE_SPIKES"]
        for col in primary: df[col] = df[col].fillna(False).astype(bool)
        df["FLAGGED"] = df[primary].any(axis=1)
        # === END FLAGGING ===

        # --- Plotting ---
        plot_title = f"Data from {actual_start_date_str} to {actual_end_date_str}"
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['DISCHARGE'], mode='lines',
            line=dict(color='lightgray', width=1.5), name='Value',
            connectgaps=False, hoverinfo='skip'
        ))

        hover_tmpl = f'<b>Date:</b> %{{x|%Y-%m-%d}}<br><b>Value:</b> %{{y:.2f}} {units}<br><b>Flag Type:</b> %{{fullData.name}}<extra></extra>'
        flags_cfg = {
            'FLAG_NEGATIVE': ('red', 'Below Capacity (-)'),
            'FLAG_ZERO': ('blue', 'Value = 0'),
            'FLAG_REPEATED': ('green', 'Repeated (≥4 non-0)'),
            'FLAG_ABOVE_MAX_OVERLAP': ('purple', 'Over Capacity'),
            'FLAG_LARGE_SPIKES': ('orange', 'Large Spikes')
        }

        min_over_capacity_value = np.nan
        max_mask = df['FLAG_ABOVE_MAX_OVERLAP']
        if max_mask.any():
            min_over_capacity_value = pd.to_numeric(df.loc[max_mask, "DISCHARGE"], errors='coerce').min()
            if pd.notna(min_over_capacity_value) and np.isfinite(min_over_capacity_value) and min_over_capacity_value > 0:
                buffer_percentage = 0.10 # Using 10% buffer
                buffer_value = buffer_percentage * min_over_capacity_value
                gradient_start_rgba = (128, 0, 128, 0.25)
                gradient_end_rgba = (255, 255, 255, 0.0)
                gradient_bands = 15

                add_gradient_buffer(
                    fig=fig, dates=df['Date'], mean_value=min_over_capacity_value, buffer=buffer_value,
                    start_color_rgba=gradient_start_rgba, end_color_rgba=gradient_end_rgba, num_bands=gradient_bands
                )

                min_plot_dt, max_plot_dt = df["Date"].min(), df["Date"].max()
                if pd.notna(min_plot_dt) and pd.notna(max_plot_dt):
                    fig.add_trace(go.Scatter(
                        x=[min_plot_dt, max_plot_dt], y=[min_over_capacity_value, min_over_capacity_value], mode="lines",
                        line=dict(color="purple", width=2, dash="dash"), hoverinfo='skip', showlegend=False
                    ))

                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode='lines',
                    line=dict(color='purple', width=2, dash='dash'), fill='toself',
                    fillcolor=interpolate_color(gradient_start_rgba, gradient_end_rgba, 0.1),
                    name=f'Est. Max Cap. ({min_over_capacity_value:.1f}) ±{buffer_percentage*100:.0f}% Buf.'
                ))
            else:
                app.logger.warning(f"Could not determine valid min_over_capacity_value for site {site_id}. Skipping buffer/line.")
                min_over_capacity_value = np.nan


        for flag, (color, legend) in flags_cfg.items():
            if flag in df.columns and df[flag].any():
                subset = df.loc[df[flag]]
                count = len(subset)
                show_flag_legend = True
                if flag == 'FLAG_ABOVE_MAX_OVERLAP' and pd.notna(min_over_capacity_value):
                    show_flag_legend = False

                fig.add_trace(go.Scatter(
                    x=subset['Date'], y=subset['DISCHARGE'], mode='markers',
                    marker=dict(color=color, size=7, symbol='circle'),
                    name=f"{legend} [{count}]", hovertemplate=hover_tmpl,
                    showlegend=show_flag_legend
                ))

        # Update layout
        fig.update_layout(
            title=dict(
                text=plot_title, x=0.5, y=0.95, font_size=28
            ),
            xaxis=dict(
                title_text="Date", title_font_size=24, tickfont_size=18,
                # **** MODIFICATION: Hide main line, show bold zero line ****
                showline=False, # Hide the main axis line
                zeroline=True, zerolinewidth=2, zerolinecolor='black' # Show bold black zero line
            ),
            yaxis=dict(
                title_text=f"{units}", title_font_size=24, tickfont_size=18,
                 # **** MODIFICATION: Hide main line, show bold zero line ****
                showline=False, # Hide the main axis line
                zeroline=True, zerolinewidth=2, zerolinecolor='black' # Show bold black zero line
            ),
            legend=dict(
                title_text="Legend", title_font_size=22, font_size=25,
                x=1.02, y=1, xanchor="left", yanchor="top"
            ),
            template="plotly_white",
            margin=dict(t=80, r=250, b=80, l=80),
            height=700,
            hovermode='closest'
        )

        app.logger.info(f"Plot generated successfully for {site_id} [{actual_start_date_str} to {actual_end_date_str}]")
        # **** MODIFICATION: Return units ****
        return fig, None, station_name, actual_start_date_str, actual_end_date_str, units

    except requests.exceptions.RequestException as e:
        err = f"Network error fetching data: {e}"
        app.logger.error(f"API Request failed for site {site_id}: {e}", exc_info=True)
        name = metadata.get('station_name', 'N/A') if 'metadata' in locals() else None
        # **** Return default units on error ****
        return None, err, name, start_date_str_requested, end_date_str_requested, units
    except Exception as e:
        err = f"Unexpected error during plot generation process."
        name = metadata.get('station_name', 'N/A') if 'metadata' in locals() else None
        app.logger.error(f"Plot generation internal error for site {site_id}: {e}", exc_info=True)
        final_start = actual_start_date_str if 'actual_start_date_str' in locals() else start_date_str_requested
        final_end = actual_end_date_str if 'actual_end_date_str' in locals() else end_date_str_requested
        # **** Return determined or default units on error ****
        final_units = units if 'units' in locals() and units != 'Unknown Units' else 'Unknown Units'
        return None, err, name, final_start, final_end, final_units


# --- Flask Route: /plot (Handling units return value) ---
@app.route('/plot')
def show_plot():
    """
    Flask route to display the data plot.
    Redirects to include default dates if only site_id is provided.
    """
    site_id = request.args.get('id')
    start_date_req = request.args.get('start_date')
    end_date_req = request.args.get('end_date')
    is_reset = request.args.get('reset') == 'true'

    app.logger.info(f"Request received: id={site_id}, start={start_date_req}, end={end_date_req}, reset={is_reset}")

    if not site_id:
        app.logger.info("No Site ID provided. Rendering initial form.")
        today = datetime.now(); one_yr_ago = today - timedelta(days=365)
        def_start = one_yr_ago.strftime('%Y-%m-%d'); def_end = today.strftime('%Y-%m-%d')
        # **** Pass default units to index render ****
        return render_template_string(HTML_TEMPLATE, site_id=None, station_name="Data Quality Analysis", start_date=def_start, end_date=def_end, error=None, plot_div=None, units='Unknown Units')

    if not site_id.strip():
        app.logger.warning("Empty Site ID provided.")
        err_msg = "Site ID cannot be empty."
        today = datetime.now(); one_yr_ago = today - timedelta(days=365)
        def_start = one_yr_ago.strftime('%Y-%m-%d'); def_end = today.strftime('%Y-%m-%d')
        start_render = start_date_req if start_date_req else def_start
        end_render = end_date_req if end_date_req else def_end
        # **** Pass default units to error render ****
        return render_template_string(HTML_TEMPLATE, site_id=site_id, station_name="Data Quality Analysis", start_date=start_render, end_date=end_render, error=err_msg, plot_div=None, units='Unknown Units'), 400

    if not is_reset and not start_date_req and not end_date_req:
        today_str = datetime.now().strftime('%Y-%m-%d')
        default_start_date = '1900-01-01'
        target_url = url_for('show_plot', id=site_id, start_date=default_start_date, end_date=today_str)
        app.logger.info(f"ID '{site_id}' provided without dates. Redirecting to: {target_url}")
        return redirect(target_url)

    err_msg = None; fig = None; plot_div = None
    st_name = "Data Quality Analysis"; status = 200
    start_proc = None; end_proc = None
    start_render = start_date_req; end_render = end_date_req
    units_val = 'Unknown Units' # Initialize units for template context

    if not is_reset:
        start_dt = validate_date(start_date_req)
        end_dt = validate_date(end_date_req)
        if not start_dt or not end_dt:
            err_msg = "Valid Start and End Dates required (YYYY-MM-DD format)."; status = 400
            app.logger.warning(f"Invalid/incomplete dates provided: Start='{start_date_req}', End='{end_date_req}'")
            start_render = start_date_req or ""; end_render = end_date_req or ""
        elif start_dt > end_dt:
            err_msg = "Start date cannot be after end date."; status = 400
            app.logger.warning(f"Date range error: Start={start_dt}, End={end_dt}")
        else:
            start_proc = start_dt.strftime('%Y-%m-%d')
            end_proc = end_dt.strftime('%Y-%m-%d')
            app.logger.info(f"Using specific date range for processing: {start_proc} to {end_proc}")
    else:
        app.logger.info("Reset requested. Will use full data range.")

    if not err_msg:
        app.logger.info(f"Calling generate_plot_for_site: id={site_id}, start={start_proc}, end={end_proc}, is_reset={is_reset}")
        try:
            # **** MODIFICATION: Unpack units value ****
            fig, err_func, name_func, final_start, final_end, units_val = \
                generate_plot_for_site(site_id, start_proc, end_proc, is_reset=is_reset)

            if name_func and name_func != 'N/A': st_name = name_func
            if err_func:
                err_msg = err_func
                if "API Error" in err_msg or "Network error" in err_msg: status = 502
                elif "JSON Decode Error" in err_msg: status = 500
                elif "Unexpected" in err_msg: status = 500
                elif "No data" in err_msg or "empty" in err_msg: status = 200
                elif "Critical error" in err_msg: status = 500
                else: status = 400

            start_render = final_start
            end_render = final_end
            app.logger.info(f"Processing complete. Render dates (actual range used) set to: Start={start_render}, End={end_render}, Units: {units_val}")

        except Exception as e:
            app.logger.error(f"Unhandled exception during plot generation: {e}", exc_info=True)
            err_msg = "Unexpected server error during plot generation."; status = 500
            start_render = final_start if 'final_start' in locals() and final_start else (start_date_req or "")
            end_render = final_end if 'final_end' in locals() and final_end else (end_date_req or "")
            # units_val remains 'Unknown Units' (its initial value)

    if fig:
        try:
            plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': True, 'scrollZoom': True})
            app.logger.info(f"Plot converted to HTML div (using CDN) for {site_id}.")
        except Exception as plot_e:
            app.logger.error(f"Error converting plot to HTML: {plot_e}", exc_info=True)
            err_msg = "Error preparing plot for display."; status = 500; plot_div = None
            fig = None

    if not fig and not err_msg:
        range_str = f"[{start_render} to {end_render}]" if start_render and end_render else "[full range requested]"
        err_msg = f"No plot generated. This might mean no data exists for site {site_id} within the range {range_str}, or another issue occurred."
        app.logger.info(f"Final state: No plot generated, no specific error message for {site_id}.")
        if status == 200: status = 200

    start_final = start_render if start_render is not None else ""
    end_final = end_render if end_render is not None else ""

    # **** MODIFICATION: Pass units to template ****
    return render_template_string(HTML_TEMPLATE,
                                  site_id=site_id, station_name=st_name,
                                  start_date=start_final, end_date=end_final,
                                  error=err_msg, plot_div=plot_div, units=units_val), status


# --- Flask Route: / (Index - remains the same, units passed in show_plot) ---
@app.route('/')
def index():
    """ Renders the initial form page with default dates (last year). """
    today_dt = datetime.now()
    today_str = today_dt.strftime('%Y-%m-%d')
    one_year_ago_str = (today_dt - timedelta(days=365)).strftime('%Y-%m-%d')
    app.logger.info("Rendering index page with default date range (last year).")
    # Pass default units here too for consistency
    return render_template_string(HTML_TEMPLATE,
                                  site_id=None, station_name="Data Quality Analysis",
                                  start_date=one_year_ago_str, end_date=today_str,
                                  error=None, plot_div=None, units='Unknown Units')

# --- Run the App (remains the same) ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
