# -*- coding: utf-8 -*-
# --- Imports ---
from flask import Flask, render_template_string, request, redirect, url_for, flash, jsonify
import logging
import os
import sys
from datetime import datetime, timedelta
import uuid # For unique plot IDs
from pathlib import Path # <--- Added import

# --- Import functions from threshold_manager and plot_generator ---
# Ensure threshold_manager.py and plot_generator.py are in the same directory or Python path

# Import threshold-related functions and constants
try:
    from threshold_manager import (
        load_thresholds,          # Function to load/reload thresholds
        update_threshold_in_csv,  # Function to update thresholds in the CSV
        THRESHOLDS_CSV_PATH,      # Path constant for the thresholds file
        HAS_FCNTL                 # Boolean indicating if file locking is available
    )
    print("INFO: Successfully imported from threshold_manager.py", file=sys.stderr)
except ImportError as e:
    # Log critical error and exit if essential threshold code cannot be imported
    print(f"FATAL ERROR: Could not import required items from threshold_manager.py. Ensure it exists and is accessible.", file=sys.stderr)
    print(f"Error details: {e}", file=sys.stderr)
    sys.exit(1)

# Import plot generation function
try:
    from plot_generator import (
        generate_plot_for_site   # The main function to create the plot
    )
    print("INFO: Successfully imported from plot_generator.py", file=sys.stderr)
except ImportError as e:
    # Log critical error and exit if essential plot code cannot be imported
    print(f"FATAL ERROR: Could not import required items from plot_generator.py. Ensure it exists and is accessible.", file=sys.stderr)
    print(f"Error details: {e}", file=sys.stderr)
    sys.exit(1)

# --- Basic Logging Setup (Configure as needed) ---
# You might have more sophisticated logging already, adjust accordingly
logging.basicConfig(level=logging.INFO, # Use DEBUG for more verbose logs if needed
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stderr)]) # Log to stderr
logger = logging.getLogger('main_app') # Use a specific logger name

# --- Flask App Setup ---
app = Flask(__name__)
# Consider setting a secret key if using flash messages or sessions
# app.secret_key = os.urandom(24)

# --- Load Thresholds at Startup with Diagnostics ---
logger.info("=" * 20 + " APPLICATION STARTUP " + "=" * 20)
try:
    logger.info("-" * 10 + " Running Startup Diagnostics for Threshold File " + "-" * 10)
    logger.info(f"Using threshold path constant: {THRESHOLDS_CSV_PATH}")
    logger.info(f"Path object type: {type(THRESHOLDS_CSV_PATH)}")

    if isinstance(THRESHOLDS_CSV_PATH, Path):
         path_str = str(THRESHOLDS_CSV_PATH)
         logger.info(f"Checking existence of resolved path: {path_str}")
         exists = os.path.exists(path_str)
         logger.info(f"Path exists (os.path.exists): {exists}")
         if exists:
              is_file = os.path.isfile(path_str)
              logger.info(f"Is file (os.path.isfile): {is_file}")
              if is_file:
                  # Crucial check: Can the *current running process* read it?
                  can_read = os.access(path_str, os.R_OK)
                  logger.info(f"Read access granted (os.access): {can_read}")
                  if not can_read:
                      logger.error("PERMISSION DENIED: Read access check failed for the threshold file path!")
              else:
                   logger.error("PATH IS NOT A FILE: The specified path exists but is not a file.")
         else:
              logger.error("PATH NOT FOUND: The specified threshold file path does not exist.")
    else:
         logger.error("CONFIGURATION ERROR: THRESHOLDS_CSV_PATH imported from threshold_manager is not a valid Path object!")

    logger.info("-" * 10 + " Attempting to Call load_thresholds " + "-" * 10)

    # --- Call the actual load_thresholds function ---
    # Pass the logger instance you configured above
    loaded_df = load_thresholds(THRESHOLDS_CSV_PATH, logger)
    # -----------------------------------------------

    logger.info("-" * 10 + " Result of load_thresholds Call " + "-" * 10)
    if loaded_df is None:
        logger.error("load_thresholds returned None. Check previous logs for loading errors (permissions, file format, missing columns etc.).")
        # Consider adding a flag or state here if needed by your app logic
        # to know that thresholds failed to load.
    elif loaded_df.empty:
        logger.warning("load_thresholds returned an EMPTY DataFrame. The file might exist and be readable but contain no data rows.")
    else:
        logger.info(f"load_thresholds returned a DataFrame successfully. Shape: {loaded_df.shape}")
        # Optional: Log columns if needed for debugging
        # logger.debug(f"Loaded columns: {loaded_df.columns.tolist()}")
    logger.info("-" * 60)

except Exception as startup_err:
    logger.error(f"UNEXPECTED ERROR DURING STARTUP THRESHOLD LOADING/DIAGNOSTICS: {startup_err}", exc_info=True)
    logger.info("-" * 60)
    # Depending on severity, you might want to exit or prevent the app from fully starting:
    # sys.exit("Critical error during threshold loading.")

logger.info("=" * 20 + " STARTUP COMPLETE (ROUTES NEXT) " + "=" * 20)




# --- Flask App Setup ---
app = Flask(__name__)

# Configure logging BEFORE using app.logger
# Basic config for root logger (might be overridden by GCP logging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Set Flask app logger level
app.logger.setLevel(logging.INFO)

# Secret key for flash messages
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24)) # Good practice to use env var or generate random

# --- Log fcntl status ---
# Use app.logger now that it's configured
if not HAS_FCNTL:
    app.logger.warning("fcntl not available. File locking disabled. This might cause issues if multiple instances write thresholds simultaneously.")
else:
    app.logger.info("fcntl available. File locking enabled.")

# --- Load Thresholds AT STARTUP ---  <<<<<<<<< CORRECT LOCATION
# This code now runs when Gunicorn imports the module OR when executed directly
app.logger.info(f"Attempting initial threshold load from: {THRESHOLDS_CSV_PATH}")
if load_thresholds(THRESHOLDS_CSV_PATH, app.logger) is None:
    # Log critical error, but allow app to potentially start to show the error message in the route
    app.logger.critical(f"CRITICAL STARTUP FAILURE: Initial threshold load failed from '{THRESHOLDS_CSV_PATH}'. Check file existence, path, permissions, and format in logs.")
else:
    app.logger.info("Initial threshold load successful.")
# --- END Load Thresholds AT STARTUP ---

# --- HTML Template Definition ---
# Ensure the triple quotes below correctly enclose the entire HTML string
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ station_name | default('Data Quality Analysis', true) }} (Site ID {{ site_id | default('N/A', true) }}) {{ start_date }} to {{ end_date }}</title>
    <style>
        /* --- Base Styles (Desktop First) --- */
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 20px 40px; line-height: 1.6; background-color: #f4f7f6; color: #333; }
        h1 { text-align: center; margin-block: 0.67em; line-height: 1.3; color: #2c3e50; }
        .error { color: #c0392b; text-align: center; font-weight: bold; margin: 15px auto; border: 1px solid #e74c3c; padding: 10px; background-color: #fbecec; border-radius: 4px; max-width: 800px; }
        .success { color: #27ae60; text-align: center; font-weight: bold; margin: 15px auto; border: 1px solid #2ecc71; padding: 10px; background-color: #eafaf1; border-radius: 4px; max-width: 800px; }
        #plot_container { margin-top: 25px; background-color: #ffffff; width: 100%; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border-radius: 4px; }

        /* --- Controls Styling --- */
        .controls { text-align: center; margin-bottom: 25px; padding: 20px; border: 1px solid #dadedd; border-radius: 5px; background-color: #ffffff; box-shadow: 0 2px 5px rgba(0,0,0,0.05); display: flex; flex-direction: column; align-items: center; gap: 25px; }
        .date-controls-wrapper { display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 15px; width: 100%; padding-bottom: 15px; border-bottom: 1px solid #eee; }
        .date-controls-wrapper form { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 12px; }
        .controls label { margin-right: 5px; font-weight: 600; white-space: nowrap; font-size: 0.9em; color: #555;}
        .controls input[type="text"], .controls input[type="date"] { padding: 6px 10px; border: 1px solid #ccc; border-radius: 3px; font-size: 0.95em;}
        .controls input[type="submit"], .controls button { padding: 7px 18px; cursor: pointer; font-size: 0.95em; border: 1px solid #adadad; border-radius: 4px; background-color: #e7e7e7; color: #333; transition: background-color 0.2s ease; }
        .controls input[type="submit"]:hover, .controls button:hover { background-color: #dcdcdc;}
        .quick-date-buttons button { font-size: 0.85em; padding: 6px 14px;}

        .threshold-controls { text-align: left; padding: 20px; border: 1px dashed #bdc3c7; border-radius: 5px; background-color: #fdfdfd; width: 100%; max-width: 800px; box-sizing: border-box; }
        .threshold-controls h3 { margin-top: 0; text-align: center; font-size: 1.15em; margin-bottom: 20px; color: #34495e;}
        .threshold-stats-wrapper { display: flex; gap: 25px; align-items: flex-start; flex-wrap: nowrap; }
        .threshold-stats-wrapper form { flex: 2; max-width: 450px; }
        .threshold-stats-wrapper .statistics-display { flex: 1; border: 1px solid #e0e0e0; border-radius: 4px; padding: 15px; background-color: #f9f9f9; min-width: 200px; }
        .threshold-controls form div { margin-bottom: 10px; display: flex; flex-wrap: wrap; align-items: center; justify-content: flex-start; gap: 8px;}
        .threshold-controls form label { display: inline-block; width: 150px; text-align: right; font-size: 0.9em; margin-right: 5px;}
        .threshold-controls form input[type="number"] { width: 100px; padding: 5px; font-size: 0.9em; border: 1px solid #ccc; border-radius: 3px;}
        .threshold-controls form small { font-size: 0.85em; color: #777; }
        .threshold-controls form input[type="submit"] { display: block; margin: 20px 0 0 158px; padding: 8px 22px; font-size: 0.95em; background-color: #e0e0e0; border: 1px solid #adadad;}
        .threshold-controls form input[type="submit"]:hover { background-color: #d0d0d0; }

        .statistics-display h4 { margin-top: 0; margin-bottom: 10px; font-size: 1.05em; font-weight: 600; color: #333; text-align: center; border-bottom: 1px solid #ddd; padding-bottom: 8px; }
        .statistics-display p { margin: 5px 0; font-size: 0.9em; color: #444; line-height: 1.5; }
        .statistics-display span { font-weight: 600; color: #000; display: inline-block; min-width: 50px; text-align: right; padding-left: 5px; }

        h1 span:first-child { font-size: 0.8em; display: block; font-weight: normal; color: #7f8c8d; }
        .header-link { font-size: 1.1em; color: #2980b9; text-decoration: none; } .header-link:hover { text-decoration: underline; color: #3498db; }
        .header-text-no-link { font-size: 1.1em; color: #2c3e50; }

        .modal { display: none; position: fixed; z-index: 1000; left: 50%; top: 50%; transform: translate(-50%, -50%); width: 320px; max-width: 90%; padding: 20px; background-color: #fefefe; border: 3px solid #888; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); border-radius: 5px; text-align: center; box-sizing: border-box;}
        .modal-content { position: relative; } .modal-content h4 { margin-top: 0; color: #333;} .modal-content p { margin: 10px 0; font-size: 14px; word-wrap: break-word; text-align: left; } .modal-close { color: #aaa; position: absolute; top: 5px; right: 10px; font-size: 24px; font-weight: bold; line-height: 1; cursor: pointer; padding: 0 5px; } .modal-close:hover, .modal-close:focus { color: black; text-decoration: none; }
        .modal-button { display: inline-block; padding: 6px 12px; font-size: 13px; font-weight: bold; font-family: sans-serif; margin: 5px 3px; cursor: pointer; background-color: #ecf0f1; color: #333; border: 1px solid #bdc3c7; border-radius: 4px; text-decoration: none; text-align: center; line-height: 1.4; white-space: nowrap; box-shadow: 0 1px 1px rgba(0,0,0,0.1); transition: all 0.2s ease; } .modal-button:hover { background-color: #dcdcdc; border-color: #95a5a6; color: #000; box-shadow: 0 1px 1px rgba(0,0,0,0.2); } .modal-button:active { background-color: #bdc3c7; box-shadow: inset 0 1px 2px rgba(0,0,0,0.1); }

        /* Media Queries */
        @media (max-width: 900px) { .threshold-stats-wrapper { flex-direction: column; align-items: center; gap: 20px; } .threshold-stats-wrapper form, .threshold-stats-wrapper .statistics-display { flex-basis: auto; width: 100%; max-width: 450px; } .threshold-controls form input[type="submit"] { margin-left: 0; display: flex; margin: 15px auto;} }
        @media (max-width: 768px) { body { margin: 15px; font-size: 95%; } h1 { font-size: 1.6em; } h1 span:first-child { font-size: 0.75em; } .header-link, .header-text-no-link { font-size: 1em; } .controls { padding: 15px; gap: 20px; } .date-controls-wrapper { flex-direction: column; gap: 15px; } .date-controls-wrapper form { display: flex; flex-direction: column; align-items: stretch; width: 100%; max-width: 350px; gap: 8px; } .controls label { text-align: left; margin-right: 0; margin-bottom: 3px; width: auto; } .controls input[type="text"], .controls input[type="date"] { width: 100%; box-sizing: border-box; } .controls input[type="submit"] { width: 100%; margin-top: 5px; } .quick-date-buttons { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; } .quick-date-buttons button { padding: 6px 10px; font-size: 0.85em; } .threshold-controls { max-width: 100%; padding: 15px; } .threshold-controls h3 { font-size: 1.1em; margin-bottom: 15px;} .threshold-controls form div { flex-direction: column; align-items: stretch; gap: 5px; margin-bottom: 12px; } .threshold-controls form label { width: auto; text-align: left; margin-bottom: 3px; } .threshold-controls form input[type="number"] { width: 100%; box-sizing: border-box; } .threshold-controls form small { display: block; text-align: left; margin-top: 2px; } .statistics-display { text-align: left; padding: 10px 15px; } .statistics-display h4 { text-align: center; } }
        @media (max-width: 480px) { body { margin: 10px; font-size: 90%; } h1 { font-size: 1.4em; } .header-link, .header-text-no-link { font-size: 0.95em; } .controls label { font-size: 0.9em; } .controls input, .controls button { font-size: 0.9em; } .modal { width: 90%; } .modal-button { font-size: 12px; padding: 5px 10px; } }
    </style>
</head>
<body>
    <h1>
        <span style="font-size: 0.8em; display: block; font-weight: normal; color: #7f8c8d;">Data Quality Analysis for Measurement Site</span>
        {% if site_id and site_id != 'N/A' and site_id is not none %}
            <a href="https://waterrights.utah.gov/cgi-bin/dvrtview.exe?Modinfo=StationView&STATION_ID={{ site_id }}" target="_blank" rel="noopener noreferrer" class="header-link">
                 {% if units and units != 'Unknown Units' %}{{ units }} - {% endif %}{{ station_name | default('Unknown Station', true) }} (Site ID={{ site_id }})
            </a>
        {% else %}
            <span class="header-text-no-link">
                 {% if units and units != 'Unknown Units' %}{{ units }} - {% endif %}{{ station_name | default('Unknown Station', true) }} (Site ID={{ site_id | default('N/A', true) }})
            </span>
        {% endif %}
    </h1>

    {# --- Display Flash Messages --- #}
    {% with messages = get_flashed_messages(with_categories=true) %}
      {% if messages %}
        <div style="text-align:center; margin-bottom: 15px;">
        {% for category, message in messages %}
          <div class="{{ category }}">{{ message }}</div>
        {% endfor %}
        </div>
      {% endif %}
    {% endwith %}
    {# Display error passed from route #}
    {% if error %} <p class="error">Error: {{ error }}</p> {% endif %}

    <div class="controls">
        <div class="date-controls-wrapper"> {# Wrapper for date form + quick buttons #}
            <form method="GET" action="/plot" id="plotForm">
                <label for="id">Site ID:</label>
                <input type="text" id="id" name="id" value="{{ site_id | default('', true) }}" required>
                <label for="start_date">Start Date:</label>
                <input type="date" id="start_date" name="start_date" value="{{ start_date | default('', true) }}" required>
                <label for="end_date">End Date:</label>
                <input type="date" id="end_date" name="end_date" value="{{ end_date | default('', true) }}" required>
                <input type="submit" value="Update Plot">
            </form>
            <div class="quick-date-buttons"> {# Container for quick buttons #}
                 <button type="button" onclick="resetDates()">Full Range (Reset)</button>
                 <button type="button" onclick="setQuickRange('year')">Last Year</button>
                 <button type="button" onclick="setQuickRange('month')">Last Month</button>
            </div>
        </div>

        {# --- Threshold & Stats Section --- #}
        {# Only show if site_id exists and thresholds were potentially loaded #}
        {% if site_id and current_thresholds %}
        <div class="threshold-controls">
            <h3>Adjust QC Thresholds & View Statistics for Site {{ site_id }}</h3>
            <div class="threshold-stats-wrapper">
                <form method="POST" action="/update_thresholds">
                    <input type="hidden" name="site_id" value="{{ site_id }}">
                    <input type="hidden" name="start_date" value="{{ start_date | default('', true) }}">
                    <input type="hidden" name="end_date" value="{{ end_date | default('', true) }}">
                    <div>
                        <label for="max_val">Max Capacity:</label>
                        <input type="number" step="any" id="max_val" name="max_val" value="{{ current_thresholds.max_val }}" required>
                        <small>({{ units if units != 'Unknown Units' else 'units' }})</small>
                    </div>
                    <div>
                        <label for="spike_unusual">Unusual Spike RoC:</label>
                        <input type="number" step="any" id="spike_unusual" name="spike_unusual" value="{{ current_thresholds.spike_unusual }}" required>
                         <small>({{ units if units != 'Unknown Units' else 'units' }}/day)</small>
                   </div>
                    <div>
                        <label for="repeated_days">Repeated Value Days:</label>
                        <input type="number" step="1" min="2" id="repeated_days" name="repeated_days" value="{{ current_thresholds.repeated_days }}" required>
                        <small>(days, min 2)</small>
                    </div>
                    <input type="submit" value="Update Thresholds">
                </form>
                {% if stats_dict %}
                <div class="statistics-display">
                     <h4>Statistics ({{ stats_dict.units }})</h4>
                     <p>Record Count: <span>{{ stats_dict.count }}</span></p>
                     <p>Mean Daily: <span>{{ stats_dict.mean }}</span></p>
                     <p>Min Value: <span>{{ stats_dict.min }}</span></p>
                     <p>Max Value: <span>{{ stats_dict.max }}</span></p>
                </div>
                {% else %}
                 <div class="statistics-display"><h4>Statistics</h4><p>Statistics not available.</p></div>
                {% endif %}
            </div> {# End threshold-stats-wrapper #}
        </div> {# End threshold-controls div #}
        {% elif site_id and not error %} {# Show message if site ID entered but thresholds could not be loaded for editing #}
        <div class="threshold-controls">
             <p style="text-align: center; color: #888;">Threshold editing is unavailable. Thresholds may not have loaded correctly for Site ID {{ site_id }}.</p>
        </div>
        {% endif %}
    </div> {# End controls div #}

    {# --- Plot Area and Modal --- #}
    {% if plot_div %}
        {# The plot_div below will contain the div with id=plot_output_id #}
        <div id='plot_container'>{{ plot_div | safe }}</div>

        {# --- Modal --- #}
        <div id="pointActionModal" class="modal">
            <div class="modal-content">
                <span id="closeModal" class="modal-close" title="Close">&times;</span>
                <h4>Quality Control Decision</h4>
                <p id="modalPointInfo">Point details will appear here.</p>
                <div id="modalActions"></div>
            </div>
        </div>

        {# --- JavaScript Section --- #}
        <script>
            // Keep your existing Javascript here
            document.addEventListener('DOMContentLoaded', function() {
                console.log("[Debug] DOM fully loaded.");

                function formatDate(date) { try { const y=date.getFullYear(),m=String(date.getMonth()+1).padStart(2,'0'),d=String(date.getDate()).padStart(2,'0'); return `${y}-${m}-${d}`; } catch (e) { console.error("[Debug] Error formatting date:",e); return null; } }
                function getSiteIdValue() { const i=document.getElementById('id'); if (!i) { console.error("Site ID input not found."); alert("Internal error."); return null; } const v=i.value.trim(); if (!v) { console.warn("Site ID empty."); alert("Please enter Site ID."); return null; } return v; }

                window.setQuickRange = function(p) { const id=getSiteIdValue(); if (!id) return; const t=new Date(), e=formatDate(t); let s=new Date(); if (!e) { alert("Error formatting end date."); return; } if (p==='year') s.setFullYear(s.getFullYear()-1); else if (p==='month') s.setDate(s.getDate()-30); else { console.error("Invalid period:",p); return; } const sd=formatDate(s); if (!sd) { alert("Error formatting start date."); return; } const url=`/plot?id=${encodeURIComponent(id)}&start_date=${sd}&end_date=${e}`; console.log("Redirecting (Quick Range):",url); try { window.location.href=url; } catch (e) { console.error("Redirect error:",e); alert("Redirect error."); } }
                window.resetDates = function() { const id=getSiteIdValue(); if (!id) return; const url=`/plot?id=${encodeURIComponent(id)}&reset=true`; console.log("Redirecting (Reset):",url); try { window.location.href=url; } catch (e) { console.error("Redirect error:",e); alert("Redirect error."); } }

                window.pointAction = function(a, i, d, v) { console.log("Action:",a,"Index:",i,"Date:",d,"Value:",v); let vs=typeof v==='number'?v.toFixed(2):String(v); alert(`Action: ${a}\nDate: ${d}\nValue: ${vs}\nIndex: ${i}\n(Backend action not implemented)`); closeActionModal(); }

                const modal=document.getElementById('pointActionModal'),closeModalBtn=document.getElementById('closeModal'),modalPointInfo=document.getElementById('modalPointInfo'),modalActions=document.getElementById('modalActions');
                function showActionModal(p) { if (!modal || !modalPointInfo || !modalActions) { console.error("Modal elements missing!"); return; } const d=p.x, vs=typeof p.y==='number'?p.y.toFixed(2):String(p.y), f=(p.data&&p.data.meta)?p.data.meta:'Unknown Flag', i=p.pointNumber; modalPointInfo.innerHTML=`<b>Date:</b> ${d}<br><b>Value:</b> ${vs}<br><b>Flag:</b> ${f}`; modalActions.innerHTML=''; ['Approve','Interpolate','Delete'].forEach(t => { var b=document.createElement('button'); b.className='modal-button'; b.innerText=t; b.onclick=()=>pointAction(t,i,d,p.y); modalActions.appendChild(b); }); modal.style.display='block'; console.log("Modal displayed for point:",p); }
                function closeActionModal() { if (modal) modal.style.display='none'; }
                if (closeModalBtn) closeModalBtn.onclick=closeActionModal;
                window.addEventListener('click', function(e) { if (modal && e.target === modal) closeActionModal(); });

                // Use the plot_output_id passed from the template context
                const plotOutputId = {{ plot_output_id | tojson | safe }};
                console.log("[Init] Plot div ID:", plotOutputId);
                if (plotOutputId) {
                    setTimeout(() => { // Delay to allow Plotly.js potentially loaded via CDN to initialize
                        const plotDiv = document.getElementById(plotOutputId);
                        if (!plotDiv) { console.error(`Plot div '${plotOutputId}' NOT FOUND after delay!`); return; }
                        console.log("Found plotDiv element:", plotDiv);

                        // Check if Plotly has attached methods to the div
                        if (typeof plotDiv.on === 'function') {
                            console.log("Attaching 'plotly_click' listener...");
                            plotDiv.on('plotly_click', function(data) {
                                console.log("==== Plotly CLICK Event Raw Data ====", data);
                                if (!data || !data.points || data.points.length === 0) { console.log("Click not on data point."); return; }
                                var point = data.points[0];
                                console.log("Clicked point data:", point);
                                // Check if clicked on a flagged point (marker trace)
                                if (point.curveNumber > 0 && point.data && 'meta' in point.data && point.fullData && point.fullData.mode && point.fullData.mode.includes('markers')) {
                                    console.log("Clicked flagged marker point. Showing modal."); showActionModal(point);
                                } else {
                                    console.log("Clicked base line/threshold/non-marker. Hiding modal."); closeActionModal();
                                }
                            });
                            console.log("'plotly_click' listener attached.");
                            plotDiv.on('plotly_afterplot', function() { console.log("---- Plotly AFTERPLOT Event ----"); });
                        } else { console.error("Plotly .on method not found on div. Plotly may not have initialized correctly."); }
                    }, 500); // 500ms delay might need adjustment
                } else { console.log("No plot_output_id, skipping plot interaction setup."); }
            });
        </script>

    {# Handle cases where no plot is generated #}
    {% elif not error and not site_id %}
        <p style="text-align: center;">Please enter a Site ID and select a date range above.</p>
    {% elif not error and site_id and not plot_div %}
        {# Message when site ID entered but no plot generated (e.g., no data in range, etc.) #}
        <p style="text-align: center;">No plot generated. Check if data exists for the selected Site ID and date range, or if there was an error fetching/processing data.</p>
    {% endif %}

</body>
</html>
""" # --- END HTML Template ---


# --- Flask Route: Update Thresholds ---
@app.route('/update_thresholds', methods=['POST'])
def update_thresholds():
    # (Keep route function code here - same as previous version)
    site_id = request.form.get('site_id')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    if not site_id:
        flash("Error: Site ID missing for update.", "error")
        return redirect(url_for('index')) # Redirect to index if no site ID
    try:
        new_vals = {
            'max_val': float(request.form['max_val']),
            'spike_unusual': float(request.form['spike_unusual']),
            'repeated_days': int(request.form['repeated_days'])
        }
        if new_vals['repeated_days'] < 2:
            flash("Error: 'Repeated Value Days' must be 2 or greater.", "error")
            raise ValueError("Repeated days threshold must be >= 2")
        if new_vals['max_val'] < 0 or new_vals['spike_unusual'] < 0:
            flash("Error: Threshold values cannot be negative.", "error")
            raise ValueError("Negative threshold value provided")

    except (ValueError, TypeError, KeyError) as e:
        app.logger.error(f"Invalid threshold format/value submitted for SiteID {site_id}: {e}")
        # Avoid duplicate flashing
        existing_flashes = [m[1] for m in get_flashed_messages(with_categories=True)]
        if 'error' not in existing_flashes:
             flash("Error: Invalid number format or missing value for thresholds.", "error")
        return redirect(url_for('show_plot', id=site_id, start_date=start_date, end_date=end_date))

    app.logger.info(f"Attempting to update thresholds for SiteID {site_id} with values: {new_vals}")
    success, message = update_threshold_in_csv(site_id, new_vals, app.logger)
    flash(message, "success" if success else "error")
    return redirect(url_for('show_plot', id=site_id, start_date=start_date, end_date=end_date))


# --- Flask Route: Show Plot ---
@app.route('/plot')
def show_plot():
    # (Keep route function code here - same as previous version)
    site_id = request.args.get('id')
    start_date_req = request.args.get('start_date')
    end_date_req = request.args.get('end_date')
    is_reset = request.args.get('reset') == 'true'
    app.logger.info(f"Plot Request Received: id={site_id}, start={start_date_req}, end={end_date_req}, reset={is_reset}")

    plot_output_id = None; err_msg = None; fig = None; plot_div = None; status = 200
    st_name = "Data Quality Analysis"; units_val = 'Unknown Units'
    current_thresholds_for_template = None; stats_dict_for_template = None
    start_render = start_date_req; end_render = end_date_req
    final_start = None; final_end = None

    if not site_id or not site_id.strip():
        if not site_id: app.logger.info("No Site ID provided. Rendering initial form.")
        else: app.logger.warning("Empty Site ID provided."); err_msg = "Site ID cannot be empty."; status = 400
        today=datetime.now(); def_end=today.strftime('%Y-%m-%d'); def_start=(today-timedelta(days=30)).strftime('%Y-%m-%d')
        start_render = start_date_req if start_date_req else def_start
        end_render = end_date_req if end_date_req else def_end
        return render_template_string(HTML_TEMPLATE,
                                      site_id=site_id, station_name=st_name, start_date=start_render, end_date=end_render,
                                      error=err_msg, plot_div=None, units=units_val,
                                      current_thresholds=None, stats_dict=None, plot_output_id=None), status

    start_proc, end_proc = None, None
    if is_reset or not start_date_req or not end_date_req:
        if not is_reset: app.logger.info(f"Site '{site_id}' missing date range in request. Resetting to full range.")
        else: app.logger.info(f"Reset requested for Site '{site_id}'. Using full range.")
        is_reset = True
        start_render, end_render = None, None
    else:
        start_proc, end_proc = start_date_req, end_date_req
        start_render, end_render = start_proc, end_proc
        app.logger.info(f"Using requested date range for processing: {start_proc} to {end_proc}")

    app.logger.info(f"Calling generate_plot_for_site: id={site_id}, start={start_proc}, end={end_proc}, reset={is_reset}")
    try:
        fig, err_func, name_func, final_start, final_end, units_val, current_thresholds_for_template, stats_dict_for_template = generate_plot_for_site(
            site_id, start_proc, end_proc, is_reset=is_reset, logger=app.logger
        )

        if final_start and final_end: start_render, end_render = final_start, final_end
        if name_func and name_func != 'N/A': st_name = name_func
        units_val = units_val or 'Unknown Units'

        if err_func:
            err_msg = err_func
            if "CRITICAL ERROR" in err_msg: status = 500
            elif "API Error" in err_msg or "Network error" in err_msg: status = 502
            elif "JSON Decode Error" in err_msg or "Threshold data" in err_msg or "Unexpected" in err_msg: status = 500
            elif "Could not find or validate" in err_msg or f"SiteID {site_id} not found" in err_msg or "Error: Threshold data missing" in err_msg: status = 404
            elif "No data" in err_msg or "No date range" in err_msg: status = 200
            else: status = 400
            app.logger.warning(f"Error handled from generate_plot_for_site: {err_msg} (Status: {status})")
        elif fig: app.logger.info(f"Plot generation complete. Actual range plotted: {final_start} to {final_end}")

    except Exception as e:
        app.logger.error(f"Unhandled exception during generate_plot_for_site call for {site_id}: {e}", exc_info=True)
        err_msg = "Unexpected server error occurred during plot generation."; status = 500
        start_render, end_render = start_date_req or "", end_date_req or ""
        units_val = 'Unknown Units'; current_thresholds_for_template = None; stats_dict_for_template = None
        fig = None

    if fig:
        try:
            plot_output_id = f"plotly-plot-{uuid.uuid4()}"
            plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn', config={'displayModeBar': True, 'scrollZoom': True, 'responsive': True}, div_id=plot_output_id)
            app.logger.info(f"Plotly figure converted to HTML div (id='{plot_output_id}') for site {site_id}")
        except Exception as plot_e:
            app.logger.error(f"Error converting plot to HTML for {site_id}: {plot_e}", exc_info=True)
            err_msg = (err_msg + "; Additionally, an error occurred preparing the plot for display.") if err_msg else "Error preparing plot for display."
            status = 500; plot_div = None; fig = None; plot_output_id = None

    start_final = final_start if final_start is not None else start_render if start_render is not None else ""
    end_final = final_end if final_end is not None else end_render if end_render is not None else ""

    app.logger.debug(f"Final Rendering Variables: status={status}, plot_id={plot_output_id}, error={err_msg is not None}")

    return render_template_string(HTML_TEMPLATE,
                                  site_id=site_id, station_name=st_name, start_date=start_final, end_date=end_final,
                                  error=err_msg, plot_div=plot_div, units=units_val,
                                  current_thresholds=current_thresholds_for_template,
                                  stats_dict=stats_dict_for_template,
                                  plot_output_id=plot_output_id
                                  ), status


# --- Flask Route: Index / Root ---
@app.route('/')
def index():
    # (Keep route function code here - same as previous version)
    today=datetime.now(); today_str=today.strftime('%Y-%m-%d'); month_ago=(today-timedelta(days=30)).strftime('%Y-%m-%d')
    app.logger.info("Rendering index page.")
    return render_template_string(HTML_TEMPLATE,
                                  site_id=None, station_name="Data Quality Analysis", start_date=month_ago, end_date=today_str,
                                  error=None, plot_div=None, units='Unknown Units', current_thresholds=None, stats_dict=None,
                                  plot_output_id=None)


# --- Main Execution Block (for local 'python main.py' execution ONLY) ---
if __name__ == '__main__':
    # This block is NOT executed by Gunicorn in production.
    # The essential threshold loading now happens reliably outside this block.
    # We just start the dev server here. We can trust the logs generated during
    # the import/startup phase regarding threshold loading success/failure.
    app.logger.info(f"Starting Flask development server on http://127.0.0.1:5000")
    # Use Flask's built-in server for local testing and debugging
    # debug=True enables auto-reloading and interactive debugger (DO NOT use in production)
    app.run(host='127.0.0.1', port=5000, debug=True)