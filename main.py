# -*- coding: utf-8 -*-
# --- Imports ---
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback, no_update, ctx
import dash_bootstrap_components as dbc

import logging
import os
import sys
from datetime import datetime, timedelta, date # Added date
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from typing import Dict, Any, Tuple, Optional, List
from urllib.parse import parse_qs, urlparse
from collections import Counter # Added for duplicate date checking

# --- Import functions ---
try:
    # Make sure you have renamed the file to threshold_manager_dash.py
    from threshold_manager_dash import (
        load_thresholds,
        update_threshold_in_csv,
        get_site_thresholds,
        THRESHOLDS_CSV_PATH,
        HAS_FCNTL,
        DEFAULT_REPEATED_THRESHOLD
    )
    print("INFO: Successfully imported from threshold_manager_dash.py", file=sys.stderr)
except ImportError as e:
    print(f"FATAL ERROR: Could not import required items from threshold_manager_dash.py. Error: {e}", file=sys.stderr)
    # Fallback code
    def load_thresholds(path, logger): logger.error(f"Threshold load failed: {path} not found."); return None
    def update_threshold_in_csv(site_id, new_vals, logger): logger.error("Threshold update unavailable."); return False, "Threshold update unavailable."
    def get_site_thresholds(site_id, logger): logger.warning(f"Thresholds unavailable for site {site_id}."); return {}
    THRESHOLDS_CSV_PATH = "thresholds_placeholder.csv"
    HAS_FCNTL = False
    DEFAULT_REPEATED_THRESHOLD = 3

# Import plot generation and flagging functions
try:
    # Make sure you have renamed the file to plot_table_generator.py
    from plot_table_generator import (
        generate_plot_for_site,
        apply_flagging
    )
    print("INFO: Successfully imported from plot_table_generator.py", file=sys.stderr)
except ImportError as e:
    print(f"FATAL ERROR: Could not import required items from plot_table_generator.py. Error: {e}", file=sys.stderr)
    # Fallback code
    def generate_plot_for_site(site_id, start, end, reset, logger, thresholds_override=None):
        logger.error("Plot generation unavailable.")
        fig = go.Figure().update_layout(title=f"Error: Plot generator unavailable for {site_id}")
        # Ensure fallback returns all expected values
        today_fb = date.today() # Use date directly
        start_fb = (today_fb - timedelta(days=30)).strftime('%Y-%m-%d')
        end_fb = today_fb.strftime('%Y-%m-%d')
        return fig, pd.DataFrame(), "Plot generator unavailable", site_id, start_fb, end_fb, "?", thresholds_override or {}, None
    def apply_flagging(df, thresholds, logger):
        logger.warning("Flagging unavailable.")
        if 'FLAGGED' not in df.columns: df['FLAGGED'] = False
        return df

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, # Set to DEBUG if needed to see logger.debug messages
                    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                    handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger('dash_app')

# --- Initialize Dash App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# --- Client-side callback for scrolling ---
# <<< ADDED: Client-side callback for scroll functionality >>>
app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks && n_clicks > 0) {
            // Introduce a small delay (e.g., 100 milliseconds)
            setTimeout(function() {
                const element = document.getElementById('data-table-section-header');
                if (element) {
                    console.log("Found element, attempting scroll..."); // Added for debugging
                    element.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                } else {
                    // This warning might still appear if 100ms isn't enough,
                    // but it's less likely.
                    console.warn("Scroll target 'data-table-section-header' not found after delay.");
                }
            }, 100); // Delay in milliseconds
        }
        return window.dash_clientside.no_update;
    }
    """,
    Input('edit-data-button', 'n_clicks'),
    prevent_initial_call=True
)
# <<< END: Client-side callback section >>>


# --- Log fcntl status ---
if not HAS_FCNTL: logger.warning("fcntl module not available. File locking DISABLED.")
else: logger.info("fcntl module available. File locking ENABLED.")

# --- Load Thresholds AT STARTUP ---
logger.info(f"Attempting initial threshold load from: {THRESHOLDS_CSV_PATH}")
if 'threshold_manager_dash' not in sys.modules:
    logger.warning(f"Skipping initial threshold load as threshold_manager_dash module was not found.")
elif load_thresholds(THRESHOLDS_CSV_PATH, logger) is None:
    logger.critical(f"CRITICAL STARTUP WARNING: Initial threshold load failed from '{THRESHOLDS_CSV_PATH}'.")
else:
    logger.info("Initial threshold load attempt complete.")


# --- Helper Function: create_dash_data_table ---
def create_dash_data_table(df: pd.DataFrame, units: str, table_id: str, logger: logging.Logger) -> Any:
    """Creates a Dash DataTable component or an Alert if df is invalid."""
    # ... (rest of function unchanged) ...
    logger.info(f"Creating Dash DataTable with id='{table_id}'...")
    if df is None or df.empty:
        logger.warning("Input DataFrame is empty or None. Skipping DataTable creation.")
        return dbc.Alert("No data available to display in the table.", color="info", className="mt-3")

    # Ensure Date column is datetime for formatting
    if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
        logger.debug("Converting 'Date' column to datetime objects for table creation.")
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df['Date'].isnull().any():
            logger.warning("Some 'Date' values could not be converted to datetime.")
    elif 'Date' not in df.columns:
        logger.error("Missing 'Date' column in DataFrame for table creation.")
        return dbc.Alert("Data error: Missing 'Date' column.", color="danger", className="mt-3")

    # Identify which flag columns actually exist in the DataFrame
    possible_flag_cols = ['FLAG_LESS_THAN_Min._Value', 'FLAG_ZERO', 'FLAG_REPEATED', 'FLAG_GREATER_THAN_MaxValue', 'UNUSUAL_SPIKE', 'FLAG_BELOW_CAPACITY', 'FLAGGED']
    existing_flag_cols = [col for col in possible_flag_cols if col in df.columns]
    logger.debug(f"Flag columns found for table: {existing_flag_cols}")

    # Prepare data for the table display
    table_data = pd.DataFrame(index=df.index) # Preserve original index if needed

    # Use existing Date column (already validated)
    # Note: If Date column contains time, this will truncate it for display
    table_data['display_date'] = df['Date'].dt.strftime('%Y-%m-%d')

    if 'DISCHARGE' in df.columns:
        table_data['Discharge'] = df['DISCHARGE']
    else:
        logger.warning("Missing 'DISCHARGE' column for table.")
        table_data['Discharge'] = [np.nan] * len(df)

    # Use .get with default Series to handle potentially missing columns gracefully
    table_data['Review Status'] = df.get('ReviewStatus', pd.Series(['Unknown'] * len(df), index=df.index))
    # Ensure Qualifiers column exists before checking for notna
    if 'Qualifiers' in df.columns:
        table_data['Qualified'] = df['Qualifiers'].notna().map({True: 'Yes', False: 'No'})
    else:
        table_data['Qualified'] = pd.Series(['No'] * len(df), index=df.index) # Default if no Qualifiers column

    table_data['FLAGGED'] = df.get('FLAGGED', pd.Series([False] * len(df), index=df.index))

    # Generate 'Active Flags' string
    active_flags_list = []
    flag_map = {
        'FLAG_LESS_THAN_Min._Value': 'Below Min',
        'FLAG_ZERO': 'Zero',
        'FLAG_REPEATED': 'Repeated',
        'FLAG_GREATER_THAN_MaxValue': 'Above Max',
        'UNUSUAL_SPIKE': 'Spike',
        'FLAG_BELOW_CAPACITY': 'Below Capacity'
    }
    # Only check flags that exist in the DataFrame
    flag_cols_to_check = [col for col in flag_map if col in existing_flag_cols]

    if flag_cols_to_check:
        for index, row in df.iterrows(): # Iterate safely
            try:
                # Check boolean flags correctly
                active = [flag_map[col] for col in flag_cols_to_check if pd.notna(row.get(col)) and row.get(col) == True]
                active_flags_list.append(', '.join(active) if active else 'None')
            except Exception as e:
                logger.error(f"Error processing flags for table display row index {index}: {e}")
                active_flags_list.append('Error')
    else: # No flag columns found in data
        active_flags_list = ['N/A'] * len(df)

    # Handle potential length mismatch if errors occurred
    if len(active_flags_list) != len(df):
        logger.error("Length mismatch creating Active Flags column. Padding with 'Error'.")
        active_flags_list.extend(['Error'] * (len(df) - len(active_flags_list)))

    table_data['Active Flags'] = active_flags_list

    # Define table columns structure
    columns = [
        {"name": "Date", "id": "display_date", "editable": False},
        {"name": f"Discharge ({units})", "id": "Discharge", "type": "numeric", "format": {'specifier': '.2f'}, "editable": True},
        {"name": "Review Status", "id": "Review Status", "editable": False},
        {"name": "Qualified", "id": "Qualified", "editable": False},
        {"name": "Active Flags", "id": "Active Flags", "editable": False}
    ]
    # Convert prepared data to dictionary format for DataTable
    data_records = table_data.to_dict('records')

    # Create the DataTable component
    dash_table_component = dash_table.DataTable(
        id=table_id,
        columns=columns,
        data=data_records,
        editable=True, # Only Discharge column is effectively editable via 'columns' config
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        row_selectable=False,
        row_deletable=False,
        page_action="native",
        page_current=0,
        page_size=25, # Keep page_size to set initial size
        style_table={'overflowX': 'auto', 'minWidth': '100%'},
        style_cell={
            'textAlign': 'left',
            'padding': '5px',
            'minWidth': '100px',
            'width': '150px',
            'maxWidth': '200px',
            'whiteSpace': 'normal',
            'height': 'auto',
            'fontFamily': 'sans-serif',
            'fontSize': '0.9rem'
        },
        style_header={
            'backgroundColor': 'paleturquoise',
            'fontWeight': 'bold',
            'border': '1px solid grey'
        },
        style_data={'border': '1px solid lightgrey'},
        style_data_conditional=[
            {'if': {'filter_query': '{FLAGGED} = True'}, 'backgroundColor': '#ffcccb'}, # Check FLAGGED column from data
            {'if': {'column_editable': True}, 'backgroundColor': '#f0f8ff', 'border': '1px solid blue'} # Highlight editable column
        ],
        persistence=True,
        persistence_type='session', # or 'local' or 'memory'
        persisted_props=['page_current', 'sort_by', 'filter_query'],
    )
    logger.info(f"Dash DataTable '{table_id}' created successfully with {len(data_records)} rows.")
    return dash_table_component

# --- Helper Function/Constants for Multi-Add Modal ---
NUM_INPUT_ROWS = 15 # Provide 15 blank rows for pasting
INPUT_TABLE_COLUMNS = [
    {"name": "Date (YYYY-MM-DD)", "id": "Date", "editable": True, "type": "text"}, # Use text for pasting flexibility, validate later
    {"name": "Discharge", "id": "Discharge", "editable": True, "type": "numeric"},
    {"name": "Qualifier (Optional)", "id": "Qualifier", "editable": True, "type": "text", "presentation": "input"} # 'input' allows empty strings easily
]

def get_initial_input_table_data(num_rows: int = NUM_INPUT_ROWS) -> List[Dict[str, Any]]:
    """Generates the initial empty data structure for the input DataTable."""
    return [{'Date': None, 'Discharge': None, 'Qualifier': None} for _ in range(num_rows)]


# --- Helper function for building threshold form ---
def build_threshold_form(site_id, thresholds, units):
    # ... (rest of function unchanged) ...
    if not site_id:
        return html.P("Enter a Site ID and load data to view/edit thresholds.")
    thresholds = thresholds if isinstance(thresholds, dict) else {}
    unit_str = f" ({units})" if units and units != '?' else ""
    roc_unit_str = f" ({units}/day)" if units and units != '?' else " (units/day)"
    # Use .get() with default None for safety
    max_val = thresholds.get('max_val')
    spike_unusual = thresholds.get('spike_unusual')
    repeated_days = thresholds.get('repeated_values_threshold', DEFAULT_REPEATED_THRESHOLD)

    # Adjusted column widths and added subscribe buttons
    form_content = dbc.Form([
        dcc.Input(id='threshold-site-id-hidden', type='hidden', value=site_id),
        dbc.Row([
            dbc.Label(f"Max Capacity{unit_str}:", width=4, className="text-end"),
            dbc.Col(dbc.Input(id='threshold-max-val', type='number', value=max_val, required=True, step='any', placeholder="e.g., 10000"), width=4),
            dbc.Col(dbc.Button("Subscribe", id='subscribe-max-val-button', color="success", size="sm"), width=3) # Added Button
        ], className="mb-2 align-items-center"),
        dbc.Row([
            dbc.Label(f"Unusual Spike RoC{roc_unit_str}:", width=4, className="text-end"),
            dbc.Col(dbc.Input(id='threshold-spike-unusual', type='number', value=spike_unusual, required=True, step='any', placeholder="e.g., 5000"), width=4),
            dbc.Col(dbc.Button("Subscribe", id='subscribe-spike-button', color="success", size="sm"), width=3) # Added Button
        ], className="mb-2 align-items-center"),
        dbc.Row([
            dbc.Label("Repeated Value Days:", width=4, className="text-end"),
            dbc.Col(dbc.Input(id='threshold-repeated-days', type='number', value=repeated_days, required=True, step=1, min=2, placeholder="e.g., 3"), width=4),
            dbc.Col(dbc.Button("Subscribe", id='subscribe-repeated-button', color="success", size="sm"), width=3), # Added Button
            # Removed the hint column for space, consider adding it back differently if needed
        ], className="mb-2 align-items-center"),
        dbc.Button("Update Thresholds", id="update-thresholds-button", color="warning", className="mt-3")
    ])

    if not thresholds:
        return html.Div([ html.P(f"Thresholds not currently loaded or set for site {site_id}.", className="text-warning"), form_content])
    else:
        return form_content


# --- Dash App Layout ---
app.layout = dbc.Container([
    # ... (Stores, Header, Notification Area - unchanged) ...
     # Stores and Location
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='data-store'),
    dcc.Store(id='site-info-store'),
    dcc.Store(id='thresholds-store'),
    dcc.Store(id='clicked-point-store'),

    # Header and Disclaimer
    html.H3("Disclaimer: App for internal use, testing, and demonstration purposes", style={'color': 'red', 'textAlign': 'center'}),
    html.H1(id='main-title', children="Data Quality Analysis", style={'textAlign': 'center'}),
    html.Hr(),

    # Notification Area
    dbc.Row(dbc.Col(html.Div(id='notification-area'), width=12)),

    # --- Control Card ---
    dbc.Card(dbc.CardBody([
        dbc.Row([
            # Site ID Input
            dbc.Col([
                dbc.Label("Site ID:", html_for="site-id-input", className="fw-bold"),
                dbc.Input(id="site-id-input", type="text", placeholder="Enter Site ID", required=False, persistence=True, persistence_type='session')
            ], md=2),
            # Date Pickers
            dbc.Col([
                dbc.Label("Start Date:", html_for="start-date-picker", className="fw-bold"),
                dcc.DatePickerSingle(id='start-date-picker', display_format='YYYY-MM-DD', persistence=True, persistence_type='session')
            ], md=2),
            dbc.Col([
                dbc.Label("End Date:", html_for="end-date-picker", className="fw-bold"),
                dcc.DatePickerSingle(id='end-date-picker', display_format='YYYY-MM-DD', persistence=True, persistence_type='session')
            ], md=2),

            # --- Action Buttons Column ---
            dbc.Col([
                # --- Plot Group ---
                dbc.Label("Plot:", className="fw-bold d-block"),
                html.Div([
                    dbc.Button("Update Plot", id="update-button", color="primary", className="me-2"),
                    dbc.Button("Reset Range", id="reset-button", color="secondary", outline=True, className=""),
                ], className="mt-1 d-flex flex-wrap"),

                # --- Data Group (MODIFIED with Edit Button) ---
                dbc.Label("Data:", className="fw-bold d-block mt-3"),
                html.Div([
                     dbc.Button("Enter Measurement", id="open-enter-data-modal-button", color="info", outline=True, className="me-2", n_clicks=0),
                     dbc.Button("Upload Data", id="open-add-multiple-modal-button", color="success", outline=True, className="me-2", n_clicks=0),
                     dbc.Button("Edit Data", id="edit-data-button", color="warning", outline=True, className="", n_clicks=0) # Added Edit Button
                ], className="mt-1 d-flex flex-wrap")

            ], md=4),
            # --- End Action Buttons Column ---

            # Quick Date Selection
            dbc.Col([
                dbc.Label("Quick Dates:", className="fw-bold d-block"),
                dbc.ButtonGroup([
                    dbc.Button("Last Year", id="quick-year-button", outline=True, color="info", size="sm"),
                    dbc.Button("Last Month", id="quick-month-button", outline=True, color="info", size="sm")
                ], className="mt-2")
            ], md=2, className="text-center"),
        ], align="start", className="mb-3"),
    ]), className="mb-3 shadow-sm"), # <<< END OF CONTROL CARD

    # <<< --- ADDED TEST COMPONENTS --- >>>
    html.Div([
        html.Button("Test CS Callback", id="test-cs-button", n_clicks=0),
        html.Div(id="test-cs-output", style={'marginTop': '10px', 'border': '1px solid blue', 'padding': '5px'})
    ], style={'padding': '20px', 'border': '2px dashed red', 'margin': '10px 0'}),
    # <<< --- END OF TEST COMPONENTS --- >>>

    # --- Thresholds & Stats Row ---
     dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Adjust QC Thresholds", className="card-title"),
            html.Div(id='threshold-form-content', children=html.P("Load data to view/edit thresholds."))
        ]), className="mb-3 shadow-sm"), md=6),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Statistics", className="card-title"),
            html.Div(id="statistics-display", children="Load data to view statistics.")
        ]), className="mb-3 shadow-sm"), md=6),
    ]),

    # --- Main Plot ---
    dbc.Row(dbc.Col(
        dcc.Loading(
            id="loading-plot-main",
            # ... (Loading children and spinner - unchanged) ...
             children=[ # The component(s) being loaded
                 html.Div(dcc.Graph(id='main-plot', config={'scrollZoom': True}), id='plot-output-container')
            ],
            custom_spinner=html.Div( # Define the custom spinner + message
                [
                    dbc.Spinner(size="lg", color="primary", spinnerClassName="me-2"), # Use spinnerClassName
                    html.Span("Please wait....Data is Loading") # Your message
                ],
                # Optional: Add some styling to center it
                style={'textAlign': 'center', 'padding': '50px'}
            )
        ),
        width=12
    )),

    html.Hr(),

    # --- Data Table Section ---
    dbc.Row([
        dbc.Col([
            # <<< MODIFIED: Added ID for scrolling target >>>
            dbc.Row([
                dbc.Col(html.H4("Data Table"), width="auto"),
                dbc.Col(dbc.Button("Show/Hide Table", id="toggle-table-button", color="secondary", outline=True, size="sm", n_clicks=0), width="auto")
            ], align="center", className="mt-3 mb-2", id="data-table-section-header"), # Added id here
            # <<< END: Modified Row >>>
            dbc.Collapse(
                id="table-collapse",
                is_open=True,
                children=[
                    # ... (Table content unchanged) ...
                    html.P("(Discharge column is editable)", className="small text-muted"),
                    dbc.Alert(id='table-edit-status', children="Click 'Save Changes' below after editing.", color="info", is_open=False, dismissable=True),
                    html.Div(id='table-container', children=dbc.Alert("Load data to view table.", color="secondary")),
                    dbc.Button("Save Changes", id="save-button", color="success", className="mt-2", n_clicks=0, disabled=True),
                    html.Div(id="save-status", className="mt-1")
                ]
            )
        ], width=12)
    ]),

    # --- Modals ---
    # ... (QC Modal unchanged) ...
     # QC Action Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Quality Control Action")),
        dbc.ModalBody("Point details...", id="qc-modal-body"),
        dbc.ModalFooter([
            dbc.Button("Approve", id="qc-approve-button", color="success", className="ms-1", n_clicks=0),
            dbc.Button("Interpolate", id="qc-interpolate-button", color="warning", className="ms-1", n_clicks=0),
            dbc.Button("Delete (Set to NaN)", id="qc-delete-button", color="danger", className="ms-1", n_clicks=0),
            dbc.Button("Close", id="qc-close-button", color="secondary", className="ms-auto", n_clicks=0)
        ])
    ], id="qc-action-modal", is_open=False, centered=True),

    # <<< MODIFIED: Enter Data Modal includes Time input >>>
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Enter New Measurement")),
        dbc.ModalBody([
             dbc.Alert(id="enter-data-modal-alert", color="danger", is_open=False, duration=4000),
             dbc.Row([
                 dbc.Label("Date:", width=2),
                 dbc.Col(dcc.DatePickerSingle(id='enter-date-picker', display_format='YYYY-MM-DD', date=date.today().isoformat()), width=10)
             ], className="mb-3"),
             dbc.Row([
                 dbc.Label("Time (HH:MM):", width=2),
                 dbc.Col(dbc.Input(id='enter-time-input', type='text', placeholder="HH:MM", value="00:00"), width=10) # Added Time input
             ], className="mb-3"),
             dbc.Row([
                 dbc.Label("Discharge:", width=2),
                 dbc.Col(dbc.Input(id='enter-discharge-input', type='number', placeholder="Enter value", step="any"), width=10)
             ], className="mb-3"),
                 dbc.Row([
                 dbc.Label("Qualifier:", width=2),
                 dbc.Col(dbc.Input(id='enter-qualifier-input', type='text', placeholder="Optional: e.g., Manual Entry, Ice", value="Manual Entry"), width=10)
             ], className="mb-3"),
         ]),
        dbc.ModalFooter([
            dbc.Button("Submit Measurement", id="submit-enter-data-button", color="primary", n_clicks=0),
            dbc.Button("Cancel", id="cancel-enter-data-button", color="secondary", className="ms-auto", n_clicks=0)
        ])
    ], id="enter-data-modal", is_open=False, centered=True),
     # <<< END: Modified Enter Data Modal >>>

    # ... (Add Multiple Data Modal unchanged) ...
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Add Multiple Measurements (Paste from Spreadsheet)")),
        dbc.ModalBody([
            # Feedback area inside this modal
            dbc.Alert(id="add-multiple-data-modal-alert", color="danger", is_open=False, duration=5000),
            # Instructions
            html.P([
                "Paste data from a spreadsheet (up to ",
                f"{NUM_INPUT_ROWS} rows) into the table below. ",
                html.Strong("Required columns: Date (YYYY-MM-DD), Discharge."),
                " Qualifier is optional."]),
            # Input DataTable
            dash_table.DataTable(
                id='add-data-input-table',
                columns=INPUT_TABLE_COLUMNS,
                data=get_initial_input_table_data(), # Initialize with blank rows
                editable=True,
                row_deletable=False, # Keep simple for now, user can clear rows
                style_table={'overflowX': 'auto', 'maxHeight': '50vh', 'overflowY': 'auto'}, # Make table scrollable
                 style_cell={
                     'textAlign': 'left',
                     'padding': '5px',
                     'minWidth': '100px',
                     'width': '150px',
                     'maxWidth': '200px',
                 },
                 style_header={
                     'backgroundColor': 'rgb(230, 230, 230)',
                     'fontWeight': 'bold'
                 },
                 style_data_conditional=[ # Highlight editable
                     {'if': {'column_editable': True}, 'backgroundColor': 'rgb(240, 248, 255)'}
                 ]
            )
        ]),
        dbc.ModalFooter([
            dbc.Button("Submit Added Data", id="submit-multiple-data-button", color="primary", n_clicks=0),
            dbc.Button("Cancel", id="cancel-multiple-data-button", color="secondary", className="ms-auto", n_clicks=0)
        ])
    ], id="add-multiple-data-modal", is_open=False, centered=True, size="lg"),

], fluid=True)

# --- END app.layout ---


# --- Callbacks ---

# *** Main callback to handle initial load and updates ***
# <<< MODIFIED: Added detailed logging >>>
@callback(
    [Output('main-plot', 'figure'),
     Output('table-container', 'children'),
     Output('data-store', 'data'),
     Output('site-info-store', 'data'),
     Output('statistics-display', 'children'),
     Output('threshold-form-content', 'children'),
     Output('main-title', 'children'),
     Output('notification-area', 'children', allow_duplicate=True),
     Output('save-button', 'disabled'),
     Output('table-edit-status', 'is_open', allow_duplicate=True),
     Output('site-id-input', 'value'),
     Output('start-date-picker', 'date'),
     Output('end-date-picker', 'date'),
    ],
    [Input('url', 'href'),
     Input('update-button', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input('quick-year-button', 'n_clicks'),
     Input('quick-month-button', 'n_clicks'),
     Input('thresholds-store', 'data')],
    [State('site-id-input', 'value'),
     State('start-date-picker', 'date'),
     State('end-date-picker', 'date'),
     State('thresholds-store', 'data')],
    prevent_initial_call='initial_duplicate'
)
def update_data_and_plots(
    href,
    update_clicks, reset_clicks, year_clicks, month_clicks, threshold_data_input,
    state_site_id, state_start_date_str, state_end_date_str, current_thresholds_state):

    triggered_input_id = ctx.triggered_id or 'N/A'
    logger.info(f"update_data_and_plots triggered by: {triggered_input_id}")
    logger.debug(f"Input states: site='{state_site_id}', start='{state_start_date_str}', end='{state_end_date_str}'")
    logger.debug(f"Input href: {href}")
    logger.debug(f"Thresholds state: {current_thresholds_state}") # Log incoming thresholds state

    # --- Default values and initial setup ---
    # ... (defaults unchanged) ...
    empty_fig = go.Figure(layout={
        'xaxis': {'visible': False},
        'yaxis': {'visible': False},
        'annotations': [{'text': 'No Data Loaded or Found', 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 20}}]
    })
    initial_table_msg = dbc.Alert("Enter Site ID and Date Range, then click 'Update Plot'.", color="info")
    error_alert = None
    site_info_output = None
    stats_display = html.P("Load data to view statistics.")
    threshold_form = build_threshold_form(state_site_id, current_thresholds_state, "?") # Build initial form based on state
    main_title = "Data Quality Analysis"
    save_disabled = True
    edit_status_open = False

    site_id_to_load = None
    start_date_to_load = None
    end_date_to_load = None
    is_reset_action = False
    output_site_id = no_update
    output_start_date = no_update
    output_end_date = no_update

    is_initial_load = triggered_input_id == 'url'
    is_threshold_update = triggered_input_id == 'thresholds-store'

    try:
        today = date.today()
        default_end = today
        default_start = today - timedelta(days=30)
        logger.debug(f"Defaults: start={default_start.isoformat()}, end={default_end.isoformat()}")

        # Determine parameters based on trigger
        if is_initial_load:
            logger.info("Handling initial load from URL.")
            # ... (URL parsing logic unchanged, add debug logs if needed) ...
            if href:
                try:
                    parsed_url = urlparse(href)
                    query_params = parse_qs(parsed_url.query)
                    url_site_id = query_params.get('id', [None])[0]
                    url_start_str = query_params.get('start_date', [None])[0]
                    url_end_str = query_params.get('end_date', [None])[0]
                    logger.debug(f"Parsed URL params: id='{url_site_id}', start='{url_start_str}', end='{url_end_str}'")

                    if url_site_id:
                        site_id_to_load = url_site_id
                        output_site_id = url_site_id

                        temp_start_date = default_start
                        temp_end_date = default_end

                        if url_start_str:
                           try:
                               parsed_start = datetime.strptime(url_start_str, '%Y-%m-%d').date()
                               temp_start_date = parsed_start
                               logger.debug(f"Parsed URL start_date: {temp_start_date}")
                           except ValueError:
                               logger.warning(f"Invalid start_date format '{url_start_str}' in URL. Using default.")
                               error_alert = dbc.Alert(f"Warning: Invalid start date '{url_start_str}' in URL. Using default.", color="warning", dismissable=True, duration=5000)

                        if url_end_str:
                           try:
                               parsed_end = datetime.strptime(url_end_str, '%Y-%m-%d').date()
                               temp_end_date = parsed_end
                               logger.debug(f"Parsed URL end_date: {temp_end_date}")
                           except ValueError:
                               logger.warning(f"Invalid end_date format '{url_end_str}' in URL. Using default.")
                               error_alert = dbc.Alert(f"Warning: Invalid end date '{url_end_str}' in URL. Using default.", color="warning", dismissable=True, duration=5000)

                        if temp_start_date > temp_end_date:
                            logger.warning(f"URL dates invalid (start > end: {temp_start_date} > {temp_end_date}). Using defaults.")
                            start_date_to_load = default_start.strftime('%Y-%m-%d')
                            end_date_to_load = default_end.strftime('%Y-%m-%d')
                            output_start_date = default_start.isoformat()
                            output_end_date = default_end.isoformat()
                            error_alert = dbc.Alert(f"Warning: Start date {temp_start_date.isoformat()} is after end date {temp_end_date.isoformat()}. Using defaults.", color="warning", dismissable=True, duration=5000)
                        else:
                            start_date_to_load = temp_start_date.strftime('%Y-%m-%d')
                            end_date_to_load = temp_end_date.strftime('%Y-%m-%d')
                            output_start_date = temp_start_date.isoformat()
                            output_end_date = temp_end_date.isoformat()
                    else:
                        logger.info("No site_id found in URL.")
                        output_site_id = None
                        output_start_date = default_start.isoformat()
                        output_end_date = default_end.isoformat()

                except Exception as e_url:
                    logger.exception(f"Error parsing URL '{href}': {e_url}")
                    error_alert = dbc.Alert(f"Error parsing URL parameters: {e_url}", color="danger", dismissable=True)
                    output_site_id = None
                    output_start_date = default_start.isoformat()
                    output_end_date = default_end.isoformat()
            else:
                 logger.warning("href is empty on initial load trigger.")
                 output_site_id = None
                 output_start_date = default_start.isoformat()
                 output_end_date = default_end.isoformat()

        elif triggered_input_id == 'reset-button':
            logger.info(f"Reset button triggered for site '{state_site_id}'.")
            if not state_site_id:
                 error_alert = dbc.Alert("Please enter a Site ID before resetting.", color="warning", dismissable=True)
                 logger.warning("Reset requested but no site ID.")
                 return no_update, no_update, no_update, no_update, no_update, no_update, no_update, error_alert, True, False, state_site_id, state_start_date_str, state_end_date_str
            site_id_to_load = state_site_id
            start_date_to_load = None # Signal reset
            end_date_to_load = None
            is_reset_action = True
            output_site_id = state_site_id

        elif triggered_input_id in ['quick-year-button', 'quick-month-button']:
             logger.info(f"Quick date '{triggered_input_id}' triggered for site '{state_site_id}'.")
             if not state_site_id:
                  error_alert = dbc.Alert("Please enter a Site ID before using Quick Dates.", color="warning", dismissable=True)
                  logger.warning("Quick date requested but no site ID.")
                  return no_update, no_update, no_update, no_update, no_update, no_update, no_update, error_alert, True, False, state_site_id, state_start_date_str, state_end_date_str
             site_id_to_load = state_site_id
             end_dt_obj = today
             start_dt_obj = today - timedelta(days=365 if triggered_input_id == 'quick-year-button' else 30)
             start_date_to_load = start_dt_obj.strftime('%Y-%m-%d')
             end_date_to_load = end_dt_obj.strftime('%Y-%m-%d')
             output_site_id = state_site_id
             output_start_date = start_dt_obj.isoformat()
             output_end_date = end_dt_obj.isoformat()
             logger.debug(f"Quick dates set: start={start_date_to_load}, end={end_date_to_load}")

        elif triggered_input_id == 'update-button' or is_threshold_update:
            if is_threshold_update:
                logger.info(f"Threshold update triggered plot refresh for site '{state_site_id}'.")
            else:
                logger.info(f"'Update Plot' button triggered for site '{state_site_id}'.")

            if not state_site_id:
                error_alert = dbc.Alert("Please enter a Site ID.", color="danger", dismissable=True)
                logger.warning("Update/Threshold trigger but no site ID.")
                current_start = state_start_date_str or default_start.isoformat()
                current_end = state_end_date_str or default_end.isoformat()
                threshold_form = build_threshold_form(None, {}, "?") # Update form for empty site
                return empty_fig, initial_table_msg, None, None, stats_display, threshold_form, main_title, error_alert, True, False, None, current_start, current_end

            site_id_to_load = state_site_id
            output_site_id = state_site_id

            if not state_start_date_str or not state_end_date_str:
                error_alert = dbc.Alert("Please select both Start and End Dates.", color="danger", dismissable=True)
                logger.warning("Update/Threshold trigger but missing date(s).")
                current_start = state_start_date_str or default_start.isoformat()
                current_end = state_end_date_str or default_end.isoformat()
                threshold_form = build_threshold_form(site_id_to_load, current_thresholds_state, "?") # Update form
                return empty_fig, initial_table_msg, None, {'site_id': site_id_to_load}, stats_display, threshold_form, f"Error - {site_id_to_load}", error_alert, True, False, site_id_to_load, current_start, current_end

            try:
                logger.debug(f"Parsing dates: start='{state_start_date_str}', end='{state_end_date_str}'")
                start_dt_obj = datetime.strptime(state_start_date_str[:10], '%Y-%m-%d').date()
                end_dt_obj = datetime.strptime(state_end_date_str[:10], '%Y-%m-%d').date()
                logger.debug(f"Parsed date objects: start={start_dt_obj}, end={end_dt_obj}")
                if start_dt_obj > end_dt_obj:
                    error_alert = dbc.Alert("Start Date cannot be after End Date.", color="danger", dismissable=True)
                    logger.error(f"Date validation failed: start > end ({start_dt_obj} > {end_dt_obj})")
                    threshold_form = build_threshold_form(site_id_to_load, current_thresholds_state, "?") # Update form
                    return empty_fig, initial_table_msg, None, {'site_id': site_id_to_load}, stats_display, threshold_form, f"Date Error - {site_id_to_load}", error_alert, True, False, site_id_to_load, state_start_date_str, state_end_date_str
                start_date_to_load = start_dt_obj.strftime('%Y-%m-%d')
                end_date_to_load = end_dt_obj.strftime('%Y-%m-%d')
                output_start_date = state_start_date_str
                output_end_date = state_end_date_str

            except (ValueError, TypeError) as e_parse:
                 logger.exception(f"Invalid date format during parsing: {e_parse}")
                 error_alert = dbc.Alert(f"Invalid date format selected: {e_parse}", color="danger", dismissable=True)
                 threshold_form = build_threshold_form(site_id_to_load, current_thresholds_state, "?") # Update form
                 return empty_fig, initial_table_msg, None, {'site_id': site_id_to_load}, stats_display, threshold_form, f"Date Error - {site_id_to_load}", error_alert, True, False, site_id_to_load, state_start_date_str, state_end_date_str
        else:
             # This case should not be reached if prevent_initial_call='initial_duplicate' works correctly
             logger.warning(f"Callback triggered by unexpected source '{triggered_input_id}' or state indicates no action needed.")
             return (no_update,) * 13


    except Exception as e_param:
        logger.exception(f"Error processing callback inputs/parameters: {e_param}")
        error_alert = dbc.Alert(f"Error processing inputs: {e_param}", color="danger", dismissable=True)
        out_site = state_site_id if triggered_input_id != 'url' else None
        out_start = state_start_date_str or default_start.isoformat()
        out_end = state_end_date_str or default_end.isoformat()
        thresh = build_threshold_form(out_site, current_thresholds_state, "?")
        return empty_fig, initial_table_msg, None, {'site_id': out_site}, stats_display, thresh, f"Input Error - {out_site}", error_alert, True, False, out_site, out_start, out_end

    # --- Data Fetching ---
    if not site_id_to_load:
         logger.info("No site ID determined. Skipping data fetch.")
         final_output_start = output_start_date if output_start_date is not no_update else default_start.isoformat()
         final_output_end = output_end_date if output_end_date is not no_update else default_end.isoformat()
         final_output_site_id = output_site_id if output_site_id is not no_update else None
         threshold_form = build_threshold_form(None, {}, "?") # Update form for empty site
         return empty_fig, initial_table_msg, None, None, stats_display, threshold_form, main_title, error_alert, True, False, final_output_site_id, final_output_start, final_output_end

    logger.info(f"--> Calling generate_plot_for_site: Site='{site_id_to_load}', Start='{start_date_to_load}', End='{end_date_to_load}', Reset={is_reset_action}, ThresholdTrigger={is_threshold_update}")
    try:
        thresholds_to_use = current_thresholds_state if is_threshold_update and current_thresholds_state else None
        logger.debug(f"Using thresholds_override: {thresholds_to_use is not None}")

        # Main data processing function call
        fig, df_processed, err_func, name_func, final_start, final_end, units_val, found_thresholds, stats_dict = generate_plot_for_site(
            site_id_to_load,
            start_date_to_load,
            end_date_to_load,
            is_reset_action,
            logger,
            thresholds_override=thresholds_to_use
            )

        logger.info(f"<-- Returned from generate_plot_for_site. Error: '{err_func}', SiteName: '{name_func}', Range: {final_start}-{final_end}, Units: {units_val}")
        df_shape_log = f"DataFrame shape: {df_processed.shape}" if df_processed is not None else "DataFrame is None"
        logger.debug(f"Returned data: {df_shape_log}, Found Thresholds: {found_thresholds is not None}, Stats: {stats_dict is not None}")


        # Store comprehensive site info
        site_info_output = {'site_id': site_id_to_load, 'name': name_func, 'units': units_val, 'start': final_start, 'end': final_end}
        # Determine thresholds to display in form
        thresholds_for_form = found_thresholds if found_thresholds else (current_thresholds_state if current_thresholds_state else {})
        logger.debug(f"Thresholds for form generation: {thresholds_for_form}")
        threshold_form = build_threshold_form(site_id_to_load, thresholds_for_form, units_val or "?")

        # Validate returned dates
        # ... (validation logic unchanged) ...
        valid_final_start = None
        valid_final_end = None
        if final_start:
            try: datetime.strptime(final_start, '%Y-%m-%d'); valid_final_start = final_start
            except (ValueError, TypeError): logger.warning(f"generate_plot_for_site returned invalid final_start: {final_start}")
        if final_end:
             try: datetime.strptime(final_end, '%Y-%m-%d'); valid_final_end = final_end
             except (ValueError, TypeError): logger.warning(f"generate_plot_for_site returned invalid final_end: {final_end}")


        # Update UI date pickers if needed (Reset action)
        if is_reset_action and valid_final_start and valid_final_end:
            logger.debug(f"Reset action: Updating date pickers to {valid_final_start} - {valid_final_end}")
            output_start_date = valid_final_start
            output_end_date = valid_final_end
        elif is_initial_load or triggered_input_id in ['quick-year-button', 'quick-month-button']:
             logger.debug("Initial load or Quick Date: Keeping previously determined output dates.")
             pass
        else:
             logger.debug("Update/Threshold trigger: Keeping state dates for output.")
             output_start_date = state_start_date_str
             output_end_date = state_end_date_str

        # Handle errors returned from function
        if err_func:
            logger.error(f"Error returned from generate_plot_for_site for {site_id_to_load}: {err_func}")
            error_alert = dbc.Alert(f"Error loading data for {site_id_to_load}: {err_func}", color="danger", dismissable=True)
            main_title = f"{name_func or '?'} ({site_id_to_load}) - Data Error"
            # Ensure date pickers retain values
            final_output_start = output_start_date if output_start_date is not no_update else state_start_date_str
            final_output_end = output_end_date if output_end_date is not no_update else state_end_date_str
            return empty_fig, initial_table_msg, None, site_info_output, stats_display, threshold_form, main_title, error_alert, True, False, site_id_to_load, final_output_start, final_output_end

        # Handle case where no data found
        if fig is None or df_processed is None or df_processed.empty:
            logger.warning(f"No data found or returned for site {site_id_to_load} in period {final_start}-{final_end}.")
            display_start = valid_final_start or "N/A"
            display_end = valid_final_end or "N/A"
            error_alert = dbc.Alert(f"No data found for site {site_id_to_load} in the selected period ({display_start} to {display_end}).", color="warning", dismissable=True)
            main_title = f"{name_func or '?'} ({site_id_to_load})"
            # Ensure date pickers retain values
            final_output_start = output_start_date if output_start_date is not no_update else state_start_date_str
            final_output_end = output_end_date if output_end_date is not no_update else state_end_date_str
            return empty_fig, initial_table_msg, None, site_info_output, stats_display, threshold_form, main_title, error_alert, True, False, site_id_to_load, final_output_start, final_output_end

        # --- Success Case ---
        logger.info(f"Data loaded successfully for {site_id_to_load}. Preparing outputs.")
        logger.debug("Calling create_dash_data_table...")
        table_component = create_dash_data_table(df_processed.copy(), units_val, 'editable-data-table', logger)
        if not isinstance(table_component, dash_table.DataTable):
             logger.error("Failed to create data table component.")
             error_alert = table_component # Show alert from helper if it failed
             table_component = html.Div("Error generating table.") # Prevent error
             save_disabled = True
             edit_status_open = False
        else:
             save_disabled = False
             edit_status_open = True

        logger.debug("Preparing data for dcc.Store...")
        df_store = df_processed.copy()
        if 'Date' in df_store.columns:
            # Ensure conversion to string for JSON compatibility
            df_store['Date'] = df_store['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        stored_data = df_store.to_json(orient='split', date_format='iso')
        logger.debug(f"Stored data size: {len(stored_data)} chars")

        # Build Stats display
        if stats_dict:
            logger.debug(f"Generating stats display from: {stats_dict}")
            # ... (stats display generation unchanged) ...
            stats_display = html.Div([
                 html.P([html.Strong("Count: "), html.Span(stats_dict.get('count'))]),
                 html.P([html.Strong(f"Mean ({stats_dict.get('units', '?')}): "), html.Span(f"{stats_dict.get('mean'):.2f}" if isinstance(stats_dict.get('mean'), (int, float)) else stats_dict.get('mean'))]),
                 html.P([html.Strong(f"Min ({stats_dict.get('units', '?')}): "), html.Span(f"{stats_dict.get('min'):.2f}" if isinstance(stats_dict.get('min'), (int, float)) else stats_dict.get('min'))]),
                 html.P([html.Strong(f"Max ({stats_dict.get('units', '?')}): "), html.Span(f"{stats_dict.get('max'):.2f}" if isinstance(stats_dict.get('max'), (int, float)) else stats_dict.get('max'))])
             ], className="small")
        else:
            logger.warning("No stats_dict returned from generate_plot_for_site.")
            stats_display = html.P("Statistics not available.")

        main_title = f"{name_func or '?'} ({site_id_to_load}) | {final_start} to {final_end}"

        # Ensure final date outputs are correct ISO strings
        final_output_start = output_start_date if output_start_date is not no_update else valid_final_start
        final_output_end = output_end_date if output_end_date is not no_update else valid_final_end
        logger.info("Successfully prepared all outputs.")
        return fig, table_component, stored_data, site_info_output, stats_display, threshold_form, main_title, error_alert, save_disabled, edit_status_open, site_id_to_load, final_output_start, final_output_end

    except Exception as e:
        # <<< MODIFIED: Use logger.exception to include traceback >>>
        logger.exception(f"Unhandled exception in update_data_and_plots callback for site '{site_id_to_load}': {e}")
        error_alert = dbc.Alert(f"An unexpected server error occurred processing the request. Please check logs.", color="danger", dismissable=True)
        main_title = f"Error Processing {site_id_to_load or 'Unknown Site'}"
        # Attempt to build threshold form even on error
        try:
             thresholds_state_to_use = current_thresholds_state if current_thresholds_state else (get_site_thresholds(site_id_to_load, logger) if site_id_to_load else {})
             if thresholds_state_to_use is None: thresholds_state_to_use = {}
        except Exception as te:
             logger.error(f"Failed to get thresholds after main error: {te}")
             thresholds_state_to_use = {}
        threshold_form = build_threshold_form(site_id_to_load, thresholds_state_to_use, "?")

        out_site = site_id_to_load
        default_start_iso = (date.today() - timedelta(days=30)).isoformat()
        default_end_iso = date.today().isoformat()
        # Ensure date pickers retain state or default
        out_start = output_start_date if output_start_date is not no_update else (state_start_date_str or default_start_iso)
        out_end = output_end_date if output_end_date is not no_update else (state_end_date_str or default_end_iso)

        return empty_fig, initial_table_msg, None, {'site_id': site_id_to_load}, stats_display, threshold_form, main_title, error_alert, True, False, out_site, out_start, out_end
# <<< END: Modified update_data_and_plots >>>

# --- Other Existing Callbacks ---
# ... (toggle_table_collapse unchanged) ...
# ... (update_site_thresholds unchanged) ...
# ... (handle_table_edit unchanged) ...
# ... (save_data unchanged) ...
# ... (display_click_data unchanged) ...
# ... (handle_qc_action unchanged) ...
# ... (update_url_on_data_change unchanged) ...
# ... (open_enter_data_modal unchanged) ...
# ... (handle_submit_new_data unchanged) ...
# ... (cancel_enter_data_modal unchanged) ...
# ... (open_add_multiple_data_modal unchanged) ...
# ... (handle_submit_multiple_data unchanged) ...
# ... (cancel_add_multiple_data_modal unchanged) ...


# <<< --- ADDED TEST CALLBACK DEFINITION --- >>>
app.clientside_callback(
    """
    function(n) {
        console.log("Test CS Callback Fired! Clicks:", n); // Add log
        // Simple action: update the div content
        if (n > 0) {
             return 'Test CS Callback Fired Successfully! Clicks: ' + n;
        }
        return window.dash_clientside.no_update; // Or return "Not fired yet" initially
    }
    """,
    Output('test-cs-output', 'children'),
    Input('test-cs-button', 'n_clicks'),
    prevent_initial_call=True
)
# <<< --- END OF TEST CALLBACK DEFINITION --- >>>


# --- Main Execution Block ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    # Default to False for production, True for local dev if env var not set
    debug_env = os.environ.get("DASH_DEBUG", "False").lower()
    debug_mode = debug_env == "true"
    # Determine host based on environment (e.g., for Google Cloud Run)
    # Cloud Run expects host '0.0.0.0' to be accessible externally
    host = '0.0.0.0' if 'K_SERVICE' in os.environ else '127.0.0.1'

    logger.info(f"Starting Dash server on http://{host}:{port}")
    logger.info(f" -> Debug mode: {'ON' if debug_mode else 'OFF'}")
    logger.info(f" -> Log Level: {logging.getLevelName(logger.getEffectiveLevel())}") # Log the effective level
    # Use app.run for local development with Gunicorn or similar in production
    app.run(host=host, port=port, debug=debug_mode)

# --- END main.py ---