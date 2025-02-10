import time

import dash
import pandas as pd
import numpy as np
from dash import dcc, html, callback, ctx
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import ast

from utility import *
from global_config import *

from queries import links_with_filter, vsl_speeds

### Custom Functions (may belong in other files) ###


def threshold_display(threshold):
    return [html.Span(f'Delta Threshold: {threshold}')]

#TODO: trigger this properly
def anomaly_status(status):
    print('Calling')
    if status is None: 
        return 
    return [html.Span(f'Latest anomaly at milemarker {status['milemarker']}, lane {status['lane']}. {status['total_anomalies']} total anomalies detected in the chosen lookback period.')]
# ------------------------------------------------------------------
#                Plot configuration values
# ------------------------------------------------------------------
# TODO: set these values when you create a new page
site_path = "/anomaly_rds"       # Must begin with '/'
initial_update_seconds = 60            # Must be less than slider maximum and multiple of slider step value
initial_lookback_hours = 24                     # Must be less than slider maximum and multiple of slider step value
initial_selected_database = 'aidss-prod'         # Must be in Postgres instance
title = "Anomaly Detection Dashboard"
subtitle = """
Displaying live graph neural network anomaly detection predictions on RDS data."""
initial_selected_anomalies_source = 'gcn'
anomalies_source_options = {'gcn': 'Graph Convolutional Network', 'gat': 'Spatiotemporal Graph Attention Network', 
                          'rgcn': 'Relational Spatiotemporal Autoencoder', 'Ensemble': 'Ensemble'}
initial_delta_threshold = 0.0
initial_selected_lane = '1'
lanes_source_options = {'1': 'Lane 1', '2': 'Lane 2', '3': 'Lane 3', '4': 'Lane 4'}
def anomalies_source_status_display(anomalies_source_description):
    return [html.Span(html.Strong('Current model:'))]
def lane_source_status_display(lane_source_description):
    return [html.Span(html.Strong('Detections in:'))]
# ------------------------------------------------------------------
# ------------------------------------------------------------------

# ------------------------------------------------------------------
#                ID definitions for layout components
# ------------------------------------------------------------------
# Register the page with the Flask site for rendering.
dash.register_page(__name__, path=site_path, description=' '.join(site_path[1:].split('_')).upper())
# >>> Note: these ID values must be unique between pages!
# Use site path stub as a prefix in order to generate uniqueness.
assert site_path[0] == '/', "Site path must begin with '/'."
site_path_stub = site_path.strip('/')
metrics_text_div_id = site_path_stub + '-text'
interval_id = site_path_stub + '-reload-interval'
last_refresh_store_id = site_path_stub + '-refresh-dt-store'
settings_database_store_id = site_path_stub + '-setting-database-store'
settings_interval_store_id = site_path_stub + '-setting-interval-store'
settings_lookback_store_id = site_path_stub + '-setting-lookback-store'
refresh_button_id = site_path_stub + '-refresh-button'
refresh_slider_id = site_path_stub + '-refresh-slider'
refresh_status_id = site_path_stub + '-refresh-display'
lookback_slider_id = site_path_stub + '-lookback-slider'
lookback_status_id = site_path_stub + '-lookback-display'
# TODO: if you need to implement additional plots or caches, put them in the lines below.
west_speed_ts_id = site_path_stub + '-west-speed-ts'
west_data_store_id = site_path_stub + '-west-data-store'
anomalies_source_select_id = site_path_stub + '-anomalies-select'
anomalies_source_status_id = site_path_stub + '-anomalies-source-status'
anomalies_store_id = site_path_stub + '-anomalies-store'
settings_anomalies_source_store_id = site_path_stub + '-setting-anomalies-select-store'
threshold_slider_id = site_path_stub + '-threshold-slider'
threshold_status_id = site_path_stub + '-threshold-status'
settings_threshold_store_id = site_path_stub + '-setting-threshold-store'
anomaly_status_id = site_path_stub + '-anomaly-status'
settings_anomaly_status_id = site_path_stub + '-setting-anomaly-status'
lane_source_select_id = site_path_stub + '-lane-select'
lane_source_status_id = site_path_stub + '-lane-source-status'
settings_lane_source_store_id = site_path_stub + '-setting-lane-select-store'

# ------------------------------------------------------------------
# ------------------------------------------------------------------


# ------------------------------------------------------------------
#                Settings components configuration
# ------------------------------------------------------------------
# >>> Note: only need to change these if you want different limits on your page (e.g., because of longer query time)
# ------------------------------------------------------------------
# Database selection dropdown configuration values.
assert initial_selected_database in available_database_dict.keys(), "Default database not found in available databases."
# Refresh interval slider configuration values.
min_refresh = 60
max_refresh = 600
step_refresh = 30
refresh_marks = {i: '{}:{:02d}'.format(i // 60, i % 60) for i in
                 range(min_refresh, max_refresh + step_refresh, step_refresh)}
refresh_marks[0] = 'Off'
assert min_refresh <= initial_update_seconds <= max_refresh and initial_update_seconds % step_refresh == 0, \
    "Invalid default update interval based on maximum and step value."
# Generate values for the actual interval component, since it needs different information.
init_update_interval_millis, init_refresh_active, init_refresh_display = interval_update_time_active_display(
    interval_seconds_value=initial_update_seconds)
# Lookback period slider configuration values.
min_lookback = 2
max_lookback = 96
step_lookback = 6
lookback_marks = {i: '{}hr'.format(i) for i in range(2, 24, step_lookback)}
lookback_marks.update({i: '{}hr'.format(i) for i in range(24, max_lookback+1, step_lookback*2)})
assert initial_lookback_hours < max_lookback and initial_lookback_hours % step_lookback == 0, \
    "Invalid default lookback interval based on maximum and step value."
min_threshold = -0.3
max_threshold = 0.3
step_threshold = 0.025
threshold_range = np.arange(min_threshold, max_threshold+step_threshold, step_threshold).round(3)
threshold_range = threshold_range[threshold_range != -0.0]
threshold_marks = {float(x): str(float(x)) if x in [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3] else '' for x in threshold_range}
threshold_marks[0] = '0'
# ------------------------------------------------------------------
# ------------------------------------------------------------------


# ------------------------------------------------------------------
#                     Layout definition
# ------------------------------------------------------------------
# >>> Note: change only if you need to add more components
# ------------------------------------------------------------------
layout = html.Div([
    navbar_to_home,
    html.Br(),
    # Title and subtitle, next to database selection and manual refresh button
    dbc.Row([
        dbc.Col([
            html.H2(title),
            html.H4(subtitle)
        ]),
        dbc.Col([
            html.Button("Refresh dashboard", id=refresh_button_id),
            html.Div(id=anomalies_source_status_id,
                     children=anomalies_source_status_display(anomalies_source_options[initial_selected_anomalies_source])),
            dcc.Dropdown(id=anomalies_source_select_id, placeholder="Select anomaly detection model",
                         value=initial_selected_anomalies_source,
                         options=[{'label': v, 'value': k} for k, v in anomalies_source_options.items()]),
            html.Div(id=lane_source_status_id,
                     children=lane_source_status_display(lanes_source_options[initial_selected_lane])),
            dcc.Dropdown(id=lane_source_select_id, placeholder="Select lane number",
                         value=initial_selected_lane,
                         options=[{'label': v, 'value': k} for k, v in lanes_source_options.items()]),
        ])
    ]),
    # Refresh interval slider and display, next to lookback period slider and display
    dbc.Row([
        dbc.Col([
            html.H4("Refresh interval (minutes:seconds)"),
            dcc.Slider(0, max_refresh, step_refresh, id=refresh_slider_id, marks=refresh_marks,
                       value=initial_update_seconds, updatemode='drag'),
            html.Div(id=refresh_status_id, children=init_refresh_display)
        ]),
        dbc.Col([
            html.H4("Lookback period (hours)"),
            dcc.Slider(min_lookback, max_lookback, step_lookback, id=lookback_slider_id, marks=lookback_marks,
                       value=initial_lookback_hours, updatemode='drag'),
            html.Div(id=lookback_status_id, children=lookback_display(initial_lookback_hours))
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            html.H4("Delta threshold (MSE)"),
            dcc.Slider(min_threshold, max_threshold, step_threshold, id=threshold_slider_id, marks=threshold_marks,
                       value=initial_delta_threshold, updatemode='drag'),
            html.Div(id=threshold_status_id, children=threshold_display(initial_delta_threshold)),
            html.Div(id=anomaly_status_id, children=anomaly_status(None))
        ]),
    ]),
    # Current run text display and main plot
    html.Br(),
    html.Div([
        html.H3(children='Westbound direction', style={'textAlign': 'center'}),
        dcc.Graph(id=west_speed_ts_id),
        html.Div(id=metrics_text_div_id),
    ]),
    # Invisible components that need to be added to the page so the browser handles them in the background.
    # Create the refresh interval timer based on the initial values
    dcc.Interval(id=interval_id, interval=init_update_interval_millis, disabled=init_refresh_active,
                 n_intervals=0, max_intervals=-1),
    # Create the browser side data and settings storage
    # TODO: if you need to add caches, they should go here
    dcc.Store(id=west_data_store_id, storage_type='memory'),
    dcc.Store(id=anomalies_store_id, storage_type='memory'),
    dcc.Store(id=last_refresh_store_id, storage_type='memory'),
    dcc.Store(id=settings_interval_store_id, storage_type='memory', data=initial_update_seconds),
    dcc.Store(id=settings_lookback_store_id, storage_type='memory', data=initial_lookback_hours),
    dcc.Store(id=settings_threshold_store_id, storage_type='memory', data=initial_delta_threshold),
    dcc.Store(id=settings_anomalies_source_store_id, storage_type='memory', data=initial_selected_anomalies_source),
    dcc.Store(id=settings_lane_source_store_id, storage_type='memory', data=initial_selected_lane),
])
# ------------------------------------------------------------------
# ------------------------------------------------------------------


# ------------------------------------------------------------------
#                Callback functions for settings components
# ------------------------------------------------------------------
# >>> Note: these shouldn't need to be changed for any reason
# ------------------------------------------------------------------
@callback([Output(interval_id, 'interval'), Output(interval_id, 'disabled'),
           Output(refresh_status_id, 'children'), Output(settings_interval_store_id, 'data')],
          [Input(refresh_slider_id, 'value'), State(settings_interval_store_id, 'data')])
def update_interval(value, current_setting):
    """NO NEED TO UPDATE THIS FUNCTION (handles update interval slider value)"""
    print("DEBUG: called new update interval. Current value is {}.".format(current_setting))
    # Attempt to suppress unnecessary callbacks that would trigger a plot update (e.g., initial page load)
    if value == current_setting:
        print("DEBUG: suppressed null update to update interval.")
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    interval_val, interval_active, interval_status = interval_update_time_active_display(interval_seconds_value=value)
    print("DEBUG: New update interval is {} seconds.".format(value))
    # Return the interval milliseconds, disabled T/F, and refresh display text into their elements; plus new setting.
    return interval_val, interval_active, interval_status, value


@callback([Output(lookback_status_id, 'children'), Output(settings_lookback_store_id, 'data')],
          [Input(lookback_slider_id, 'value'), State(settings_lookback_store_id, 'data')])
def update_lookback(value, current_setting):
    """NO NEED TO UPDATE THIS FUNCTION (handles lookback period slider value)"""
    print("DEBUG: called new lookback period. Current value is {}.".format(current_setting))
    # Attempt to suppress unnecessary callbacks that would trigger a plot update (e.g., initial page load)
    if value == current_setting:
        print("DEBUG: suppressed null update to lookback period.")
        return dash.no_update, dash.no_update
    print("DEBUG: new lookback period is {} hours.".format(value))
    lookback_disp = lookback_display(value)
    return lookback_disp, value


@callback([Output(anomalies_source_status_id, 'children'), Output(settings_anomalies_source_store_id, 'data')],
          [Input(anomalies_source_select_id, 'value'), State(settings_anomalies_source_store_id, 'data')])
def update_anomalies_source_selected(value, current_setting):
    print("DEBUG: called new anomalies source selection. Current value is {}.".format(current_setting))
    # Attempt to suppress unnecessary callbacks that would trigger a plot update (e.g., initial page load)
    if value == current_setting:
        print("DEBUG: suppressed null update to anomalies source selection.")
        return dash.no_update, dash.no_update
    print("DEBUG: New anomalies source selected is {}.".format(value))
    # Return a display of the name/description of the database.
    return anomalies_source_status_display(anomalies_source_options[value]), value

@callback([Output(lane_source_status_id, 'children'), Output(settings_lane_source_store_id, 'data')],
          [Input(lane_source_select_id, 'value'), State(settings_lane_source_store_id, 'data')])
def update_lane_selected(value, current_setting):
    print("DEBUG: called new anomalies source selection. Current value is {}.".format(current_setting))
    # Attempt to suppress unnecessary callbacks that would trigger a plot update (e.g., initial page load)
    if value == current_setting:
        print("DEBUG: suppressed null update to anomalies source selection.")
        return dash.no_update, dash.no_update
    print("DEBUG: New anomalies source selected is {}.".format(value))
    # Return a display of the name/description of the database.
    return lane_source_status_display(lanes_source_options[value]), value

@callback([Output(threshold_status_id, 'children'), Output(settings_threshold_store_id, 'data')],
          [Input(threshold_slider_id, 'value'), State(settings_threshold_store_id, 'data')])
def update_threshold_slider(value, current_setting):
    print("DEBUG: called new threshold value. Current value is {}.".format(current_setting))
    # Attempt to suppress unnecessary callbacks that would trigger a plot update (e.g., initial page load)
    if value == current_setting:
        print("DEBUG: suppressed null update to threshold.")
        return dash.no_update, dash.no_update
    print("DEBUG: new delta threshold is {} MSE.".format(value))
    lookback_disp = threshold_display(value)
    return lookback_disp, value

# ------------------------------------------------------------------
# ------------------------------------------------------------------


# ------------------------------------------------------------------
#       Callback function for updating system run metadata
# ------------------------------------------------------------------
# >>> Note: you shouldn't need to update this unless you want to add more information
# ------------------------------------------------------------------
@callback([Output(metrics_text_div_id, 'children')],
          [Input(interval_id, 'n_intervals'), Input(refresh_button_id, 'n_clicks'),
           State(last_refresh_store_id, 'data')],
          prevent_initial_call=False)
def update_system_run_metadata(interval_number, click_number, last_data_refresh):
    """
    Updates the display of textual summary of current system run metadata. Runs on interval timer or when database
    selection is changed. Runs independent of plot update function, but on the same interval timer.
    :param interval_number: (unused) number of times interval has fired; irrelevant, but required for  callback
    :param click_number: (unused) number of time refresh button has been clicked; irrelevant, but required for callback
    :param database_display_value: (unused) display for DB selection; used to trigger plot update on database change
    :param current_database: state of page storage for current database value
    :param last_data_refresh: state of page storage for datetime value at which plot was last refreshed
    :return: elements to be placed into HTML <div>
    """
    print(f"DEBUG: {site_path_stub} function update_system_run_metadata called.")
    return get_system_run_text(current_database=initial_selected_database, current_last_data_refresh=last_data_refresh)


# !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !
# TODO: update this function with your desired inputs and query, specific to your dashboard page and data cache
# >>> Note: if you change the function call signature, change it in the update_graph_live() function below
# !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !
def database_query_rds(road_direction, start_datetime_exclusive, end_datetime_inclusive, database_name):
    # Get the RDS filter for the gantry range from the global config
    mm_low, mm_high, link_low, link_high = rds_filter_parameters['gantries']
    return links_with_filter(road_direction=road_direction, rds_measures='speed',
                             start_datetime_exclusive=start_datetime_exclusive,
                             end_datetime_inclusive=end_datetime_inclusive, database_name=database_name,
                             rds_filter_mm_low=mm_low, rds_filter_mm_high=mm_high,
                             rds_filter_link_id_min=link_low, rds_filter_link_id_max=link_high)

def get_anomaly_data(lookback_hours:int):
    query_formatted = """
        SELECT *
        FROM evaluations.ft_aed
        WHERE rds_update_time >= NOW() - INTERVAL '%(lookback_timedelta)s hours'
        """
    query_input_dict = {'lookback_timedelta': lookback_hours}
    query_columns = ['write_time', 'db_update_id', 'rds_update_time', 'milemarker', 'lane_id',
                     'direction', 'reconstruction_error_rgcn', 'reconstruction_error_gat',
                     'reconstruction_error_gcn', 'threshold_rgcn', 'threshold_gat', 'threshold_gcn']
    
    config_override = {
        'DATABASE': {
            'database_host': '10.80.3.22',
            'database_port': 5432,
            'database_username': 'austin',
            'database_password': 'austin4829'
        }
    }

    dict_anomaly = make_database_query(database_to_connect='aidss-prod',
                                     query_text=query_formatted,
                                     query_inputs=query_input_dict,
                                     dataframe_or_lists='df',
                                     dataframe_column_names=query_columns, 
                                     config_override=config_override)
    return dict_anomaly

# !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !
# TODO: update this function with your plot(s) and data cache structure
# >>> Note: if you change the function call signature, change it in the update_graph_live() function below
# !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !  !

def determine_anomalies(lane_data, model, delta_threshold):
    if model == 'Ensemble':
        lane_data.loc[:, 'anomaly1'] = lane_data['reconstruction_error_gcn'] > (lane_data['threshold_gcn'] + delta_threshold)
        lane_data.loc[:, 'anomaly2'] = lane_data['reconstruction_error_gat'] > (lane_data['threshold_gat'] + delta_threshold)
        lane_data.loc[:, 'anomaly3'] = lane_data['reconstruction_error_rgcn'] > (lane_data['threshold_rgcn'] + delta_threshold)
        lane_data.loc[:, 'anomaly'] = lane_data['anomaly1'] & lane_data['anomaly2'] & lane_data['anomaly3']
    else:
        lane_data.loc[:, 'anomaly'] = lane_data[f'reconstruction_error_{model}'] > (lane_data[f'threshold_{model}'] + delta_threshold)


    lane_data['anomaly'] = lane_data['anomaly'].astype(int)
    print("ANOMALIES:", np.sum(lane_data['anomaly']))
    return lane_data[lane_data['anomaly']==1]


def generate_figure(rds_cache_to_plot, dt_low_bound, dt_high_bound, anomaly_data, delta_threshold, model, lane):
    """
    Function to generate plot from data cache (and any secondary inputs).
    :param data_cache_to_plot: current data cache (likely as dataframe)
    :param dt_low_bound: lower datetime bound for plot x-axis
    :param dt_high_bound: upper datetime bound for plot y-axis
    :return: Plotly figure
    """
    if rds_cache_to_plot is None:
        return make_subplots(rows=1, cols=1) 

    rds_plot_df = rds_cache_to_plot.groupby('milemarker')[['speed', 'link_update_time']].resample(
        '30s', on='link_update_time').mean()
    rds_plot_df = pd.pivot_table(rds_plot_df, values='speed', index=['link_update_time'], columns=['milemarker'])

    #TODO: check the plot is correct
    lane_data = anomaly_data[anomaly_data['lane_id']==int(lane)]
    anomaly_points = determine_anomalies(lane_data, model, delta_threshold)

    # convert to central time
    rds_plot_df.index = rds_plot_df.index.tz_convert(local_tz_name)
    anomaly_points.set_index('rds_update_time', inplace=True)
    anomaly_points.index = anomaly_points.index.tz_convert(local_tz_name)

    status = {
        'milemarker': anomaly_points.iloc[-1]['milemarker'],
        'lane': anomaly_points.iloc[-1]['lane_id'],
        'total_anomalies': np.sum(anomaly_points['anomaly']),
    }
    anomaly_status(status)

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, shared_yaxes=True, y_title='Nashville   ---   Murfreesboro')
    # fig.update_layout(height=800, width=1200)
    fig.update_layout(height=800, autosize=True)
    fig.add_trace(
        go.Scatter(
            x=anomaly_points.index,  # Use the timestamp column
            y=anomaly_points['milemarker'],  # Use milemarker as y-axis
            mode='markers',
            marker=dict(color='purple', size=5, symbol='x'),
            name="Detected Anomalies"
        ),
        row=1, col=1
    )
    fig.add_trace(trace=px.imshow(rds_plot_df.T, origin='lower', zmin=0, zmax=85,
                                  labels=dict(x="Time", y="Mile marker", color='speed')).data[0],
                  row=1, col=1)
    fig.update_traces(xaxis='x2')
    fig.update_layout(xaxis_nticks=36, xaxis_range=[dt_low_bound, dt_high_bound])
    fig.update_layout(yaxis_nticks=6, yaxis_range=[53, 71])
    fig.update_layout(coloraxis=dict(colorscale='RdYlGn', cmin=0, cmax=85), showlegend=False)
    fig.update_xaxes(showgrid=True, showspikes=True, spikemode='across')

    return fig

# Callback function for data update (triggers the plot update)
@callback([Output(west_data_store_id, 'data'), Output(anomalies_store_id, 'data'), Output(last_refresh_store_id, 'data')],
          [Input(interval_id, 'n_intervals'), Input(refresh_button_id, 'n_clicks'),
           Input(lookback_status_id, 'children'), State(settings_lookback_store_id, 'data'),
           State(settings_interval_store_id, 'data'), State(last_refresh_store_id, 'data'), 
           State(west_data_store_id, 'data'), State(anomalies_store_id, 'data')],         # Added
          prevent_initial_call=False)
def update_data(interval_number, click_number, lookback_display_value,
                      current_lookback, current_update_interval_seconds,
                      last_data_refresh, existing_west_data_cache_dict, 
                      anomalies_cache_dict):
    """
    Callback function for graph updating. Fires on interval timer, manual refresh button, change in database selection,
        or change in lookback period selection.
    >> Note: updates that are called by interval timer less than 75% of interval time since last update are suppressed.
    :param interval_number: (unused) number of times interval has fired; irrelevant, but required for  callback
    :param click_number: (unused) number of time refresh button has been clicked; irrelevant, but required for callback
    :param database_display_value: (unused) display for DB selection; used to trigger plot update on database change
    :param lookback_display_value: (unused) display for lookback; used to trigger plot update on lookback period change
    :param vsl_display_value: (unused) display for VSL source; used to trigger plot update and cache dump on change
    :param current_database: state of page storage for current database value
    :param current_lookback: state of page storage for current lookback period value
    :param current_update_interval_seconds: state of page storage for current update interval (in seconds)
    :param last_data_refresh: state of page storage for datetime value at which plot was last refreshed
    :param current_vsl_source: state of page storage item for selected VSL source
    :param existing_west_data_cache_dict: state of page storage for east data cache
    :param existing_west_vsl_cache_dcit: state of page storage for east VSL cache
    :return: new Plotly figure, updated data cache storage value, updated refresh datetime value
    """
    def no_plot_update():
        """Needs to match the number of output/return values from this plot update function."""
        # TODO: if you update callback output count, update number of values in this helper function
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, \
            dash.no_update, dash.no_update

    print(f"DEBUG: {site_path_stub} function update_graph_live called by {ctx.triggered_id} component.")
    # Gather the timing information for the query and data cache.
    now_dt = now_tz()
    new_lookback_dt = now_dt - dt.timedelta(hours=current_lookback)
    last_refresh_dt = dt_from_str(last_data_refresh) if last_data_refresh is not None else new_lookback_dt

    # If the interval was the component that fired, check time since last update is close enough to the update interval.
    if ctx.triggered_id == interval_id:
        seconds_since_last_refresh = (now_dt - last_refresh_dt).total_seconds()
        if seconds_since_last_refresh < current_update_interval_seconds * 0.5:
            print("DEBUG: insufficient time since last refresh.")
            return no_plot_update()

    # If the database or lookback just changed and triggered the callback, we should dump the existing cache.

    # Get the updated dataframe for the data cache. This adds recent data and drops off any stale data.
    query_start_time = time.time()
    # Run the database query function to get the updated dataframe
    # time lower bound = time from the last query upper bound, i.e., last refresh timestamp
    # time upper bound = time at which this update is being made; this will be the new "last refresh timestamp"
    # TODO: update this call to database_query() if you need additional arguments
    west_new_data_cache = database_query_rds(start_datetime_exclusive=new_lookback_dt, end_datetime_inclusive=now_dt,
                                            database_name=initial_selected_database, road_direction='W')
    # print(extract_lane_data(west_new_data_cache).columns)
    # Concatenate the new and existing dataframes
    # >>> Note: all dataframes should be time sorted in descending order if you want to maintain order

    if west_new_data_cache is None:
        return no_plot_update()
    
    # anomaly_data = get_anomaly_data(current_lookback)
    anomaly_data = get_anomaly_data(1)
    print(f"DEBUG: {site_path_stub} query took {time.time() - query_start_time} seconds to run.")

    new_last_refresh = dt_to_str(now_dt)

    west_rds_cache = dataframe_to_page_storage(west_new_data_cache)
    anomaly_cache = dataframe_to_page_storage(anomaly_data)

    return west_rds_cache, anomaly_cache, new_last_refresh


# Callback function for plot update
# Updates the graph without necessarily querying more data
# Useful when moving the threshold or selecting a new model
@callback([Output(west_speed_ts_id, 'figure')], 
          [Input(interval_id, 'n_intervals'), Input(refresh_button_id, 'n_clicks'),
           Input(anomalies_source_status_id, 'children'), Input(threshold_status_id, 'children'), Input(lane_source_status_id, 'children'),
           Input(west_data_store_id, 'data'), Input(anomalies_store_id, 'data'),
           State(settings_lookback_store_id, 'data'), State(settings_anomalies_source_store_id, 'data'), State(settings_threshold_store_id, 'data'), State(settings_lane_source_store_id, 'data')])
def update_graph_live(interval_number, click_number, anomalies_display, threshold_display, lane_display,
                      west_data, anomaly_data, lookback, anomalies_display_value, threshold_display_value, lane_display_value):
    print(f"DEBUG: {site_path_stub} function update_graph_live called by {ctx.triggered_id} component.")

    west_data = load_dataframe_from_page_storage(west_data, 'link_update_time')
    anomaly_data = load_dataframe_from_page_storage(anomaly_data, 'rds_update_time')

    now_dt = now_tz()
    west_new_figure = generate_figure(rds_cache_to_plot=west_data, dt_low_bound=lookback, 
                                      dt_high_bound=now_dt, anomaly_data=anomaly_data,
                                      delta_threshold=threshold_display_value, model=anomalies_display_value, lane=lane_display_value)

    return (west_new_figure,)
# ------------------------------------------------------------------
# ------------------------------------------------------------------