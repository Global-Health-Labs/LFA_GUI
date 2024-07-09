#!/usr/bin/env python3

import pandas as pd
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Load the data
df = pd.read_csv('./20240531_113642.158/imganl4_output_data.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1("Intensity Line Chart over Time"),
    dcc.Graph(id='intensity-line-chart'),
    html.Label('Select Channel:'),
    dcc.Dropdown(
        id='channel-dropdown',
        options=[
            {'label': 'Blue', 'value': 'Blue'},
            {'label': 'Green', 'value': 'Green'},
            {'label': 'Red', 'value': 'Red'},
            {'label': 'All Channels', 'value': 'All'}
        ],
        value='All'
    ),
    html.Label('Select Rectangle:'),
    dcc.Dropdown(
        id='rectangle-dropdown',
        options=[{'label': f'Rectangle {rect}', 'value': rect} for rect in df['Rectangle'].unique()],
        value=df['Rectangle'].unique()[0]
    ),
    html.Label('Select Time:'),
    dcc.Slider(
        id='time-slider',
        min=df['Time (s)'].min(),
        max=df['Time (s)'].max(),
        step=1,
        marks={int(time): str(time) for time in df['Time (s)'].unique()},
        value=df['Time (s)'].min(),
    )
])

# Define callback to update the plot
@app.callback(
    Output('intensity-line-chart', 'figure'),
    [Input('channel-dropdown', 'value'),
     Input('rectangle-dropdown', 'value'),
     Input('time-slider', 'value')]
)
def update_graph(selected_channel, selected_rectangle, selected_time):
    filtered_df = df[(df['Rectangle'] == selected_rectangle) & (df['Time (s)'] == selected_time)]
    fig = go.Figure()

    colors = {'Blue': 'blue', 'Green': 'green', 'Red': 'red'}

    if selected_channel == 'All':
        for channel in ['Blue', 'Green', 'Red']:
            fig.add_trace(go.Scatter(
                x=filtered_df['Pixel'],
                y=filtered_df[channel],
                mode='lines',
                name=f'{channel} Channel',
                line=dict(color=colors[channel])
            ))
        title = f'Intensity of All Channels for Rectangle {selected_rectangle} at Time {selected_time}'
    else:
        fig.add_trace(go.Scatter(
            x=filtered_df['Pixel'],
            y=filtered_df[selected_channel],
            mode='lines',
            name=f'{selected_channel} Channel',
            line=dict(color=colors[selected_channel])
        ))
        title = f'Intensity {selected_channel} for Rectangle {selected_rectangle} at Time {selected_time}'

    fig.update_layout(
        title=title,
        xaxis_title='Pixel',
        yaxis_title='Intensity'
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
