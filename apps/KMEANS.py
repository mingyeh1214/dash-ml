import os

import numpy as np
import pandas as pd

import dash
import dash_bootstrap_components as dbc

from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objs as go

import pathlib

from app import app

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

def createClusteredData(N, K, sd):
  ###建立虛構的聚類資料####
  # N為資料筆數
  # K為聚類個數
  # sd為依每個聚類產生資料的標準差
  #########################
  pointsPerCluster = float(N)/K# 每個聚類產生資料筆數
  x_range = [0.0, 100]# x範圍
  y_range = [0.0, 100]# y範圍
  x_sd = y_sd = sd# 依每個聚類產生資料的標準差
  data = []# 建立空白list以儲存資料
  for i in range (K):
    xCentroid = np.random.uniform(x_range[0], x_range[1])# 依均勻分布產生聚類x座標
    yCentroid = np.random.uniform(y_range[0], y_range[1])# 依均勻分布產生聚類y座標
    for j in range(int(pointsPerCluster)):
      # 依常態分布產生資料
      data.append([np.random.normal(xCentroid, x_sd), np.random.normal(yCentroid, y_sd)])
  data = np.array(data)
  return data

def initial_centroids(X, K):
  ind = np.random.randint(X.shape[0], size = K)
  centroids = X[ind]
  return centroids

def find_closest_centroids(X, centroids):  
  m = X.shape[0]# m筆資料
  K = centroids.shape[0]# K個聚類
  N = np.zeros((m, K))# 建立一個全為0的mxk array用來紀錄每一筆資料與每一個聚類的距離
  for k in range(K):
    N[:,k] = np.linalg.norm(X-centroids[k], ord = 2, axis=1)
    dist = np.amin(N, axis = 1)#min_distance_between_x_and_centroids
    idx = np.argmin(N, axis = 1)
    J = np.mean(dist)
  return {'J':J, 'idx':idx}

def compute_new_centroids(X, idx, centroids):
  m, n = X.shape
  K = centroids.shape[0]
  new_centroids = np.zeros((K, n))
  for k in range(K):
    groupX = X[idx == k]
    if groupX.shape[0] == 0:
      new_centroids[k,:] = centroids[k,]
    else:
      new_centroids[k,:] = np.mean(groupX , axis = 0)
  return new_centroids

def KMeans(X, K, max_iters, max_run):
  m, n = X.shape
  J_record = np.zeros((max_run, max_iters))
  centroids_record = np.zeros((max_run, max_iters, K, n))
  idx_record = np.zeros((max_run, max_iters, m))
  for j in range(max_run):
    centroids = initial_centroids(X, K)
    centroids_record[j, 0, :] = centroids
    for i in range(max_iters):
      parameter = find_closest_centroids(X, centroids)
      J = parameter['J']
      idx = parameter['idx']
      centroids = compute_new_centroids(X, idx, centroids)
      J_record[j, i] = J
      centroids_record[j, i, :] = centroids
      idx_record[j, i, :] = idx
  min_J_in_run = np.argmin(np.amin(J_record, axis = 1))
  J = J_record[min_J_in_run,:]
  centroids = centroids_record[min_J_in_run, max_iters - 1,:]
  idx = idx_record[min_J_in_run, max_iters - 1,:]
  parameter = {
    'min_J_in_run':min_J_in_run,
    'centroids':centroids,
    'idx':idx,
    'J':J,
    'centroids_record':centroids_record,
    'idx_record':idx_record,
    'J_record':J_record,
  }
  return parameter

def find_KMeans_K(X, max_iters, max_run, max_kmeans_cluster):
  J ={}
  for k in range(1, max_kmeans_cluster + 1):
    J[k] = KMeans(X, k, max_iters, max_run)['J'].min()
  return J

def plot_X(X):
  trace0 = go.Scatter(
      x = X[:,0],
      y = X[:,1],
      mode = 'markers',
      name = 'data',
      opacity = 0.5,
      marker = dict(      # change the marker style
          size = 6,
          # color = 'lightskyblue',
          # symbol = 'circle',
          line = dict(
              width = 1,
              # color = 'midnightblue'
          )
      )
  )
  layout = go.Layout(
    title = 'Random Clustered Data',
    width=400, height=400,
    template='plotly',
    margin = dict(l = 0, r = 0, t = 25, b = 0, pad = 0),
    xaxis = dict(range = [-5, 105]),
    yaxis = dict(range = [-5, 105])
  )
  traces = [trace0]
  fig = go.Figure(data = traces, layout = layout)
  return fig

def plot_groupX_and_centroids(X, centroids, idx):
  trace0 = go.Scatter(
      x = X[:,0],
      y = X[:,1],
      mode = 'markers',
      name = 'data',
      opacity = 0.5,
      marker = dict(      # change the marker style
          size = 6,
          color = idx,
          # symbol = 'circle',
          line = dict(
              width = 1,
              # color = 'midnightblue'
          )
      )
  )
  trace1 = go.Scatter(
      x = centroids[:,0],
      y = centroids[:,1],
      mode = 'markers',
      name = 'centroids',
      marker = dict(      # change the marker style
          size = 6,
          color = 'red',
          # symbol = 'circle',
          line = dict(
              width = 1,
              # color = 'midnightblue'
          )
      )
  )
  layout = go.Layout(
    title = 'Random Clustered Data',
    width=400, height=400,
    template='plotly',
    margin = dict(l = 0, r = 0, t = 25, b = 0, pad = 0),
    xaxis = dict(range = [-5, 105]),
    yaxis = dict(range = [-5, 105]),
    legend=dict(
      yanchor="top",
      y=0.99,
      xanchor="left",
      x=0.01
      )
  )
  traces = [trace0 ,trace1]
  fig = go.Figure(data = traces, layout = layout)
  return fig

def plot_J(J):
  trace0 = go.Scatter(
      x = list(J.keys()),
      y = list(J.values()),
      mode = 'markers+lines',
      name = 'lines'
  )
  traces = [trace0]  # assign traces to data
  layout = go.Layout(
    title = 'J',
    width = 400, 
    height = 400,
    template = 'plotly',
    margin = dict(l = 0, r = 0, t = 25, b = 0, pad = 0)
  )
  fig = go.Figure(data = traces, layout = layout)
  return fig

createClusteredData_N_input = {
  1: '100',
  2: '500',
  3: '1000',
  4: '5000'
}

content =  dbc.Container([
  html.H1('K-means'),
  # row 1
  dbc.Row([
    # row 1 col 1
    dbc.Col([
      html.Label('Number od Data:'),
      dcc.Slider(
        id = 'createClusteredData_N',
        min = 1,
        max = 4,
        step = 1,
        marks = createClusteredData_N_input,
        value = 1
      ),
      html.Label('Number of Clusters:'),
      dcc.Slider(
        id = "createClusteredData_K", 
        min = 1,
        max = 10,
        step = 1,
        marks = {i:str(i) for i in range(1, 11)},
        value = 3,
      ),
      html.Label('Standard Deviation:'),
      dcc.Slider(
        id = "createClusteredData_sd", 
        min = 1,
        max = 10,
        step = 1,
        marks = {i:str(i) for i in range(1, 11)},
        value = 3,
      ),
      dbc.Button(
        id = 'submit-button',
        color = "primary",
        n_clicks = 0,
        children = 'Generate Data'
      ),
      ], 
      width = 2
    ),
    # row 1 col 2
    dbc.Col([
      dbc.Spinner(
        dcc.Graph(id = "graph",
          config={
            'displayModeBar': False
          }
        ),
        color = "primary",
        size = "sm"
      )
    ], width = 5),
    dbc.Col([
      dbc.Spinner(
        dcc.Graph(id = "find_KMeans_K",
          config={
            'displayModeBar': False
          }
        ),
        color = "primary",
        size = "sm"
      )
    ], width = 5)
  ]),
  
  html.Hr(),
  
  # row 2
  dbc.Row([
    # row 2 col 1
    dbc.Col([
      
      dbc.InputGroup(
        [
          dbc.InputGroupAddon("聚類個數", addon_type="prepend"),
          dbc.Input(
            id = "KMeans_K",
            name = "KMeans_K",
            type = "number",
            placeholder = "聚類個數",
            value = 2,
            min = 1,
            step = 1,
            max = 10
          )
        ],
        className = "mb-1",
      ),
      
      dbc.InputGroup(
        [
          dbc.InputGroupAddon("迭代次數", addon_type="prepend"),
          dbc.Input(
            id = "KMeans_max_iters", 
            name = "KMeans_max_iters",
            type = "number",
            placeholder = "迭代次數",
            value = 10,
            min = 1,
            step = 1,
            max = 10
          )
        ],
        className = "mb-1",
      ),
      
      dbc.InputGroup(
        [
          dbc.InputGroupAddon("執行次數", addon_type="prepend"),
          dbc.Input(
            id = "KMeans_max_run", 
            name = "KMeans_max_run",
            type = "number",
            placeholder = "執行次數",
            value = 10,
            min = 1,
            step = 1,
            max = 100
          )
        ],
        className = "mb-1",
      ),
          
      dbc.Button(
        id = 'submit-button2',
        color="primary",
        n_clicks = 0,
        children = '執行分類'
      )
      
    ], 
    width = 2
    ),
    dbc.Col([
      dbc.Spinner(
        dcc.Graph(
          id = "graph2",
          config = {
            'displayModeBar': False
            }
        ),
        color = "primary",
        size = "sm"
      )
    ], width = 5),
    dbc.Col([
      "NONE"
    ], width = 5),
  ])
  ],
  fluid = True)

# app = dash.Dash(
#   __name__,
#   external_stylesheets=[dbc.themes.BOOTSTRAP],
#   meta_tags=[
#     {'name': 'viewport',
#     'content': 'width=device-width, initial-scale=1.0'}
#   ]
# )


layout = html.Div([
 content
  ])


@app.callback(
    [
      Output('graph', 'figure'),
      Output('find_KMeans_K', 'figure')
      ],
    [
      Input('submit-button', 'n_clicks')
      ],
    [
      State('createClusteredData_N', 'value'),
      State('createClusteredData_K', 'value'),
      State('createClusteredData_sd', 'value')
      ]
)
def output(n_clicks, createClusteredData_N, createClusteredData_K, createClusteredData_sd):
  data = createClusteredData(createClusteredData_N_input[createClusteredData_N], createClusteredData_K, createClusteredData_sd)
  pd.DataFrame(data).to_csv(DATA_PATH.joinpath("ClusteredData.csv"), index = False)
  data = pd.read_csv(DATA_PATH.joinpath("ClusteredData.csv")).to_numpy()
  fig = plot_X(data)
  J = find_KMeans_K(data, 10, 10, 10)
  fig2 = plot_J(J)
  return fig, fig2

@app.callback(
    Output('graph2', 'figure'),
    [
      Input('submit-button2', 'n_clicks')
    ],
    [
      State('KMeans_K', 'value'),
      State('KMeans_max_iters', 'value'),
      State('KMeans_max_run', 'value')
    ]
)
def output2(n_clicks, KMeans_K, KMeans_max_iters, KMeans_max_run):
  if n_clicks:
    data = pd.read_csv(DATA_PATH.joinpath("ClusteredData.csv")).to_numpy()
    parameter = KMeans(data, KMeans_K, KMeans_max_iters, KMeans_max_run)
    fig = plot_groupX_and_centroids(data, parameter['centroids'], parameter['idx'])
    return fig
  return {}

# if __name__ == '__main__':
#   app.run_server(debug = False)
