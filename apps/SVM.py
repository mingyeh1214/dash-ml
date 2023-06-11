import os

import numpy as np
import pandas as pd


import dash
import dash_bootstrap_components as dbc

from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objs as go

import colorlover as cl

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

import pathlib

from app import app

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

plot_width = plot_height = 350
bright_cscale = [[0, '#FF0000'], [1, '#0000FF']]

def generate_data(n_samples, dataset, noise):
  if dataset == 'moons':
    X, y = datasets.make_moons(
      n_samples = n_samples,
      noise = noise,# Standard deviation of Gaussian noise added to the data.
      random_state = 0,
    )
    
  elif dataset == 'circles':
    X, y = datasets.make_circles(
      n_samples = n_samples,
      noise = noise,
      random_state = 1,
      factor = 0.5,# Scale factor between inner and outer circle in the range (0, 1).
    )

  elif dataset == 'linear':
    X, y = datasets.make_classification(
      n_samples = n_samples,# The number of samples.
      n_features = 2,# The total number of features.
      n_informative = 2,# The number of informative features.
      n_redundant = 0,# The number of redundant features.
      n_repeated = 0,# The number of duplicated features, drawn randomly from the informative and the redundant features.
      n_classes = 2,# The number of classes (or labels) of the classification problem.
      n_clusters_per_class = 1,# The number of clusters per class.
      flip_y = 0.01,# The fraction of samples whose class is assigned randomly.
      class_sep = 1.0,# The factor multiplying the hypercube size.
      random_state = 2,
    )
    rng = np.random.RandomState(2)
    X += noise * rng.uniform(size = X.shape)
  return X, y

def plot_X_y(X, y):
  trace0 = go.Scatter(
    x = X[:,0],
    y = X[:,1],
    mode = 'markers',
    name = 'data',
    marker = dict(
      size = 6,
      color = y,
      colorscale = bright_cscale,
      opacity = 0.5,
      symbol = 'circle',
      line = dict(
          width = 1,
          color = 'black',
      )
    )
  )
  layout = go.Layout(
    title = None,
    xaxis=dict(
            range = [X[:, 0].min() - 0.2, X[:, 0].max() + 0.2],
            zeroline=False,
        ),
        yaxis=dict(
            range = [X[:, 1].min() - 0.2, X[:, 1].max() + 0.2],
            zeroline=False,
        ),
    width = plot_width, 
    height = plot_height,
    template = 'plotly',
    margin = dict(l = 0, r = 0, t = 0, b = 0, pad = 0),
  )
  traces = [trace0]
  fig = go.Figure(data = traces, layout = layout)
  return fig

def Graph(id):
  Graph = dbc.Spinner(
    dcc.Graph(
      id = id,
      config = {'displayModeBar': False}
    ),
    color = "primary",
    size = "sm"
  )
  return Graph


def svm(data,
        kernel,
        C_coef = 1,
        C_power = 1,
        gamma_coef = 0.1,
        gamma_power = 0.5,
        degree = 3,
        shrinking = False):
  X, y = data.iloc[:, 0:2].to_numpy(), data.iloc[:,2].to_numpy()
  h = .3  # step size in the mesh

  # Data Pre-processing
  X = StandardScaler().fit_transform(X)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)

  x_min = X[:, 0].min() - 1
  x_max = X[:, 0].max() + 1
  y_min = X[:, 1].min() - 1
  y_max = X[:, 1].max() + 1
  xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
  )
  C = C_coef * 10 ** C_power
  gamma = gamma_coef * 10 ** gamma_power

  # Train SVM
  clf = SVC(
      C=C,
      kernel=kernel,
      degree=degree,
      gamma=gamma,
      shrinking=shrinking
  )
  clf.fit(X_train, y_train)
  
  y_pred_train = clf.predict(X_train)
  y_pred_test = clf.predict(X_test)
  train_score = accuracy_score(y_true=y_train, y_pred=y_pred_train)
  test_score = accuracy_score(y_true=y_test, y_pred=y_pred_test)

  # Plot the decision boundary. For that, we will assign a color to each
  # point in the mesh [x_min, x_max]x[y_min, y_max].
  if hasattr(clf, "decision_function"):
      Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
  else:
      Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
  return {
  'clf':clf, 
  'X':X, 
  'X_train':X_train, 
  'X_test':X_test, 
  'y_train':y_train, 
  'y_test':y_test, 
  'xx':xx, 
  'yy':yy, 
  'Z':Z, 
  'train_score':train_score, 
  'test_score':test_score,
  'C':C,
  }


def predict_plot(clf, X, X_train, X_test, y_train, y_test, xx, yy, Z, train_score, test_score, C):
  mesh_step = 0.3
  # Colorscale
  bright_cscale = [[0, '#FF0000'], [1, '#0000FF']]
  colorscale_zip = zip(
    np.arange(0, 1.01, 1 / 8),
    cl.scales['9']['div']['RdBu'])
  cscale = list(map(list, colorscale_zip))
  # Create the plot
  # Plot the prediction contour of the SVM
  trace0 = go.Contour(
      x=np.arange(xx.min(), xx.max(), mesh_step),
      y=np.arange(yy.min(), yy.max(), mesh_step),
      z=Z.reshape(xx.shape),
      # zmin=scaled_threshold - range,
      # zmax=scaled_threshold + range,
      name = 'prediction contour',
      hoverinfo='none',
      showscale=False,
      contours=dict(
          showlines=False
      ),
      colorscale=cscale,
      opacity=0.9
  )

  # Plot Training Data
  trace2 = go.Scatter(
      x=X_train[:, 0],
      y=X_train[:, 1],
      mode='markers',
      name=f'Train(acc={train_score:.3f})',
      marker=dict(
          size = 6,
          color=y_train,
          colorscale=bright_cscale,
          opacity = 0.5,
          line=dict(
              width=1
          )
      )
  )

  # Plot Test Data
  trace3 = go.Scatter(
      x=X_test[:, 0],
      y=X_test[:, 1],
      mode='markers',
      name=f'Test(acc={test_score:.3f})',
      marker=dict(
          size=6,
          symbol='triangle-up',
          color=y_test,
          colorscale=bright_cscale,
          opacity = 0.5,
          line=dict(
              width=1
          ),
      )
  )

  layout = go.Layout(
        xaxis=dict(
            # scaleanchor="y",
            # scaleratio=1,
            # ticks='',
            # showticklabels=False,
            # showgrid=False,
            range = [X[:, 0].min() - 0.2, X[:, 0].max() + 0.2],
            zeroline=False,
        ),
        yaxis=dict(
            # ticks='',
            # showticklabels=False,
            # showgrid=False,
            range = [X[:, 1].min() - 0.2, X[:, 1].max() + 0.2],
            zeroline=False,
        ),
        title = None,
        width = plot_width, 
        height = plot_height,
        hovermode='closest',
        template = 'plotly',
        legend=dict(x=0, y=1.1, orientation="h"),
        margin=dict(l=0, r=0, t=25, b=0),
    )
  data = [trace0, trace2,trace3 ]
  fig = go.Figure(data=data, layout = layout)
  return fig



content =  dbc.Container([
  html.H1('SVM'),
  dbc.Row([
    dbc.Col([
      html.Label('Sample Size:'),
      dcc.Slider(
        id = 'slider-dataset-sample-size',
        min = 100,
        max = 500,
        step = 100,
        marks = {i: str(i) for i in [100, 200, 300, 400, 500]},
        value = 300,
      ),
      html.Label('Noise Level:'),
      dcc.Slider(
        id = 'slider-dataset-noise-level',
        min = 0,
        max = 1,
        marks = {i / 10: str(i / 10) for i in range(0, 11, 2)},
        step = 0.2,
        value = 0.2,
      ),
      html.Label('C_power:'),
      dcc.Slider(
        id='slider-svm-parameter-C-power',
        min=-2,
        max=4,
        step = 1,
        value=0,
        marks={i: '{}'.format(10 ** i) for i in range(-2, 5)}
      ),
      html.Label('C_coef:'),
      dcc.Slider(
        id='slider-svm-parameter-C-coef',
        min=1,
        max=9,
        step = 1,
        value=1,
      ),
      html.Label('Degree for Poly Kernel:'),
      dcc.Slider(
        id = 'degree_for_ploy_kernel',
        min = 2,
        max = 10,
        value = 3,
        step = 1,
        marks = {i: str(i) for i in range(2, 11, 2)},
      ),
    ], width = 2),
    dbc.Col([
      Graph('graph_moons'),
      Graph('linear_moons'),
      Graph('rbf_moons'),
      Graph('poly_moons'),
      Graph('sigmoid_moons'),
    ], width = 3),
    dbc.Col([
      Graph('graph_circles'),
      Graph('linear_circles'),
      Graph('rbf_circles'),
      Graph('poly_circles'),
      Graph('sigmoid_circles'),
    ], width = 3),
    dbc.Col([
      Graph('graph_linear'),
      Graph('linear_linear'),
      Graph('rbf_linear'),
      Graph('poly_linear'),
      Graph('sigmoid_linear'),
    ], width = 3)
  ])
], fluid = True)


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

@app.callback(Output('slider-svm-parameter-C-coef', 'marks'),
              [Input('slider-svm-parameter-C-power', 'value')])
def update_slider_svm_parameter_C_coef(power):
    scale = 10 ** power
    return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}

@app.callback(
  [
    Output('graph_moons', 'figure'),
    Output('graph_circles', 'figure'),
    Output('graph_linear', 'figure'),
    Output('linear_moons', 'figure'),
    Output('linear_circles', 'figure'),
    Output('linear_linear', 'figure'),
    Output('rbf_moons', 'figure'),
    Output('rbf_circles', 'figure'),
    Output('rbf_linear', 'figure'),
    Output('poly_moons', 'figure'),
    Output('poly_circles', 'figure'),
    Output('poly_linear', 'figure'),
    Output('sigmoid_moons', 'figure'),
    Output('sigmoid_circles', 'figure'),
    Output('sigmoid_linear', 'figure'),
  ],
  [
    Input('slider-dataset-sample-size', 'value'),
    Input('slider-dataset-noise-level', 'value'),
    Input('slider-svm-parameter-C-coef', 'value'),
    Input('slider-svm-parameter-C-power', 'value'),
    Input('degree_for_ploy_kernel', 'value'),
  ],
)
def render_plot(slider_dataset_sample_size, slider_dataset_noise_level, slider_svm_parameter_C_coef, slider_svm_parameter_C_power, degree_for_ploy_kernel):
  X, y = generate_data(slider_dataset_sample_size, 'moons', slider_dataset_noise_level)
  pd.DataFrame({'x1':X[:,0], 'x2':X[:,1], 'y':y}).to_csv(DATA_PATH.joinpath("SVM_moons.csv"), index = False)
  graph_moons = plot_X_y(X, y)
  linear_moons = predict_plot(
    **svm(pd.read_csv(DATA_PATH.joinpath("SVM_moons.csv")), kernel = 'linear', C_coef = slider_svm_parameter_C_coef, C_power = slider_svm_parameter_C_power)
  )
  rbf_moons = predict_plot(
    **svm(pd.read_csv(DATA_PATH.joinpath("SVM_moons.csv")), kernel = 'rbf', C_coef = slider_svm_parameter_C_coef, C_power = slider_svm_parameter_C_power)
  )
  poly_moons = predict_plot(
    **svm(pd.read_csv(DATA_PATH.joinpath("SVM_moons.csv")), kernel = 'poly', C_coef = slider_svm_parameter_C_coef, C_power = slider_svm_parameter_C_power, degree = degree_for_ploy_kernel)
  )
  sigmoid_moons = predict_plot(
    **svm(pd.read_csv(DATA_PATH.joinpath("SVM_moons.csv")), kernel = 'sigmoid', C_coef = slider_svm_parameter_C_coef, C_power = slider_svm_parameter_C_power)
  )
  
  X, y = generate_data(slider_dataset_sample_size, 'circles', slider_dataset_noise_level)
  pd.DataFrame({'x1':X[:,0], 'x2':X[:,1], 'y':y}).to_csv(DATA_PATH.joinpath("SVM_circles.csv"), index = False)
  graph_circles = plot_X_y(X, y)
  linear_circles = predict_plot(
    **svm(pd.read_csv(DATA_PATH.joinpath("SVM_circles.csv")), kernel = 'linear', C_coef = slider_svm_parameter_C_coef, C_power = slider_svm_parameter_C_power)
  )
  rbf_circles = predict_plot(
    **svm(pd.read_csv(DATA_PATH.joinpath("SVM_circles.csv")), kernel = 'rbf', C_coef = slider_svm_parameter_C_coef, C_power = slider_svm_parameter_C_power)
  )
  poly_circles = predict_plot(
    **svm(pd.read_csv(DATA_PATH.joinpath("SVM_circles.csv")), kernel = 'poly', C_coef = slider_svm_parameter_C_coef, C_power = slider_svm_parameter_C_power, degree = degree_for_ploy_kernel)
  )
  sigmoid_circles = predict_plot(
    **svm(pd.read_csv(DATA_PATH.joinpath("SVM_circles.csv")), kernel = 'sigmoid', C_coef = slider_svm_parameter_C_coef, C_power = slider_svm_parameter_C_power)
  )
  
  X, y = generate_data(slider_dataset_sample_size, 'linear', slider_dataset_noise_level)
  pd.DataFrame({'x1':X[:,0], 'x2':X[:,1], 'y':y}).to_csv(DATA_PATH.joinpath("SVM_linear.csv"), index = False)
  graph_linear = plot_X_y(X, y)
  linear_linear = predict_plot(
    **svm(pd.read_csv(DATA_PATH.joinpath("SVM_linear.csv")), kernel = 'linear', C_coef = slider_svm_parameter_C_coef, C_power = slider_svm_parameter_C_power)
  )
  rbf_linear = predict_plot(
    **svm(pd.read_csv(DATA_PATH.joinpath("SVM_linear.csv")), kernel = 'rbf', C_coef = slider_svm_parameter_C_coef, C_power = slider_svm_parameter_C_power)
  )
  poly_linear = predict_plot(
    **svm(pd.read_csv(DATA_PATH.joinpath("SVM_linear.csv")), kernel = 'poly', C_coef = slider_svm_parameter_C_coef, C_power = slider_svm_parameter_C_power, degree = degree_for_ploy_kernel)
  )
  sigmoid_linear = predict_plot(
    **svm(pd.read_csv(DATA_PATH.joinpath("SVM_linear.csv")), kernel = 'sigmoid', C_coef = slider_svm_parameter_C_coef, C_power = slider_svm_parameter_C_power)
  )
  return graph_moons, graph_circles, graph_linear, linear_moons, linear_circles, linear_linear, rbf_moons, rbf_circles, rbf_linear, poly_moons, poly_circles, poly_linear, sigmoid_moons, sigmoid_circles, sigmoid_linear
# 
# if __name__ == '__main__':
#   app.run_server(debug = False)


