import os

import dash_bootstrap_components as dbc
import dash
from dash import Dash, dcc, html, Input, Output, State
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objs as go

from app import app
from app import server
from apps import HOME, KMEANS, SVM

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

# https://dash-bootstrap-components.opensource.faculty.ai/docs/components/dropdown_menu/
DropDownMenu_MachineLearning = [
  # dbc.DropdownMenuItem("Home", href = "/apps/HOME"),
  dbc.DropdownMenuItem("K Means", href = "/apps/KMEANS"),
  dbc.DropdownMenuItem("SVM", href = "/apps/SVM"),
]

DropDown = html.Div([
  dbc.DropdownMenu(
    label = "Machine Learning",
    color = "primary",
    bs_size = ['sm', 'md', 'lg'][0],
    direction = ['up', 'down', 'left', 'right'][1],
    right = False,
    nav = False,
    in_navbar = True,
    children = DropDownMenu_MachineLearning,
  )
])

Navbar = dbc.Navbar(
  dbc.Container([
    html.Img(src = PLOTLY_LOGO, height = "30px"),
    dbc.NavbarBrand("plotly", href = "https://plot.ly"),
    dbc.NavbarToggler(id = "navbar-toggler"),
    dbc.Collapse(
      dbc.Row([
        dbc.Col([
          DropDown
        ]),
        ],
        no_gutters = True,
        # add a top margin to make things look nice when the navbar
        # isn't expanded (mt-3) remove the margin on medium or
        # larger screens (mt-md-0) when the navbar is expanded.
        # keep button and search box on same row (flex-nowrap).
        # align everything on the right with left margin (ml-auto).
        className = "ml-auto flex-nowrap mt-3 mt-md-0"
      ),
      id = "navbar-collapse",
      navbar = True,
    ),
  ]),
  className = "Navbar"
)

Content = html.Div(
  id = "page-content",
  className = "Content",
  children = []
)

app.layout = html.Div([
  dcc.Location(id = "url", refresh = False),
  Navbar,
  dbc.Container([Content], fluid = True)
  ])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == "/":
        return KMEANS.layout
    elif pathname == "/apps/HOME":
        return KMEANS.layout
    elif pathname == "/apps/KMEANS":
        return KMEANS.layout
    elif pathname == "/apps/SVM":
        return SVM.layout
    return dbc.Jumbotron([
    html.H1("404: Not found", className = "text-danger"),
    html.Hr(),
    html.P(f"The pathname {pathname} was not recognised..."),
    ])

@app.callback(
  Output(f"navbar-collapse", "is_open"),
  [Input(f"navbar-toggler", "n_clicks")],
  [State(f"navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

if __name__ == '__main__':
  app.run_server(debug = True)

