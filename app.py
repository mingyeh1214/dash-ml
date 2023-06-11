import os

import dash
import dash_bootstrap_components as dbc

app = dash.Dash(
  __name__, 
  suppress_callback_exceptions = True,
  # https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/
  external_stylesheets = [dbc.themes.BOOTSTRAP],
  # meta_tags are required for the app layout to be mobile responsive
  meta_tags = [{
    'name': 'viewport',
    'content': 'width=device-width, initial-scale=1.0'
  }]
)
server = app.server

# pip freeze > requirements.txt
# git add .
# git commit -m 'update'
# git push heroku master





