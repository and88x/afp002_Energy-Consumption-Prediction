import pandas as pd
import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# data = read_csv("./databases/AEP_hourly.csv")
# df = pd.read_parquet("./databases/est_hourly.paruqet", engine="pyarrow")

df = pd.read_csv("./databases/PJME_hourly.csv")

df.Datetime = pd.to_datetime(df.Datetime)
df.set_index("Datetime")

# from statsmodels.tsa.stattools import adfuller
# statistical_result = adfuller(df.PJME_MW)
# print('ADF Statistic: %f' % statistical_result[0])
# print('p-value: %f' % statistical_result[1])
# print('Critical Values:')
# for key, value in statistical_result[4].items():
#     print('\t%s: %.3f' % (key, value))
# ADF Statistic: -18.828913
# p-value: 0.000000
# Critical Values:
#         1%: -3.430
#         5%: -2.862
#         10%: -2.567


fig = px.line(df, y="PJME_MW")

# Edit the layout
fig.update_layout(
    title="PJM East Region: 2001-2018 (PJME)",
    xaxis_title="Hourly Date Time",
    yaxis_title="Power consumption [MW]",
)

fig2 = px.histogram(df, x="PJME_MW", nbins=100, opacity=0.8)
fig3 = px.histogram(df.diff(periods=1), x="PJME_MW", nbins=100, opacity=0.8)

app = dash.Dash()
app.layout = html.Div(
    [
        dcc.Graph(figure=fig),
        html.P("Mean:"),
        dcc.Graph(figure=fig2),
        html.P("Mean:"),
        dcc.Graph(figure=fig3),
    ]
)

app.run_server(debug=True)

app = dash.Dash(__name__)
