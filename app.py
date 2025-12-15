from gaugetools import GaugeDataModel, Trainer, CNN_LSTM
import json
import torch
from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dash import Dash, html, dcc, Input, Output, callback


data_files = [{'path': '/home/jo/cub/research/projects/turner-basinpredict/data/cached_data/site_dict.json',
               'conversion_factor': 0.0283168466,
               'data_key': 'discharge'}]
target_site = "07374000"
start_date = "2010-01-01"
end_date = "2025-01-01"
tz="UTC"
sequence_length = 90
forecast_horizon = 15
cutoff_date = np.datetime64('2020-01-01')
na_filter = 0.25
conversion_factor = 0.0283168466
epochs = 100
cluster_columns = ['log_drain_area', 'dec_lat_va', 'dec_long_va']
n_clusters = 50
cluster_state = 1

subsample_path = Path("/home/jo/cub/research/projects/turner-basinpredict/GaugePredict/subsamples/subsamples.json")
weight_dir = Path("/home/jo/cub/research/projects/turner-basinpredict/GaugePredict/subsamples")
results_path = Path("/home/jo/cub/research/projects/turner-basinpredict/GaugePredict/subsamples/evaluation_results.csv")
shaps_path = Path("/home/jo/cub/research/projects/turner-basinpredict/GaugePredict/subsamples/shaps.npy")
sites_path = Path('/home/jo/cub/research/projects/turner-basinpredict/GaugePredict/site_positions.csv')
with open(subsample_path, 'r') as f:
     subsamples = json.load(f)

# gdms = []
# trainers = []
# evals = []
# for i, subsample in enumerate(subsamples):
#     gdm = GaugeDataModel(data_files,
#                      target_site,
#                      start_date,
#                      end_date,
#                      tz,
#                      sequence_length,
#                      forecast_horizon,
#                      cutoff_date,
#                      site_filter=subsample
#                      )
#     gdm.setup()
#     model = CNN_LSTM(gdm.train_dataset.input_channels, sequence_length)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1.15e-6, weight_decay=0.5e-4)
#     criterion = torch.nn.MSELoss()
#     trainer = Trainer(model, gdm, gdm.scaler_y, criterion, optimizer)
#     #trainer.fit(epochs)
#     weight_path = weight_dir / f"gp_subset_{i}.pt"
#     trainer.model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
#     evals.append(trainer.model.eval())
#     gdms.append(gdm)
#     trainers.append(trainer)

# targets, preds = trainers[0].evaluate()
# columns: target,pred,subset
shaps = np.load(shaps_path)
evaluation_df = pd.read_csv(results_path)
site_df = pd.read_csv(sites_path, dtype={'site': str})
print(site_df)
site_dfs = [site_df[site_df['site'].isin(subsample)] for subsample in subsamples]
app = Dash()
app.layout = html.Div([
    dcc.Dropdown(evaluation_df.subset.unique(), id='subset-dropdown'),
    dcc.Graph(id='performance-graph'),
    dcc.Graph(id='basin-map'),
    dcc.Graph(id='shap-graph')
])



def get_day_lookback_means(shap, day):
    day_shaps = shap[day]
    return np.mean(np.abs(day_shaps), axis=(1,2))

def get_day_basin_means(shap, day):
    day_shaps = shap[day]
    return np.mean(np.abs(day_shaps),axis=(0,2))

@callback(
    Output('basin-map', 'figure'),
    Input('subset-dropdown', 'value'),
    Input('performance-graph', 'hoverData')
)
def update_map(subset_value, hover_data):
    if subset_value is None:
        subset_value = 0
    subset_shaps = shaps[subset_value]
    if hover_data is None:
        day = 0
    else:
        day = hover_data['points'][0]['pointIndex']
    sites = subsamples[subset_value]
    basin_means = get_day_basin_means(subset_shaps, day)[:50]
    site_df = site_dfs[subset_value].merge(pd.DataFrame({'site': sites, 'shap_mean': basin_means}), on='site')
    fig = px.scatter_mapbox(site_df, lat='latitude', lon='longitude', color='shap_mean',
                            size='shap_mean', hover_name='site', zoom=3,
                            mapbox_style='carto-positron', color_continuous_scale='Viridis')
    print("i updated the map!")
    return fig
    

@callback(
    Output('performance-graph', 'figure'),
    Input('subset-dropdown', 'value')
)   
def update_graph(subset_value):
    if subset_value is None:
        subset_value = 0
    length = shaps.shape[1]
    df = evaluation_df[evaluation_df.subset == subset_value][:length]
    fig = go.Figure()
    fig.add_trace(go.Line(y=df['target'], x=df.index, name='Target'))
    fig.add_trace(go.Line(y=df['pred'], x=df.index,  name='Prediction'))
    return fig

@callback(
    Output('shap-graph', 'figure'),
    Input('subset-dropdown', 'value'),
    Input('performance-graph', 'hoverData')
)
def update_shap_graph(subset_value, hover_data):
    if subset_value is None:
        subset_value = 0
    subset_shaps = shaps[subset_value]
    if hover_data is None:
        day = 0
    else:
        day = hover_data['points'][0]['pointIndex']
    day_means = get_day_lookback_means(subset_shaps, day)
    fig = go.Figure()
    fig.add_trace(go.Line(x=list(range(len(day_means))), y=day_means))
    print("i updated the shap graph!")
    return fig


if __name__ == '__main__':
    app.run(debug=True)
