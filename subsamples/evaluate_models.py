from gaugetools import GaugeDataModel, Trainer, CNN_LSTM
import json
import torch
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import shap

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
NUM_BACKGROUND_BATCHES = 1
NUM_EXPLAIN_BATCHES = 2

subsample_path = Path("/home/jo/cub/research/projects/turner-basinpredict/GaugePredict/subsamples/subsamples.json")
weight_dir = Path("/home/jo/cub/research/projects/turner-basinpredict/GaugePredict/subsamples") 
with open(subsample_path, 'r') as f:
    subsamples = json.load(f)

gdms = []
trainers = []
targets = []
preds = []
subsets = []
explainers = []
shaps = []
for i, subsample in enumerate(subsamples):
    print(f"On {i}")
    gdm = GaugeDataModel(data_files,
                     target_site,
                     start_date,
                     end_date,
                     tz,
                     sequence_length,
                     forecast_horizon,
                     cutoff_date,
                     site_filter=subsample
                     )
    gdm.setup()
    model = CNN_LSTM(gdm.train_dataset.input_channels, sequence_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.15e-6, weight_decay=0.5e-4)
    criterion = torch.nn.MSELoss()
    trainer = Trainer(model, gdm, gdm.scaler_y, criterion, optimizer)
    weight_path = weight_dir / f"gp_subset_{i}.pt"
    trainer.model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    trainer.model.eval()
    gdms.append(gdm)
    trainers.append(trainer)
    cur_targets, cur_preds = trainer.evaluate()
    targets.append(cur_targets.flatten())
    preds.append(cur_preds.flatten())
    subsets += [i]*len(cur_targets)
    test_loader_shuff = DataLoader(gdm.test_dataset, batch_size = gdm.batch_size, shuffle=True)
    background_batches = []
    for j, b in enumerate(test_loader_shuff):
        if j>NUM_BACKGROUND_BATCHES:
            break
        background_batches.append(b[0])
    background_gauges = torch.concat(background_batches)
    print(f"creating explainer for {i}")
    e = shap.GradientExplainer(trainer.model, background_gauges)
    explainers.append(e)
    eval_batches = []
    for j, b in enumerate(trainer.test_dataloader):
        if j > NUM_EXPLAIN_BATCHES:
            break
        eval_batches.append(b[0])
    explain_guages = torch.concat(eval_batches)
    print(f"explaining {i}")
    shaps.append(e.shap_values(explain_guages))

targets = np.concatenate(targets).flatten()
preds = np.concatenate(preds).flatten()
test_results = pd.DataFrame({'target': targets, 'pred': preds, 'subset': subsets})
test_results.to_csv("evaluation_results.csv", index=False)
np.save("shaps.py", np.array(shaps))

    


