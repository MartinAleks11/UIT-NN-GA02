#Small script to check if model is initializing correctly

import torch.nn as nn
import json
from deepql_model import DeepQ_Model

with open('model_config/v17.1.json', 'r') as f:
        m = json.loads(f.read())

model = DeepQ_Model(m)

print(model)