# Imports

import torch.nn as nn
import predictor
import simulator

predictor = predictor.Predictor('models/nll_network_0911.pt', None)
simulator.simulate(2009, 0, predictor)
