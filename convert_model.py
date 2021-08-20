#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

import hls4ml
import os.path
from MLP import MLP
from hls4ml.converters import pytorch_to_hls

def main():
    config = {
        'PytorchModel': os.path.dirname(__file__)+'/pt/mlp_30ts_v0.pt',
        # 'PytorchModel': os.path.dirname(__file__)+'/pt/mlp_3000ts_v0.pt',
        'InputShape': '60,1',
        # 'InputShape': '6000,1',
        'HLSConfig': {
            'Model': {
                "Precision": "ap_fixed<16,6>",
                "ReuseFactor": 1
            }
        },
        "ProjectName": "Readout_MLP",
        "OutputDir": os.path.dirname(__file__) + '/output',
    }
    model = pytorch_to_hls(config)
    
    if model is not None:
        model.write()

if __name__== "__main__":
    main()
