# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import os
import textwrap

import torch
from sagemaker_inference import content_types, decoder, default_inference_handler, encoder

INFERENCE_ACCELERATOR_PRESENT_ENV = "SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT"
DEFAULT_MODEL_FILENAME = "model.pt"
DEFAULT_MODEL_DEFINITION = "code/model.py"


class context(object):
    manifest = None
    system_properties = None

    def __init__(self, model_dir, model_def_file=DEFAULT_MODEL_DEFINITION):
        self.manifest = {
                        'model': {
                            'modelFile': model_def_file,
                            'serializedFile': DEFAULT_MODEL_FILENAME,
                        }
                    }

        self.system_properties = {
                        'model_dir': model_dir,
                    }


def model_fn(model_dir):
        
    if os.getenv(INFERENCE_ACCELERATOR_PRESENT_ENV) == "true":
        model_path = os.path.join(model_dir, DEFAULT_MODEL_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError("Failed to load model with default model_fn: missing file {}."
                                    .format(DEFAULT_MODEL_FILENAME))

        # Client-framework is CPU only. But model will run in Elastic Inference server with CUDA.
        return torch.jit.load(model_path, map_location=torch.device('cpu'))
    else:
        from handler import Handler
        
        ctxt = context(model_dir, 'code/model.py')
        hdlr = Handler()
        hdlr.initialize(context=ctxt)
            
        return hdlr.model


def input_fn(input_data, content_type):

        np_array = decoder.decode(input_data, content_type)
        tensor = torch.FloatTensor(
            np_array) if content_type in content_types.UTF8_TYPES else torch.from_numpy(np_array).float()
        return tensor
    
    
def predict_fn(data, model):

    with torch.no_grad():
        if os.getenv(INFERENCE_ACCELERATOR_PRESENT_ENV) == "true":
            device = torch.device("cpu")
            model = model.to(device)
            input_data = data.to(device)
            model.eval()
            with torch.jit.optimized_execution(True, {"target_device": "eia:0"}):
                output = model(input_data)
        else:
            from handler import Handler

            hdlr = Handler()
                
            if not hdlr.initialized:
                hdlr.initialize(model=model)

            if data is None:
                return None

            output = hdlr.inference(data)

    return output


def output_fn(prediction, accept):
    if type(prediction) == torch.Tensor:
        prediction = prediction.detach().cpu().numpy().tolist()
    encoded_prediction = encoder.encode(prediction, accept)
    if accept == content_types.CSV:
        encoded_prediction = encoded_prediction.encode("utf-8")

    return encoded_prediction

