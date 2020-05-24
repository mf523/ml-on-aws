import io
import logging
import numpy as np
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

logger = logging.getLogger(__name__)


class Handler(object):
    """
    FaceGenerator handler class. This handler takes list of noises
    and returns a corresponding list of images
    """

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False

    def initialize(self, *, context=None, model=None):
        """First try to load torchscript else load eager mode state_dict based model"""
        import os
        import json

        if context is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info('Available device {}'.format(self.device))
            if model is None:
                raise RuntimeError(f"Missing context and model")
            self.model = model.to(self.device)
        else:
            self.manifest = context.manifest
            properties = context.system_properties
            gpu_id = properties.get("gpu_id")
            if gpu_id is None:
                gpu_id = 0
            model_dir = properties.get("model_dir")
            
            from os import listdir
            from os.path import isfile, join
            entries = [f for f in listdir(model_dir) if join(model_dir, f)]

            self.device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
            logger.info('Available device {}'.format(self.device))

            # Read model serialize/pt file
            serialized_file = self.manifest['model']['serializedFile']
            model_pt_path = os.path.join(model_dir, serialized_file)
            json_manifest = json.dumps(self.manifest)
            json_properties = json.dumps(properties)
            json_entries = json.dumps(entries)
            logger.info(f'{json_manifest}')
            logger.info(f'{json_properties}')
            logger.info(f'{json_entries}')
            if not os.path.isfile(model_pt_path):
                raise RuntimeError(f"Missing the serialized model file '{model_pt_path}'")

            try:
                logger.info('Loading torchscript model to device {}'.format(self.device))
                self.model = torch.jit.load(model_pt_path)
            except Exception as e:
                # Read model definition file
                model_py_file = self.manifest['model']['modelFile']
                model_def_path = os.path.join(model_dir, model_py_file)
                if not os.path.isfile(model_def_path):
                    raise RuntimeError(f"Missing the model.py file '{model_def_path}'")

                from model_tools import load_model
                from model import Generator
                import json

                with open(os.path.join(model_dir, 'code', 'hps.json')) as fp:
                    hps = json.load(fp)
                    fp.close()

                params = {'nz': hps['nz'], 'nc': hps['nc'], 'ngf': hps['ngf'], 'num_classes': hps['num-classes']}

                self.model = load_model(model_pt_path, model_cls=Generator, params=params, device=self.device)

            self.model.eval()
            logger.debug('Model file {0} loaded successfully'.format(model_pt_path))

        self.initialized = True

    def preprocess(self, request, content_type='application/python-pickle'):
        import torch
        from serde import deserialize

        data = deserialize(request, content_type)
        
        noises = data['noises']
        labels = data['labels']

        noises = torch.Tensor(noises)

        return noises.view(noises.size(0), noises.size(1), 1, 1), torch.LongTensor(labels)

    def inference(self, noises, labels):
        import torch
        
        noises = noises.to(self.device)
        labels = labels.to(self.device)

        with torch.no_grad():
            generated_images = self.model.forward(noises, labels)

        return generated_images.to('cpu')

    def postprocess(self, inference_output, accept='application/python-pickle'):
        import torch
        from serde import serialize

        detached_inference_output = inference_output.detach()

        serialized_data = serialize(detached_inference_output.numpy(), accept)
        
        return serialized_data
    
    def _process(self, request):
        noises, labels = self.preprocess(request)
        data = self.inference(noises, labels)
        data = self.postprocess(data)
        
        return data


_service = Handler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context=context)

    if data is None:
        return None

    data = _service._process(data)

    return data
