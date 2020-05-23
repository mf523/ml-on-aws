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

        if context is None:
            if model is None:
                raise RuntimeError(f"Missing context and model")
            self.model = model
        else:
            self.manifest = context.manifest
            properties = context.system_properties
            gpu_id = properties.get("gpu_id")
            if gpu_id is None:
                gpu_id = 0
            model_dir = properties.get("model_dir")
            self.device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")

            # Read model serialize/pt file
            serialized_file = self.manifest['model']['serializedFile']
            model_pt_path = os.path.join(model_dir, serialized_file)
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

                from progressive_gan.networks.progressive_conv_net import GNet
                from model_tools import create_gnet_512_512

                self.model = create_gnet_512_512(GNet, model_pt_path, device=self.device)

            self.model.eval()
            logger.debug('Model file {0} loaded successfully'.format(model_pt_path))

        self.initialized = True

    def preprocess(self, request):

        import json
        from io import BytesIO
        import torch

        noises = []
        for idx, data in enumerate(request):
            raw_data = data.get("data")
            if raw_data is None:
                raw_data = data.get("body")

            stream = BytesIO(raw_data)

            np_output = np.load(stream, allow_pickle=True)
            break
        
        noises = torch.Tensor(np_output).float()

        return noises

    def inference(self, noises, labels):
        import torch
        
        noises.to(self.device)
        labels.to(self.device)
        with torch.no_grad():
            generated_images = self.model.forward(noises, labels)

        return generated_images

    def postprocess(self, inference_output):
        num_img, c, h, w = inference_output.shape
        detached_inference_output = inference_output.detach()
        output_classes = []
#         for i in range(num_img):
#             out = inference_output[i].numpy().tobytes()
#             _, y_hat = out.max(1)
#             predicted_idx = str(y_hat.item())
        output_classes.append(detached_inference_output.numpy())
    
        return output_classes


_service = Handler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context=context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data
