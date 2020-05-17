import io
import logging
import numpy as np
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from ts.torch_handler.encoder import encode

logger = logging.getLogger(__name__)


class PganFaceGenerator(object):
    """
    FaceGenerator handler class. This handler takes list of noises
    and returns a corresponding list of images
    """

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False

    def initialize(self, context):
        """First try to load torchscript else load eager mode state_dict based model"""
        import os

        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        try:
            logger.info('Loading torchscript model to device {}'.format(self.device))
            self.model = torch.jit.load(model_pt_path)
        except Exception as e:
            # Read model definition file
            model_file = self.manifest['model']['modelFile']
            model_def_path = os.path.join(model_dir, model_file)
            if not os.path.isfile(model_def_path):
                raise RuntimeError("Missing the model.py file")

            state_dict = torch.load(model_pt_path, map_location=self.device)
            
            from model import GNet
            
            self.model = GNet(512, 512)
            self.model.addScale(512)
            self.model.addScale(512)
            self.model.addScale(512)
            self.model.addScale(256)
            self.model.addScale(128)
            self.model.addScale(64)
            self.model.addScale(32)
            self.model.load_state_dict(state_dict)

        self.model.eval()
        logger.debug('Model file {0} loaded successfully'.format(model_pt_path))

        self.initialized = True

    def preprocess(self, request):
        """
         Scales, crops, and normalizes a PIL image for a PyTorch model,
         returns an Numpy array
        """

        import numpy as np

        noises = []
        for idx, data in enumerate(request):
            raw_data = data.get("data")
            if raw_data is None:
                raw_data = data.get("body")

            noises = np.frombuffer(raw_data, dtype=np.uint8)
#             my_preprocess = transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.CenterCrop(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#             ])
#             input_image = Image.open(io.BytesIO(image))
#             input_image = my_preprocess(input_image).unsqueeze(0)

#             if input_image.shape is not None:
#                 if image_tensor is None:
#                     image_tensor = input_image
#                 else:
#                     image_tensor = torch.cat((image_tensor, input_image), 0)

#         inputLatent = torch.randn(5, 512)

        return noises

    def inference(self, x):
        return self.model.forward(x)

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


_service = PganFaceGenerator()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data
