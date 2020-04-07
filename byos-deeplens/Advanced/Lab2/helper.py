def download(url, path):
    import requests
    
    r = requests.get(url, allow_redirects=True)
    if len(r.content) < 1024:
        raise Exception((path.split('/')[-1]).split('.')[0])
    else:
        open(path, 'wb').write(r.content)
        
        
def upload_to_s3(local_path, s3_data_location, *, search=None):
    import os
    from tqdm import tqdm
    from sagemaker.s3 import S3Uploader as s3up
    
    for root, dirs, files in os.walk(local_path):
        if len(files) > 0:
            idx = len(local_path)
            for name in tqdm(files, desc=f"Uploading folder '{root}'"):
                file_path = os.path.join(root, name)
                s3_path = os.path.join(s3_data_location, root[idx:])
                s3up.upload(file_path, s3_path)


def create_presigned_url(bucket_name, object_name, expiration=3600):
    
    import boto3
    from botocore.exceptions import ClientError
    from sagemaker.s3 import parse_s3_url


    # Generate a presigned URL for the S3 object
    s3_client = boto3.client('s3')
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except ClientError as e:
        print(e)
        return None

    # The response contains the presigned URL
    return response


def get_last_object_by_name(s3_location):
    
    import os
    from sagemaker.s3 import S3Downloader as s3down
    
    object_list = s3down.list(s3_location)

    object_list.sort()
    obj = object_list.pop()
    
    return obj


def get_basepath(s3_location):
    
    import os

    url = os.path.dirname(s3_location)
    
    return url


def get_object_path_by_filename(s3_location, filename):

    import os
    from sagemaker.s3 import S3Downloader as s3down
    
    object_list = s3down.list(s3_location)

    for url in object_list:
        if os.path.basename(url) == filename:
            return url
    
    return None


def load_model(model_cls, params, path, *, filename=None, device=None):

    import os
    import torch
    from dcgan.model import Generator

    model = model_cls(**params)

    if not filename is None:
        path = os.path.join(path, filename)

    with open(path, 'rb') as f:
        model.load(path)
    return model.to(device)


def generate_fake_handwriting(model, *, batch_size, nz, device=None):

    import torch
    import torchvision.utils as vutils
    from io import BytesIO
    from PIL import Image
    

    z = torch.randn(batch_size, nz, 1, 1, device=device)
    fake = model(z)

    imgio = BytesIO()
    vutils.save_image(fake.detach(), imgio, normalize=True, format="PNG")
    img = Image.open(imgio)
    
    return img