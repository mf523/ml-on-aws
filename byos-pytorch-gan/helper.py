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


def load_model(path, *, model_cls=None, params=None, filename=None, device=None):

    import os
    import torch
    
    model_pt_path = path
    if not filename is None:
        model_pt_path = os.path.join(path, filename)
        
    if device is None:
        device = 'cpu'
        
    if not model_cls is None:
        model = model_cls(**params)
        model.load_state_dict(torch.load(model_pt_path, map_location=torch.device(device)))
    else:
        model = torch.jit.load(model_pt_path, map_location=torch.device(device))

    model.to(device)
    
    return model

