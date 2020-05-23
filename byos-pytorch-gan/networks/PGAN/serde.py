def serialize(nparray, content_type='application/python-pickle'):
    from io import BytesIO
    import numpy as np
    import pickle
    import json

    if content_type == 'application/json':
        serialized = json.dumps(nparray.tolist())
    elif content_type == 'application/x-npy':
        array_like = nparray.tolist()
        buffer = BytesIO()
        np.save(buffer, array_like)
        serialized = buffer.getvalue()
    else:
        serialized = pickle.dumps(nparray)

    return serialized


def deserialize(serialized, content_type='application/python-pickle'):
    from io import BytesIO
    import numpy as np
    from botocore.response import StreamingBody
    import json
    import pickle

    if isinstance(serialized, StreamingBody):
        serialized = serialized.read()

    if content_type == 'application/json':
        deserialized = json.loads(serialized)
    elif content_type == 'application/x-npy':
        stream = BytesIO(serialized)
        deserialized = np.load(stream, allow_pickle=True)
    else:
        deserialized = pickle.loads(serialized)
    
    return deserialized