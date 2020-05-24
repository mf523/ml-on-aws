# generate a list of random noises
def generate_noises(num_images=1, dim=512):
    import torch
    
    noises = torch.randn(num_images, dim)
    
    return noises


def generate_labels(num_images=1):
    import torch

    labels = torch.LongTensor(torch.randint(0, 10, (num_images,)))
    
    return labels


# plot images using torchvision and matplotlib
def show_multiple_pictures(pics, nrow=10):
    import matplotlib.pyplot as plt
    import torchvision
    import torch

    generated_images = torch.Tensor(pics)
    grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1),
                                       scale_each=True, normalize=True, nrow=nrow)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.show()
    


def generate_fake_handwriting(model, *, batch_size, nz, device=None):

    import torch
    import torchvision.utils as vutils
    from io import BytesIO
    from PIL import Image
    

    z = torch.randn(batch_size, nz, 1, 1, device=device)
    labels = torch.LongTensor(
                torch.randint(0, 10, (batch_size,))).to(device)
    fake = model(z, labels)

    imgio = BytesIO()
    vutils.save_image(fake.detach(), imgio, normalize=True, format="PNG")
    img = Image.open(imgio)
    
    return img


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

