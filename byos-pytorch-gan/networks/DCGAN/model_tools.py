

def generate_fake_handwriting(model, *, num_images, nz, device=None):

    import torch
    import torchvision.utils as vutils
    from io import BytesIO
    from PIL import Image
    

    z = torch.randn(num_images, nz, 1, 1, device=device)
    fake = model(z)

    imgio = BytesIO()
    vutils.save_image(fake.detach(), imgio, normalize=True, format="PNG")
    img = Image.open(imgio)
    
    return img
