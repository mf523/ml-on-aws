# generate a list of random noises
def generate_noises(num_images=1, dim=512):
    import torch
    
    noises = torch.randn(num_images, dim)
    
    return noises

# plot images using torchvision and matplotlib
def show_multiple_pictures(pics):
    import matplotlib.pyplot as plt
    import torchvision
    import torch

    generated_images = torch.Tensor(pics)
    grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1),
                                       scale_each=True, normalize=True)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.show()
    
    
def create_pgan_netG_512_512(gnet_cls, model_pt_path=None, device='cpu'):
    import torch
    
    model = gnet_cls(512, 512) #, True, 0.2, True)
    model.addScale(512)
    model.addScale(512)
    model.addScale(512)
    model.addScale(256)
    model.addScale(128)
    model.addScale(64)
    model.addScale(32)
    
    if not model_pt_path is None:
        state_dict = torch.load(model_pt_path, map_location=device)
        model.load_state_dict(state_dict)
        
    model.to(device)
    
    return model


def load_pgan_netG(model_pt_path, device='cpu'):
    import torch
    
    if not model_pt_path is None:
        model = torch.jit.load(model_pt_path, map_location=device)
        
    model.to(device)
    
    return model
