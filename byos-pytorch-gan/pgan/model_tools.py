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
