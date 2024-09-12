def move_model_to_cpu(model):
    for param in model.parameters():
        param.data = param.data.to('cpu')
        if param.grad is not None:
            param.grad.data = param.grad.data.to('cpu')
    model.to('cpu')