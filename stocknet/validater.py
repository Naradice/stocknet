import numpy

def validate(self, model_name):
    # load model
    ## check if file exists
    # load client
    ## check if file exists
    ## create data loader for validation

    mean_loss = 0
    out_ = {}
    ans_ = {}
    output_shape = val_loader.dataset[0][1].shape#tuple(batch, input, output]
    output_size = output_shape.numel()
    for index in range(0, output_size):
        out_[index] = numpy.array([])
        ans_[index] = numpy.array([])
        
    viewer = Rendere()

    with torch.no_grad():
        count = 0
        for values, ans in val_loader:
            outputs = model(values).to(self.device)
            ans = ans.to(self.device)
            loss = self.loss_fn(outputs, ans)
            mean_loss += loss.item()
            #output: [batchDim, outputDim]
            for index in range(0, output_size):
                out_[index] = numpy.append(out_[index], outputs.to('cpu').detach().numpy().copy())
                ans_[index] = numpy.append(ans_[index], ans.to('cpu').detach().numpy().copy())
            count += 1
    
    print('--------------------------------------------------')
    print(f'mean loss: {mean_loss/count}')
    print(f'mean dif ({index}): {[(out_[index] - ans_[index]).mean() for index in range(0, output_size)]}, var: {[(out_[index] - ans_[index]).var() for index in range(0, output_size)]}')
    print('--------------------------------------------------')