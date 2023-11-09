

def check_sizes(dataloader):
    train_size = 0
    test_size = 0
    if hasattr(dataloader,"train"):
        for databatch in dataloader.train():
            batchsize = databatch[0].size(0)
            train_size+= batchsize
    if hasattr(dataloader,"test"):
        for databatch in dataloader.test():
            batchsize = databatch[0].size(0)
            test_size+= batchsize
    return train_size,test_size