import torch

def select_label(dataloader_0,label_to_see,sample_size=20,train=True):
    if train:
        dataloader_iterator = dataloader_0.train()
    else:
        dataloader_iterator = dataloader_0.test()

    images = []
    remaining = sample_size
    for databatch_0 in dataloader_iterator:
        images_ = databatch_0[0]
        labels_ = databatch_0[1]
        selected_index = labels_ == label_to_see
        num_images_encountered = selected_index.float().sum()

        if num_images_encountered > 0:
            num_select = min(num_images_encountered,remaining)
            selected_images = images_[selected_index]
            selected_images = selected_images[:num_select]
            images.append(selected_images)

            remaining -= num_select
            if remaining > 0:
                break

    images = torch.cat(images,dim=0)
    return images