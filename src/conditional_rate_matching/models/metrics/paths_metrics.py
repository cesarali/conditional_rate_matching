import torch
from torch.nn import functional as F
from conditional_rate_matching.models.metrics.fid_metrics import load_classifier

def map_proportion_nist(crm,device,max_number_of_batches=1):
    number_of_source_labels = crm.config.data0.number_of_labels
    number_of_target_labels = crm.config.data1.number_of_labels
    classifier = load_classifier(crm.config.data1.dataset_name ,device)  # CLASSIFIES TARGET AT TIME 1

    label_to_label_histograms = {label_source :torch.zeros(number_of_target_labels) for label_source in range(number_of_source_labels)}
    source_label_numbers = {label_source :0. for label_source in range(number_of_source_labels)}

    number_of_batches = 0
    for databatch_0 in crm.dataloader_0.test():
        images_ = databatch_0[0]
        labels_ = databatch_0[1]

        # evolves from the whole batch then performs statistics
        x_f = crm.pipeline(100 ,train=True ,x_0=images_)
        for label_to_see in range(number_of_source_labels):
            selected_index = labels_ == label_to_see
            selected_images = images_[selected_index]
            selected_target = x_f[selected_index]
            num_images_encountered = selected_images.size(0)

            if num_images_encountered > 0:
                source_label_numbers[label_to_see] += num_images_encountered # how many images of that label in source
                y = classifier(selected_target.view(-1 ,1 ,28 ,28))
                y = torch.argmax(y ,dim=1)
                label_to_label_histograms[label_to_see] += F.one_hot(y,number_of_target_labels).sum(axis=0) # how many of the target images are encountered from that source

        number_of_batches +=1

        if number_of_batches >= max_number_of_batches:
            break

    for source_label in range(number_of_source_labels):
        label_to_label_histograms[source_label] = label_to_label_histograms[source_label] / source_label_numbers[source_label]
        print(label_to_label_histograms[source_label])

    return label_to_label_histograms