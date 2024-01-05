import torch
import torch.nn as nn 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import tqdm

def train_classifier(model, 
                     train_dataloader, 
                     test_dataloader, 
                     device = 'cpu',
                     save_as='model.pth', 
                     max_epochs=10, 
                     early_stopping=None,
                     accuracy_goal=None,
                     lr=0.001):

    early_stopping = max_epochs if early_stopping is None else early_stopping
    accuracy_goal = 1 if accuracy_goal is None else accuracy_goal

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    max_accuracy = 0
    patience = 0

    for epoch in tqdm.tqdm(range(1, max_epochs), desc="Epochs"):

        for (data, target) in train_dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

        accuracy = get_model_accuracy(model, device, test_dataloader)

        if accuracy > max_accuracy:
            patience = 0
            max_accuracy = accuracy
            torch.save(model.state_dict(), save_as)
            print('INFO: current max accuracy: {}%'.format(100. * accuracy))
        else:
            patience += 1
            if patience > early_stopping:
                print('INFO: accuracy has not improved in {} epochs. Stopping training at {} epochs'.format(early_stopping, epoch))
                break
        if accuracy > accuracy_goal:
            print('INFO: accuracy goal reached. Stopping training at {} epochs'.format(epoch))
            break
    print('===================================')
    print('INFO: final max accuracy: {}%'.format(max_accuracy))
    print('===================================')


@torch.no_grad()
def get_model_accuracy(model, device, test_dataloader):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_dataloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item() 
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    accuracy = correct / len(test_dataloader.dataset)
    return accuracy


def plot_uncolor_images(images, title,  cmap="gray", figsize=(4, 4)):
    _, axes = plt.subplots(8, 8, figsize=figsize)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(images[i].squeeze(), cmap=cmap)
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

def plot_color_images(images, title, figsize=(4, 4)):
    fig, axes = plt.subplots(8, 8, figsize=figsize)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        img = images[i].permute(1, 2, 0)
        ax.imshow(img)
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

def plot_images(images, title, figsize=(4, 4), cmap=None):
    fig, axes = plt.subplots(8, 8, figsize=figsize)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        img = images[i].permute(1, 2, 0)
        if cmap is not None: ax.imshow(img, cmap=cmap)
        else: ax.imshow(img)
        ax.axis('off')
    plt.suptitle(title)
    plt.show()