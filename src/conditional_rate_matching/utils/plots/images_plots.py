# Display some generated samples
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def plot_sample(sample):
    if sample.shape[0] >= 64:
        plt.figure(figsize=(8, 8))
        for i in range(64):
            plt.subplot(8, 8, i + 1)
            plt.imshow(sample[i].view(28, 28), cmap='gray')
            plt.axis('off')
        plt.show()
    else:
        print("Not Enough Images in Sample")

def mnist_grid(sample,save_path=None):
    assert sample.size(0) >= 16
    sample = sample[:16]
    img = make_grid(sample.view(16, 1, 28, 28))
    npimg = np.transpose(img.detach().cpu().numpy(),(1,2,0))
    if save_path is None:
        plt.imshow(npimg)
        plt.show()
    else:
        plt.imsave(save_path,npimg)

def mnist_noise_bridge(x_hist,ts,steps_of_noise_to_see,number_of_images,save_path=None):
    fig, axs = plt.subplots(number_of_images, steps_of_noise_to_see, figsize=(steps_of_noise_to_see, number_of_images))
    for j in range(steps_of_noise_to_see):
        images_at_noise_level = x_hist[:,j,:]
        for i in range(number_of_images):
            img_noisy = images_at_noise_level[i].view(1,28,28).detach().numpy()
            axs[i, j].imshow(img_noisy.squeeze(), cmap='gray')
            if i == 0:
                axs[i, j].set_title(r'$\tau = {0}$'.format(round(ts[j].item(),2)))
            axs[i, j].axis('off')
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
