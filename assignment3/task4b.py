import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np


image = Image.open("images/zebra.jpg")
print("Image shape:", image.size)

model = torchvision.models.resnet18(pretrained=True)
print(model)
first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]
print("Image shape:", image.shape)

activation = first_conv_layer(image)
print("Activation shape:", activation.shape)


def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image


###############################################################################
# Task 4b: Visualize the activation of the first convolutional layer
###############################################################################

def plot_filter_and_activation(activations, indices, plotname, weights=torch.Tensor([])):
    """
    Function for making plots in task 4b and 4c. 
    
    Args: 
        activations: tensor conaining activations 
        indices: list of indices in the layers you want to plot 
        plotname: name of the plot to save
        weighst (optional): tensor of weights. If not given, only activations will be plotted
    """
    fig, axs = plt.subplots(nrows=2 if weights.nelement() else 1,
                            ncols=len(indices),
                            figsize=(12, 4))
    fig.suptitle('Indices', fontsize=14, fontweight='bold')

    a_subplot_index = 1 if weights.nelement() else 0
    
    if weights.nelement():
        ax0 = axs[0,:]
        # Adding subplot title
        ax0[0].annotate('Filter weights', xy=(0, 0.5), xytext=(-ax0[0].yaxis.labelpad - 5, 0),
                                xycoords=ax0[0].yaxis.label, textcoords='offset points',
                                size='large', ha='right', va='center', fontsize=12, fontweight='bold')
    
    ax1 = axs[a_subplot_index,:] if weights.nelement() else axs
    # Adding subplot title
    ax1[0].annotate('Activations', xy=(0, 0.5), xytext=(-ax1[0].yaxis.labelpad - 5, 0),
                            xycoords=ax1[0].yaxis.label, textcoords='offset points',
                            size='large', ha='right', va='center', fontsize=12, fontweight='bold')
    
    #Plotting subplots with title corresponding to the indices they represent.
    for i, index in enumerate(indices):
        if weights.nelement():
            w_img = torch_image_to_numpy(weights[index])
            ax0[i].imshow(w_img)
            ax0[i].set_title('{}'.format(index))
            ax0[i].set_xticks([])
            ax0[i].set_yticks([])

        a_img = torch_image_to_numpy(activations[0][index])
        ax1[i].imshow(a_img, cmap='gray')
        ax1[i].set_title('' if weights.nelement() else '{}'.format(index))
        ax1[i].set_xticks([])
        ax1[i].set_yticks([])
        
    plt.savefig(plotname)


indices = [14, 26, 32, 49, 52]
plot_filter_and_activation(activation,
                           indices, 
                           "plots/task4b_test.png",
                           first_conv_layer.weight,)

# Get the activations from the last convolutional layer
activation = image
for layer in list(model.children())[:-2]:
    activation = layer(activation)
print('Activation shape: ', activation.shape)     

indices = range(10)
plot_filter_and_activation(activation,
                           indices, 
                           "plots/task4c.png") 