import torch
import torch.nn
import numpy as np
import matplotlib.pyplot as plt
import tracemalloc

def zero_gradients(in_tensor):
    """Zeroes out the gradients of the input tensor.

    Args:
        in_tensor (torch.Tensor): input image in Tensor form.
    """
    if isinstance(in_tensor, torch.Tensor):  #Make sure type is correct
        if in_tensor.grad is not None:
            in_tensor.grad.detach_()
            in_tensor.grad.zero_() #Zero the gradients

    elif not isinstance(in_tensor, torch.Tensor):
        raise TypeError("Input tensor must be in form torch.Tensor")

def visualise(in_tensor, attack_tensor, perturbation, epsilon, in_label, attack_label, in_prob, attack_prob, mean, std):
    """Display the image, perturbation and advsersarial image in human readable form.

    Args:
        in_tensor (torch.Tensor): input image in Tensor form.
        attack_tensor (torch.Tensor): adversarial image in Tensor form.
        perturbation (torch.Tensor): perturbation image in Tensor form.
        epsilon (float): value to multiply the perturbation by before adding it to input image to find adversarial.
        in_label (str): top prediction of the input image.
        attack_label (str): top prediction of the adversarial image.
        in_prob (float): probability that the input image is the predicted class.
        attack_prob (float): probability that the adversarial image is the predicted class.
        mean (list): mean values of imagenet images.
        std (std): standard deviation of imagenet images.
    """
    in_tensor = in_tensor.squeeze(0) #Remove dimension in first position
    
    #Compete the reverse of the normalisation process
    in_tensor = in_tensor.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()
    in_tensor = np.transpose( in_tensor , (1,2,0)) #Transpose the tensor so it may be displayed
    in_tensor = np.clip(in_tensor, 0, 1) #Clip with 0 and 1 as min and max values
    
    attack_tensor = attack_tensor.squeeze(0) #Remove dimension in first position
    
    #Complete the reverse of the normalisation process
    attack_tensor = attack_tensor.mul(torch.FloatTensor(std).view(3,1,1)).add(torch.FloatTensor(mean).view(3,1,1)).numpy()
    attack_tensor = np.transpose( attack_tensor , (1,2,0)) #Transpose the tensor so it may be displayed
    attack_tensor = np.clip(attack_tensor, 0, 1) #Clip with 0 and 1 as min and max values
    
    perturbation = perturbation.squeeze(0).numpy() #Remove dimension in first position
    perturbation = np.transpose(perturbation, (1,2,0)) #Transpose the tensor so it may be displayed
    perturbation = np.clip(perturbation, 0, 1) #Clip with 0 and 1 as min and max values
    
    figure, ax = plt.subplots(1,3, figsize=(14,8)) #Define the figure
    ax[0].imshow(in_tensor) #Show the original image
    ax[0].set_title('Input', fontsize=20) #Set title
    
    
    ax[1].imshow(perturbation) #Show the perturbation
    ax[1].set_title('perturbation', fontsize=20) #Set title
    ax[1].set_yticklabels([]) #Remove ticks
    ax[1].set_xticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    
    ax[2].imshow(attack_tensor) #Show the adversarial example
    ax[2].set_title('Adversarial', fontsize=20) #Set title
    
    ax[0].axis('off') #Disable axis
    ax[2].axis('off') #Disable axis

    #Show the addition of epsilon multiplied perturbation, coordinates, text, size, alignment, 
    ax[0].text(1.1,0.5, "+{}*".format(round(epsilon,3)), size=15, ha="center",transform=ax[0].transAxes)
    
     #Show the prediction and probability of the original image
    ax[0].text(0.5,-0.13, "{}\n{}%".format(in_label, round(in_prob,2)), size=15, ha="center", transform=ax[0].transAxes)
    
    ax[1].text(1.1,0.5, " = ", size=15, ha="center", transform=ax[1].transAxes)

     #Show prediction and probability of the adversarial image
    ax[2].text(0.5,-0.13, "{}\n{}%".format(attack_label, round(attack_prob,2)), size=15, ha="center", 
         transform=ax[2].transAxes)
    
    plt.show() #Show the figure


def tracing_start():
    """Stop any active tracing and to start new trace.
    """
    tracemalloc.stop() #Stop any tracing
    tracemalloc.start() #Start new tracing

def trace_peak_memory():
    """Find and show the peak memory usage of the measured code in Megabytes.
    """
    _, peak = tracemalloc.get_traced_memory() #Find the peak of the traced memory
    mb_peak = peak/(1024*1024) #Convert from bytes to Megabytes
    print("Peak memory usage in MB: ", mb_peak) #Print the usage
    tracemalloc.stop()
    return mb_peak