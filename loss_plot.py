import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_loss(directory):


    Loss = torch.load(directory)
    plot_y = list(Loss[0])
    plot_x = range(0,len(plot_y))
    plt.plot(plot_x, plot_y, '.-')
    plt_title = ''
    plt.title(plt_title)
    plt.xlabel('per batch')
    plt.ylabel('LOSS')
    # plt.savefig(file_name)
    plt.show()


plot_loss('loss/loss_bsz8_lr0.001')