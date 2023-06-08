import numpy as np
import pandas
import cv2 as cv
import matplotlib.pyplot as plt

from skimage.color import rgb2lab
from skimage.color import lab2rgb


def rescale(img):
    return (img - img.min()) / (img.max() - img.min())

## Loading images
def load_rgb(path):
    img = cv.imread(path)
    img = img[..., ::-1]
    
    img = rescale(img)
    return img

def load_lab(path):
    img = cv.imread(path)
    # img = cv.cvtColor(img, cv.COLOR_BGR2LAB) 
    
    img = img[..., ::-1]
    img = rgb2lab(img)
    
    img = rescale(img)
    return img

def load_gray(path):
    img = cv.imread(path, 0)
    
    img = rescale(img)
    return img


## Plotting stuff
def plot_rgb(img, figsize=None, title=None, xlabel=None, ylabel=None, save=False):
    fig = plt.figure(num=None, figsize=figsize, dpi=100)
    
    # img = rescale(img)
    plt.imshow(img)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if save:
        plt.imsave(f'../data/outs/{title}.png', img)
        
    plt.show()
    plt.close(fig)

def plot_grayscale(img, figsize=None, title=None, xlabel=None, ylabel=None, save=False):
    fig = plt.figure(num=None, figsize=figsize, dpi=100)
    
    # img = rescale(img)
    plt.imshow(img, cmap='gray')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if save:        
        # plt.savefig(f'../imgs/{title}.png', dpi=100, bbox_inches=0) # shit method
        plt.imsave(f'../data/outs/{title}.png', img, cmap='gray')
        
    plt.show()
    plt.close(fig)
    
def plot_covoled(transformed, titles, sup_title, save=False, save_name=None):
    fig,ax = plt.subplots(1, len(transformed),figsize=(30,10))
    fig.suptitle(sup_title)
    
    for i in range(len(transformed)):
        ax[i].imshow(transformed[i])
        ax[i].set_title(titles[i])
        ax[i].axis('off')
        
    if save:
        plt.savefig(f'../data/outs/{save_name}.png', dpi=100, bbox_inches=0) # shit method
        # plt.imsave(f'../data/outs/{title}.png', img, cmap='gray)
        
    # plt.show()
    plt.close(fig)


    
from multiscale import laplacian_stacks, residual, local_energy, robust_transfer, warp_stacks, warp_residual, aggregate_stacks
from dlib_landmarks import face_landmarks
from warp import warp_style
from background import exchange_baground


def main(style_name, input_name):
    
    ## Reading Images in RGB colorspace
    style_rgb = load_rgb(f'../data/inputs/{style_name}.png')
    input_rgb = load_rgb(f'../data/inputs/{input_name}.png')

    R_style, G_style, B_style = cv.split(style_rgb)
    R_input, G_input, B_input = cv.split(input_rgb)
    
    ## Reading Images in Lab colorspaces 
    style_lab = load_lab(f'../data/inputs/{style_name}.png')
    input_lab = load_lab(f'../data/inputs/{input_name}.png')

    L_style, a_style, b_style = cv.split(style_lab)
    L_input, a_input, b_input = cv.split(input_lab)
    
    
    ## Laplacian Stacks
    n = 6

    # L channel
    L_style_stacks = laplacian_stacks(L_style, n)
    L_style_residual = residual(L_style, n)

    L_input_stacks = laplacian_stacks(L_input, n)
    L_input_residual = residual(L_input, n)

    # a channel
    a_style_stacks = laplacian_stacks(a_style, n)
    a_style_residual = residual(a_style, n)

    a_input_stacks = laplacian_stacks(a_input, n)
    a_input_residual = residual(a_input, n)

    # b channel
    b_style_stacks = laplacian_stacks(b_style, n)
    b_style_residual = residual(b_style, n)

    b_input_stacks = laplacian_stacks(b_input, n)
    b_input_residual = residual(b_input, n)
    
    # RGB channel
    rgb_style_stacks = laplacian_stacks(style_rgb, n)
    rgb_style_residual = residual(style_rgb, n)

    rgb_input_stacks = laplacian_stacks(input_rgb, n)
    rgb_input_residual = residual(input_rgb, n)
    
    
    ## Local Energy
    
    # L channel
    L_style_energy = local_energy(L_style_stacks)
    L_input_energy = local_energy(L_input_stacks)

    # a channel
    a_style_energy = local_energy(a_style_stacks)
    a_input_energy = local_energy(a_input_stacks)

    # b channel
    b_style_energy = local_energy(b_style_stacks)
    b_input_energy = local_energy(b_input_stacks)
    
    # RGB channel
    rgb_style_energy = local_energy(rgb_style_stacks)
    rgb_input_energy = local_energy(rgb_input_stacks)
    
    
    ## Reading Images in Grayscale
    style_gray = np.uint8(np.round((load_gray(f'../data/inputs/{style_name}.png') * 255)))
    input_gray = np.uint8(np.round((load_gray(f'../data/inputs/{input_name}.png') * 255)))
    
    
    ## Import Image Detector
    # !wget -nc https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat -O ../data/inputs/shape_predictor_68_face_landmarks.dat
    
    ## Get Image Landmarks
    style_landmarks = np.array(face_landmarks(style_gray),dtype='float32')
    input_landmarks = np.array(face_landmarks(input_gray),dtype='float32')
    
    ## Wrap image and operator
    warp_img, xx, yy, vx, vy = warp_style(style_img=style_rgb, input_img=input_rgb,style_lm=style_landmarks,input_lm=input_landmarks)
    warp_img = rescale(warp_img)
    
    # L channel
    L_warp_style_energy = warp_stacks(L_style_stacks, xx, yy, vx, vy)
    L_warp_style_residual = warp_residual(L_style_residual, xx, yy, vx, vy)

    # a channel
    a_warp_style_energy = warp_stacks(a_style_stacks, xx, yy, vx, vy)
    a_warp_style_residual = warp_residual(a_style_residual, xx, yy, vx, vy)

    # b channel
    b_warp_style_energy = warp_stacks(b_style_stacks, xx, yy, vx, vy)
    b_warp_style_residual = warp_residual(b_style_residual, xx, yy, vx, vy)
    
    ## RGB channels
    rgb_warp_style_energy = np.zeros(rgb_style_stacks.shape)

    rgb_warp_style_energy[..., 0] = warp_stacks(rgb_style_stacks[..., 0], xx, yy, vx, vy)
    rgb_warp_style_energy[..., 1] = warp_stacks(rgb_style_stacks[..., 1], xx, yy, vx, vy)
    rgb_warp_style_energy[..., 2] = warp_stacks(rgb_style_stacks[..., 2], xx, yy, vx, vy)

    rgb_warp_style_residual = np.zeros(rgb_style_residual.shape)

    rgb_warp_style_residual[..., 0] = warp_residual(rgb_style_residual[..., 0], xx, yy, vx, vy)
    rgb_warp_style_residual[..., 1] = warp_residual(rgb_style_residual[..., 1], xx, yy, vx, vy)
    rgb_warp_style_residual[..., 2] = warp_residual(rgb_style_residual[..., 2], xx, yy, vx, vy)
    
    
    ## Robust Transfer
    # L channel
    L_output_stacks = robust_transfer(L_input_stacks, L_warp_style_energy, L_input_energy)

    # a channel
    a_output_stacks = robust_transfer(a_input_stacks, a_warp_style_energy, a_input_energy)

    # b channel
    b_output_stacks = robust_transfer(b_input_stacks, b_warp_style_energy, b_input_energy)
    
    # RGB channel
    rgb_output_stacks = robust_transfer(rgb_input_stacks, rgb_warp_style_energy, rgb_input_energy)
    
    
    ## Aggregate Output stacks
    # L channel
    L_output = aggregate_stacks(L_output_stacks[1:], L_warp_style_residual)

    # a channel
    a_output = aggregate_stacks(a_output_stacks[3:], a_warp_style_residual)

    # b channel
    b_output = aggregate_stacks(b_output_stacks[3:], b_warp_style_residual)
    
    ## LAB output
    lab_output = np.dstack([L_output, a_output, b_output])
    
    ## RGB output
    rgb_output = aggregate_stacks(rgb_output_stacks[1:], rgb_warp_style_residual)
    
    
    ## Output image in RGB colorspace
    output_rgb1 = lab2rgb(lab_output)
    # plot_rgb(rescale(output_rgb1), title=f'RGB Colorspace Output (no changes)')

    output_rgb2 = lab2rgb(lab_output * 255 - 128)
    # plot_rgb(rescale(output_rgb2), title=f'input', save=True)
    
    
    ## Background
    output_img = np.uint8(np.round(rescale(output_rgb2) * 255))
    output_bg = exchange_baground(np.uint8(np.round(input_rgb * 255)), output_img, np.uint8(np.round(style_rgb * 255)))
    
    
    
    ## Plots
    plot_covoled([input_rgb, style_rgb, output_rgb2, rgb_output], ["Input","Style","Lab Transfer","RGB Transfer"],"Style Transfer", save=True, save_name=f'{input_name}_{style_name}')
    plot_covoled([style_rgb, output_rgb2, output_bg], ["Style","Lab Transfer","With Background"],"Adding Background Mask", save=True, save_name=f'{input_name}_{style_name}_bg')
    
    return


if __name__ == "__main__":
    
    for i in range(6):
        style_name = f'example_{i+1}'
        input_name = f'input_{i+1}'
    
        main(style_name, input_name)
