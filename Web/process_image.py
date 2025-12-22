import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import streamlit as st

def process_image(canvas_result):
    """
    process the canvas result image to a tensor of shape (1, 1, 28, 28)
    """
    # numpy (H, W, 4)
    img = canvas_result.image_data.astype(np.uint8)

    # remove alpha
    img = img[:, :, :3]

    # create new image from data object
    pil_img = Image.fromarray(img)

    # convert to grayscale
    gray = TF.rgb_to_grayscale(pil_img, num_output_channels=1)

    # convert to tensor
    tensor = transforms.ToTensor()(gray)


    # find bounding box to heightlight text
    mask = tensor > 0.1
    coords = torch.nonzero(mask[0])
    # if there is no text, return None
    if coords.numel() == 0:
        return None
    
    y_min = coords[:, 0].min()
    y_max = coords[:, 0].max()
    x_min = coords[:, 1].min()
    x_max = coords[:, 1].max()

    text = tensor[:, y_min:y_max+1, x_min:x_max+1] # (1, H, W) since tensor vectorized

    # padding to square
    _, h, w = text.shape
    size = max(h, w)

    padded = torch.zeros((1, size, size))

    # center the text when padding
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    padded[:, y_offset:y_offset+h, x_offset:x_offset+w] = text

    # resize to 28*28
    text_28 = TF.resize(
        padded,
        size = [28, 28],
        interpolation = TF.InterpolationMode.BILINEAR
    )

    # normalize since the model expects input in range [-1, 1]
    text_28 = (text_28 - 0.5) / 0.5

    input = text_28.unsqueeze(0) # (1, 1, 28, 28)

    return input