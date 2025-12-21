import streamlit as st
from streamlit_drawable_canvas import st_canvas

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

canvas_result = st_canvas(
    stroke_color='#FFFFFF',
    background_color= 'rgb(0, 0, 0)',
    update_streamlit=False,
    height=420, # 28 * 15
    width=420
)

if st.button('Predict'):
    if canvas_result.image_data is not None:
        # numpy (H, W, 4)
        img = canvas_result.image_data.astype(np.uint8)

        # remove alpha
        img = img[:, :, :3]

        # create new image from data object
        pil_img = Image.fromarray(img)

        # convert to grayscale
        gray = TF.rgb_to_grascale(pil_img, num_output_channels=1)

        # convert to tensor
        to_tensor = transforms.ToTensor()
        tensor = to_tensor(gray)

        # binarization to set pixel values to 0 or 1
        # avoid gray gradient appears at the edges
        binary = (tensor > 0.05).float()

        