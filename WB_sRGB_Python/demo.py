import gradio as gr
import cv2
import numpy as np
from classes import WBsRGB as wb_srgb

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

# Initialize the model with default parameters
gamut_mapping_default = 2
upgraded_model_default = 0
wbModel = wb_srgb.WBsRGB(gamut_mapping=gamut_mapping_default, upgraded=upgraded_model_default)

def white_balance(input_image):
    I = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    outImg = wbModel.correctImage(I)  # White balance it
    result_image = (outImg * 255).astype(np.uint8)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR format for display
    return result_image

# Create Gradio interface
interface = gr.Interface(
    fn=white_balance,
    inputs=gr.Image(type="numpy", label="Input Image"),
    outputs=gr.Image(type="numpy", label="Output Image"),
    title="White Balance Correction",
    description="Upload an image to correct its white balance."
)

# Launch the interface
interface.launch()
