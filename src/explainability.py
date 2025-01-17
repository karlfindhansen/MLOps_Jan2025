import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def compute_gradcam(model, input_tensor, target_class, layer_name="layer4"):
    """
    Computes Grad-CAM for explainability purposes.
    """
    gradcam_output = None
    gradients = None

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    def forward_hook(module, input, output):
        nonlocal gradcam_output
        gradcam_output = output

    # Register hooks
    target_layer = dict(model.named_modules())[layer_name]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    model.eval()
    output = model(input_tensor)

    # Backward pass for the target class
    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()

    # Compute Grad-CAM
    weights = gradients.mean(dim=[2, 3], keepdim=True)
    gradcam = F.relu((weights * gradcam_output).sum(dim=1, keepdim=True))
    gradcam = F.interpolate(gradcam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
    gradcam = gradcam.squeeze().detach().cpu().numpy()

    return gradcam


def overlay_gradcam_on_image(image, gradcam):
    """
    Overlays the Grad-CAM heatmap on the original image.
    """
    gradcam_rescaled = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())
    gradcam_rescaled = np.uint8(255 * gradcam_rescaled)

    heatmap = Image.fromarray(gradcam_rescaled).resize(image.size, resample=Image.BICUBIC)
    heatmap = np.asarray(heatmap)
    heatmap = plt.cm.jet(heatmap)[:, :, :3]  # Use jet colormap

    image_array = np.array(image)
    overlay = heatmap * 0.4 + image_array / 255.0 * 0.6
    overlay = np.uint8(overlay * 255)

    return Image.fromarray(overlay)
