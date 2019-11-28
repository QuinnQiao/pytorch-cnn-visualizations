from PIL import Image
from torchvision import models
import numpy as np
import torch

from misc_functions import get_example_params, save_class_activation_images


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def forward_pass(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        # only support target_layer == layer4
        conv_output = x  # Save the convolution output on that layer

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        return conv_output, x

    def get_weights(self, index):
        return self.model.fc.weight.data[index, :]


class Cam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        weight_softmax = self.extractor.get_weights(target_class)
        weight_softmax = weight_softmax.data.numpy() # C
        conv_output = conv_output.data.squeeze(0).numpy() # C*H*W
        c, h, w = conv_output.shape
        cam = weight_softmax.dot(conv_output.reshape(c, h*w)) # (C)x(C*HW) -> HW
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))#/255
        return cam


if __name__ == '__main__':
    # cat_dog
    target_example = 1
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example, 'resnet50')
    cam = Cam(pretrained_model, target_layer='layer4')
    # Generate cam mask
    mask = cam.generate_cam(prep_img, target_class)
    # Save mask
    save_class_activation_images(original_image, mask, file_name_to_export)
    print('Cam completed')
