from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

from misc_functions import get_example_params, save_class_activation_images


target_example = 1
(original_image, prep_img, target_class, file_name_to_export, net) =\
    get_example_params(target_example, 'resnet50')

finalconv_name = 'layer4'

net.eval()

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy()) # fc.weight, -1: fc.bias

logits = net(prep_img)

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 224 * 224
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape # bz = 1
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(np.array(Image.fromarray(cam_img).resize(size_upsample, Image.ANTIALIAS)))
    return output_cam

# generate class activation mapping for the top1 prediction
cam = returnCAM(features_blobs[0], weight_softmax, [target_class])[0]

save_class_activation_images(original_image, cam, file_name_to_export)
