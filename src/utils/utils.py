import numpy as np
from PIL import Image

img_to_array = lambda image: np.asarray(image)
array_to_img = lambda array: Image.fromarray(array)


# Source: https://discuss.pytorch.org/t/is-there-something-like-keras-utils-to-categorical-in-pytorch/5960/2
# Also check: https://github.com/keras-team/keras/blob/master/keras/utils/np_utils.py#L9-L37
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]
