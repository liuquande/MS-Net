import tensorflow as tf
from tensorflow.contrib import slim
from scipy import misc
import os, random
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import logging
from medpy import metric
# https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
# https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/

def _label_decomp(label_vol, num_class):
    """decompose label for softmax classifier
        original labels are batchsize * W * H * 1, with label values 0,1,2,3...
        this function decompse it to one hot, e.g.: 0,0,0,1,0,0 in channel dimension
        numpy version of tf.one_hot
    """
    #label_vol[label_vol == 2] = 1
    _batch_shape = list(label_vol.shape)
    _vol = np.zeros(_batch_shape)
    _vol[label_vol == 0] = 1
    _vol = _vol[..., np.newaxis]
    for i in range(num_class):
        if i == 0:
            continue
        _n_slice = np.zeros(label_vol.shape)
        _n_slice[label_vol == i] = 1
        _vol = np.concatenate((_vol, _n_slice[..., np.newaxis]), axis=3)
    return np.float32(_vol)
    
def parse_fn(data_path):
    '''
    :param image_path: path to a folder of a patient
    :return: normalized entire image with its corresponding label
    In an image, the air region is 0, so we only calculate the mean and std within the brain area
    For any image-level normalization, do it here
    '''
    path = data_path.split(",")
    image_path = path[0]
    label_path = path[1]
    itk_image = sitk.ReadImage(image_path)#os.path.join(image_path, 'T1_unbiased_brain_rigid_to_mni.nii.gz'))
    itk_mask = sitk.ReadImage(label_path)#os.path.join(image_path, 'T1_brain_seg_rigid_to_mni.nii.gz'))
    # itk_image = sitk.ReadImage(os.path.join(image_path, 'T2_FLAIR_unbiased_brain_rigid_to_mni.nii.gz'))

    image = sitk.GetArrayFromImage(itk_image)
    mask = sitk.GetArrayFromImage(itk_mask)
    return image.transpose([2,1,0]), mask.transpose([2,1,0])

def load_test_data(image_path, size_h=256, size_w=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size_h, size_w])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img

def preprocessing(x):
    x = x/127.5 - 1 # -1 ~ 1
    return x

def augmentation(image, aug_img_h, aug_img_w):
    seed = random.randint(0, 2 ** 31 - 1)
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize_images(image, [aug_img_h, aug_img_w])
    image = tf.random_crop(image, ori_image_shape, seed=seed)
    return image

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2

def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 1))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img[:,:,0]

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        print ("Allocating '{:}'".format(log_dir))
        os.makedirs(log_dir)
    return log_dir

def read_labeled_image_list(fid):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    if not os.path.isfile(fid):
        return None
    with open(fid, 'r') as fd:
        _list = fd.readlines()

    my_list = []
    for _item in _list:
        if len(_item) < 5:
            _list.remove(_item)
        my_list.append(_item.split('\n')[0])
    return my_list

def _read_lists(fid):
    """ read train list and test list """
    if not os.path.isfile(fid):
        return None
    with open(fid, 'r') as fd:
        _list = fd.readlines()

    my_list = []
    for _item in _list:
        if len(_item) < 5:
            _list.remove(_item)
        my_list.append(str(_item.split('\n')[0]))
    return my_list


def _eval_dice(gt_y, pred_y, detail=False):

    class_map = {  # a map used for mapping label value to its name, used for output
        "0": "bg",
        "1": "CZ",
        "2": "prostate"
    }

    dice = []

    for cls in xrange(1,3):

        gt = np.zeros(gt_y.shape)
        pred = np.zeros(pred_y.shape)

        gt[gt_y == cls] = 1
        pred[pred_y == cls] = 1

        dice_this = 2*np.sum(gt*pred)/(np.sum(gt)+np.sum(pred))
        dice.append(dice_this)

        if detail is True:
            #print ("class {}, dice is {:2f}".format(class_map[str(cls)], dice_this))
            logging.info("class {}, dice is {:2f}".format(class_map[str(cls)], dice_this))
    return dice

def _connectivity_region_analysis(mask):
    s = [[0,1,0],
         [1,1,1],
         [0,1,0]]
    label_im, nb_labels = ndimage.label(mask)#, structure=s)

    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))

    # plt.imshow(label_im)        
    label_im[label_im != np.argmax(sizes)] = 0
    label_im[label_im == np.argmax(sizes)] = 1

    return label_im

def _eval_average_surface_distances(reference, result, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    return metric.binary.asd(result, reference)