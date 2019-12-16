import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import os
import scipy.ndimage
import matplotlib.pyplot as plt


def _load_data(datalist, patch_size, batch_size, num_class):

    '''
    :param datalist:
    :param num_patches:
    :param patch_size:
    :param batch_size:
    :param num_class:
    :return: a tuple with (image, label) for a mini-batch
    image.shape=(batch_size,patch_size[0],patch_size[1],patch_size[2],1), dtype=tf.float32
    label.shape=(batch_size,patch_size[0],patch_size[1],patch_size[2],num_class, dtype=tf.int16)
    '''

    with open(datalist, 'r') as fp:
        rows = fp.readlines()
    image_list = [row[:-1] for row in rows]
    # image_list = [image_list[0]] * 2 ## UNCOMMENT THIS TO SUPPLY ONLY ONE IMAGE, for debugging

    buffer_size = 150  # IMGS2BATCH * num_patches  # shuffle patches from 50 different images
    num_parallel_calls = 5          # num threads
    
    get_patches_fn = lambda filename: tf.py_func(extract_patch, [filename, patch_size, num_class], [tf.float32, tf.float32])

    dataset =  (tf.data.Dataset.from_tensor_slices(image_list)
               .shuffle(buffer_size=buffer_size) # 5. shuffle the data # 1. all filenames go into the buffer (for good shuffling)
               .repeat() # repeat indefinitely (train.py will count the epochs)
               .map(get_patches_fn, num_parallel_calls=num_parallel_calls) # 3. return the patches
               .apply(tf.contrib.data.unbatch()) # 4. unbatch the patches so ones from same image are separate
               .batch(batch_size)  # 6. set batch size
               #.repeat() # repeat indefinitely (train.py will count the epochs)
               .prefetch(50) # 7. always have one batch ready to go
                )

    iterator = dataset.make_one_shot_iterator()
    samples = iterator.get_next()

    return samples

def extract_patch(filename, patch_size, num_class, num_patches=1, augmentation=False):
    """Extracts a patch of given resolution and size at a specific location."""

    image, mask = parse_fn(filename) # get the image and its mask
    image_patches = []
    mask_patches = []
    num_patches_now = 0
    patch_wise_image = []
    patch_wise_mask = []

    while num_patches_now < num_patches:
        z = random_patch_center_z(mask, patch_size=patch_size) # define the centre of current patch
        # x, y, z = [kk//2 for kk in mask.shape] # only select the centroid location
        image_patch = image[:, :, z-1:z+2]
        mask_patch  =  mask[:, :, z]
        
        image_patches.append(image_patch)
        mask_patches.append(mask_patch)
        num_patches_now += 1
    image_patches = np.stack(image_patches) # make into 4D (batch_size, patch_size[0], patch_size[1], patch_size[2])
    mask_patches = np.stack(mask_patches) # make into 4D (batch_size, patch_size[0], patch_size[1], patch_size[2])

    mask_patches = _label_decomp(mask_patches, num_cls=num_class) # make into 5D (batch_size, patch_size[0], patch_size[1], patch_size[2], num_classes)
    #print image_patches.shape
    return image_patches.astype(np.float32), mask_patches.astype(np.float32)


def random_patch_center_z(mask, patch_size):
    # bounded within the brain mask region
    limX, limY, limZ = np.where(mask>0)
    if (np.min(limZ) + patch_size[2] // 2 + 1) < (np.max(limZ) - patch_size[2] // 2):
        z = np.random.randint(low = np.min(limZ) + patch_size[2] // 2 + 1, high = np.max(limZ) - patch_size[2] // 2)
    else:
        z = np.random.randint(low = patchsize[2]//2, high = mask.shape[2] - patchsize[2]//2)

    limX, limY, limZ = np.where(mask>0)

    z = np.random.randint(low = max(1, np.min(limZ)), high = min(np.max(limZ), mask.shape[2] - 2))

    return z


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
    #itk_image = zoom2shape(image_path, [512,512])#os.path.join(image_path, 'T1_unbiased_brain_rigid_to_mni.nii.gz'))
    #itk_mask = zoom2shape(label_path, [512,512], label=True)#os.path.join(image_path, 'T1_brain_seg_rigid_to_mni.nii.gz'))
    itk_image = sitk.ReadImage(image_path)#os.path.join(image_path, 'T1_unbiased_brain_rigid_to_mni.nii.gz'))
    itk_mask = sitk.ReadImage(label_path)#os.path.join(image_path, 'T1_brain_seg_rigid_to_mni.nii.gz'))
    # itk_image = sitk.ReadImage(os.path.join(image_path, 'T2_FLAIR_unbiased_brain_rigid_to_mni.nii.gz'))

    image = sitk.GetArrayFromImage(itk_image)
    mask = sitk.GetArrayFromImage(itk_mask)
    #image[image >= 1000] = 1000
    binary_mask = np.ones(mask.shape)
    mean = np.sum(image * binary_mask) / np.sum(binary_mask)
    std = np.sqrt(np.sum(np.square(image - mean) * binary_mask) / np.sum(binary_mask))
    image = (image - mean) / std  # normalize per image, using statistics within the brain, but apply to whole image

    # image = (image - mean) / std  # normalize per image, using statistics within the brain, but apply to whole image

    #image = (image - np.min(image)) / ((np.max(image) - np.min(image)) / 2.0 ) - 1
    #mask = sitk.GetArrayFromImage(itk_mask)
    #image[mask==0] = 0
    mask[mask==2] = 1
   # binary_mask = np.ones(mask.shape)
    #binary_mask[np.where(mask > 0)] = 1.0
    # mean = np.sum(image * binary_mask) / np.sum(binary_mask)
    # std = np.sqrt(np.sum(np.square(image - mean) * binary_mask) / np.sum(binary_mask))

    # image = (image - mean) / std  # normalize per image, using statistics within the brain, but apply to whole image

    return image.transpose([1,2,0]), mask.transpose([1,2,0]) # transpose the orientation of the


def _label_decomp(label_vol, num_cls):
    """
    decompose label for softmax classifier
    original labels are batchsize * W * H * 1, with label values 0,1,2,3...
    this function decompse it to one hot, e.g.: 0,0,0,1,0,0 in channel dimension
    numpy version of tf.one_hot
    """
    one_hot = []
    for i in xrange(num_cls):
        _vol = np.zeros(label_vol.shape)
        _vol[label_vol == i] = 1
        one_hot.append(_vol)

    return np.stack(one_hot, axis=-1)

def zoom2shape(img_nii, new_shape, label=False):
    sitk_img = sitk.ReadImage(img_nii)
    img = sitk.GetArrayFromImage(sitk_img)  # indexes are z,y,x (notice the ordering)
    if (label == True):
        img[img==2] = 1
    origin = sitk_img.GetOrigin()           # x,y,z  Origin in world coordinates (mm)
    spacing = sitk_img.GetSpacing()         # spacing of voxels in world coor. (mm)
    direction = sitk_img.GetDirection()     

    old_shape_z, old_shape_y, old_shape_x = img.shape
    old_shape = [old_shape_x, old_shape_y, old_shape_z]
    new_shape = np.array([new_shape[0], new_shape[1], old_shape_z])
    new_spacing = np.array(spacing) / (new_shape * 1.0 / np.array(old_shape))

    sitk_img = sitk.GetImageFromArray(img)
    sitk_img.SetOrigin(origin)
    sitk_img.SetSpacing(spacing)
    sitk_img.SetDirection(direction)

    resample = sitk.ResampleImageFilter()

    if (label == True):
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetOutputDirection(sitk_img.GetDirection())
    resample.SetOutputOrigin(sitk_img.GetOrigin())
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_shape)
    newimage = resample.Execute(sitk_img)
    return newimage

if __name__ == "__main__":

    # # Here are code for debugging the functions in this file
    with tf.Session() as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        data = _load_data(datalist='/research/pheng4/qdliu/Dou_Project/3D-UNet/data/mr_train_list', patch_size=[96, 96, 96], batch_size=2, num_class=5)

        for _ in xrange(5):

            image, label = sess.run(data)
            plt.subplot(151)
            plt.imshow(image[1,:,:,32,0], cmap='gray')
            plt.subplot(152)
            plt.imshow(label[1,:,:,32,0], cmap='gray')
            plt.subplot(153)
            plt.imshow(label[1,:,:,32,1], cmap='gray')
            plt.subplot(154)
            plt.imshow(label[1,:,:,32,2], cmap='gray')
            plt.subplot(155)
            plt.imshow(label[1,:,:,32,3], cmap='gray')
            plt.show()


    ## plain debug without CPU Qeuen backend
    # with open('/vol/biomedic2/qdou20/projects/test_specific/data/train_list.txt', 'r') as fp:
    #     rows = fp.readlines()
    # image_list = [row[:-1] for row in rows]

    # for filename in image_list:
    #     image, label = extract_patch(filename, patch_size=[96,96,96], num_class=4, num_patches=10, augmentation=True)
    #     plt.subplot(151)
    #     plt.imshow(image[1, :, :, 32, 0], cmap='gray')
    #     plt.subplot(152)
    #     plt.imshow(label[1, :, :, 32, 0], cmap='gray')
    #     plt.subplot(153)
    #     plt.imshow(label[1, :, :, 32, 1], cmap='gray')
    #     plt.subplot(154)
    #     plt.imshow(label[1, :, :, 32, 2], cmap='gray')
    #     plt.subplot(155)
    #     plt.imshow(label[1, :, :, 32, 3], cmap='gray')
    #     plt.show()