import os
import time
import numpy as np
import logging
import tensorflow as tf
import re
import data_loader as data_loader
import SimpleITK as sitk
from scipy import ndimage

from lib.layers  import *
from lib.utils import _label_decomp, _eval_dice, _read_lists, _connectivity_region_analysis, _eval_average_surface_distances

class_map = {  # a map used for mapping label value to its name, used for output
    "0": "bg",
    "1": "prostate"
}

decomp_feature = { # configuration for decoding tf_record file
            'dsize_dim0': tf.FixedLenFeature([], tf.int64),
            'dsize_dim1': tf.FixedLenFeature([], tf.int64),
            'dsize_dim2': tf.FixedLenFeature([], tf.int64),
            'lsize_dim0': tf.FixedLenFeature([], tf.int64),
            'lsize_dim1': tf.FixedLenFeature([], tf.int64),
            'lsize_dim2': tf.FixedLenFeature([], tf.int64),
            'data_vol': tf.FixedLenFeature([], tf.string),
            'label_vol': tf.FixedLenFeature([], tf.string)}

class Tester(object):

    def __init__(self, args, sess, network = None):
        self.args = args
        self.sess = sess
        self.net = network
        self.save_freq = args.save_freq # intervals between saving a checkpoint and decaying learning rate
        self.display_freq = args.display_freq
        self.n_class = args.n_class

        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.iteration = args.iteration
        self.lr = args.lr
        self.lr_decay = args.lr_decay 
        self.restored_model = args.restored_model
        self.volume_size = args.volume_size
        self.label_size = args.label_size
        self.dropout = args.dropout
        self.opt_kwargs = args.opt_kwargs
        self.cost_kwargs = args.cost_kwargs

        self.source_1_train_list = args.source_1_train_list
        self.source_1_val_list = args.source_1_val_list
        self.source_2_train_list = args.source_2_train_list
        self.source_2_val_list = args.source_2_val_list
        self.source_3_train_list = args.source_3_train_list
        self.source_3_val_list = args.source_3_val_list

        with open(args.test_1_list, 'r') as fp:
            rows = fp.readlines()
        self.test_1_list  = [row[:-1] if row[-1] == '\n' else row for row in rows]
        with open(args.test_2_list, 'r') as fp:
            rows = fp.readlines()
        self.test_2_list  = [row[:-1] if row[-1] == '\n' else row for row in rows]
        with open(args.test_3_list, 'r') as fp:
            rows = fp.readlines()
        self.test_3_list  = [row[:-1] if row[-1] == '\n' else row for row in rows]

        self.log_dir = args.log_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir

    def _get_optimizers(self):

        self.global_step = tf.Variable(0, name = "global_step")

        self.learning_rate = self.lr
        self.learning_rate_node = tf.Variable(self.learning_rate, name = "learning_rate")

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer_teacher_segmentation = tf.train.AdamOptimizer(learning_rate = self.learning_rate_node).minimize(
                                                                           loss = self.net.source_1_teacher_total_loss + self.net.source_2_teacher_total_loss + self.net.source_3_teacher_total_loss, \
                                                                           var_list = self.net.teacher_variables, \
                                                                           global_step = self.global_step, \
                                                                           colocate_gradients_with_ops=True)
            optimizer_student_segmentation = tf.train.AdamOptimizer(learning_rate = self.learning_rate_node).minimize(
                                                                           loss = self.net.source_student_total_loss, \
                                                                           var_list = self.net.student_variables, 
                                                                           colocate_gradients_with_ops=True)

        return optimizer_teacher_segmentation, optimizer_student_segmentation



    def restore_model(self, restored_model):

        if restored_model is not None:
            saver = tf.train.Saver()
            saver.restore(self.sess, restored_model)
            logging.info("Fine tune the segmenter, model restored from %s" % restored_model)
        else:
            logging.info("Training the segmenter model from scratch")


    def test(self):

        # self.optimizer_teacher, self.optimizer_student = self._get_optimizers()
        # self._init_tfboard()

        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = False

        if self.restored_model is not None:
            self.restore_model(self.restored_model)
        else:
            start_epoch = 0
            init_glb = tf.global_variables_initializer()
            init_loc = tf.variables_initializer(tf.local_variables())
            self.sess.run([init_glb, init_loc])

        train_summary_writer = tf.summary.FileWriter(self.log_dir + "/train_log", graph=self.sess.graph)
        val_summary_writer = tf.summary.FileWriter(self.log_dir + "/val_log", graph=self.sess.graph)

        self.source_1_train = data_loader._load_data(datalist=self.source_1_train_list, patch_size=self.volume_size, batch_size=self.batch_size, num_class=self.n_class)
        self.source_1_val = data_loader._load_data(datalist=self.source_1_train_list, patch_size=self.volume_size, batch_size=self.batch_size, num_class=self.n_class)
        self.source_2_train = data_loader._load_data(datalist=self.source_2_train_list, patch_size=self.volume_size, batch_size=self.batch_size, num_class=self.n_class)
        self.source_2_val = data_loader._load_data(datalist=self.source_2_train_list, patch_size=self.volume_size, batch_size=self.batch_size, num_class=self.n_class)
        self.source_3_train = data_loader._load_data(datalist=self.source_3_train_list, patch_size=self.volume_size, batch_size=self.batch_size, num_class=self.n_class)
        self.source_3_val = data_loader._load_data(datalist=self.source_3_train_list, patch_size=self.volume_size, batch_size=self.batch_size, num_class=self.n_class)

        # threads = tf.train.start_queue_runners(sess = self.sess, coord = coord)
        best_dice = 0

        with open(os.path.join(self.log_dir, 'eva.txt'), 'a') as f:
            dice_teacher_1, dice_arr_teacher_1, dice_student_1, dice_arr_student_1, asd_teacher_1, asd_student_1 = self.test_single(self.test_1_list, 1)
            print >> f, "step ", self.restored_model
            print >> f, "    source 1 dice student is: ", dice_student_1, dice_arr_student_1
            dice_teacher_2, dice_arr_teacher_2, dice_student_2, dice_arr_student_2, asd_teacher_2, asd_student_2 = self.test_single(self.test_2_list, 2)
            print >> f, "    source 2 dice student is: ", dice_student_2, dice_arr_student_2  
            dice_teacher_3, dice_arr_teacher_3, dice_student_3, dice_arr_student_3, asd_teacher_3, asd_student_3 = self.test_single(self.test_3_list, 3)
            print >> f, "    source 3 dice student is: ", dice_student_3, dice_arr_student_3
            print >> f, "    source ave dice student is: ", (dice_student_1 + dice_student_2 + dice_student_3) / 3

        return 0

    def test_single(self, test_list, domain, test_model=None):

        if domain == 1:
            student_pred_mask = self.net.source_1_student_pred_compact
            student_pred_map = self.net.source_1_student_softmaxpred
            data_place = self.net.source_1

        elif domain == 2:
            student_pred_mask = self.net.source_2_student_pred_compact
            student_pred_map = self.net.source_2_student_softmaxpred
            data_place = self.net.source_2

        elif domain == 3:
            student_pred_mask = self.net.source_3_student_pred_compact
            student_pred_map = self.net.source_3_student_softmaxpred
            data_place = self.net.source_3

        if test_model is not None:
            self.restore_model(test_model)

        dice_teacher = []
        dice_student = []
        asd_teacher = []
        asd_student = []
        
        start = time.time()
        # cost_time = 0
        for fid, filename in enumerate(test_list):
            image, mask = data_loader.parse_fn(filename)
            #print image.shape, mask.shape
            #image = np.flip(np.flip(_npz_dict['arr_0'], axis=0), axis=1)#_npz_dict['arr_0']
            # print image.shape
            #mask = np.flip(np.flip( _npz_dict['arr_1'], axis=0), axis=1)#_npz_dict['arr_1']#
            pred_teacher_y = np.zeros(mask.shape)
            pred_student_y = np.zeros(mask.shape)

            frame_list = [kk for kk in range(1, image.shape[2] - 1)]
            for ii in xrange(int(np.floor(image.shape[2] // self.net.batch_size))):
                vol = np.zeros([self.net.batch_size, self.net.volume_size[0], self.net.volume_size[1], self.net.volume_size[2]])

                for idx, jj in enumerate(frame_list[ii * self.net.batch_size: (ii + 1) * self.net.batch_size]):
                    vol[idx, ...] = image[..., jj - 1: jj + 2].copy()
                pred_student = self.sess.run((student_pred_mask), feed_dict={data_place: vol,
                                                                        self.net.keep_prob: self.dropout,
                                                                        self.net.training_mode_encoder: False,
                                                                        self.net.training_mode_decoder: False})
                # cost_time += 
                for idx, jj in enumerate(frame_list[ii * self.net.batch_size: (ii + 1) * self.net.batch_size]):
                    # pred_teacher_y[..., jj] = pred_teacher[idx, ...].copy()
                    pred_student_y[..., jj] = pred_student[idx, ...].copy()

            processed_pred_student_y = _connectivity_region_analysis(pred_student_y)

            dice_subject_student = _eval_dice(mask, processed_pred_student_y)
            asd_subject_student = _eval_average_surface_distances(mask, processed_pred_student_y)

            dice_student.append(dice_subject_student)
            asd_student.append(asd_subject_student)

            self._save_nii_prediction(mask, processed_pred_student_y, pred_student_y, self.result_dir, out_bname=str(domain) + '_' + filename[-26:-20])

        dice_avg_student = np.mean(dice_student, axis=0).tolist()[0]
        asd_avg_student = np.mean(asd_student)
        

        logging.info("dice_avg_student %.4f" % (dice_avg_student))
        logging.info("asd_avg_student %.4f" % (asd_avg_student))

        return 0,0,dice_avg_student, dice_student,0,0#dice_avg_teacher, dice_teacher, dice_avg_student, dice_student, asd_avg_teacher, asd_avg_student

    def _save_nii_prediction(self, gth, comp_pred, pre_pred, out_folder, out_bname):
        sitk.WriteImage(sitk.GetImageFromArray(gth), out_folder + out_bname + 'gth.nii.gz') 
        sitk.WriteImage(sitk.GetImageFromArray(pre_pred), out_folder + out_bname + 'premask.nii.gz') 
        sitk.WriteImage(sitk.GetImageFromArray(comp_pred), out_folder + out_bname + 'mask.nii.gz') 
