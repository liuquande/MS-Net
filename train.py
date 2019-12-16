import os
import time
import numpy as np
import logging
import tensorflow as tf
import re
import data_loader
import SimpleITK as sitk
from scipy import ndimage

from lib.layers  import *
from lib.utils import _label_decomp, _eval_dice, _read_lists, _connectivity_region_analysis

class_map = {  # a map used for mapping label value to its name, used for output
    "0": "bg",
    "1": "prostate"
}

class Trainer(object):

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

        # Here is for batch normalization, moving mean and moving variance work properly
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


    def train(self):

        self.optimizer_teacher, self.optimizer_student = self._get_optimizers()
        self._init_tfboard()

        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = False

        if self.restored_model is not None:
            self.restore_model(self.restored_model)
            start_epoch = self.global_step.eval() / self.iteration
            #self.sess.run(tf.assign(self.learning_rate_node, self.learning_rate))
        else:
            start_epoch = 0
            init_glb = tf.global_variables_initializer()
            init_loc = tf.variables_initializer(tf.local_variables())
            self.sess.run([init_glb, init_loc])


        train_summary_writer = tf.summary.FileWriter(self.log_dir + "/train_log", graph=self.sess.graph)
        val_summary_writer = tf.summary.FileWriter(self.log_dir + "/val_log", graph=self.sess.graph)

        self.source_1_train = data_loader._load_data(datalist=self.source_1_train_list, patch_size=self.volume_size, batch_size=self.batch_size, num_class=self.n_class)
        self.source_1_val = data_loader._load_data(datalist=self.source_1_val_list, patch_size=self.volume_size, batch_size=self.batch_size, num_class=self.n_class)
        self.source_2_train = data_loader._load_data(datalist=self.source_2_train_list, patch_size=self.volume_size, batch_size=self.batch_size, num_class=self.n_class)
        self.source_2_val = data_loader._load_data(datalist=self.source_2_val_list, patch_size=self.volume_size, batch_size=self.batch_size, num_class=self.n_class)
        self.source_3_train = data_loader._load_data(datalist=self.source_3_train_list, patch_size=self.volume_size, batch_size=self.batch_size, num_class=self.n_class)
        self.source_3_val = data_loader._load_data(datalist=self.source_3_val_list, patch_size=self.volume_size, batch_size=self.batch_size, num_class=self.n_class)

        best_dice = 0

        for epoch in xrange(start_epoch, self.epoch):
            for step in xrange((epoch*self.iteration), ((epoch+1)*self.iteration)):
                start = time.time()


                source_1_train_batch_x, source_1_train_batch_y = self.sess.run(self.source_1_train)
                source_2_train_batch_x, source_2_train_batch_y = self.sess.run(self.source_2_train)
                source_3_train_batch_x, source_3_train_batch_y = self.sess.run(self.source_3_train)
                #logging.info("time: %4.4f" % (time.time() - start))
                _, teacher_1_loss, teacher_2_loss, teacher_3_loss, lr= \
                    self.sess.run((self.optimizer_teacher, self.net.source_1_teacher_total_loss, self.net.source_2_teacher_total_loss, self.net.source_3_teacher_total_loss, self.learning_rate_node),\
                                                                    feed_dict={self.net.source_1: source_1_train_batch_x,
                                                                               self.net.source_1_y: source_1_train_batch_y,
                                                                               self.net.source_2: source_2_train_batch_x,
                                                                               self.net.source_2_y: source_2_train_batch_y,
                                                                               self.net.source_3: source_3_train_batch_x,
                                                                               self.net.source_3_y: source_3_train_batch_y,
                                                                               self.net.training_mode_encoder: True,
                                                                               self.net.training_mode_decoder: True,
                                                                               self.net.keep_prob: 1,
                                                                               })
                # warm up teacher model first
                if step>=500:
                    _, seg_student_total_loss, student_1_dice_loss, student_2_dice_loss, student_3_dice_loss, lr= \
                        self.sess.run((self.optimizer_student, self.net.source_student_total_loss, self.net.source_1_student_output_loss, self.net.source_2_student_output_loss, \
                                       self.net.source_3_student_output_loss, self.learning_rate_node),\
                                                                        feed_dict={self.net.source_1: source_1_train_batch_x,
                                                                                   self.net.source_1_y: source_1_train_batch_y,
                                                                                   self.net.source_2: source_2_train_batch_x,
                                                                                   self.net.source_2_y: source_2_train_batch_y,
                                                                                   self.net.source_3: source_3_train_batch_x,
                                                                                   self.net.source_3_y: source_3_train_batch_y,
                                                                                   self.net.training_mode_encoder: True,
                                                                                   self.net.training_mode_decoder: True,
                                                                                   self.net.keep_prob: 1,
                                                                                   })
                    logging.info("Epoch: [%2d] [%6d/%6d] time: %4.4f)lr:%.8f" % (epoch, step, self.iteration, (time.time()-start), lr))
                    logging.info("s_loss_total: %.8f, s_dice_loss_1: %.8f, s_dice_loss_2: %.8f, s_dice_loss_3: %.8f" % (seg_student_total_loss, student_1_dice_loss, student_2_dice_loss, student_3_dice_loss))
                else:
                    seg_student_total_loss = 0
                    student_1_loss = 0
                    student_2_loss = 0
                    student_3_loss = 0
                    logging.info("Epoch: [%2d] [%6d/%6d] time: %4.4f)lr:%.8f" % (epoch, step, self.iteration, (time.time()-start), lr))

                logging.info("t_dice_loss_total: %.8f, t_1_dice_loss: %.8f, t_2_dice_loss: %.8f, t_3_dice_loss: %.8f" \
                                                        % (teacher_1_loss+teacher_2_loss+teacher_3_loss, teacher_1_loss, teacher_2_loss, teacher_3_loss))

                if step % self.display_freq  == 0:
                    self.minibatch_stats_segmenter_source(train_summary_writer, step, source_1_train_batch_x, source_1_train_batch_y, source_2_train_batch_x, source_2_train_batch_y, source_3_train_batch_x, source_3_train_batch_y)

                if step % (self.display_freq) == 0:
                    source_1_val_batch_x, source_1_val_batch_y = self.sess.run(self.source_1_val)
                    source_2_val_batch_x, source_2_val_batch_y = self.sess.run(self.source_2_val)
                    source_3_val_batch_x, source_3_val_batch_y = self.sess.run(self.source_3_val)
                    self.minibatch_stats_segmenter_source(val_summary_writer, step, source_1_val_batch_x, source_1_val_batch_y, source_2_val_batch_x, source_2_val_batch_y, source_3_val_batch_x, source_3_val_batch_y)

                if step % (self.save_freq) == 0:
                    saver = tf.train.Saver()
                    saved_model_name = "model.cpkt"
                    save_path = saver.save(self.sess, os.path.join(self.checkpoint_dir, saved_model_name), global_step = self.global_step.eval())
                    logging.info("Model saved as step %d, save path is %s" % (self.global_step.eval(), save_path))

                if (step % self.iteration == 0) and (step != 0):
                    _pre_lr = self.sess.run(self.learning_rate_node)
                    self.sess.run(tf.assign(self.learning_rate_node, _pre_lr * 0.95))


                    with open(os.path.join(self.log_dir, 'eva.txt'), 'a') as f:
                        dice_teacher_1, dice_arr_teacher_1, dice_student_1, dice_arr_student_1 = self.test(self.test_1_list, 1)
                        print >> f, "step ", self.global_step.eval()
                        print >> f, "    source 1 dice teacher is: ", dice_teacher_1, dice_arr_teacher_1
                        print >> f, "    source 1 dice student is: ", dice_student_1, dice_arr_student_1
                        dice_teacher_2, dice_arr_teacher_2, dice_student_2, dice_arr_student_2 = self.test(self.test_2_list, 2)
                        print >> f, "    source 2 dice teacher is: ", dice_teacher_2, dice_arr_teacher_2
                        print >> f, "    source 2 dice student is: ", dice_student_2, dice_arr_student_2                        
                        dice_teacher_3, dice_arr_teacher_3, dice_student_3, dice_arr_student_3 = self.test(self.test_3_list, 3)
                        print >> f, "    source 3 dice teacher is: ", dice_teacher_3, dice_arr_teacher_3
                        print >> f, "    source 3 dice student is: ", dice_student_3, dice_arr_student_3

                        print >> f, "    source ave dice teacher is: ", (dice_teacher_1 + dice_teacher_2 + dice_teacher_3) / 3
                        print >> f, "    source ave dice student is: ", (dice_student_1 + dice_student_2 + dice_student_3) / 3

        logging.info("Modeling training Finished!")

        return 0

    def test(self, test_list, domain, test_model=None):

        if domain == 1:
            student_pred_mask = self.net.source_1_student_pred_compact
            teacher_pred_mask = self.net.source_1_teacher_pred_compact
            data_place = self.net.source_1

        elif domain == 2:
            student_pred_mask = self.net.source_2_student_pred_compact
            teacher_pred_mask = self.net.source_2_teacher_pred_compact
            data_place = self.net.source_2

        elif domain == 3:
            student_pred_mask = self.net.source_3_student_pred_compact
            teacher_pred_mask = self.net.source_3_teacher_pred_compact
            data_place = self.net.source_3

        if test_model is not None:
            self.restore_model(test_model)

        dice_teacher = []
        dice_student = []
        for fid, filename in enumerate(test_list):
            image, mask = data_loader.parse_fn(filename)
            #print image.shape, mask.shape
            #image = np.flip(np.flip(_npz_dict['arr_0'], axis=0), axis=1)#_npz_dict['arr_0']
            # print image.shape
            #mask = np.flip(np.flip( _npz_dict['arr_1'], axis=0), axis=1)#_npz_dict['arr_1']#
            pred_teacher_y = np.zeros(mask.shape)
            pred_student_y = np.zeros(mask.shape)

            frame_list = [kk for kk in range(1, image.shape[2] - 1)]
            #np.random.shuffle(frame_list)
            for ii in xrange(int(np.floor(image.shape[2] // self.net.batch_size))):
                vol = np.zeros([self.net.batch_size, self.net.volume_size[0], self.net.volume_size[1], self.net.volume_size[2]])

                for idx, jj in enumerate(frame_list[ii * self.net.batch_size: (ii + 1) * self.net.batch_size]):
                    vol[idx, ...] = image[..., jj - 1: jj + 2].copy()
                pred_teacher = self.sess.run((teacher_pred_mask), feed_dict={data_place: vol,
                                                                        self.net.keep_prob: self.dropout,
                                                                        self.net.training_mode_encoder: False,
                                                                        self.net.training_mode_decoder: False})
                pred_student = self.sess.run((student_pred_mask), feed_dict={data_place: vol,
                                                                        self.net.keep_prob: self.dropout,
                                                                        self.net.training_mode_encoder: False,
                                                                        self.net.training_mode_decoder: False})
                for idx, jj in enumerate(frame_list[ii * self.net.batch_size: (ii + 1) * self.net.batch_size]):
                    pred_teacher_y[..., jj] = pred_teacher[idx, ...].copy()
                    pred_student_y[..., jj] = pred_student[idx, ...].copy()

            processed_pred_teacher_y = _connectivity_region_analysis(pred_teacher_y)
            processed_pred_student_y = _connectivity_region_analysis(pred_student_y)
            dice_subject_teacher = _eval_dice(mask, processed_pred_teacher_y)
            dice_subject_student = _eval_dice(mask, processed_pred_student_y)

            #logging.info("dice_avg %s : %.4f" % (filename[-26:], dice_subject[0]))

            dice_teacher.append(dice_subject_teacher)
            dice_student.append(dice_subject_student)

            #self._save_nii_prediction(mask, pred_y, self.result_dir, out_bname=filename[-26:-20])
        dice_avg_teacher = np.mean(dice_teacher, axis=0).tolist()[0]
        dice_avg_student = np.mean(dice_student, axis=0).tolist()[0]
        #print np.mean(dice_teacher, axis=0).tolist()
        #print np.mean(dice_student, axis=0).tolist()
        logging.info("dice_avg_teacher %.4f" % (dice_avg_teacher))
        logging.info("dice_avg_student %.4f" % (dice_avg_student))

        return dice_avg_teacher, dice_teacher, dice_avg_student, dice_student

    def _save_nii_prediction(self, gth, comp_pred, out_folder, out_bname):
        sitk.WriteImage(sitk.GetImageFromArray(gth), out_folder + out_bname + 'gth.nii.gz') 
        sitk.WriteImage(sitk.GetImageFromArray(comp_pred), out_folder + out_bname + 'mask.nii.gz') 

    def _init_tfboard(self):
        """
        initialization and tensorboard summary
        """

        scalar_summaries = []
        train_images = []
        val_images = []

        scalar_summaries.append(tf.summary.scalar('learning_rate', self.learning_rate_node))
        scalar_summaries.append(tf.summary.scalar("source_1_teacher_output_loss", self.net.source_1_teacher_total_loss))
        scalar_summaries.append(tf.summary.scalar("source_2_teacher_output_loss", self.net.source_2_teacher_total_loss))
        scalar_summaries.append(tf.summary.scalar("source_3_teacher_output_loss", self.net.source_3_teacher_total_loss))

        scalar_summaries.append(tf.summary.scalar("source_1_student_output_loss", self.net.source_1_student_output_loss))
        # scalar_summaries.append(tf.summary.scalar("source_1_student_inter_loss", self.net.source_1_student_inter_loss))

        scalar_summaries.append(tf.summary.scalar("source_2_student_output_loss", self.net.source_2_student_output_loss))
        # scalar_summaries.append(tf.summary.scalar("source_2_student_inter_loss", self.net.source_2_student_inter_loss))

        scalar_summaries.append(tf.summary.scalar("source_3_student_output_loss", self.net.source_3_student_output_loss))
        scalar_summaries.append(tf.summary.scalar("overall_dice_loss", self.net.overall_dice_loss))
        # scalar_summaries.append(tf.summary.scalar("source_3_student_inter_loss", self.net.source_3_student_inter_loss))

        train_images.append(tf.summary.image("source_1_student_pred_train", tf.expand_dims(tf.cast(self.net.source_1_student_pred_compact, tf.float32), 3)))
        train_images.append(tf.summary.image('source_1_student_image_train', tf.expand_dims(tf.cast(self.net.source_1[:,:,:,1], tf.float32), 3)))
        train_images.append(tf.summary.image('source_1_student_gt_train', tf.expand_dims(tf.cast(self.net.source_1_y_compact, tf.float32), 3)))

        val_images.append(tf.summary.image("source_1_student_pred_val", tf.expand_dims(tf.cast(self.net.source_1_student_pred_compact, tf.float32), 3)))
        val_images.append(tf.summary.image('source_1_student_image_val', tf.expand_dims(tf.cast(self.net.source_1[:,:,:,1], tf.float32), 3)))

        train_images.append(tf.summary.image("source_2_student_pred_train", tf.expand_dims(tf.cast(self.net.source_2_student_pred_compact, tf.float32), 3)))
        train_images.append(tf.summary.image('source_2_student_image_train', tf.expand_dims(tf.cast(self.net.source_2[:,:,:,1], tf.float32), 3)))
        train_images.append(tf.summary.image('source_2_student_gt_train', tf.expand_dims(tf.cast(self.net.source_2_y_compact, tf.float32), 3)))

        val_images.append(tf.summary.image("source_2_student_pred_val", tf.expand_dims(tf.cast(self.net.source_2_student_pred_compact, tf.float32), 3)))
        val_images.append(tf.summary.image('source_2_student_image_val', tf.expand_dims(tf.cast(self.net.source_2[:,:,:,1], tf.float32), 3)))

        train_images.append(tf.summary.image("source_3_student_pred_train", tf.expand_dims(tf.cast(self.net.source_3_student_pred_compact, tf.float32), 3)))
        train_images.append(tf.summary.image('source_3_student_image_train', tf.expand_dims(tf.cast(self.net.source_3[:,:,:,1], tf.float32), 3)))
        train_images.append(tf.summary.image('source_3_student_gt_train', tf.expand_dims(tf.cast(self.net.source_3_y_compact, tf.float32), 3)))

        val_images.append(tf.summary.image("source_3_student_pred_val", tf.expand_dims(tf.cast(self.net.source_3_student_pred_compact, tf.float32), 3)))
        val_images.append(tf.summary.image('source_3_student_image_val', tf.expand_dims(tf.cast(self.net.source_3[:,:,:,1], tf.float32), 3)))

        self.source_scalar_summary_op = tf.summary.merge(scalar_summaries)
        self.source_train_image_summary_op = tf.summary.merge(train_images)
        self.source_val_image_summary_op = tf.summary.merge(val_images)

    def minibatch_stats_segmenter_source(self, summary_writer, step, batch_x_1, batch_y_1, batch_x_2, batch_y_2, batch_x_3, batch_y_3):

            summary_str, summary_img = self.sess.run([self.source_scalar_summary_op, self.source_val_image_summary_op],
                                                feed_dict={self.net.source_1: batch_x_1,
                                                           self.net.source_1_y: batch_y_1,
                                                           self.net.source_2: batch_x_2,
                                                           self.net.source_2_y: batch_y_2,
                                                           self.net.source_3: batch_x_3,
                                                           self.net.source_3_y: batch_y_3,
                                                           self.net.training_mode_encoder: False,
                                                           self.net.training_mode_decoder: False,
                                                           self.net.keep_prob: 1.})
            summary_writer.add_summary(summary_str, step)
            summary_writer.add_summary(summary_img, step)
            summary_writer.flush()
