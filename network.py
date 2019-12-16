import os
import glob
import time
import shutil
import numpy as np
import logging
import tensorflow as tf

from tensorflow.python import debug as tf_debug
from lib.layers import *

np.random.seed(0)
contour_map = { # a map used for mapping label value to its name, used for output
    "bg": 0,
    "prostate": 1,
}

class Network(object):

    def __init__(self, args):

        self.n_class = args.n_class
        self.batch_size = args.batch_size
        self.volume_size = args.volume_size
        self.label_size = args.label_size
        self.cost_kwargs = args.cost_kwargs

        self.training_mode_encoder = tf.placeholder_with_default(True, shape = None, name = "training_mode_for_bn_moving_source")
        self.training_mode_decoder = tf.placeholder_with_default(True, shape = None, name = "training_mode_for_bn_moving_decoder")
        self.training_mode_discriminator = tf.placeholder_with_default(True, shape = None, name = "training_mode_for_bn_moving_discriminator")

        self.keep_prob = tf.placeholder(dtype=tf.float32, name = "dropout_keep_rate")  # dropout (keep probability
        self.miu_gan_gen = tf.placeholder(dtype=tf.float32, name = "miu_gan_gen") 
        self.miu_gan_dis = tf.placeholder(dtype=tf.float32, name = "miu_gan_dis") 
        self.source_1 = tf.placeholder("float", shape=[self.batch_size, self.volume_size[0], self.volume_size[1], self.volume_size[2]])
        self.source_1_y = tf.placeholder("float", shape=[self.batch_size, self.label_size[0], self.label_size[1], self.n_class]) # source segmentation
        self.source_2 = tf.placeholder("float", shape=[self.batch_size, self.volume_size[0], self.volume_size[1], self.volume_size[2]])
        self.source_2_y = tf.placeholder("float", shape=[self.batch_size, self.label_size[0], self.label_size[1], self.n_class]) # source segmentation
        self.source_3 = tf.placeholder("float", shape=[self.batch_size, self.volume_size[0], self.volume_size[1], self.volume_size[2]])
        self.source_3_y = tf.placeholder("float", shape=[self.batch_size, self.label_size[0], self.label_size[1], self.n_class]) # source segmentation

        """share encoder"""
        with tf.variable_scope("student_encoder", reuse = tf.AUTO_REUSE) as scope:
            with tf.device('/device:GPU:0'):
                self.source_1_student_feature_pyramid = self.encoder(input_data = self.source_1, keep_prob = self.keep_prob, is_training = self.training_mode_encoder, bn_scope='source_1')
            with tf.device('/device:GPU:1'):
                self.source_2_student_feature_pyramid = self.encoder(input_data = self.source_2, keep_prob = self.keep_prob, is_training = self.training_mode_encoder, bn_scope='source_2')
            with tf.device('/device:GPU:2'):
                self.source_3_student_feature_pyramid = self.encoder(input_data = self.source_3, keep_prob = self.keep_prob, is_training = self.training_mode_encoder, bn_scope='source_3')

        """individual decoder"""
        with tf.device('/device:GPU:0'):
            with tf.variable_scope("teacher_1", reuse = tf.AUTO_REUSE) as scope:
                self.source_1_teacher_seg_logits = self.decoder(input_data = self.source_1_student_feature_pyramid, keep_prob = self.keep_prob, is_training = self.training_mode_decoder)
                self.source_1_teacher_softmaxpred = tf.nn.softmax(self.source_1_teacher_seg_logits)
                self.source_1_teacher_pred_compact = tf.argmax(self.source_1_teacher_softmaxpred, 3) # predictions
        with tf.device('/device:GPU:1'):
            with tf.variable_scope("teacher_2", reuse = tf.AUTO_REUSE) as scope:
                self.source_2_teacher_seg_logits = self.decoder(input_data = self.source_2_student_feature_pyramid, keep_prob = self.keep_prob, is_training = self.training_mode_decoder)
                self.source_2_teacher_softmaxpred = tf.nn.softmax(self.source_2_teacher_seg_logits)
                self.source_2_teacher_pred_compact = tf.argmax(self.source_2_teacher_softmaxpred, 3) # predictions
        with tf.device('/device:GPU:2'):
            with tf.variable_scope("teacher_3", reuse = tf.AUTO_REUSE) as scope:
                self.source_3_teacher_seg_logits = self.decoder(input_data = self.source_3_student_feature_pyramid, keep_prob = self.keep_prob, is_training = self.training_mode_decoder)
                self.source_3_teacher_softmaxpred = tf.nn.softmax(self.source_3_teacher_seg_logits)
                self.source_3_teacher_pred_compact = tf.argmax(self.source_3_teacher_softmaxpred, 3) # predictions

        """Define Network"""
        with tf.variable_scope("student_decoder", reuse = tf.AUTO_REUSE) as scope:
            with tf.device('/device:GPU:0'):
                    self.source_1_student_seg_logits = self.decoder(input_data = self.source_1_student_feature_pyramid, keep_prob = self.keep_prob, is_training = self.training_mode_decoder, bn_scope='source_1')
                    self.source_1_student_softmaxpred = tf.nn.softmax(self.source_1_student_seg_logits)
                    self.source_1_student_pred_compact = tf.argmax(self.source_1_student_softmaxpred, 3) # predictions
            with tf.device('/device:GPU:1'):
                    self.source_2_student_seg_logits = self.decoder(input_data = self.source_2_student_feature_pyramid, keep_prob = self.keep_prob, is_training = self.training_mode_decoder, bn_scope='source_2')
                    self.source_2_student_softmaxpred = tf.nn.softmax(self.source_2_student_seg_logits)
                    self.source_2_student_pred_compact = tf.argmax(self.source_2_student_softmaxpred, 3) # predictions
            with tf.device('/device:GPU:2'):
                    self.source_3_student_seg_logits = self.decoder(input_data = self.source_3_student_feature_pyramid, keep_prob = self.keep_prob, is_training = self.training_mode_decoder, bn_scope='source_3')
                    self.source_3_student_softmaxpred = tf.nn.softmax(self.source_3_student_seg_logits)
                    self.source_3_student_pred_compact = tf.argmax(self.source_3_student_softmaxpred, 3)

        self.source_1_y_compact = tf.argmax(self.source_1_y, 3)
        self.source_2_y_compact = tf.argmax(self.source_2_y, 3)
        self.source_3_y_compact = tf.argmax(self.source_3_y, 3)
         # predictions

        """ Define Loss """
        self.source_1_teacher_hard_seg_loss, self.source_1_teacher_hard_seg_dice_loss, self.source_1_teacher_hard_seg_ce_loss  = self._get_segmentation_cost(seg_logits = self.source_1_teacher_seg_logits, softmaxpred = self.source_1_teacher_softmaxpred, seg_gt = self.source_1_y)
        self.source_1_teacher_soft_seg_loss, self.source_1_teacher_soft_seg_dice_loss, self.source_1_teacher_soft_seg_ce_loss  = self._get_segmentation_cost(seg_logits = self.source_1_teacher_seg_logits, softmaxpred = self.source_1_teacher_softmaxpred, seg_gt = tf.one_hot(self.source_1_student_pred_compact, depth=2))
        self.source_2_teacher_hard_seg_loss, self.source_2_teacher_hard_seg_dice_loss, self.source_2_teacher_hard_seg_ce_loss  = self._get_segmentation_cost(seg_logits = self.source_2_teacher_seg_logits, softmaxpred = self.source_2_teacher_softmaxpred, seg_gt = self.source_2_y)
        self.source_2_teacher_soft_seg_loss, self.source_2_teacher_soft_seg_dice_loss, self.source_2_teacher_soft_seg_ce_loss  = self._get_segmentation_cost(seg_logits = self.source_2_teacher_seg_logits, softmaxpred = self.source_2_teacher_softmaxpred, seg_gt = tf.one_hot(self.source_2_student_pred_compact, depth=2))
        self.source_3_teacher_hard_seg_loss, self.source_3_teacher_hard_seg_dice_loss, self.source_3_teacher_hard_seg_ce_loss  = self._get_segmentation_cost(seg_logits = self.source_3_teacher_seg_logits, softmaxpred = self.source_3_teacher_softmaxpred, seg_gt = self.source_3_y)
        self.source_3_teacher_soft_seg_loss, self.source_3_teacher_soft_seg_dice_loss, self.source_3_teacher_soft_seg_ce_loss  = self._get_segmentation_cost(seg_logits = self.source_3_teacher_seg_logits, softmaxpred = self.source_3_teacher_softmaxpred, seg_gt = tf.one_hot(self.source_3_student_pred_compact, depth=2))

        self.source_1_student_hard_seg_loss, self.source_1_student_hard_seg_dice_loss, self.source_1_student_hard_seg_ce_loss  = self._get_segmentation_cost(seg_logits = self.source_1_student_seg_logits, softmaxpred = self.source_1_student_softmaxpred, seg_gt = self.source_1_y)
        self.source_1_student_soft_seg_loss, self.source_1_student_soft_seg_dice_loss, self.source_1_student_soft_seg_ce_loss  = self._get_segmentation_cost(seg_logits = self.source_1_student_seg_logits, softmaxpred = self.source_1_student_softmaxpred, seg_gt = tf.one_hot(self.source_1_teacher_pred_compact, depth=2))
        self.source_2_student_hard_seg_loss, self.source_2_student_hard_seg_dice_loss, self.source_2_student_hard_seg_ce_loss  = self._get_segmentation_cost(seg_logits = self.source_2_student_seg_logits, softmaxpred = self.source_2_student_softmaxpred, seg_gt = self.source_2_y)
        self.source_2_student_soft_seg_loss, self.source_2_student_soft_seg_dice_loss, self.source_2_student_soft_seg_ce_loss  = self._get_segmentation_cost(seg_logits = self.source_2_student_seg_logits, softmaxpred = self.source_2_student_softmaxpred, seg_gt = tf.one_hot(self.source_2_teacher_pred_compact, depth=2))
        self.source_3_student_hard_seg_loss, self.source_3_student_hard_seg_dice_loss, self.source_3_student_hard_seg_ce_loss  = self._get_segmentation_cost(seg_logits = self.source_3_student_seg_logits, softmaxpred = self.source_3_student_softmaxpred, seg_gt = self.source_3_y)
        self.source_3_student_soft_seg_loss, self.source_3_student_soft_seg_dice_loss, self.source_3_student_soft_seg_ce_loss  = self._get_segmentation_cost(seg_logits = self.source_3_student_seg_logits, softmaxpred = self.source_3_student_softmaxpred, seg_gt = tf.one_hot(self.source_3_teacher_pred_compact, depth=2))

        # self.source_1_student_soft_align_loss = self._get_inter_align_cost(self.source_1_student_feature_pyramid, self.source_1_teacher_feature_pyramid)
        # self.source_2_student_soft_align_loss = self._get_inter_align_cost(self.source_2_student_feature_pyramid, self.source_2_teacher_feature_pyramid)
        # self.source_3_student_soft_align_loss = self._get_inter_align_cost(self.source_3_student_feature_pyramid, self.source_3_teacher_feature_pyramid)

        self.source_1_teacher_total_loss = self.source_1_teacher_hard_seg_loss# * args.cost_kwargs["student_hard_dice"] + self.source_1_teacher_soft_seg_loss * args.cost_kwargs["student_soft_dice"]
        self.source_2_teacher_total_loss = self.source_2_teacher_hard_seg_loss# * args.cost_kwargs["student_hard_dice"] + self.source_2_teacher_soft_seg_loss * args.cost_kwargs["student_soft_dice"]
        self.source_3_teacher_total_loss = self.source_3_teacher_hard_seg_loss# * args.cost_kwargs["student_hard_dice"] + self.source_3_teacher_soft_seg_loss * args.cost_kwargs["student_soft_dice"]

        self.source_1_student_output_loss = self.source_1_student_hard_seg_loss * args.cost_kwargs["student_hard_dice"] + self.source_1_student_soft_seg_loss * args.cost_kwargs["student_soft_dice"] 
        self.source_2_student_output_loss = self.source_2_student_hard_seg_loss * args.cost_kwargs["student_hard_dice"] + self.source_2_student_soft_seg_loss * args.cost_kwargs["student_soft_dice"] 
        self.source_3_student_output_loss = self.source_3_student_hard_seg_loss * args.cost_kwargs["student_hard_dice"] + self.source_3_student_soft_seg_loss * args.cost_kwargs["student_soft_dice"] 
        # self.source_1_student_inter_loss = self.source_1_student_soft_align_loss * args.cost_kwargs["student_inter_align"] 
        # self.source_2_student_inter_loss = self.source_2_student_soft_align_loss * args.cost_kwargs["student_inter_align"] 
        # self.source_3_student_inter_loss = self.source_3_student_soft_align_loss * args.cost_kwargs["student_inter_align"] 
        
        self.source_1_student_total_loss = args.cost_kwargs["student_1"] * self.source_1_student_output_loss #+ self.source_1_student_inter_loss
        self.source_2_student_total_loss = args.cost_kwargs["student_2"] * self.source_2_student_output_loss #+ self.source_2_student_inter_loss
        self.source_3_student_total_loss = args.cost_kwargs["student_3"] * self.source_3_student_output_loss #+ self.source_3_student_inter_loss

        self.overall_dice_loss = self.source_1_student_hard_seg_loss + self.source_2_student_hard_seg_loss + self.source_3_student_hard_seg_loss
        self.source_student_total_loss = self.source_1_student_total_loss + self.source_2_student_total_loss + self.source_3_student_total_loss

        """ Define Variable"""
        self.student_variables = tf.trainable_variables(scope="student_encoder") + tf.trainable_variables(scope="student_decoder") 
        self.teacher_1_variables = tf.trainable_variables(scope="teacher_1")
        self.teacher_2_variables = tf.trainable_variables(scope="teacher_2")
        self.teacher_3_variables = tf.trainable_variables(scope="teacher_3")
        self.teacher_variables = self.teacher_1_variables + self.teacher_2_variables + self.teacher_3_variables +  tf.trainable_variables(scope="student_encoder")
        self.joint_variables = self.student_variables + self.teacher_1_variables + self.teacher_2_variables + self.teacher_3_variables
 

    def encoder(self, input_data, keep_prob, is_training, feature_base = 32, bn_scope=''):
        conv1 = conv2d(input_data, 3, feature_base, keep_prob, name = 'conv_1')
        res1 = res_block(conv1, 3, feature_base, keep_prob, is_training = is_training, scope='res_1', bn_scope=bn_scope)
        pool1 = max_pool2d(res1, n = 2)


        conv2 = conv2d(pool1, 3, feature_base * 2, keep_prob, name='conv_2')
        res2 = res_block(conv2, 3, feature_base * 2, keep_prob, is_training=is_training, scope='res_2', bn_scope=bn_scope)
        pool2 = max_pool2d(res2, n = 2)

        conv3 = conv2d(pool2, 3, feature_base * 4, keep_prob, name='conv_3')
        res3 = res_block(conv3, 3, feature_base * 4, keep_prob, is_training=is_training, scope='res_3', bn_scope=bn_scope)
        pool3 = max_pool2d(res3, n = 2)

        conv4 = conv2d(pool3, 3, feature_base * 8, keep_prob, name='conv_4')
        res4 = res_block(conv4, 3, feature_base * 8, keep_prob, is_training=is_training, scope='res_4', bn_scope=bn_scope)
        pool4 = max_pool2d(res4, n = 2)

        conv5 = conv2d(pool4, 3, feature_base * 16, keep_prob, name='conv_5')
        res5_1 = res_block(conv5, 3, feature_base * 16, keep_prob, is_training=is_training, scope='res_5_1', bn_scope=bn_scope)
        res5_2 = res_block(res5_1, 3, feature_base * 16, keep_prob, is_training=is_training, scope='res_5_2', bn_scope=bn_scope)

        return [res1, res2, res3, res4, res5_2]


    def decoder(self, input_data, keep_prob, is_training, feature_base = 32, bn_scope=''):
        input_res1, input_res2, input_res3, input_res4, input_res5_2 = input_data

        deconv6 = bn_relu_deconv2d(input_res5_2, 3, feature_base * 8, [self.batch_size, self.volume_size[0]/8, self.volume_size[1]/8, feature_base * 8], keep_prob, \
                                   is_training=is_training, stride=2, scope='up_sample_6', bn_scope=bn_scope)
        sum6 = concat2d(input_res4, deconv6)
        conv6 = conv2d(sum6, 3, feature_base * 8, keep_prob, name='conv_6')
        #conv6 = domain_adapter_specific(conv6, scope='adapter_6')
        res6 = res_block(conv6, 3, feature_base * 8, keep_prob, is_training=is_training, scope='res_6', bn_scope=bn_scope)

        deconv7 = bn_relu_deconv2d(res6, 3, feature_base * 4, [self.batch_size, self.volume_size[0]/4, self.volume_size[1]/4, feature_base * 4], keep_prob, \
                                   is_training=is_training, stride=2, scope='up_sample_7', bn_scope=bn_scope)
        sum7 = concat2d(input_res3, deconv7)
        conv7 = conv2d(sum7, 3, feature_base * 4, keep_prob, name='conv_7')
        #conv7 = domain_adapter_specific(conv7, scope='adapter_7')
        res7 = res_block(conv7, 3, feature_base * 4, keep_prob, is_training=is_training, scope='res_7', bn_scope=bn_scope)

        deconv8 = bn_relu_deconv2d(res7, 3, feature_base * 2, [self.batch_size, self.volume_size[0]/2, self.volume_size[1]/2, feature_base * 2], keep_prob, \
                                   is_training=is_training, stride=2, scope='up_sample_8', bn_scope=bn_scope)
        sum8 = concat2d(input_res2, deconv8)
        conv8 = conv2d(sum8, 3, feature_base * 2, keep_prob, name='conv_8')
        #conv8 = domain_adapter_specific(conv8, scope='adapter_8')
        res8 = res_block(conv8, 3, feature_base * 2, keep_prob, is_training=is_training, scope='res_8', bn_scope=bn_scope)

        deconv9 = bn_relu_deconv2d(res8, 3, feature_base, [self.batch_size, self.volume_size[0], self.volume_size[1], feature_base], keep_prob, \
                                   is_training=is_training, stride=2, scope='up_sample_9', bn_scope=bn_scope)
        sum9 = concat2d(input_res1, deconv9)
        conv9 = conv2d(sum9, 3, feature_base , keep_prob, name='conv_9')
        #conv9 = domain_adapter_specific(conv9, scope='adapter_9')
        res9 = res_block(conv9, 3, feature_base, keep_prob, is_training=is_training, scope='res_9', bn_scope=bn_scope)

        output = bn_relu_conv2d(res9, 3, self.n_class, keep_prob, is_training=is_training, scope='conv_final', bn_scope=bn_scope)  

        return output

    def _get_segmentation_cost(self, seg_logits, softmaxpred, seg_gt):
        """
        calculate the loss for segmentation prediction
        :param seg_logits: probability segmentation from the segmentation network
        :param seg_gt: ground truth segmentaiton mask
        :return: segmentation loss, according to the cost_kwards setting, cross-entropy weighted loss and dice loss
        """

        dice = 0

        for i in xrange(self.n_class):
            #inse = tf.reduce_sum(softmaxpred[:, :, :, i]*seg_gt[:, :, :, i])
            inse = tf.reduce_sum(softmaxpred[:, :, :, i]*seg_gt[:, :, :, i])
            l = tf.reduce_sum(softmaxpred[:, :, :, i])
            r = tf.reduce_sum(seg_gt[:, :, :, i])
            dice += 2.0 * inse/(l+r+1e-7) # here 1e-7 is relaxation eps
        dice_loss = 1 - 1.0 * dice / self.n_class

        ce_weighted = 0
        for i in xrange(self.n_class):
            gti = seg_gt[:,:,:,i]
            predi = softmaxpred[:,:,:,i]
            ce_weighted += -1.0 * gti * tf.log(tf.clip_by_value(predi, 0.005, 1))
        ce_weighted_loss = tf.reduce_mean(ce_weighted)

        total_loss =  dice_loss 


        return total_loss, dice_loss, ce_weighted_loss

    def _get_inter_align_cost(self, student_feature, teacher_feature):


        return tf.reduce_mean(tf.square(student_feature[4] - teacher_feature[4]))