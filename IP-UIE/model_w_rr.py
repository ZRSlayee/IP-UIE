#coding:utf-8
# 2degree，1channel, with r restoration
from __future__ import print_function

import os
from glob import glob
import time
import random
import argparse

from PIL import Image
import tensorflow as tf
import numpy as np

from utils import *
from datetime import datetime

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
LOG_DIR = "./logs/model_r/" + TIMESTAMP
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def concat(layers):
    return tf.concat(layers, axis=3)

def upsample_and_concat(x1, x2, output_channels, in_channels, scope_name, trainable=True):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool_size = 2
        deconv_filter = tf.get_variable('weights', [pool_size, pool_size, output_channels, in_channels], trainable= True)
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1], name=scope_name)

        deconv_output =  tf.concat([deconv, x2],3)
        deconv_output.set_shape([None, None, None, output_channels*2])

        return deconv_output

def Conv(name, x, filter_size, in_filters, out_filters, strides, padding):
    with tf.variable_scope(name):
        kernel = tf.get_variable('filter', [filter_size, filter_size, in_filters, out_filters], tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=0.01))
        bias = tf.get_variable('bias', [out_filters], tf.float32, initializer=tf.zeros_initializer())
        
        return tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding=padding) + bias
    
def Conv_transpose(name, x, filter_size, in_filters, out_filters, fraction=2, padding="SAME"):
    with tf.variable_scope(name):
        n = filter_size * filter_size * out_filters
        kernel = tf.get_variable('filter', [filter_size, filter_size, out_filters, in_filters], tf.float32,
                                 initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
        size = tf.shape(x)
        output_shape = tf.stack([size[0], size[1] * fraction, size[2] * fraction, out_filters])
        x = tf.nn.conv2d_transpose(x, kernel, output_shape, [1, fraction, fraction, 1], padding)
        
        return x

def instance_norm(x, BN_epsilon = 1e-5, name = "instance_norm"):
    
    with tf.variable_scope(name):
        N, H, W, C = x.shape
        mean, variance = tf.nn.moments(x, axes=[1, 2], keep_dims = True) # keep_dims!!!!!
        x = (x - mean) / (tf.sqrt(variance + BN_epsilon))
    return x

# input tensor [batch,in_height,in_width,in_channels]
def DecomNet(input_im, layer_num, channel=64, kernel_size=3):
    input_max = tf.reduce_max(input_im, axis=3, keepdims=True) # 取每个通道的最大值
    input_im = concat([input_max, input_im]) # 用各个通道的最大值图 来模拟亮度图 
    with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
        conv = tf.layers.conv2d(input_im, channel, kernel_size * 3, padding='same', activation=None, name="shallow_feature_extraction")
        for idx in range(layer_num):
            conv = tf.layers.conv2d(conv, channel, kernel_size, padding='same', activation=tf.nn.relu, name='activated_layer_%d' % idx)
        conv = tf.layers.conv2d(conv, 4, kernel_size, padding='same', activation=None, name='recon_layer')
    # 将卷积结果分解为亮度图和光照图
    R = tf.sigmoid(conv[:,:,:,0:3])
    L = tf.sigmoid(conv[:,:,:,3:4])
    print("R", R.shape)
    print("L", L.shape)
    return R, L

# R restoration
def RRNet(input_L, input_R, channel=64, kernel_size=3):
    input_im = concat([input_R, input_L])
    with tf.variable_scope('RestorationNet'):
        # 输入层
        conv0 = tf.layers.conv2d(input_im, channel, kernel_size, padding='same', activation=None)
        # 三次下采样
        conv1 = tf.layers.conv2d(conv0, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        # 三次上采样，最近邻插值
        up1 = tf.image.resize_nearest_neighbor(conv3, (tf.shape(conv2)[1], tf.shape(conv2)[2]))# 使用最近邻插值调整图像
        deconv1 = tf.layers.conv2d(up1, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv2
        up2 = tf.image.resize_nearest_neighbor(deconv1, (tf.shape(conv1)[1], tf.shape(conv1)[2]))
        deconv2= tf.layers.conv2d(up2, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv1
        up3 = tf.image.resize_nearest_neighbor(deconv2, (tf.shape(conv0)[1], tf.shape(conv0)[2]))
        deconv3 = tf.layers.conv2d(up3, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv0
        # 多尺度特征融合
        deconv1_resize = tf.image.resize_nearest_neighbor(deconv1, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
        deconv2_resize = tf.image.resize_nearest_neighbor(deconv2, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
        feature_gather = concat([deconv1_resize, deconv2_resize, deconv3])
        feature_fusion = tf.layers.conv2d(feature_gather, channel, 1, padding='same', activation=None)
        output = tf.layers.conv2d(feature_fusion, 3, 3, padding='same', activation=None)
    return output 

def Relight(input_I, channel=32, reuse = False):
    with tf.variable_scope("RelightNet", reuse = reuse):
        conv1 = tf.layers.conv2d(input_I, channel, 3, padding="same", activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, channel, 3, padding="same", activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, channel, 3, padding="same", activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(conv3, channel, 3, padding="same", activation=tf.nn.relu)
        conv5 = tf.layers.conv2d(concat([conv3, conv4]), channel, 3, padding="same", activation=tf.nn.relu)
        conv6 = tf.layers.conv2d(concat([conv2, conv5]), channel, 3, padding="same", activation=tf.nn.relu)
        conv7 = tf.layers.conv2d(concat([conv1, conv6]), 3, 3, padding="same", activation=None)
        
        e1, e2, e3 = conv7[:,:,:,0:1], conv7[:,:,:,1:2], conv7[:,:,:,2:3]
        print(input_I.shape)
#         I = input_I*tf.exp(e*(1-input_I))  #N*H*W*1 MIN: N*1,MAX: N*1,
        I = input_I * (tf.exp(e1 + e2*input_I + e3*input_I*input_I))
#         S = concat([I, I, I]) * input_R
        E = tf.exp(e1 + e2*input_I + e3*input_I*input_I)
        
    return I, E, e1, e2, e3

def Generator(input_I, input_R, channel=32, reuse = False):
    with tf.variable_scope("Generator", reuse = reuse):
        I_delta, E, e1, e2, e3 = Relight(input_I)
        R_delta = RRNet(I_delta, input_R)
#         I_delta, E = Relight(input_I)
        S_delta = concat([I_delta, I_delta, I_delta])*R_delta
    return I_delta, R_delta, S_delta, E, e1, e2, e3

def Discriminator(x, channel=64, kernel_size=3, reuse = False):
    
    with tf.variable_scope("Discriminator", reuse = reuse):
        x = Conv(name='conv1', x=x, filter_size=4, in_filters=3,
                     out_filters=channel, strides=2, padding="SAME")
        print(x.shape)
#         x = instance_norm(x)
        x = tf.nn.leaky_relu(x)

        n = 1

        for i in range(3):
            prev = n
            n = min(2 ** (i+1), 8)
            x = Conv(name='conv%02d' % i, x=x, filter_size=4, in_filters=channel * prev,
                     out_filters=channel * n, strides=2, padding="SAME")
#             x = instance_norm(x)
            x = tf.nn.leaky_relu(x)

        prev = n
        n = min(2 ** 3, 8)
        x = Conv(name='conv_d1', x=x, filter_size=4, in_filters=channel * prev,
                 out_filters=channel * n, strides=1, padding="SAME")
        # x = instance_norm(name = 'instance_norm_d1', x = x, dim = self.n_feats * n)
#         x = instance_norm(x)
        x = tf.nn.leaky_relu(x)

        x = Conv(name='conv_d2', x=x, filter_size=4, in_filters=channel * n,
                 out_filters=1, strides=1, padding="SAME")
        x = tf.nn.sigmoid(x)

        return x
    
class lowlight_enhance(object):
    def __init__(self, sess):
        self.sess = sess
        self.DecomNet_layer_num = 5
        self.batch_size = 16

        # build the model
        self.input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')
        self.input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')

        [R_low, I_low] = DecomNet(self.input_low, layer_num=self.DecomNet_layer_num)
        [R_high, I_high] = DecomNet(self.input_high, layer_num=self.DecomNet_layer_num)
        
        I_delta, R_delta, S, E, e1, e2, e3 = Generator(I_low, R_low)
    
        I_low_3 = concat([I_low, I_low, I_low])
        I_high_3 = concat([I_high, I_high, I_high])
        I_delta_3 = concat([I_delta, I_delta, I_delta])

        self.output_R_low = R_low
        self.output_I_low = I_low_3
        self.output_I_delta = I_delta_3
        self.output_R_delta = R_delta
        self.output_S = S
        self.output_e1 = e1
        self.output_e2 = e2
        self.output_e3 = e3
        self.output_E = E

        # loss
        self.real_prob = Discriminator(self.input_high, reuse=False)
        self.fake_prob = Discriminator(self.output_S, reuse=True)   
        # Decom-Net loss
        self.recon_loss_low = tf.reduce_mean(tf.abs(R_low * I_low_3 -  self.input_low)) # 保证分解的正确性
        self.recon_loss_high = tf.reduce_mean(tf.abs(R_high * I_high_3 - self.input_high)) # 同上
        self.recon_loss_mutal_low = tf.reduce_mean(tf.abs(R_high * I_low_3 - self.input_low)) # 保证反射图的一致性R_low == R_high
        self.recon_loss_mutal_high = tf.reduce_mean(tf.abs(R_low * I_high_3 - self.input_high)) # 同上
        self.equal_R_loss = tf.reduce_mean(tf.abs(R_low - R_high))
        self.Ismooth_loss_low = self.smooth(I_low, R_low)
        self.Ismooth_loss_high = self.smooth(I_high, R_high)
        # RRNet loss
        self.mse_R_loss = tf.reduce_mean(tf.square(R_delta - R_high))
        self.gradient_R = self.grad_loss(R_low, R_delta)
        # Relight loss
        self.relight_loss = tf.reduce_mean(tf.abs(self.output_S - self.input_high))
#         self.Ismooth_loss_I_delta = self.smooth(I_delta, R_low)
        self.Ismooth_loss_RI_delta = self.smooth(I_delta, R_delta)
        # GAN loss
        ## GP_loss
        epsilon = tf.random_uniform(shape = [self.batch_size, 1, 1, 1],
                                    minval = 0.0, maxval = 1.0)
        interpolated_input = epsilon * self.input_high + (1 - epsilon) * self.output_S
        gra = tf.gradients(Discriminator(interpolated_input, reuse = True),
                                [interpolated_input])[0]
        self.loss_GP = tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_mean(tf.square(gra), axis=[1,2,3]))-1))
        ## d_loss
        d_loss_real = - tf.reduce_mean(self.real_prob)
        d_loss_fake = tf.reduce_mean(self.fake_prob)
        ## g_loss       
        ####### exp 损失
        self.exp_loss = self.L_exp(I_delta, I_high)
        self.tv_loss = self.L_tv(E)
#         self.tv_loss = self.L_tv(e1) + self.L_tv(e2) + self.L_tv(e3)

        self.loss_Decom = self.recon_loss_low + self.recon_loss_high + 0.001 * self.recon_loss_mutal_low + 0.001 * self.recon_loss_mutal_high + 0.1 * self.Ismooth_loss_low + 0.1 * self.Ismooth_loss_high + 0.01 * self.equal_R_loss
#         self.loss_RR = self.mse_R_loss
        self.loss_D = d_loss_real + d_loss_fake
        self.loss_G = - 0.1 * d_loss_fake + 1.0 * self.relight_loss + 0.3 * self.Ismooth_loss_RI_delta + 0.1*self.mse_R_loss + 0.1*self.gradient_R + 0.001 * self.tv_loss

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        GAN_optimizer = tf.train.RMSPropOptimizer(self.lr, name="RMSProp")
#         GAN_optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        self.var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
#         self.var_RR = [var for var in tf.trainable_variables() if 'RestorationNet' in var.name]
        self.var_G = [var for var in tf.trainable_variables() if 'Generator' in var.name]
        self.var_D = [var for var in tf.trainable_variables() if 'Discriminator' in var.name]
#         self.var_max_relight = [var for var in tf.trainable_variables() if "Relight_max" in var.name]

        self.train_op_Decom = optimizer.minimize(self.loss_Decom, var_list = self.var_Decom)
#         self.train_op_RR = optimizer.minimize(self.loss_RR, var_list = self.var_RR)
        self.train_op_D = GAN_optimizer.minimize(self.loss_D, var_list = self.var_D)
        self.train_op_G = GAN_optimizer.minimize(self.loss_G, var_list = self.var_G)

        ## tensorboard
        logging_Decom_loss = tf.summary.scalar(name="Decom_loss", tensor=self.loss_Decom)
        logging_RR_loss = tf.summary.scalar(name="RR_loss", tensor=self.mse_R_loss)
        logging_RR_loss = tf.summary.scalar(name="gradient_R_loss", tensor=self.gradient_R)
        logging_D_loss = tf.summary.scalar(name="D_loss", tensor=self.loss_D)
        logging_G_loss = tf.summary.scalar(name="G_loss", tensor=self.loss_G)
        logging_relight_loss = tf.summary.scalar(name="relight_loss", tensor=self.relight_loss)
        logging_adv_loss = tf.summary.scalar(name="adv_loss", tensor=-d_loss_fake)
        logging_Ismooth_loss = tf.summary.scalar(name="I_smooth_loss", tensor=self.Ismooth_loss_RI_delta)
        logging_tv_loss = tf.summary.scalar(name="tve_loss", tensor=self.tv_loss)

        self.sess.run(tf.global_variables_initializer())

        self.saver_Decom = tf.train.Saver(var_list = self.var_Decom)
#         self.saver_RR = tf.train.Saver(var_list = self.var_RR)
        self.saver_Relight = tf.train.Saver(var_list = self.var_D + self.var_G)
#         self.saver_max_relight = tf.train.Saver(var_list = self.var_max_relight)

        print("[*] Initialize model successfully...")
    
    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
        self.smooth_kernel_y = tf.transpose(self.smooth_kernel_x, [1, 0, 2, 3]) # 转置函数

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        return tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))

    def ave_gradient(self, input_tensor, direction):
        return tf.layers.average_pooling2d(self.gradient(input_tensor, direction), pool_size=3, strides=1, padding='SAME')
    
    def grad_loss(self, input_r_low, input_r_high):
        input_r_low_gray = tf.image.rgb_to_grayscale(input_r_low)
        input_r_high_gray = tf.image.rgb_to_grayscale(input_r_high)
        x_loss = tf.square(self.gradient(input_r_low_gray, 'x') - self.gradient(input_r_high_gray, 'x'))
        y_loss = tf.square(self.gradient(input_r_low_gray, 'y') - self.gradient(input_r_high_gray, 'y'))
        grad_loss_all = tf.reduce_mean(x_loss + y_loss)
        return grad_loss_all

    def smooth(self, input_I, input_R):
        input_R = tf.image.rgb_to_grayscale(input_R)
        if input_I.shape[-1] == 3:
            input_I = tf.image.rgb_to_grayscale(input_I)
        return tf.reduce_mean(self.gradient(input_I, "x") * tf.exp(-10 * self.ave_gradient(input_R, "x")) + self.gradient(input_I, "y") * tf.exp(-10 * self.ave_gradient(input_R, "y")))
    
    def L_exp(self, input_I_low, input_I_high, patch_size=16):
#         input_I = tf.image.rgb_to_grayscale(input_I)
        mean_low = tf.layers.average_pooling2d(input_I_low, pool_size=patch_size, strides=1, padding="SAME")
        mean_high = tf.layers.average_pooling2d(input_I_high, pool_size=patch_size, strides=1, padding="SAME")
        d = tf.reduce_mean(tf.abs(mean_low - mean_high))
        return d
    
    def L_tv(self, E):
#         return tf.reduce_mean(self.gradient(E, "x") + self.gradient(E, "y"))
        return tf.reduce_mean(tf.image.total_variation(E))
        
    
    def evaluate(self, epoch_num, eval_low_data, sample_dir, train_phase):
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)

            if train_phase == "Decom":
                result_1, result_2 = self.sess.run([self.output_R_low, self.output_I_low], feed_dict={self.input_low: input_low_eval})
                print("I_low", np.mean(result_2))
            if train_phase == "Relight":
                result_1, result_2, S, E, e1, e2, e3 = self.sess.run([self.output_R_delta, self.output_I_delta, self.output_S, self.output_E, self.output_e1, self.output_e2, self.output_e3], feed_dict={self.input_low: input_low_eval})
                save_images(os.path.join(sample_dir, 'eval_%s_S_%d_%d.png' % (train_phase, idx + 1, epoch_num)), S)
                save_heatmap(os.path.join(sample_dir, 'eval_e1_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), e1)
                save_heatmap(os.path.join(sample_dir, 'eval_e2_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), e2)
                save_heatmap(os.path.join(sample_dir, 'eval_e3_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), e3)
                save_heatmap_e(os.path.join(sample_dir, 'eval_e_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), E)

            save_images(os.path.join(sample_dir, 'eval_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1, result_2)

    def train(self, train_low_data, train_high_data, eval_low_data, batch_size, patch_size, epoch, lr, sample_dir, ckpt_dir, eval_every_epoch, train_phase):
        assert len(train_low_data) == len(train_high_data)
        numBatch = len(train_low_data) // int(batch_size)

        # tensorboard
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(LOG_DIR, self.sess.graph)
        # load pretrained model
        if train_phase == "Decom":
            saver = self.saver_Decom
        elif train_phase == "RR":
            saver = self.saver_RR
        elif train_phase == "Relight":
            saver = self.saver_Relight
        
        load_model_status, global_step = self.load(saver, ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id = 0

        for epoch in range(start_epoch, epoch):
            for batch_id in range(start_step, numBatch):
                # generate data for a batch
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                for patch_id in range(batch_size):
                    h, w, _ = train_low_data[image_id].shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
            
                    rand_mode = random.randint(0, 7)
                    batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode) # 旋转270度
                    batch_input_high[patch_id, :, :, :] = data_augmentation(train_high_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    
                    image_id = (image_id + 1) % len(train_low_data)
                    if image_id == 0:
                        tmp = list(zip(train_low_data, train_high_data))
                        random.shuffle(list(tmp))
                        train_low_data, train_high_data  = zip(*tmp)

                # train
                if train_phase == "Decom":
                    _, loss = self.sess.run([self.train_op_Decom, self.loss_Decom], feed_dict={self.input_low: batch_input_low, \
                                                                               self.input_high: batch_input_high, \
                                                                               self.lr: lr[epoch]})

                    print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                          % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
#                 elif train_phase == "RR":
#                     _, loss = self.sess.run([self.train_op_RR, self.loss_RR], feed_dict={self.input_low: batch_input_low, \
#                                                                                self.input_high: batch_input_high, \
#                                                                                self.lr: lr[epoch]})

#                     print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
#                           % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                
                elif train_phase == "Relight":
                    _, D_loss, GP_loss = self.sess.run([self.train_op_D, self.loss_D, self.loss_GP], feed_dict = {self.input_low: batch_input_low, self.input_high: batch_input_high, self.lr: lr[epoch]})
                    for i in range(1):
                        _, G_loss = self.sess.run([self.train_op_G, self.loss_G],
                                                 feed_dict = {self.input_low: batch_input_low, self.input_high: batch_input_high, self.lr: lr[epoch]})
                    real, fake = self.sess.run([self.real_prob, self.fake_prob], feed_dict = {self.input_low: batch_input_low, self.input_high: batch_input_high, self.lr: lr[epoch]})
                    print("%s Epoch: [%2d] [%4d/%4d], D_loss: %.6f G_loss: %.6f GP_loss: %.6f" % \
                         (train_phase, epoch+1, batch_id+1, numBatch, D_loss, G_loss, GP_loss))
                    
                iter_num += 1

            # evalutate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data, sample_dir=sample_dir, train_phase=train_phase)
                self.save(saver, iter_num, ckpt_dir, "RetinexNet-%s" % train_phase)
            
            summary = self.sess.run(merged, feed_dict = {self.input_low: batch_input_low, self.input_high: batch_input_high})
            train_writer.add_summary(summary, epoch)
            
        print("[*] Finish training for phase %s." % train_phase)

    def save(self, saver, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s" % model_name)
        saver.save(self.sess, \
                   os.path.join(ckpt_dir, model_name), \
                   global_step=iter_num)

    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def test(self, test_low_data, test_high_data, test_low_data_names, load_dir, save_dir, decom_flag):
        tf.global_variables_initializer().run()

        print("[*] Reading checkpoint...")
        load_model_status_Decom, _ = self.load(self.saver_Decom, os.path.join(load_dir, "Decom"))
        load_model_status_Relight, _ = self.load(self.saver_Relight, os.path.join(load_dir, "Relight"))
       
        if load_model_status_Decom and load_model_status_Relight:
            print("[*] Load weights successfully...")
        
        print("[*] Testing...")
        for idx in range(len(test_low_data)):
            print(test_low_data_names[idx])
            [_, name] = os.path.split(test_low_data_names[idx])
#             suffix = name[name.find('.') + 1:]
            suffix = "png"
            name = name[:name.find('.')]

            input_low_test = np.expand_dims(test_low_data[idx], axis=0)
            [R_low, I_low, I_delta, R_delta, S, e1, e2, e3, E] = self.sess.run([self.output_R_low, self.output_I_low, self.output_I_delta, self.output_R_delta, self.output_S, self.output_e1, self.output_e2, self.output_e3, self.output_E], feed_dict = {self.input_low: input_low_test})


            if decom_flag == 1:
                print("save decom img")
                save_images(os.path.join(save_dir, "R_low", name + "_R_low." + suffix), R_low)
                save_images(os.path.join(save_dir, "I_low", name + "_I_low." + suffix), I_low)
                save_images(os.path.join(save_dir, "I_delta", name + "_I_delta." + suffix), I_delta)
                save_images(os.path.join(save_dir, "R_delta", name + "_R_delta." + suffix), R_delta)
                save_heatmap(os.path.join(save_dir, "e", name + "_e1." + suffix), e1)
                save_heatmap(os.path.join(save_dir, "e", name + "_e2." + suffix), e2)
                save_heatmap(os.path.join(save_dir, "e", name + "_e3." + suffix), e3)
                save_heatmap_e(os.path.join(save_dir, "e", name + "_E." + suffix), E)
            save_images(os.path.join(save_dir, "S_delta", name + "."   + suffix), S)
