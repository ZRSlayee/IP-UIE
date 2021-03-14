#coding:utf-8
# 2degree，1channel
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
LOG_DIR = "./logs/" + TIMESTAMP
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def concat(layers):
    return tf.concat(layers, axis=3)

# 图层文件 layer
# 定义DeblurGAN中用到的卷积，反卷积，norm层
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

    return R, L

def Generator(input_I, input_R, channel=32, reuse = False):
    with tf.variable_scope("Generator", reuse = reuse):
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
        S = concat([I, I, I]) * input_R
        E = tf.exp(e1 + e2*input_I + e3*input_I*input_I)
        
    return I, S, e1, e2, e3, E

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
        
        I_delta, S, e1, e2, e3, E = Generator(I_low, R_low)

        I_low_3 = concat([I_low, I_low, I_low])
        I_high_3 = concat([I_high, I_high, I_high])
        I_delta_3 = concat([I_delta, I_delta, I_delta])

        self.output_R_low = R_low
        self.output_I_low = I_low_3
        self.output_I_delta = I_delta_3
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
        self.relight_loss = tf.reduce_mean(tf.abs(self.output_S - self.input_high))
        self.Ismooth_loss_delta = self.smooth(I_delta, R_low)
        ####### exp 损失
        self.exp_loss = self.L_exp(I_delta, I_high)
#         self.tv_loss = self.L_tv(e)
        self.tv_loss = self.L_tv(e1) + self.L_tv(e2) + self.L_tv(e3)

        self.loss_Decom = self.recon_loss_low + self.recon_loss_high + 0.001 * self.recon_loss_mutal_low + 0.001 * self.recon_loss_mutal_high + 0.1 * self.Ismooth_loss_low + 0.1 * self.Ismooth_loss_high + 0.01 * self.equal_R_loss
        self.loss_D = d_loss_real + d_loss_fake
        self.loss_G = - 0.1 * d_loss_fake + 1.0 * self.relight_loss + 0.3 * self.Ismooth_loss_delta + 0.001 * self.tv_loss 

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        GAN_optimizer = tf.train.RMSPropOptimizer(self.lr, name="RMSProp")
#         GAN_optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        self.var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
        self.var_G = [var for var in tf.trainable_variables() if 'Generator' in var.name]
        self.var_D = [var for var in tf.trainable_variables() if 'Discriminator' in var.name]

        self.train_op_Decom = optimizer.minimize(self.loss_Decom, var_list = self.var_Decom)
        self.train_op_D = GAN_optimizer.minimize(self.loss_D, var_list = self.var_D)
        self.train_op_G = GAN_optimizer.minimize(self.loss_G, var_list = self.var_G)

        ## tensorboard
        logging_Decom_loss = tf.summary.scalar(name="Decom_loss", tensor=self.loss_Decom)
        logging_D_loss = tf.summary.scalar(name="D_loss", tensor=self.loss_D)
        logging_G_loss = tf.summary.scalar(name="G_loss", tensor=self.loss_G)
        logging_recon_loss = tf.summary.scalar(name="recon_loss", tensor=self.relight_loss)
        logging_adv_loss = tf.summary.scalar(name="adv_loss", tensor=-d_loss_fake)
        logging_Is_en_loss = tf.summary.scalar(name="I_smooth_loss", tensor=self.Ismooth_loss_delta)
        logging_tv_loss = tf.summary.scalar(name="tve_loss", tensor=self.tv_loss)

        self.sess.run(tf.global_variables_initializer())

        self.saver_Decom = tf.train.Saver(var_list = self.var_Decom)
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
#                 print("R_low", result_1)
                print("I_low", np.mean(result_2))
            if train_phase == "Relight":
                result_1, result_2, e1, e2, e3, E = self.sess.run([self.output_S, self.output_I_delta, self.output_e1, self.output_e2, self.output_e3, self.output_E], feed_dict={self.input_low: input_low_eval})
#                 print("S", result_1)
                print("I_delta", np.mean(result_2))
#                 print(result_2)
#                 save_images(os.path.join(sample_dir, 'eval_e_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_3)
                save_heatmap(os.path.join(sample_dir, 'eval_e1_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), e1)
                save_heatmap(os.path.join(sample_dir, 'eval_e2_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), e2)
                save_heatmap(os.path.join(sample_dir, 'eval_e3_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), e3)
                save_heatmap_e(os.path.join(sample_dir, 'eval_e_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), E)

#             if train_phase == "Relight_max":
#                 result_1, result_2 = self.sess.run([self.output_S_max, self.output_I_max], feed_dict={self.input_low: input_low_eval})
#                 print("S_max", result_1)
#                 print("I_max", np.mean(result_2))

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
                elif train_phase == "Relight":
#                     for t in range(5):
#                         _, D_loss, GP_loss = self.sess.run([self.train_op_D, self.loss_D, self.loss_GP],
#                                              feed_dict = {self.input_low: batch_input_low, self.input_high: batch_input_high, self.lr: lr[epoch]})
#                     _, G_loss = self.sess.run([self.train_op_G, self.loss_G],
#                                          feed_dict = {self.input_low: batch_input_low, self.input_high: batch_input_high, self.lr: lr[epoch]})
                    _, D_loss, GP_loss = self.sess.run([self.train_op_D, self.loss_D, self.loss_GP], feed_dict = {self.input_low: batch_input_low, self.input_high: batch_input_high, self.lr: lr[epoch]})
                    for i in range(1):
                        _, G_loss, tv_loss = self.sess.run([self.train_op_G, self.loss_G, self.tv_loss],
                                                 feed_dict = {self.input_low: batch_input_low, self.input_high: batch_input_high, self.lr: lr[epoch]})
                    real, fake = self.sess.run([self.real_prob, self.fake_prob], feed_dict = {self.input_low: batch_input_low, self.input_high: batch_input_high, self.lr: lr[epoch]})
                    print("%s Epoch: [%2d] [%4d/%4d], D_loss: %.6f G_loss: %.6f GP_loss: %.6f tv_loss: %.6f" % \
                         (train_phase, epoch+1, batch_id+1, numBatch, D_loss, G_loss, GP_loss, tv_loss))
                    
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
            [R_low, I_low, I_delta, S, e1, e2, e3, E] = self.sess.run([self.output_R_low, self.output_I_low, self.output_I_delta, self.output_S, self.output_e1, self.output_e2, self.output_e3, self.output_E], feed_dict = {self.input_low: input_low_test})


            if decom_flag == 1:
                print("save decom img")
                print("(min,max)")
                print(np.min(np.squeeze(I_delta)))
                print(np.max(np.squeeze(I_delta)))
                E = np.log(E)
                save_images(os.path.join(save_dir, "R_low", name + "_R_low." + suffix), R_low)
                save_images(os.path.join(save_dir, "I_low", name + "_I_low." + suffix), I_low)
                save_images(os.path.join(save_dir, "I_delta", name + "_I_delta." + suffix), I_delta)
                save_heatmap(os.path.join(save_dir, "e", name + "_e1." + suffix), e1)
                save_heatmap(os.path.join(save_dir, "e", name + "_e2." + suffix), e2)
                save_heatmap(os.path.join(save_dir, "e", name + "_e3." + suffix), e3)
                save_heatmap(os.path.join(save_dir, "e", name + "_E." + suffix), E)
            save_images(os.path.join(save_dir, "S_delta", name + "."   + suffix), S)