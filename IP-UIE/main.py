from __future__ import print_function
import os
import argparse
from glob import glob

from PIL import Image
import tensorflow as tf

from model import lowlight_enhance
from utils import *


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

parser = argparse.ArgumentParser(description='')

parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default="9", help='GPU idx')
parser.add_argument('--gpu_mem', dest='gpu_mem', type=float, default=0.9, help="0 to 1, gpu memory usage")
parser.add_argument('--phase', dest='phase', default='train', help='train or test')

parser.add_argument('--decom_epoch', dest='decom_epoch', type=int, default=100, help='number of total epoches')
parser.add_argument('--relight_epoch', dest='relight_epoch', type=int, default=80, help='number of total epoches')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='patch size')
parser.add_argument('--start_lr', dest='start_lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', default=20, help='evaluating and saving checkpoints every #  epoch')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint/', help='directory for checkpoints')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample/', help='directory for evaluating outputs')
parser.add_argument('--save_dir', dest='save_dir', default='./test_results/', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='../DATA/uw_ll/test/', help='directory for testing inputs')
parser.add_argument('--decom', dest='decom', default=1, help='decom flag, 0 for enhanced results only and 1 for decomposition results')

args = parser.parse_args()

def lowlight_train(lowlight_enhance):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    
    global_step = tf.Variable(0, trainable=False)
#     lr = args.start_lr * np.ones([args.epoch])
    lr_Decom = args.start_lr * np.ones([args.decom_epoch])
    lr_Relight = args.start_lr/10.0 * np.ones([args.relight_epoch])
#     lr_Relight = tf.train.exponential_decay(args.start_lr, global_step=global_step, decay_steps=400, decay_rate=0.9)
    # 学习率衰减 
    lr_Decom[40:] = lr_Decom[0] / 10.0

    train_low_data = []
    train_high_data = []

    train_low_data_names = glob('../DATA/uw_ll/syn/low/*.jpg')
    train_low_data_names.sort()
    train_high_data_names = glob('../DATA/uw_ll/syn/normal/*.jpg')
    train_high_data_names.sort()
    assert len(train_low_data_names) == len(train_high_data_names)
    print('[*] Number of training data: %d' % len(train_low_data_names))

    for idx in range(len(train_low_data_names)):
        low_im = load_images(train_low_data_names[idx])
        train_low_data.append(low_im)
        high_im = load_images(train_high_data_names[idx])
        train_high_data.append(high_im)

    eval_low_data = []
    eval_high_data = []

    eval_low_data_name = glob('../DATA/uw_ll/val/low/*.*')

    for idx in range(len(eval_low_data_name)):
        eval_low_im = load_images(eval_low_data_name[idx])
        eval_low_data.append(eval_low_im)

    lowlight_enhance.train(train_low_data, train_high_data, eval_low_data, batch_size=args.batch_size, patch_size=args.patch_size, epoch=int(args.decom_epoch), lr=lr_Decom, sample_dir=args.sample_dir, ckpt_dir=os.path.join(args.ckpt_dir, 'Decom'), eval_every_epoch=int(args.eval_every_epoch), train_phase="Decom")

    lowlight_enhance.train(train_low_data, train_high_data, eval_low_data, batch_size=args.batch_size, patch_size=args.patch_size, epoch=int(args.relight_epoch), lr=lr_Relight, sample_dir=args.sample_dir, ckpt_dir=os.path.join(args.ckpt_dir, 'Relight'), eval_every_epoch=int(args.eval_every_epoch), train_phase="Relight")

def lowlight_test(lowlight_enhance):
    if args.test_dir == None:
        print("[!] please provide --test_dir")
        exit(0)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        os.makedirs(args.save_dir + "/R_low")
        os.makedirs(args.save_dir + "/I_low")
        os.makedirs(args.save_dir + "/I_delta")
        os.makedirs(args.save_dir + "/S_delta")
        os.makedirs(args.save_dir + "/e")

    test_low_data_name = glob(os.path.join(args.test_dir) + '/*.*')
    test_low_data = []
    test_high_data = []
    for idx in range(len(test_low_data_name)):
        test_low_im = load_images(test_low_data_name[idx])
        test_low_data.append(test_low_im)
    
    start = time.time()
    lowlight_enhance.test(test_low_data, test_high_data, test_low_data_name, load_dir=args.ckpt_dir, save_dir=args.save_dir, decom_flag=args.decom)
    print("End---------------,time:{},average{}".format(time.time() - start, (time.time()-start)/50))


def main(_):
    if args.use_gpu:
        print("[*] GPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
#         gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
#         with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        with tf.Session(config=config) as sess:
            model = lowlight_enhance(sess)
            if args.phase == 'train':
                lowlight_train(model)
            elif args.phase == 'test':
                lowlight_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)
    else:
        print("[*] CPU\n")
        with tf.Session() as sess:
            model = lowlight_enhance(sess)
            if args.phase == 'train':
                lowlight_train(model)
            elif args.phase == 'test':
                lowlight_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)

if __name__ == '__main__':
    import time
    tf.app.run()
