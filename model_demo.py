from __future__ import division
import os
import time
from sklearn.metrics import mean_absolute_error
import scipy.io as sio
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import csv
from ops_ import *
from utils_ import *
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow_probability as tfp
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

labels = np.random.choice([0, 1], size=(400,1))

batch_shape = 40


class graph2graph(object):
    def __init__(self, sess, test_dir, train_dir, graph_size, output_size, dataset,
                 batch_size=40, sample_size=40,
                 gf_dim=10, df_dim=10, L1_lambda=10000, L1_C=100000, additionl=0.001,
                 input_c_dim=1, output_c_dim=1,
                 checkpoint_dir=None, sample_dir=None, g_train_num=10, d_train_num=1, c_train_num=2, n_input=1225 * batch_shape,
                 n_hidden=600 * batch_shape, n_hidden1=24 * batch_shape, n_output=batch_shape, n_regions = 35):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the graphs. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input graph channel. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output graph channel. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.graph_size = graph_size
        self.sample_size = sample_size
        self.output_size = output_size
        self.g_train_num = g_train_num
        self.d_train_num = d_train_num
        self.c_train_num = c_train_num
        self.test_dir = test_dir
        self.train_dir = train_dir
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.labels = labels
        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim
        self.dataset = dataset
        self.L1_lambda = L1_lambda
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_hidden1 = n_hidden1
        self.n_output = n_output
        self.L1_c = L1_C
        self.additional = additionl
        self.n_regions = n_regions
        self.vectorized_graph = int((self.n_regions * (self.n_regions - 1)) /2) #outputs an upper triangular part of the graph
        self.fully_vectorized_graph = self.n_regions*self.n_regions

        # batch normalization
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e1 = batch_norm(name='g_bn_e1')
        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')

        ##########################################

        self.g_bn_e11 = batch_norm(name='l_bn_e11')
        self.g_bn_e22 = batch_norm(name='l_bn_e22')
        self.g_bn_e33 = batch_norm(name='l_bn_e33')
        self.g_bn_e44 = batch_norm(name='l_bn_e44')

        #########################################"

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')

        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):

        n_input = 1225
        n_hidden = 1000
        n_hidden1 = 100
        n_output = 1

        self.X = tf.placeholder(tf.float32)
        self.Y = tf.placeholder(tf.float32)
        # Weights
        self.W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0), name='c_w1')
        self.W2 = tf.Variable(tf.random_uniform([n_hidden, n_hidden1], -1.0, 1.0), name='c_w2')
        self.W3 = tf.Variable(tf.random_uniform([n_hidden1, n_output], -1.0, 1.0), name='c_w3')
        # Bias
        self.b1 = tf.Variable(tf.zeros([n_hidden]), name='c_b1')
        self.b2 = tf.Variable(tf.zeros([n_hidden1]), name='c_b2')
        self.b3 = tf.Variable(tf.zeros([n_output]), name='c_b2')

        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.graph_size[0], self.graph_size[1],
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_graphs')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]  # takes the first real graph
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]  # takes the target real graph
        self.fake_B = self.generator(self.real_A)
        self.latent = self.lat(self.real_A)
        self.upper_vector_A = self.upper_triangular_extractor(self.real_A)
        self.upper_vector_B = self.upper_triangular_extractor(self.fake_B)
        self.multiplex = tf.concat([self.upper_vector_A, self.latent, self.upper_vector_B], 1)
        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
        self.D, self.D_logits = self.discriminator(self.real_AB, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)
        self.hy = self.classification_arch(self.X, self.Y)
        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.fake_B_sum = tf.summary.histogram("fake_B", self.fake_B)
        self.latent_sum = tf.summary.histogram("latent", self.latent)
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(
            self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
                      + self.L1_lambda * tf.reduce_mean(
            tf.abs(self.real_AB - self.fake_AB))

        self.classifier_loss = tf.reduce_mean(-self.Y * tf.log(tf.maximum(self.hy, 1e-9)) - (1 - self.Y) * tf.log(tf.maximum(1 - self.hy, 1e-9))) \
                               * self.L1_c + self.additional

        self.l1_regularizer = tf.contrib.layers.l1_regularizer( scale=0.005, scope=None)
        weights = tf.trainable_variables()
        self.regularization_penalty = tf.contrib.layers.apply_regularization(self.l1_regularizer, weights)
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.l_sum = tf.summary.scalar("l_loss", self.latent)
        self.classifier_loss_sum = tf.summary.scalar("c_loss", self.classifier_loss)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.l_vars = [var for var in t_vars if 'l_' in var.name]
        self.c_vars = [var for var in t_vars if 'c_' in var.name]

        self.saver = tf.train.Saver()

    def load_random_samples(self, sample_dir):
        sample_data = load_data(sample_dir)
        sample = np.random.choice(sample_data, self.batch_size)
        sample_graphs = np.array(sample).astype(np.float32)
        return sample_graphs

    def sample_model(self, sample_dir, epoch, idx):
        sample_graphs = self.load_random_samples(sample_dir)
        samples, d_loss, g_loss = self.sess.run(
            [self.fake_B_sample, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_graphs}
        )

        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def demo(self, args):
        second_view = np.zeros((1, self.n_regions, self.n_regions, 1))
        tab = np.zeros((1))
        """Train pix2pix"""
        d_optim = tf.train.AdamOptimizer(args.lr_d, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)  # minimizing the discriminator's loss using Adam optimizer.

        g_optim = tf.train.AdamOptimizer(args.lr_g, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)  # minimizing the generator's loss using Adam optimizer.

        c_optim = tf.train.AdamOptimizer(args.lr_c, beta1=args.beta1) \
            .minimize(self.classifier_loss, var_list=self.c_vars)  # minimizing the generator's loss using Adam optimizer.

        init_op = tf.global_variables_initializer()  # initialize the variable.
        self.sess.run(init_op)  # running the initializer.
        self.g_sum = tf.summary.merge([self.d__sum,
                                       self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum,
                                       self.classifier_loss_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()
        data = load_data(args.train_dir, 'train', self.graph_size[0], self.dataset)  # load the train data.
        Kf = KFold(n_splits=10)
        second_counter = 0

        for X_index, Y_index in Kf.split(data):
            X_train = data[X_index]
            X_test = data[Y_index]
            Y_train = labels[X_index]
            Y_test = labels[Y_index]
            errD_fake = 0
            errD_real = 0
            best = 4500
            errC = 0
            best_dis = 2


            # load testing input
            print("Loading testing graphs ...")
            sample_graphs_all = X_test
            batch_idxs = min(len(sample_graphs_all), args.train_size) // self.batch_size
            if self.load(self.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

            for idxx in range(0, batch_idxs):
                sample_graphs = sample_graphs_all[idxx * self.batch_size:(idxx + 1) * self.batch_size]
                sample_graphs = np.array(sample_graphs)
                view1_test = sample_graphs[:, :, :, 0:1]
                print("sampling graph ", idxx)
                samples = self.sess.run(
                    self.fake_B,
                    feed_dict={self.real_data: sample_graphs})
                second_view = np.concatenate((second_view, samples), axis=0)
                np.save('second_view', second_view)

                multi_test = self.sess.run([self.latent], feed_dict={self.real_data: sample_graphs})
                multi_test = np.reshape(multi_test, (batch_shape, self.n_regions))

                v1_upper_test = self.sess.run([self.upper_vector_A], feed_dict={self.real_A: view1_test})
                v1_upper_test = np.reshape(v1_upper_test, (batch_shape, self.vectorized_graph))

                view2_test = np.reshape(samples, (batch_shape, self.n_regions, self.n_regions, 1))
                v2_upper_test = self.sess.run([self.upper_vector_B], feed_dict={self.fake_B: view2_test})
                v2_upper_test = np.reshape(v2_upper_test, (batch_shape, self.vectorized_graph))
                multiplex_test = np.concatenate((v1_upper_test, multi_test, v2_upper_test), axis=1)

                multiplex_final = np.zeros((1, self.fully_vectorized_graph))
                if batch_shape > 1:
                    for k in range(batch_shape):
                        multiplex_full = multiplex_test[k:k + 1, :]

                        multiplex_final = np.concatenate((multiplex_final, multiplex_full), axis=1)

                multiplex_final = multiplex_final[:, self.fully_vectorized_graph:(batch_shape + 1) * self.fully_vectorized_graph]

                for m in range(batch_shape):
                    multiplex_per_sub = multiplex_final[:, m * self.fully_vectorized_graph:(m + 1) * self.fully_vectorized_graph]
                    if m == 0:
                        joined_multiplex_per_sub = multiplex_per_sub
                    else:
                        joined_multiplex_per_sub = np.concatenate((joined_multiplex_per_sub, multiplex_per_sub), axis=0)

                if idxx == 0:
                    multiplex_per_batch = joined_multiplex_per_sub
                else:
                    multiplex_per_batch = np.concatenate((multiplex_per_batch, joined_multiplex_per_sub), axis=0)

            if second_counter == 0:
                joined_multiplex_per_batch = multiplex_per_batch
                y = Y_test
            else:
                joined_multiplex_per_batch = np.concatenate((joined_multiplex_per_batch, multiplex_per_batch), axis=0)
                y = np.concatenate((y, Y_test), axis=0)
                np.save('final_second_view.npy', joined_multiplex_per_batch)
            second_counter = second_counter + 1


    def discriminator(self, graph, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            h0 = lrelu(e2e(graph, self.df_dim, k_h=self.graph_size[0], name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(e2e(h0, self.df_dim * 2, k_h=self.graph_size[0], name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(e2n(h1, self.df_dim * 2, k_h=self.graph_size[0], name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(n2g(h2, self.df_dim * 2, k_h=self.graph_size[0], name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')
            return tf.nn.sigmoid(h4), h4

    def generator(self, graph, y=None):
        with tf.variable_scope("generator") as scope:

            e1 = self.g_bn_e1(e2e(lrelu(graph), self.gf_dim, k_h=self.graph_size[0], name='g_e1_conv'))

            e2 = self.g_bn_e2(e2e(lrelu(e1), self.gf_dim * 2, k_h=self.graph_size[0], name='g_e2_conv'))
            e2_ = tf.nn.dropout(e2, keep_prob=1)

            e3 = self.g_bn_e3(e2n(lrelu(e2_), self.gf_dim * 2, k_h=self.graph_size[0], name='g_e3_conv'))

            self.d2, self.d2_w, self.d2_b = de_e2n(tf.nn.relu(e3),
                                                   [self.batch_size, self.graph_size[0], self.graph_size[0], self.gf_dim * 2], k_h=self.graph_size[0],
                                                   name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), keep_prob=1)
            d2 = tf.concat([d2, e2], 3)

            self.d3, self.d3_w, self.d3_b = de_e2e(tf.nn.relu(d2),
                                                   [self.batch_size, self.graph_size[0], self.graph_size[0], int(self.gf_dim)],
                                                   k_h=self.graph_size[0], name='g_d3', with_w=True)
            d3 = self.g_bn_d3(self.d3)
            d3 = tf.concat([d3, e1], 3)

            self.d4, self.d4_w, self.d4_b = de_e2e(tf.nn.relu(d3),
                                                   [self.batch_size, self.graph_size[0], self.graph_size[0], self.output_c_dim],
                                                   k_h=self.graph_size[0], name='g_d4', with_w=True)

            return tf.add(tf.nn.relu(self.d4), graph)

    def upper_triangular_extractor(self, graph, y=None, reuse=True):
        reshaped_output_final = np.zeros((1, self.vectorized_graph))
        if batch_shape > 1:
            for i in range(batch_shape):
                output_graph = graph[i:i + 1, :, :, :]
                output_graph = tf.reshape(output_graph, [self.n_regions, self.n_regions])
                output = upper_triang(output_graph)
                reshaped_output = tf.reshape(output, [1, self.vectorized_graph])
                reshaped_output_final = tf.concat([reshaped_output_final, reshaped_output], 0)
        reshaped_output_final = reshaped_output_final[1:batch_shape + 1, :]
        return reshaped_output_final

    def lat(self, graph, y=None):
        with tf.variable_scope("lat") as scope:
            e1 = self.g_bn_e11(e2e(lrelu(graph), self.gf_dim, k_h=self.graph_size[0], name='l_e1'))
            e2 = self.g_bn_e22(e2e(lrelu(e1), self.gf_dim * 2, k_h=self.graph_size[0], name='l_e2'))
            e2_ = tf.nn.dropout(e2, keep_prob=1)
            e3 = self.g_bn_e33(e2n(lrelu(e2_), self.gf_dim * 2, k_h=self.graph_size[0], name='l_e3'))
            latent_space = np.zeros((batch_shape, self.n_regions))
            for i in range(20):
                extracted_latent = e3[:, :, :, i:i + 1]
                extracted_latent = tf.reshape(extracted_latent, [batch_shape, self.n_regions])
                latent_space = latent_space + extracted_latent

            return latent_space

    def save(self, checkpoint_dir, step):
        model_name = "g2g.model"
        model_dir = "%s" % ('flu')
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s" % ('flu')
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def classification_arch(self, x_data, y_data):
        # Dataset
        x_data = tf.cast(x_data, tf.float32)
        self.L2 = tf.sigmoid(tf.matmul(x_data, self.W1) + self.b1)
        self.L3 = tf.sigmoid(tf.matmul(self.L2, self.W2) + self.b2)
        classification_arch_output = tf.sigmoid(tf.matmul(self.L3, self.W3) + self.b3)
        return classification_arch_output

    def classi(self, input_hy, X, Y, epoch):
        epochs = epoch
        lr = 0.01
        display_step = 100

        x_data = X
        y_data = Y
        hy = input_hy
        X = tf.placeholder(tf.float32)
        Y = tf.placeholder(tf.float32)

        cost = tf.reduce_mean(-Y * tf.log(hy) - (1 - Y) * tf.log(1 - hy))
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

        init = tf.global_variables_initializer()
        tab = np.zeros(1)
        with tf.Session() as sess:
            sess.run(init)
            for step in range(epochs):
                _, c = sess.run([optimizer, cost], feed_dict={X: x_data, Y: y_data})
                if step % display_step == 0:
                    print("Cost: ", c)
                if step % 2:
                    errg = np.array(c)
                    errg = np.reshape(errg, (1))
                    tab = np.concatenate((tab, errg), axis=0)
            answer = tf.equal(tf.floor(hy + 0.1), Y)
            accuracy = tf.reduce_mean(tf.cast(answer, "float"))
            print(sess.run([hy], feed_dict={X: x_data, Y: y_data}))
            accuracy_evaluation = accuracy.eval({X: x_data, Y: y_data})
            print("Accuracy: ", accuracy.eval({X: x_data, Y: y_data}))
        return accuracy_evaluation

