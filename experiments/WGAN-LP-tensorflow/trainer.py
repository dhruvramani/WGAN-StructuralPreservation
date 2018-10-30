import os
import tensorflow as tf
import numpy as np
import scipy.misc 
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

from data_generator import *
from model import Generator, Critic
from reg_losses import get_regularization_term

slim = tf.contrib.slim

__eval_step_list__ = [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 15000, 20000]

flags = tf.app.flags
flags.DEFINE_integer("n_epoch", 1, "Epoch to train [20000]") # TODO Change
flags.DEFINE_integer("n_batch_size", 256, "Batch size to train [256]")
flags.DEFINE_integer("latent_dimensionality", 128, "Dimensionality of the latent variables [2]")

"""
During training, 10 critic updates are performed for every generator update,
except for the first 25 generator updates,
where the critic is updated 100 times for each generator update
in order to get closer to the optimal critic in the beginning of training.
"""

flags.DEFINE_integer("begining_init_step", 25, "[25]")
flags.DEFINE_integer("n_c_iters_under_begining_init_step", 100, "[100]")
flags.DEFINE_integer("n_c_iters_over_begining_init_step", 10, "[10]")
flags.DEFINE_integer("interval_record_earth_mover", 10, "[10]")

flags.DEFINE_float("learning_rate", 5e-5, "Learning rate of optimizer [5e-5]")
flags.DEFINE_float("Lambda", 5., "Weights for critics' regularization term [5]")
flags.DEFINE_string("Regularization_type", "LP", "[no_reg, no_reg_but_clipping, LP, GP]")
flags.DEFINE_string("Purturbation_type", "dragan_only_training",
                    "[no_purf, wgan_gp, dragan_only_training, dragan_both]")
flags.DEFINE_string("dataset", 'GeneratorSwissRoll',
                    "Which dataset is used? [GeneratorGaussians8, GeneratorGaussians25, GeneratorSwissRoll]")

flags.DEFINE_string("critic_variable_scope_name", "Critic", "[Critic]")
flags.DEFINE_string("generator_variable_scope_name", "Generator", "Generator")

flags.DEFINE_bool("emd_records", False, "Whether EMD is recorded. (It takes some time...)[True, False]")
FLAGS = flags.FLAGS


class Trainer(object):
    def __init__(self):
        self.dataset_generator = None
        self.real_input = None

        self.z = None

        self.generator = None
        self.critic_x = None
        self.critic_gz = None

        self.g_loss = None
        self.c_negative_loss = None
        self.c_regularization_loss = None
        self.c_loss = None
        self.c_clipping = None
        self.x_hat = None

        self.ckpt_dir = None
        self.summary_writer = None
        self.c_summary_op = None
        self.g_summary_op = None
        self.emd_placeholder = None
        self.emd_summary = None

        self.saver = None

        self.step = None

        self.sess = None
        self.step_inc = None
        self.g_opt = None
        self.c_opt = None

        self.g_update_fetch_dict = None
        self.c_update_fetch_dict = None
        self.c_feed_dict = None

        self.coord = None
        self.threads = None
        self.feed_data = None

        self.define_dataset()
        self.define_latent()
        self.define_model()
        self.define_loss()
        self.define_optim()
        self.define_writer_and_summary()
        self.define_saver()
        self.initialize_session_and_etc()
        self.define_feed_and_fetch()

    def define_dataset(self):
        self.feed_data = CelebAFeed("./dataset/celebA_redux/train", FLAGS.n_batch_size)
        shape = self.feed_data.get_img_shape()
        self.dataset_generator = iter(self.feed_data)
        self.real_input = tf.placeholder(tf.float32, shape=(None, shape[0], shape[1], shape[2]))

    def define_latent(self):
        self.z = tf.random_normal([FLAGS.n_batch_size, FLAGS.latent_dimensionality], mean=0.0, stddev=1.0, name='z')

    def define_model(self):
        self.generator = Generator(self.z,
                                   variable_scope_name=FLAGS.generator_variable_scope_name)
        self.critic_x = Critic(self.real_input,
                               variable_scope_name=FLAGS.critic_variable_scope_name)
        self.critic_gz = Critic(self.generator.output_tensor,
                                variable_scope_name=FLAGS.critic_variable_scope_name,
                                reuse=True)

    def define_loss(self):
        self.g_loss = -tf.reduce_mean(self.critic_gz.output_tensor)
        self.c_negative_loss = -self.g_loss - tf.reduce_mean(self.critic_x.output_tensor)
        if FLAGS.Regularization_type == 'no_reg_but_clipping' or \
           FLAGS.Regularization_type == 'no_reg':
            self.c_regularization_loss = tf.Variable(0., trainable=False)
        else:
            self.c_regularization_loss, self.x_hat = get_regularization_term(
                                                training_samples=self.real_input,
                                                generated_samples=self.generator.output_tensor,
                                                reg_type=FLAGS.Regularization_type,
                                                per_type=FLAGS.Purturbation_type,
                                                critic_variable_scope_name=FLAGS.critic_variable_scope_name
                                                )

        self.c_loss = self.c_negative_loss + FLAGS.Lambda * self.c_regularization_loss

    def define_optim(self):
        self.step = tf.Variable(0, name='step', trainable=False)
        self.step_inc = tf.assign(self.step, self.step + 1)

        optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)

        self.g_opt = optimizer.minimize(self.g_loss, var_list=self.generator.var_list)
        self.c_opt = optimizer.minimize(self.c_loss, var_list=self.critic_x.var_list)

        with tf.control_dependencies([self.c_opt]):
            if FLAGS.Regularization_type == 'no_reg_but_clipping':
                self.c_clipping = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.critic_x.var_list]
            else:
                self.c_clipping = [tf.no_op()]

    def define_writer_and_summary(self):
        self.ckpt_dir = ''.join(['ckpts/',
                                 FLAGS.dataset+'_',
                                 FLAGS.Regularization_type+'_',
                                 FLAGS.Purturbation_type+'_',
                                 str(FLAGS.Lambda)+'_',
                                 str(FLAGS.emd_records),
                                 '/'])

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.summary_writer = tf.summary.FileWriter(self.ckpt_dir)

        self.c_summary_op = tf.summary.merge([
            tf.summary.scalar('loss/c', self.c_loss),
            tf.summary.scalar('loss/c_negative_loss', self.c_negative_loss),
            tf.summary.scalar('loss/c_regularization_loss', self.c_regularization_loss)
        ])
        self.g_summary_op = tf.summary.merge([
            tf.summary.scalar('loss/g', self.g_loss)
        ])

        self.emd_placeholder = tf.placeholder(tf.float32, shape=())
        self.emd_summary = tf.summary.scalar('EMD', self.emd_placeholder)

    def define_saver(self):
        self.saver = tf.train.Saver()

    def initialize_session_and_etc(self):
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=gpu_options)
        self.sess = tf.Session(config=sess_config)

        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def define_feed_and_fetch(self):
        self.g_update_fetch_dict = {
            "opt": self.g_opt,
            "z": self.z,
            "G_z": self.generator.output_tensor,
            "loss": self.g_loss,
            'summary': self.g_summary_op,
            "step": self.step
        }

        self.c_update_fetch_dict = {
            'gradient_clipping': self.c_clipping,
            "x": self.real_input,
            "G_z": self.generator.output_tensor,
            "loss": self.c_loss,
            "negative_loss": self.c_negative_loss,
            "regularization_loss": self.c_regularization_loss,
            'summary': self.c_summary_op,
            "step": self.step
        }

        self.c_feed_dict = {
            self.real_input: None
        }

    def train(self):
        try:
            c_fetch_dict = None
            print("[.] Learning Start...")
            step = 0

            if os.path.isfile("./logs/critic_train_loss.log"):
                os.remove("./logs/critic_train_loss.log")
            
            if os.path.isfile("./logs/generator_train_loss.log"):
                os.remove("./logs/generator_train_loss.log")

            while not self.coord.should_stop():
                if step > FLAGS.n_epoch:
                    break

                self.c_feed_dict[self.real_input] = next(self.dataset_generator)
                step = self.sess.run(self.step)

                n_c_iters = (FLAGS.n_c_iters_under_begining_init_step
                             if step < FLAGS.begining_init_step
                             else FLAGS.n_c_iters_over_begining_init_step)

                c_fetch_dict = None
                for i in range(n_c_iters):
                    c_fetch_dict = self.sess.run(self.c_update_fetch_dict, feed_dict=self.c_feed_dict)
                    print("Critic - epoch : {}, step: {}, loss: {}, reg_loss : {}".format(step, i, c_fetch_dict["loss"], c_fetch_dict["regularization_loss"]), end="\r")

                print("\n")
                g_fetch_dict = self.sess.run(self.g_update_fetch_dict)
                print("Generator - epoch : {}, loss: {}".format(step, g_fetch_dict["loss"]))

                # NOTE : if(step % 100 == 0): # /100
                scipy.misc.imsave("./{}.png".format(int(step)), g_fetch_dict["G_z"][0])
                self.summary_writer.add_summary(c_fetch_dict["summary"], c_fetch_dict["step"])
                self.summary_writer.add_summary(g_fetch_dict["summary"], g_fetch_dict["step"])
                self.summary_writer.flush()

                with open("./logs/critic_train_loss.log", "a+") as file:
                    file.write("{}\n".format(c_fetch_dict["loss"]))

                with open("./logs/generator_train_loss.log", "a+") as file:
                    file.write("{}\n".format(g_fetch_dict["loss"]))

                '''
                if step in __eval_step_list__:
                    self.draw_level_sets(step)

                if FLAGS.emd_records and step % FLAGS.interval_record_earth_mover == 0 and step != 0:
                    self.estimate_earth_mover_distance(step)
                ''' 

                self.sess.run(self.step_inc)

        except KeyboardInterrupt:
            print("Interrupted")
            self.coord.request_stop()
        finally:
            self.saver.save(self.sess, self.ckpt_dir)
            print('Stop')
            self.coord.request_stop()
            self.coord.join(self.threads)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    trainer.sess.close()
