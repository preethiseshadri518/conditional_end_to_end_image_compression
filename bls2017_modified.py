# Copyright 2018 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic nonlinear transform coder for RGB images.
This is a close approximation of the image compression model published in:
J. Ballé, V. Laparra, E.P. Simoncelli (2017):
"End-to-end Optimized Image Compression"
Int. Conf. on Learning Representations (ICLR), 2017
https://arxiv.org/abs/1611.01704
With patches from Victor Xing <victor.t.xing@gmail.com>
This is meant as 'educational' code - you can use this to get started with your
own experiments. To reproduce the exact results from the paper, tuning of hyper-
parameters may be necessary. To compress images with published models, see
`tfci.py`.
"""

import argparse
import csv
import glob
import sys
from os import path

from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow.compat.v1 as tf

import tensorflow_compression as tfc


def read_png(filename):
  """Loads a PNG image file."""
  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  return image

# Ideally would combine into one function with above, but had issues
# getting this work with the dataset.map(lambda x: read_png(filename, label, condition))
def read_png_conditional(filename, label):
  """Loads a PNG image file."""
  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label


def quantize_image(image):
  image = tf.round(image * 255)
  image = tf.saturate_cast(image, tf.uint8)
  return image


def write_png(filename, image):
  """Saves an image to a PNG file."""
  image = quantize_image(image)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)


class AnalysisTransform(tf.keras.layers.Layer):
  """The analysis transform."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(AnalysisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (9, 9), name="layer_0", corr=True, strides_down=4,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_0")),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="gdn_1")),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
            padding="same_zeros", use_bias=False,
            activation=None),
    ]
    super(AnalysisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor


class SynthesisTransform(tf.keras.layers.Layer):
  """The synthesis transform."""

  def __init__(self, num_filters, *args, **kwargs):
    self.num_filters = num_filters
    super(SynthesisTransform, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self._layers = [
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_0", inverse=True)),
        tfc.SignalConv2D(
            self.num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
            padding="same_zeros", use_bias=True,
            activation=tfc.GDN(name="igdn_1", inverse=True)),
        tfc.SignalConv2D(
            3, (9, 9), name="layer_2", corr=False, strides_up=4,
            padding="same_zeros", use_bias=True,
            activation=None),
    ]
    super(SynthesisTransform, self).build(input_shape)

  def call(self, tensor):
    for layer in self._layers:
      tensor = layer(tensor)
    return tensor

def conv_block(x, num_filters, i, kernel_size=3, strides=2):
  # not sure if naming layers is necessary, but thought it might be related to key not found in checkpoint error
  # note: it does not fix the issue
  x = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        name='conv_'+str(i))(x)

  x = tf.keras.layers.BatchNormalization(name='bn_'+str(i))(x)
  x = tf.keras.layers.Activation(activation='relu', name='activ_'+str(i))(x)
  return x

def deconv_block(x, num_filters, i, kernel_size=3, strides=2):
  x = tf.keras.layers.Conv2DTranspose(filters=num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  name='convT_'+str(i))(x)

  x = tf.keras.layers.BatchNormalization(name='bnT_'+str(i))(x)
  x = tf.keras.layers.Activation(activation='relu', name='activT_'+str(i))(x)
  return x


def cond_norm_conv_block(x, label, num_filters, kernel_size=3, strides=2):
  gamma = tf.keras.layers.Dense(num_filters)(label)
  gamma = tf.keras.layers.Reshape((1,1,num_filters))(gamma) # prob not necessary with broadcasting, but just in case
  beta = tf.keras.layers.Dense(num_filters)(label)
  beta = tf.keras.layers.Reshape((1,1,num_filters))(beta)

  x = tf.keras.layers.Conv2D(filters=num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same')(x)

  x = tf.keras.layers.BatchNormalization(center=False, scale=False)(x)
  x = tf.keras.layers.Multiply()([x, gamma]) # scale by gamma
  x = tf.keras.layers.Add()([x, beta]) # shift/offset with beta
  x = tf.keras.layers.Activation(activation='relu')(x)
  return x


def cond_norm_deconv_block(x, label, num_filters, kernel_size=3, strides=2):
  gamma = tf.keras.layers.Dense(num_filters)(label)
  gamma = tf.keras.layers.Reshape((1,1,num_filters))(gamma) # prob not necessary with broadcasting, but just in case
  beta = tf.keras.layers.Dense(num_filters)(label)
  beta = tf.keras.layers.Reshape((1,1,num_filters))(beta)

  x = tf.keras.layers.Conv2DTranspose(filters=num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same')(x)

  x = tf.keras.layers.BatchNormalization(center=False, scale=False)(x)
  x = tf.keras.layers.Multiply()([x, gamma]) # scale by gamma
  x = tf.keras.layers.Add()([x, beta]) # shift/offset with beta
  x = tf.keras.layers.Activation(activation='relu')(x)
  return x


def InferenceNetwork(input_size, conditional, num_classes, num_filters, filter_dims):
  print('INPUTS:', input_size, conditional, num_classes, num_filters, filter_dims)
  # conditional means using conditional batch norm approach
  encoder_input = tf.keras.Input(shape=input_size, name='encoder_input')
  if conditional: # args.conditional
    label = tf.keras.Input(shape=(num_classes,))
    layer = cond_norm_conv_block(encoder_input, label, num_filters[0], kernel_size=filter_dims[0])
    layer = cond_norm_conv_block(layer, label, num_filters[1], kernel_size=filter_dims[1])
    layer = x = tf.keras.layers.Conv2D(filters=num_filters[2],
          kernel_size=filter_dims[2],
          strides=2,
          padding='same')(layer)

    return tf.keras.Model([encoder_input, label], layer)
  else:
    layer = conv_block(encoder_input, num_filters[0], 0, kernel_size=filter_dims[0])
    layer = conv_block(layer, num_filters[1], 1, kernel_size=filter_dims[1])
    layer = x = tf.keras.layers.Conv2D(filters=num_filters[2],
          kernel_size=filter_dims[2],
          strides=2,
          padding='same',
          name='conv_2')(layer)

    return tf.keras.Model(encoder_input, layer)


def GenerativeNetwork(input_size, conditional, num_classes, num_filters, filter_dims):
  print('INPUTS:', input_size, conditional, num_classes, num_filters, filter_dims)
  # conditional means using conditional batch norm approach
  decoder_input = tf.keras.Input(shape=input_size, name='decoder_input')
  if conditional:
    label = tf.keras.Input(shape=(num_classes,))
    layer = cond_norm_deconv_block(decoder_input, label, num_filters[2], kernel_size=filter_dims[2])
    layer = cond_norm_deconv_block(layer, label, num_filters[1], kernel_size=filter_dims[1])
    layer = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=filter_dims[0], strides=2, padding='same')(layer)

    return tf.keras.Model([decoder_input, label], layer)
  else:
    layer = deconv_block(decoder_input, num_filters[2], 0, kernel_size=filter_dims[2])
    layer = deconv_block(layer, num_filters[1], 1, kernel_size=filter_dims[1])
    layer = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=filter_dims[0], strides=2, padding='same', name='output')(layer)

    return tf.keras.Model(decoder_input, layer)


def train(args):
  """Trains the model."""
  print('num_filters', args.num_filters)
  print('filter_dims', args.filter_dims)
  print('patchsize', args.patchsize)
  print('conditional', args.conditional)
  print('num_classes', args.num_classes)

  # WHaving trouble using a lambda function accpeting arguments for patchsize and conditional bool
  # TODO: See if there's a cleaner way to implement this
  def crop(image):
      return tf.random_crop(image, (args.patchsize, args.patchsize, 3))

  def crop_conditional(image, label):
      return tf.random_crop(image, (args.patchsize, args.patchsize, 3)), label

  if args.verbose:
    tf.logging.set_verbosity(tf.logging.INFO)

  # Create input data pipeline.
  with tf.device("/cpu:0"):
    train_files = glob.glob(args.train_glob)
    if not train_files:
      raise RuntimeError(
          "No training images found with glob '{}'.".format(args.train_glob))

    if args.conditional:
      print('inside conditional')
      # Assumes label are part of filename, e.g. {common_path}/{class_num}_{image_num}.jpg
      # TODO: Should improve this, but not sure what to do instead?
      labels = np.array([int(x.split('/')[-1].split('_')[0]) for x in train_files])
      labels = tf.one_hot(labels, args.num_classes)
      print(labels.shape)

      train_dataset = tf.data.Dataset.from_tensor_slices((train_files, labels))
      train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
      # TODO: combine functions for conditional and unconditional if possible
      train_dataset = train_dataset.map(
            read_png_conditional, num_parallel_calls=args.preprocess_threads)
      train_dataset = train_dataset.map(
            crop_conditional, num_parallel_calls=args.preprocess_threads)

      train_dataset = train_dataset.batch(args.batchsize)
      train_dataset = train_dataset.prefetch(32)
      print(train_dataset)
      x, label = train_dataset.make_one_shot_iterator().get_next()
      print('Final shape:', x.shape, label.shape)

    else:
      print('inside unconditional')
      train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
      train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
      train_dataset = train_dataset.map(
            read_png, num_parallel_calls=args.preprocess_threads)
      train_dataset = train_dataset.map(
            crop, num_parallel_calls=args.preprocess_threads)

      train_dataset = train_dataset.batch(args.batchsize)
      train_dataset = train_dataset.prefetch(32)
      print(train_dataset)
      x = train_dataset.make_one_shot_iterator().get_next()
      print('Final shape:', x.shape)

  print('data processing is done')
  encoder_input = (args.patchsize, args.patchsize, 3)
  if args.conditional:
    y = InferenceNetwork(encoder_input, args.conditional, args.num_classes, args.num_filters, args.filter_dims)([x, label])
    print('y', y.shape)
    entropy_bottleneck = tfc.EntropyBottleneck()
    y_tilde, likelihoods = entropy_bottleneck(y, training=True)
    print('y_tilde', y_tilde.shape)
    x_tilde = GenerativeNetwork(y_tilde.shape[1:], args.conditional, args.num_classes, args.num_filters, args.filter_dims)([y_tilde, label])
    print('x_tilde', x_tilde.shape)
  else:
    y = InferenceNetwork(encoder_input, args.conditional, args.num_classes, args.num_filters, args.filter_dims)(x)
    print('y', y.shape)
    entropy_bottleneck = tfc.EntropyBottleneck()
    y_tilde, likelihoods = entropy_bottleneck(y, training=True)
    print('y_tilde', y_tilde.shape)
    x_tilde = GenerativeNetwork(y_tilde.shape[1:], args.conditional, args.num_classes, args.num_filters, args.filter_dims)(y_tilde)
    print('x_tilde', x_tilde.shape)

  num_pixels = args.batchsize * args.patchsize ** 2

  # Total number of bits divided by number of pixels.
  train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

  # Mean squared error across pixels.
  train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
  # Multiply by 255^2 to correct for rescaling.
  train_mse *= 255 ** 2

  # The rate-distortion cost.
  train_loss = args.lmbda * train_mse + train_bpp

  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  main_step = main_optimizer.minimize(train_loss, global_step=step)

  aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
  aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

  train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

  tf.summary.scalar("loss", train_loss)
  tf.summary.scalar("bpp", train_bpp)
  tf.summary.scalar("mse", train_mse)

  tf.summary.image("original", quantize_image(x))
  tf.summary.image("reconstruction", quantize_image(x_tilde))

  hooks = [
      tf.train.StopAtStepHook(last_step=args.last_step),
      tf.train.NanTensorHook(train_loss),
  ]

  with tf.train.MonitoredTrainingSession(
      hooks=hooks, checkpoint_dir=args.checkpoint_dir,
      save_checkpoint_secs=300, save_summaries_secs=60) as sess:
    while not sess.should_stop():
      sess.run(train_op)


def compress(args):
  """Compresses an image."""

  # Load input image and add batch dimension.
  x = read_png(args.input_file)
  x = tf.expand_dims(x, 0)
  x.set_shape([1, None, None, 3])
  x_shape = tf.shape(x)

  # Instantiate model.
  analysis_transform = AnalysisTransform(args.num_filters)
  entropy_bottleneck = tfc.EntropyBottleneck()
  synthesis_transform = SynthesisTransform(args.num_filters)

  # Transform and compress the image.
  y = analysis_transform(x)
  string = entropy_bottleneck.compress(y)

  # Transform the quantized image back (if requested).
  y_hat, likelihoods = entropy_bottleneck(y, training=False)
  x_hat = synthesis_transform(y_hat)
  x_hat = x_hat[:, :x_shape[1], :x_shape[2], :]

  num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

  # Total number of bits divided by number of pixels.
  eval_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_pixels)

  # Bring both images back to 0..255 range.
  x *= 255
  x_hat = tf.clip_by_value(x_hat, 0, 1)
  x_hat = tf.round(x_hat * 255)

  mse = tf.reduce_mean(tf.squared_difference(x, x_hat))
  psnr = tf.squeeze(tf.image.psnr(x_hat, x, 255))
  msssim = tf.squeeze(tf.image.ssim_multiscale(x_hat, x, 255))

  with tf.Session() as sess:
    # Load the latest model checkpoint, get the compressed string and the tensor
    # shapes.
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    tensors = [string, tf.shape(x)[1:-1], tf.shape(y)[1:-1]]
    arrays = sess.run(tensors)

    # Write a binary file with the shape information and the compressed string.
    packed = tfc.PackedTensors()
    packed.pack(tensors, arrays)
    with open(args.output_file, "wb") as f:
      f.write(packed.string)

    # If requested, transform the quantized image back and measure performance.
    if args.verbose:
      eval_bpp, mse, psnr, msssim, num_pixels = sess.run(
          [eval_bpp, mse, psnr, msssim, num_pixels])

      # The actual bits per pixel including overhead.
      bpp = len(packed.string) * 8 / num_pixels

      print("Mean squared error: {:0.4f}".format(mse))
      print("PSNR (dB): {:0.2f}".format(psnr))
      print("Multiscale SSIM: {:0.4f}".format(msssim))
      print("Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssim)))
      print("Information content in bpp: {:0.4f}".format(eval_bpp))
      print("Actual bits per pixel: {:0.4f}".format(bpp))


def decompress(args):
  """Decompresses an image."""

  # Read the shape information and compressed string from the binary file.
  string = tf.placeholder(tf.string, [1])
  x_shape = tf.placeholder(tf.int32, [2])
  y_shape = tf.placeholder(tf.int32, [2])
  with open(args.input_file, "rb") as f:
    packed = tfc.PackedTensors(f.read())
  tensors = [string, x_shape, y_shape]
  arrays = packed.unpack(tensors)

  # Instantiate model.
  entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)
  synthesis_transform = SynthesisTransform(args.num_filters)

  # Decompress and transform the image back.
  y_shape = tf.concat([y_shape, [args.num_filters]], axis=0)
  y_hat = entropy_bottleneck.decompress(
      string, y_shape, channels=args.num_filters)
  x_hat = synthesis_transform(y_hat)

  # Remove batch dimension, and crop away any extraneous padding on the bottom
  # or right boundaries.
  x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]

  # Write reconstructed image out as a PNG file.
  op = write_png(args.output_file, x_hat)

  # Load the latest model checkpoint, and perform the above actions.
  with tf.Session() as sess:
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    sess.run(op, feed_dict=dict(zip(tensors, arrays)))


def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--verbose", "-V", action="store_true",
      help="Report bitrate and distortion when training or compressing.")
  parser.add_argument("--num_filters", nargs=3, type=int, default=[64, 64, 64],
      help="Number of filters per layer; should be 3 integers")
  parser.add_argument('--filter_dims', nargs=3, type=int, default=[5, 5, 5],
      help='dimensions of filters in the encoding/analysis network; should be 3 integers')
  parser.add_argument('--conditional', type=bool, default=False,
      help='whether or not to use conditional architecture')
  parser.add_argument('--num_classes', type=int, default=1,
        help='number of labels/classes for conditional architecture')
  parser.add_argument(
      "--checkpoint_dir", default="train",
      help="Directory where to save/load model checkpoints.")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "PNG format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options. Invoke '<command> -h' for more information.")

  # 'train' subcommand.
  train_cmd = subparsers.add_parser(
      "train",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Trains (or continues to train) a new model.")
  train_cmd.add_argument(
      "--train_glob", default="images/*.png",
      help="Glob pattern identifying training data. This pattern must expand "
           "to a list of RGB images in PNG format.")
  train_cmd.add_argument(
      "--batchsize", type=int, default=8,
      help="Batch size for training.")
  train_cmd.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training.")
  train_cmd.add_argument(
      "--lambda", type=float, default=0.01, dest="lmbda",
      help="Lambda for rate-distortion tradeoff.")
  train_cmd.add_argument(
      "--last_step", type=int, default=1000000,
      help="Train up to this number of steps.")
  train_cmd.add_argument(
      "--preprocess_threads", type=int, default=16,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")

  # 'compress' subcommand.
  compress_cmd = subparsers.add_parser(
      "compress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a PNG file, compresses it, and writes a TFCI file.")

  # 'decompress' subcommand.
  decompress_cmd = subparsers.add_parser(
      "decompress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a TFCI file, reconstructs the image, and writes back "
                  "a PNG file.")

  # Arguments for both 'compress' and 'decompress'.
  for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
    cmd.add_argument(
        "input_file",
        help="Input filename.")
    cmd.add_argument(
        "output_file", nargs="?",
        help="Output filename (optional). If not provided, appends '{}' to "
             "the input filename.".format(ext))

  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args


def main(args):
  # Invoke subcommand.
  if args.command == "train":
    train(args)
  elif args.command == "compress":
    if not args.output_file:
      args.output_file = args.input_file + ".tfci"
    compress(args)
  elif args.command == "decompress":
    if not args.output_file:
      args.output_file = args.input_file + ".png"
    decompress(args)


if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
