import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
def gen_mnist_train_tfrecord(mnist):

    # image, label = mnist.train.next_batch(batch_size)
    image = mnist.train.images
    label = np.argmax(mnist.train.labels,1)

    num_trains = mnist.train.num_examples
    num_shards = 2
    instances_per_shard = num_trains //num_shards

    for j in range(num_shards):
        writer = tf.python_io.TFRecordWriter \
            ("../data/mnist_train_tfrecord/mnist.tfrecords-%.5d-of-%.5d" %(j, num_shards))
        start = int(instances_per_shard*j)
        end = int(instances_per_shard*(j+1))
        image_shard = image[start:end]
        label_shard = label[start:end]

        for i in range(instances_per_shard):
            image_x = image_shard[i].tostring()
            label_y = label_shard[i]
            example = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_x])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label_y]))
            }))
            writer.write(example.SerializeToString())

        writer.close()

def gen_mnist_test_tfrecord(mnist):

    # image, label = mnist.train.next_batch(batch_size)
    image = mnist.test.images
    label = mnist.test.labels

    num_trains = mnist.test.num_examples
    num_shards = 1
    instances_per_shard = num_trains /num_shards

    for j in range(num_shards):
        writer = tf.python_io.TFRecordWriter \
            ("../data/mnist_train_tfrecord/mnist.tfrecords-%.5d-of-%.5d" %(j, num_shards))

        image = image[instances_per_shard*j,instances_per_shard*(j+1)]
        label = label[instances_per_shard*j,instances_per_shard*(j+1)]

        for i in range(instances_per_shard):
            image = image[i].eval().tostring()
            label = label[i].eval()
            example = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))
            writer.write(example.SerializeToString())

        writer.close()


if __name__ == '__main__':
    mnist = input_data.read_data_sets("../data/mnist_01/",one_hot=True)
    gen_mnist_train_tfrecord(mnist)