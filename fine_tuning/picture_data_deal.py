import numpy

import tensorflow as tf
import os
import numpy as np
# import tensorflow.contrib.slim.python.slim.nets.inception_v3

TEST_DATA_PERCENTAGE = 20
VALIDATION_DATA_PERCENTAGE = 10

def read_picture_files(file_names):
    # 1 列队读取图片
    # 创建文件队列
    file_queue = tf.train.string_input_producer(file_names,num_epochs=1,seed=1,capacity=64)
    # 构建图片问价能读取器
    reader = tf.WholeFileReader()
    # 从队列中读取文件
    _,value = reader.read(file_queue)
    # 解析 分为 png 和jpg
    decoded_images = tf.image.decode_jpeg(value,channels=3)
    if decoded_images.dtype != tf.float32:
        decoded_images = tf.image.convert_image_dtype(decoded_images,tf.float32)
    # 定义图片的像素
    images = tf.image.resize_images(decoded_images,[299,299])
    images.set_shape([299,299,3])

    # 批处理
    image_batch = tf.train.batch([images],12,3,64)

    return image_batch

def transform__to_tfrecord(file_names,label,out_file_name):
    image_batch = read_picture_files(file_names)

    writer = tf.python_io.TFRecordWriter(out_file_name)
    for i in range(12):
        image = image_batch[i].eval().tostring()

        example = tf.train.Example(feaures=tf.train.Features(feature={
            "image":tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        }))
        writer.write(example.SerializeToString())
    writer.close()

def gen_tfrecord_file():
    sub_dirs = [x[0] for x in os.walk("../data/flowers/")]
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        i = 0  # label
        for sub_dir in sub_dirs[1:]:
            files = os.listdir(sub_dir)
            file_list = [os.path.join(sub_dir, file) for file in files]
            image_batch = read_picture_files(file_list)

            out_file_train_name = os.path.join("../data/flower_tfrecord/train_data.tf.records")
            out_file_test_name = os.path.join("../data/flower_tfrecord/test_data.tf.records")
            out_file_validation_name = os.path.join("../data/flower_tfrecord/validation_data.tf.records")
            # 将数据分为训练集,验证集和测试集
            chance = numpy.random.randint(100)
            if chance < VALIDATION_DATA_PERCENTAGE:
                transform__to_tfrecord(file_list, i, out_file_validation_name)
            elif chance <(VALIDATION_DATA_PERCENTAGE+TEST_DATA_PERCENTAGE):
                transform__to_tfrecord(file_list, i, out_file_test_name)
            else:
                transform__to_tfrecord(file_list,i,out_file_train_name)
            i += 1
        coord.request_stop()
        coord.join(threads)

def _parse_picture(filename,label):
    image_concent = tf.read_file(filename)
    # image_decoded = tf.image.decode_image(image_concent)
    # file_queue = tf.train.string_input_producer([filename])
    #
    # reader = tf.WholeFileReader()
    # _,value = reader.read(file_queue)
    # value = tf.gfile.FastGFile(filename.eval(),"rb").read()

    image_decoded = tf.image.decode_jpeg(image_concent,3)

    if image_decoded.dtype != tf.float32:
        image_decoded = tf.image.convert_image_dtype(image_decoded,tf.float32)
    image = tf.image.resize_images(image_decoded,[299,299])
    image.set_shape([299,299,3])

    return image,label

def build_datasets():
    # 从张量中构建数据集
    sub_dirs = [x[0] for x in os.walk("../data/flowers/")]
    filenames = []
    labels = []
    for i,sub_dir in enumerate(sub_dirs[1:]):
        files = os.listdir(sub_dir)
        file_list = [os.path.join(sub_dir, file) for file in files]
        label_list = np.array([i]*len(file_list))
        filenames.extend(file_list)
        labels.extend(label_list)

    filenames_placeholder = tf.placeholder(tf.string,(None,))
    labels_placeholder = tf.placeholder(tf.int32,(None,))
    # dataset = tf.data.Dataset.from_tensor_slices((filenames_placeholder,labels_placeholder))
    # files_tensor = tf.constant(filenames)
    # labels_tensor = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices([filenames,labels])
    dataset.map(_parse_picture)
    dataset.shuffle(100)
    dataset.batch(12)

    # iterator = dataset.make_initializable_iterator()
    iterator = dataset.make_one_shot_iterator()
    image_batch,label_batch = iterator.get_next()

    with tf.Session() as sess:
        # sess.run(iterator.initializer,feed_dict={filenames_placeholder:filenames,
        #                                          labels_placeholder:labels})
        # while True:
            # try:
            #     print(sess.run([image_batch,label_batch]))
            # except tf.errors.OutOfRangeError:
            #     break
        print(sess.run([image_batch, label_batch]))






if __name__ == '__main__':

    # gen_tfrecord_file()
    build_datasets()





