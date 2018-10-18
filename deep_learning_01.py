import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class MYEstimatorModel(object):
    """
    Estimator自定义模型
    """
    def __init__(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.mnist = input_data.read_data_sets("./data/mnist_01/",one_hot=False)
        self.model_params = {
            "learning_rate":0.01
        }

    def le_net(self,input_tensor,is_training):
        x = tf.reshape(input_tensor,[-1,28,28,1])
        # 神经网络,卷积
        conv_1 = tf.layers.conv2d(x,filters=32,kernel_size=5,padding="SAME",activation=tf.nn.relu)
        pool_1 = tf.layers.max_pooling2d(conv_1,2,2)
        conv_2 = tf.layers.conv2d(pool_1,64,3,activation=tf.nn.relu)
        pool_2 = tf.layers.max_pooling2d(conv_2,2,2)
        # 将输出从4威武降为2为
        low_flatten = tf.contrib.layers.flatten(pool_2)
        # 全连接
        fc_1 = tf.layers.dense(low_flatten,1024)
        dropout = tf.layers.dropout(fc_1,rate=0.4,training=is_training)
        return tf.layers.dense(dropout,10)

    def model_fn(self,features,labels,mode,params):
        y_predict = self.le_net(features["image"],mode==tf.estimator.ModeKeys.TRAIN)

        # 如果是测试模式
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"result":tf.argmax(y_predict,1)}
            )
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=y_predict
        ))
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())

        eval_metric_ops = {
            "my_metric":tf.metrics.accuracy(
                tf.argmax(y_predict,1),labels
            )
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops
        )
    def run(self):
        estimator = tf.estimator.Estimator(model_fn=self.model_fn,params=self.model_params)

        train_data = self.gen_train_data()

        estimator.train(train_data,steps=10000)
        test_data = self.gen_test_data()

        accuracy_score = estimator.evaluate(test_data)["my_metric"]
        print("test accuracy is : %g %%" %(accuracy_score*100))

        predict_data = self.gen_predict_data()
        predictions = estimator.predict(predict_data,"数字")
        for i in predictions:
            print("prediction-{}:\n {}".format(i+1,predictions))

    def gen_train_data(self):
        return tf.estimator.inputs.numpy_input_fn(
            x = {"image":self.mnist.train.images},
            y = self.mnist.train.labels.astype(np.int32),
            batch_size=128,
            num_epochs=1,
            shuffle=True,
            queue_capacity=1000,
            num_threads=5
        )

    def gen_test_data(self):
        return tf.estimator.inputs.numpy_input_fn(
            x={"image": self.mnist.test.images},
            y=self.mnist.test.labels.astype(np.int32),
            batch_size=128,
            num_epochs=1,
            shuffle=False,
            # queue_capacity=1000,
            num_threads=5
        )

    def gen_predict_data(self):
        return tf.estimator.inputs.numpy_input_fn(
            x={"image": self.mnist.test.images[:10]},
            # y=self.mnist.test.labels,
            # batch_size=128,
            num_epochs=1,
            shuffle=False,
            # queue_capacity=1000,
            # num_threads=5
        )

if __name__ == '__main__':
    my_estimator = MYEstimatorModel()
    my_estimator.run()