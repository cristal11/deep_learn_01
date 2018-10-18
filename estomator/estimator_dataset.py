import tensorflow as tf
import numpy as np

class Iris(object):

    """

    """
    def __init__(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        self.file_path = "../data/iris/iris_train.csv",
        self.repeat_count = 1
        self.batch_size = 32
        # self.classifier = None

    def my_input_fn(self,file_path,perform_shuffle,repeat_count=1):
        dataset = tf.data.TextLineDataset(file_path,).skip(1).map(self.decode_csv)

        if perform_shuffle:
            dataset = dataset.shuffle(buffer_size=256)
        dataset = dataset.repeat(repeat_count)
        dataset = dataset.batch(self.batch_size)
        # dataset.map()

        iterator =  dataset.make_one_shot_iterator()
        feature_batch,label_batch = iterator.get_next()

        return feature_batch,label_batch

    def decode_csv(self,line):
        # 对单行进行解析
        record_defaults = [[0.],[0.],[0.],[0.],[0]]
        parsed_line = tf.decode_csv(line,record_defaults)
        return {"x":parsed_line[:-1]},parsed_line[-1]

    def train(self):
        feature_columns = [tf.feature_column.numeric_column("x",shape=(4,))]
        classifier = tf.estimator.DNNClassifier(
            hidden_units=[20,20,10],
            feature_columns=feature_columns,
            n_classes=3
        )
        # x = self.my_input_fn()
        classifier.train(
            input_fn=lambda :self.my_input_fn("../data/iris/iris_train.csv",True,10),
        )

        test_result = classifier.evaluate(
            input_fn=lambda :self.my_input_fn("../data/iris/iris_test.csv",False)
        )
        print("test accuracy is %g %%" %(test_result["accuracy"]*100))

if __name__ == '__main__':
    my_estimator = Iris()
    my_estimator.train()
    # my_estimator.evaluate()