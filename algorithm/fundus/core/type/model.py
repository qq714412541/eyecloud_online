import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets

from .configs import FLAGS, network_configs


class StageModel:
    def build(self, num_classes):
        network = FLAGS['network']
        config = network_configs[network]
        net, arg_scope = self.network_select(network)

        with tf.variable_scope('Input'):
            x = tf.placeholder(
                tf.float32,
                [
                    None,
                    config['INPUT_WIDTH'],
                    config['INPUT_HEIGHT'],
                    3
                ],
                name='input'
            )
            y_ = tf.placeholder(
                tf.int32,
                [None, num_classes],
                name='ground_truth'
            )
            is_training = tf.placeholder_with_default(
                True, [], name='is_training'
            )

        with slim.arg_scope(arg_scope()):
            logits, _ = net(x, num_classes, is_training)
            if network[:9] == 'ResNet-V2':
                logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

        if FLAGS['transfer_learning']:
            self.variable_to_restore = slim.get_variables_to_restore(
                include=self.include, exclude=self.exclude
            )

        with tf.variable_scope('softmax'):
            softmax = tf.nn.softmax(logits)

        with tf.variable_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=y_,
                logits=logits
            )
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('cross_entropy', cross_entropy_mean)

        with tf.variable_scope('evaluate'):
            inferred_class = tf.argmax(softmax, 1)
            true_class = tf.argmax(y_, 1)
            correct_prediction = tf.equal(inferred_class, true_class)
            accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32)
            )
            tf.summary.scalar('accuracy', accuracy)

        self.config = config
        self.input = x
        self.ground_truth = y_
        self.is_training = is_training
        self.logits = logits
        self.softmax = softmax
        self.loss = cross_entropy_mean
        self.inferred_class = inferred_class
        self.accuracy = accuracy

    def train_operation(self):
        with tf.variable_scope('train_layer'):
            opt = FLAGS['optimizer']
            learning_rate = FLAGS['learning_rate']

            global_step = tf.Variable(0, name='global_step')
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            trainable_variables = tf.trainable_variables()
            self.trainable_variables = trainable_variables
            self.update_ops = update_ops

            if FLAGS['transfer_learning'] and not FLAGS['fine_tune']:
                def should_train(var):
                    name = var.name
                    for exclude in self.exclude:
                        if exclude in name:
                            return True
                    for include in self.include:
                        if include in name:
                            return False

                trainable_variables = [
                    var for var in trainable_variables
                    if should_train(var)
                ]
                update_ops = [
                    var for var in update_ops
                    if should_train(var)
                ]

            with tf.control_dependencies(update_ops):
                if opt == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                elif opt == 'sgd':
                    optimizer = tf.train.GradientDescentOptimizer(
                        learning_rate
                    )
                elif opt == 'sgdr':
                    first_decay_steps = FLAGS['first_decay_steps']
                    lr = tf.train.cosine_decay_restarts(
                        learning_rate,
                        global_step,
                        first_decay_steps
                    )
                    optimizer = tf.train.GradientDescentOptimizer(lr)

                self.train_step = optimizer.minimize(
                    self.loss,
                    global_step=global_step,
                    var_list=trainable_variables
                )
                self.global_step = global_step

    def network_select(self, network):
        if network == 'Inception-V1':
            net = nets.inception.inception_v1
            arg_scope = nets.inception.inception_v1_arg_scope
            self.include = ['InceptionV1']
            self.exclude = ['InceptionV1/Logits', 'InceptionV1/AuxLogits']
        elif network == 'Inception-V3':
            net = nets.inception.inception_v3
            arg_scope = nets.inception.inception_v3_arg_scope
            self.include = ['InceptionV3']
            self.exclude = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']
        elif network == 'Inception-ResNet-V2':
            import slim_nets.inception_resnet_v2 as inception_resnet_v2
            net = inception_resnet_v2.inception_resnet_v2
            arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope
            self.include = ['InceptionResnetV2']
            self.exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
        elif network == 'ResNet-V2-50':
            net = nets.resnet_v2.resnet_v2_50
            arg_scope = nets.resnet_v2.resnet_arg_scope
            self.include = ['resnet_v2_50']
            self.exclude = ['resnet_v2_50/logits']
        elif network == 'ResNet-V2-152':
            net = net.resnet_v2.resnet_v2_152
            arg_scope = nets.resnet_v2.resnet_arg_scope
            self.include = ['resnet_v2_152']
            self.exclude = ['resnet_v2_152/logits']

        return net, arg_scope
