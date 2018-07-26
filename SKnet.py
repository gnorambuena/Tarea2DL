import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

config_sess = tf.ConfigProto()
config_sess.gpu_options.allow_growth = True

with tf.device('/device:GPU:0'):
	with tf.Session(config=config_sess) as sess:

		# Training Parameters
		learning_rate = 0.0001
		num_epochs = 5
		batch_size = 64
		buffer_size = 10**5//batch_size

		# Network Parameters
		num_classes = 100 # QUICKDRAW total classes (1-100 weas)
		dropout = 0.25 # Dropout, probability to drop a unit

		#gaussian weights 
		def gaussian_weights(shape,  mean, stddev):
			return tf.truncated_normal(shape, 
									   mean = mean, 
									   stddev = stddev)
			

		#convolution layer using stride = 1
		def conv_layer(input, shape, name, stride = 1, is_training = True):
			#weights are initialized according to a gaussian distribution
			W =  tf.Variable(gaussian_weights(shape, 0.0, 0.01), name=name)     
			#weights for bias ares fixed as constants 0
			b = tf.Variable(tf.zeros(shape[3]), name='bias_'+name)
			return tf.nn.relu(
					tf.layers.batch_normalization(
						tf.add(tf.nn.conv2d(
								input, 
								W, 
								strides=[1, stride, stride, 1], 
								padding='SAME'), b), scale = False, training = is_training))

		#pooling layer that uses max_pool
		def max_pool_layer(input, kernel, stride):
			return tf.nn.max_pool(input,  
								  [1, kernel, kernel, 1], 
								  [1, stride, stride, 1], 
								  padding = 'SAME' )

		#fully-connected layer fc
		def fc_layer(input, size, name, use_relu=True): 
			layer_shape_in =  input.get_shape()
			# shape is a 1D tensor with 4 values
			num_features_in = layer_shape_in[1:4].num_elements()
			#reshape to  1D vector
			input_reshaped = tf.reshape(input, [-1, num_features_in])
			shape = [num_features_in, size]
			W = tf.Variable(gaussian_weights(shape, 0.0, 0.02), name=name)     
			b = tf.Variable(tf.zeros(size))
			#
			layer = tf.add( tf.matmul(input_reshaped, W) ,  b)    
			
			if use_relu:
				layer=tf.nn.relu(layer)
			return  layer

		#dropout
		def dropout_layer(input, prob):
			return tf.nn.dropout(input, prob)

		def conv_net(x, y_true, is_training, input_shape = [None, 128, 128]):
			with tf.variable_scope('ConvNet'):      
			
				x_tensor = tf.reshape(x, [-1, x.get_shape().as_list()[1], x.get_shape().as_list()[2], 1 ] )
				
				conv1_1 = conv_layer(x_tensor, shape = [3, 3, 1, 64], name='conv1_1', is_training = is_training)
				conv1_2 = conv_layer(conv1_1, shape = [3, 3, 64, 64], name='conv1_2', is_training = is_training)
				max_pool1 = max_pool_layer(conv1_2, 3, 2) 
				
				conv2_1 = conv_layer(max_pool1, shape = [3, 3, 64, 128], name = 'conv2_1', is_training = is_training)
				conv2_2 = conv_layer(conv2_1, shape = [3, 3, 128, 128], name = 'conv2_2', is_training = is_training)
				max_pool2 = max_pool_layer(conv2_2, 3, 2) 
				
				conv3_1 = conv_layer(max_pool2, shape = [3, 3, 128, 128], name = 'conv3_1', is_training = is_training)
				conv3_2 = conv_layer(conv3_1, shape = [3, 3, 128, 128], name = 'conv3_2', is_training = is_training)
				max_pool3 = max_pool_layer(conv3_2, 3, 2)

				conv4_1 = conv_layer(max_pool3, shape = [3, 3, 128, 256], name = 'conv4_1', is_training = is_training)
				conv4_2 = conv_layer(conv4_1, shape = [3, 3, 256, 256], name = 'conv4_2', is_training = is_training)
				max_pool4 = max_pool_layer(conv4_2, 3, 2)
				
				fc1 = fc_layer(max_pool4, 1024, name = 'fc1')
				fc2 = fc_layer(fc1, 100, name = 'fc2', use_relu = False)

				return fc2

		def model_fn(features, labels, mode):
			tf.logging.set_verbosity(tf.logging.INFO)

			logits = conv_net(features, num_classes,is_training = (mode == tf.estimator.ModeKeys.TRAIN))

			pred_classes = tf.argmax(logits, axis=1)
			pred_probas = tf.nn.softmax(logits)

			# If prediction mode, early return
			if mode == tf.estimator.ModeKeys.PREDICT:
				return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

			# Define loss and optimizer
			loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels))
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	  
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				train_op = optimizer.minimize(loss_op,
											  global_step=tf.train.get_global_step())

			accuracy = tf.metrics.accuracy(labels=tf.argmax(labels,axis=1),
								   predictions=pred_classes,
								   name='acc_op')
			metrics = {'accuracy': accuracy}

			tf.summary.scalar('accuracy', accuracy[1])
			tf.summary.scalar('loss', loss_op)
	  
			if mode == tf.estimator.ModeKeys.EVAL:
				return tf.estimator.EstimatorSpec(
					mode, loss = loss_op, eval_metric_ops = metrics, predictions = pred_classes)
			   
			logging_hook = tf.train.LoggingTensorHook({"loss": loss_op, "accuracy": accuracy[1]}, every_n_iter=100)
			return tf.estimator.EstimatorSpec(mode, loss=loss_op, train_op=train_op,
											training_hooks=[logging_hook,], eval_metric_ops={'accuracy': accuracy})

		# Build the Estimator
		config = tf.estimator.RunConfig(
			save_checkpoints_steps = 1000,
			keep_checkpoint_max = 10,
		)

		model = tf.estimator.Estimator(model_fn, config = config)

		def parser_tfrecord(serialized_example):
			features = tf.parse_example([serialized_example],
										features={
											'train/image': tf.FixedLenFeature([], tf.string),
											'train/label': tf.FixedLenFeature([], tf.int64),
										})
			image = tf.decode_raw(features['train/image'], tf.uint8)
			image = tf.reshape(image, [128, 128])
			image = tf.cast(image, tf.float32)
			image = image * 1.0 / 255.0

			label = tf.one_hot(tf.cast(features['train/label'], tf.int32), 100)
			label = tf.reshape(label, [100])
			label = tf.cast(label, tf.float32)

			return image, label

		def train_input_fn():
			dataset = tf.data.TFRecordDataset("train/train.tfrecords")
			dataset = dataset.map(parser_tfrecord)

			dataset = dataset.shuffle(buffer_size)
			dataset = dataset.batch(batch_size)
			dataset = dataset.repeat(num_epochs)

			iterator = dataset.make_one_shot_iterator()

			features, labels = iterator.get_next()
			return features, labels
		  
		def test_input_fn():
			dataset = tf.data.TFRecordDataset("test/test.tfrecords")
			dataset = dataset.map(parser_tfrecord)

			dataset = dataset.batch(batch_size)

			iterator = dataset.make_one_shot_iterator()

			features, labels = iterator.get_next()
			return features, labels
		  
	model.train(train_input_fn)

	e = model.evaluate(test_input_fn)
	print("Testing Accuracy:", e['accuracy'])