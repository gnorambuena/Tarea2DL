import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

with tf.device('/device:GPU:0'):
	def buildConv2DLayer(input, out_size):
		conv = tf.layers.conv2d(input,out_size,3,
				padding = "SAME", activation = tf.nn.relu,
				kernel_initializer = tf.contrib.layers.xavier_initializer())

		return tf.layers.batch_normalization(conv,training = True, fused = True)

	def buildMaxPool(input):
		return tf.layers.max_pooling2d(input, 3, 2, padding = "SAME")

	img_size = 128
	channels = 1
	classes = 100
	x = tf.placeholder(tf.float32, shape = [None,img_size,img_size,channels])
	y = tf.placeholder(tf.float32, shape = [None,classes])

	x_norm = tf.layers.batch_normalization(x,training = True)

	conv1_1 = buildConv2DLayer(x_norm,64)
	conv1_2 = buildConv2DLayer(conv1_1,64)
	maxpool1 = buildMaxPool(conv1_2)

	conv2_1 = buildConv2DLayer(maxpool1,128)
	conv2_2 = buildConv2DLayer(conv2_1,128)
	maxpool2 = buildMaxPool(conv2_2)

	conv3_1 = buildConv2DLayer(maxpool2,128)
	conv3_2 = buildConv2DLayer(conv3_1,128)
	maxpool3 = buildMaxPool(conv3_2)

	conv4_1 = buildConv2DLayer(maxpool3,256)
	conv4_2 = buildConv2DLayer(conv4_1,256)
	maxpool4 = buildMaxPool(conv4_2)

	flt = tf.contrib.layers.flatten(maxpool4)

	fc_1 = tf.layers.dense(flt,1024,activation = tf.nn.relu,
			kernel_initializer = tf.contrib.layers.xavier_initializer())
	fc_2 = tf.layers.dense(fc_1,100,
			kernel_initializer = tf.contrib.layers.xavier_initializer())

	y_pred = tf.nn.softmax(fc_2)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc_2, labels=y))

	learning_rate = 0.03
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
