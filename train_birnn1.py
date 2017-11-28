import pickle as pkl
import numpy as np
from time import time
import sys
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
#import GettingData1
from PIL import Image

imageLocation = '/home/ss5408/ADEChallengeData2016/images/training/ADE_train_'
annotationLocation = '/home/ss5408/ADEChallengeData2016/annotations/training/ADE_train_'

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W , strideShape):
    return tf.nn.conv2d(x, W, strides=strideShape, padding='SAME')

def deconv(x,w,output_shape,strides) :
    return tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=strides, padding='SAME')

def ConvertToOneHotEncoding(img_labels,NumOfClasses) :
    batch,dim1,dim2 = img_labels.shape
    newLabel = np.zeros([batch,dim1,dim2,NumOfClasses])
    for i in range(batch):
        for j in range(dim1):
            for k in range(dim2):
				newLabel[i,j,k,int(img_labels[i,j,k])] = 1
	return newLabel

def readImage(index) :
	global imageLocation
	global annotationLocation
	try :
		filename = str('00000000'+ str(index+1))
		filename = filename[len(filename)-8:]
		img = Image.open(imageLocation+filename+'.jpg')
		annotation = Image.open(annotationLocation+filename+'.png')
		img = img.resize((384,384))
		annotation = np.array(annotation.resize( (384,384) ) )
		img = np.array(img)
		#img = np.transpose(img,(2,0,1))
		return img,annotation
	except :
		print 'error occured in reading images'
		return 'error','error'

def getTrainingData(batch_size) :
	global start
	first = start
	start = start+batch_size
	global trainSize
	global dataIndices
	if start > trainSize :
		first = 0
		np.random.shuffle(dataIndices)
		start = batch_size
	end = start
	dataindex = dataIndices[first:end]
	data = []
	label = []
	for i in range(batch_size) :
		img,annotation = readImage(dataindex[i])
		if img == 'error' :
			continue
		data.append(img)
		label.append(annotation)

	data = np.array(data)
	label = np.array(label)

	return data,label

batch_size = 32
dim1 = 384
dim2 = 384
nf = 16
nclass=151
hidden_size = 128
learning_rate = 0.0005
lr_decay_rate = 0.75
lr_decay_step = 4000
CheckpointStep = 500
training_iters = 10000
displayStep = 1
checkPointFile = '/home/ss5408/Checkpoints/model1.ckpt'
start = 0
trainSize = 20000

dataIndices = []
errorFiles = [1700,3019,8454,13507]
for i in range(trainSize) :
	if not i in errorFiles :
		dataIndices.append(i)
dataIndices = np.array(dataIndices)
np.random.shuffle(dataIndices)

# with tf.device('/cpu:0'):

# with tf.device('/cpu:0'):

l_in = tf.placeholder(tf.float32, shape=(None, dim1,dim2,3))
#n_batch, n_steps, in_size = l_in.get_shape()
w_conv1 = weight_variable([3, 3, 3, 8])
h_conv1 = conv2d(l_in, w_conv1 , [1, 3, 3, 1])

w_conv2 = weight_variable([3, 3, 8, nf])
h_conv2 = conv2d(h_conv1, w_conv2 , [1, 2, 2, 1])
targets = tf.placeholder(tf.float32,shape=(None,dim1,dim2,nclass))
targets_ = tf.reshape(targets,[-1,nclass])
convOut = tf.transpose(h_conv2,[0,3,1,2])


#transpose to (dim1, batch,dim2, nf )


rnn_input_tr = tf.transpose(convOut,[2,0,3,1])
rnn_input_re = tf.reshape(rnn_input_tr,[-1,nf])
rnn_input = tf.split(0, dim1/6, rnn_input_re)
with tf.variable_scope('hori'):
	lstm_fw_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)
	lstm_bw_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)
		# Get lstm cell output
	lstm_outputs, _, _ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, rnn_input, dtype=tf.float32)
	lstm_outputs_re=tf.reshape(lstm_outputs, [dim1/6, -1, dim2/6, 2*hidden_size])
	lstm_output_tr=tf.transpose(lstm_outputs_re, [2,1,0,3])

rnn_input_re2 = tf.reshape(lstm_output_tr,[-1,nf])
rnn_input2 = tf.split(0, dim2/6, rnn_input_re2)
with tf.variable_scope('vert'):
	lstm_fw_cell2 = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)
	lstm_bw_cell2 = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0)
		# Get lstm cell output
	lstm_outputs2, _, _ = rnn.bidirectional_rnn(lstm_fw_cell2, lstm_bw_cell2, rnn_input2, dtype=tf.float32)
	lstm_outputs_re2=tf.reshape(lstm_outputs2, [dim2/6, -1, dim1/6, 2*hidden_size])
	lstm_output_tr2=tf.transpose(lstm_outputs_re, [1,2,0,3])


W_deconv1 = weight_variable([3, 3, 2*hidden_size, 2*hidden_size])
h_deconv1 = deconv(lstm_output_tr2,W_deconv1,[batch_size,128,128,2*hidden_size],[1, 2, 2, 1])

W_deconv2 = weight_variable([3, 3, 2*hidden_size, 2*hidden_size])
h_deconv2 = deconv(h_deconv1,W_deconv2,[batch_size,dim1,dim2,2*hidden_size],[1, 3, 3, 1])

W2 = weight_variable([2*hidden_size,nclass])
b2 = bias_variable([nclass])
l_reshape3 = tf.reshape(h_deconv2,[-1,2*hidden_size] )
Final_output = tf.nn.softmax(tf.matmul(l_reshape3,W2) + b2)

cross_entropy = -tf.reduce_sum(targets_*tf.log(Final_output))


global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(learning_rate,
								global_step,
								lr_decay_step,
								lr_decay_rate,
								staircase=True)

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(targets_,1), tf.argmax(Final_output,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
saver = tf.train.Saver()
init = tf.initialize_all_variables()
sess.run(init)


for i in range(training_iters) :
	#datadict = save_img.get_batch(p)
	# if i == 1:
	# 	print ("Input shape: " + str(l_in.get_shape()))
	# 	print ("hconv1 shape: " + str(h_conv1.get_shape()))
	# 	print ("hconv2 shape: " + str(convOut.get_shape()))
	# 	print("RNN input shape: " + str(rnn_input_re.get_shape()))
	# 	print("LSTM output RE: " + str(lstm_outputs_re.get_shape()))
	# 	print("LSTM output TR: " + str(lstm_output_tr.get_shape()))
    #
	# 	print("RNN input 2: " + str(rnn_input_2.get_shape()))
	# 	print("LSTM output RE: " + str(lstm_outputs_re2.get_shape()))
	# 	print("LSTM output TR: " + str(lstm_output_tr2.get_shape()))

	# if i== 0 :
	# 	saver.restore(sess, "Checkpoints/model.ckpt-501")
	data_x,data_y = getTrainingData(batch_size)
	#spx=datadict['x'].shape
	#if(spx[0]==0):
	#	continue
	'''
	print 'received data'
	data_x = datadict['x']
	print data_x.shape
	data_y = datadict['y']
	print data_y.shape
	'''
	data_y = ConvertToOneHotEncoding(data_y,nclass)
	#print('batch_size = %d'%len(datadict['x']))
	if i %displayStep == 0:
		print('%d steps reached'%i)
		correct = sess.run([accuracy],feed_dict={l_in: data_x, targets: data_y})
		print('after %d steps the accuracy is %g'%(i,correct[0]))
	if i % CheckpointStep == 0 or i == training_iters - 1:
		save_path = saver.save(sess,  checkPointFile,
					   global_step= 1+ i)
		print("Model saved in file: %s" % save_path)
	sess.run([train_step],feed_dict={l_in: data_x, targets: data_y})
	del data_x
	del data_y

print 'training done'
