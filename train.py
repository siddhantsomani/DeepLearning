import pickle as pkl
import numpy as np
from time import time
import sys
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from PIL import Image
import scipy.misc
import time
import sys



############################# PROBLEM PARAMETERS #############################
dim1 = 384
dim2 = 384
nclass = 151
##############################################################################

########################## LEARNING RATE PARAMETERS ##########################
learning_rate = 0.0005 * (0.75)**5
lr_decay_rate = 0.75
lr_decay_step = 300
##############################################################################


################################# MODEL PARAMETERS ###########################

batch_size = 64

#Number of iterations after which the model is stored
CheckpointStep = 100

#Directory to store checkpoints in
checkPointFile = 'Checkpoints/model1.ckpt'

#total number of iterations
training_iters = 50000

#number of steps after which accuracy is displayed
displayStep = 1

#Number of training images
trainSize = 20000
##############################################################################


#################################### AUXILIARY PARAMETERS #############################
#starting index, used in get images in batch
start = 0

#logs the step accuracy with timestamp
fname = "logs/logs-" + str(time.ctime()).replace(" ", "_") + ".txt"
flogs = open(fname,"w")

imageLocation = 'ADEChallengeData2016/images/training/ADE_train_'
annotationLocation = 'ADEChallengeData2016/annotations/training/ADE_train_'
#######################################################################################

# distorted images, so ignored
errorFiles = [1700,3019,8454,13507]

dataIndices = []
for i in range(trainSize) :
    if not i in errorFiles :
        dataIndices.append(i)
dataIndices = np.array(dataIndices)



def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def ConvertToOneHotEncoding(img_labels,NumOfClasses) :
    batch,dim1,dim2 = img_labels.shape
    newLabel = np.zeros([batch,dim1,dim2,NumOfClasses])
    for i in range(batch) :
        for j in range(dim1):
            for k in range(dim2):
                newLabel[i,j,k,int(img_labels[i,j,k])] = 1
    return newLabel

# reads the image into numpy array given image index
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

        img = img.astype(dtype=np.float32)
        annotation = annotation.astype(dtype=np.float32)

        return img,annotation
    except Exception as e:
        print(str(e))
        return 'error','error'
# normalizes image data to [0,1]
def normalize(data):
    data = data / 255.0
    return data

#gets training data in batches
def getTrainingData(batch_size):


    global start
    first = start
    start = start + batch_size
    global trainSize
    global dataIndices
    np.random.shuffle(dataIndices)
    if start > trainSize or first > dataIndices.shape[0] or start > dataIndices.shape[0]:
        first = 0
        start = batch_size
    end = start

    #first and end maintain index bounds
    dataindex = dataIndices[first:end]

    #images used in current batch
    #print(dataindex)

    ##########################################
    #data contains images
    data = []
    #labels contains labelled per pixel map
    label = []
    ##########################################

    for i in range(batch_size) :
        img,annotation = readImage(dataindex[i])
        if img == 'error' :
            continue
        data.append(img)
        label.append(annotation)

    data = np.array(data)
    label = np.array(label)

    return data,label

#both inputs are of size [batch_size X 384 X 384, 151]
def calculateKAccuracy(Final_output, targets_, k = 1):

	#vector of ground truth labels per pixel
	target_labels = tf.cast(tf.argmax(targets_, axis = 1), tf.int64)

	#find the top k most probable labels for each pixel
	#size of Final_output: [batch_size X 384 X 384, 151]
	#size of indices: [batch_size X 384 X 384, k]
	(_, indices) = tf.nn.top_k(Final_output, k = k)
	indices = tf.cast(indices, tf.int64)

	truth_vector = tf.zeros(shape = (indices.shape[0], ), dtype = tf.int64)
	

	for i in range(k):

		#compare ith most probable pixel
		bool_vector = tf.equal(indices[:,i], target_labels)

		truth_vector = truth_vector + tf.cast(bool_vector, tf.int64)

	#per pixel we have k values
	#truth_matrix = tf.equal(indices, tf.cast(tf.argmax(targets_, axis = 1), tf.int64))
	
	#change true, false to 1,0
	truth_vector = tf.cast(truth_vector, tf.float32)

	#any of k predicted labels was right
	# truth_matrix = tf.reduce_sum(truth_matrix, axis = 1)

	return tf.reduce_mean(truth_vector)


print("-----------------------------INPUT LAYER-----------------------------------")
l_in = tf.placeholder(tf.float32, shape=(None, dim1,dim2,3))
print(l_in.get_shape())
print("---------------------------------------------------------------------------")

targets = tf.placeholder(tf.float32,shape=(None,dim1,dim2,nclass))
targets_ = tf.reshape(targets,[-1,nclass])
#valid = tf.reduce_sum(targets_, axis = 1)

print("--------------------CONVOLUTION LAYER (RELU)-------------------------------")

with tf.name_scope('conv1_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 16], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(l_in, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv1_1 = tf.nn.relu(out, name=scope)

    print(conv1_1.get_shape())
print("---------------------------------------------------------------------------")


print("--------------------CONVOLUTION LAYER (RELU) -------------------------------")

# conv1_2
with tf.name_scope('conv1_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 16, 16], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv1_2 = tf.nn.relu(out, name=scope)

    print(conv1_2.get_shape())
print("-----------------------------------------------------------------------------")


print("-----------------------------------POOL---------------------------------------")

# pool1
pool1 = tf.nn.max_pool(conv1_2,
                       ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1],
                       padding='SAME',
                       name='pool1')

#pool1: [batch_size, 192, 192, 16]
print(pool1.get_shape())
print("-------------------------------------------------------------------------------")

#rnn_input: [batch_size, 96, 96 * 128]
rnn_input = tf.reshape(pool1, [batch_size, 192, 192*16])

print("------------------------------------ HORIZONTAL RNN -------------------------------------")
#rnn_input: [192, batch_size, 192*16]
rnn_input = tf.transpose(rnn_input, [1,0,2])
rnn_input = tf.split(rnn_input, 192)
rnn_input = [tf.squeeze(x) for x in rnn_input]
print(len(rnn_input))
print(rnn_input[0].get_shape())


with tf.variable_scope('horizontal'):
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(672, forget_bias=1.0)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(672, forget_bias=1.0)

    lstm_outputs,_,_ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, rnn_input, dtype=tf.float32)

print(len(lstm_outputs))
print(lstm_outputs[0].get_shape())

print("------------------------------------------------------------------------------------------")
print("------------------------------------Vertical RNN-------------------------------------------")
#[192, batch_size, 192*7]
lstm_outputs = tf.stack(lstm_outputs)
lstm_outputs = tf.reshape(lstm_outputs, [192,batch_size, 192, 7])
lstm_outputs = tf.transpose(lstm_outputs, [2,1,0,3])
lstm_outputs = tf.reshape(lstm_outputs, [192, batch_size, 192*7])


lstm_outputs = tf.split(lstm_outputs, 192)
lstm_outputs = [tf.squeeze(x) for x in lstm_outputs]
print(len(lstm_outputs))
print(lstm_outputs[0].get_shape())

with tf.variable_scope('vertical'):
    lstm_fw_cell2 = tf.contrib.rnn.BasicLSTMCell(1536, forget_bias=1.0)
    lstm_bw_cell2 = tf.contrib.rnn.BasicLSTMCell(1536, forget_bias=1.0)

    #[192] [batch_size, 192*16]
    lstm_outputs2,_,_ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell2, lstm_bw_cell2, lstm_outputs, dtype=tf.float32)

#lstm_outputs: [192, batch_size, 192*16]
lstm_outputs2 = tf.stack(lstm_outputs2)


#[192, batch_size, 192, 16]
lstm_outputs2 = tf.reshape(lstm_outputs2, [192, batch_size, 192, 16])
#[batch_size, 192, 192, 16]
lstm_outputs2 = tf.transpose(lstm_outputs2, [1,2,0,3])
print(lstm_outputs2.get_shape())


print("---------------------------------------------------------------------------------------------------------")

print("-------------------------------------------CONVOLUTION LAYER (RELU)--------------------------------------")

# conv2_1
with tf.name_scope('conv2_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 16, 32], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(lstm_outputs2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv2_1 = tf.nn.relu(out, name=scope)

    print(conv2_1.get_shape())


print("--------------------------------------------------------------------------------------------------------")
print("--------------------------------------CONVOLUTION LAYER (RELU)------------------------------------------")

# conv2_1_1
with tf.name_scope('conv2_1_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 32, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv2_1_1 = tf.nn.relu(out, name=scope)

    print(conv2_1_1.get_shape())
print("---------------------------------------------------------------")



print("----------------------------------------CONVOLUTION LAYER (RELU)-----------------------------------------")
# conv2_2
with tf.name_scope('conv2_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv2_1_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv2_2 = tf.nn.relu(out, name=scope)

    print(conv2_2.get_shape())

print("----------------------------------------------------------------------------------------------------------")


print("--------------------------------------------POOL----------------------------------------------------------")
# pool2
pool2 = tf.nn.max_pool(conv2_2,
                       ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1],
                       padding='SAME',
                       name='pool2')

print(pool2.get_shape())
print("----------------------------------------------------------------------------------------------------------")




print("------------------------------------------CONVOLUTION LAYER (RELU)-----------------------------------------")

# conv3_1
with tf.name_scope('conv3_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_1 = tf.nn.relu(out, name=scope)

    print(conv3_1.get_shape())

print("-----------------------------------------------------------------------------------------------------------")



print("----------------------------------------CONVOLUTION LAYER (RELU)-------------------------------------------")
# conv3_2
with tf.name_scope('conv3_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv3_2 = tf.nn.relu(out, name=scope)

    print(conv3_2.get_shape())



print("-----------------------------------------------------------------------------------------------------------")

print("--------------------------------------------------POOL-----------------------------------------------------")

# pool3
pool3 = tf.nn.max_pool(conv3_2,
                       ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1],
                       padding='SAME',
                       name='pool3')

print(pool3.get_shape())


print("-----------------------------------------------------------------------------------------------------------")



print("--------------------------------------CONVOLUTION LAYER (RELU)---------------------------------------------")
# conv4_1
with tf.name_scope('conv4_1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_1 = tf.nn.relu(out, name=scope)

    print(conv4_1.get_shape())


print("-----------------------------------------------------------------------------------------------------------")


print("------------------------------------CONVOLUTION LAYER (RELU)-----------------------------------------------")

# conv4_2
with tf.name_scope('conv4_2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                         trainable=True, name='biases')
    out = tf.nn.bias_add(conv, biases)
    conv4_2 = tf.nn.relu(out, name=scope)

    print(conv4_2.get_shape())



print("-----------------------------------------------------------------------------------------------------------")

print("###########################################################################################################")

print("---------------------------------------- DECONVOLUTION LAYER ----------------------------------------------")
upsampled_pool5 = tf.layers.conv2d_transpose(conv4_2, filters=512, strides=[2,2], kernel_size=[2,2])
print(upsampled_pool5.get_shape())
print("-----------------------------------------------------------------------------------------------------------")

print("----------------------------------------- DECONVOLUTION LAYER ---------------------------------------------")
deconv5 = tf.layers.conv2d_transpose(upsampled_pool5, filters=256, strides=[2,2], kernel_size=[2,2])
print(deconv5.get_shape())
print("-----------------------------------------------------------------------------------------------------------")


print("------------------------------------------ DECONVOLUTION LAYER --------------------------------------------")
deconv4 = tf.layers.conv2d_transpose(deconv5, filters=151, strides=[2,2], kernel_size=[2,2])
print(deconv4.get_shape())
print("-----------------------------------------------------------------------------------------------------------")


#[batch_size X 384 X 384, 151]
l_reshape = tf.reshape(deconv4, [-1,151])
print(l_reshape.get_shape())

W = weight_variable(shape=(151,151))
b = bias_variable(shape=(151,))

#[batch_size X 384 X 384, 151]
l_reshape = tf.matmul(l_reshape,W) + b


#[batch_size X 384 X 384, 151]
Final_output = tf.nn.softmax(l_reshape)

cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=targets_, logits=l_reshape))



#used in learning rate decay and checkpoint stamping
global_step = tf.Variable(0, trainable=False)

#exponential decay function for learning rate
lr = tf.train.exponential_decay(learning_rate,global_step,lr_decay_step,lr_decay_rate,staircase=True)

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(targets_,1), tf.argmax(Final_output,1))

#mean pixel accuraacy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
topkaccuracy = calculateKAccuracy(Final_output, targets_)

sess = tf.Session()
saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(training_iters) :

	#restore the model if needed
    if i == 0 and str(sys.argv[1]) == 'restore':
   	
	try:  
    		#prints the checkpoint used to restore the model
	    	print(tf.train.latest_checkpoint("Checkpoints/"))
    		saver.restore(sess, tf.train.latest_checkpoint("Checkpoints/"))
	catch Exception as e:
		print("Checkpoint was not restored")
    
    data_x, data_y = getTrainingData(batch_size)
    data_x = normalize(data_x)
    data_y = ConvertToOneHotEncoding(data_y, nclass)

    if i % displayStep == 0 :
        print('{0} steps reached'.format(i))
        (correct, correctk) = sess.run([accuracy, topkaccuracy], feed_dict = {l_in: data_x, targets: data_y})
        

        #stdout
        print('After {0} steps the mean pixel accuracy is {1}'.format(i, correct))
        print('After {0} steps the top 5 accuracy is {1}'.format(i, correctk))

        #write to log file
        flogs.write('After {0} steps the accuracy is {1}\n'.format(i, correct))
        flogs.write('After {0} steps the top 5 accuracy is {1}\n'.format(i, correctk))

    if i % CheckpointStep == 0 or i == training_iters - 1:
        save_path = saver.save(sess, checkPointFile, global_step = i + 1)
        print("Model saved in file: {0}".format(save_path))

    ce = sess.run([cross_entropy],feed_dict = {l_in: data_x, targets: data_y})[0]
    print("Loss:{0}".format(ce))


    #Free the memory
    del data_x
    del data_y

print('training done')
