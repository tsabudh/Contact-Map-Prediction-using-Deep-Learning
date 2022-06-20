#pip install tensorflow==1.14
#import tensorflow.compat.v1 as tf
# from re import I
import tensorflow as tf
import numpy as np
from time import gmtime, strftime
import time
import os
import io
import pickle as pickle
import glob
import math
import codecs
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
tf.compat.v1.disable_eager_execution()
#tf.disable_v2_behavior()

tf.compat.v1.flags.DEFINE_string('f','','')
# ******************************************CONFIG.PY***************************************************************************************

# import tensorflow.compat.v1 as tf

# from absl import app

# if __name__ == '__main__':
    
#     app.run(main)

# database flags

tf.app.flags.DEFINE_string(
    'train_file', 'pdb25-6767-train.release.contactFeatures.pkl',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'valid_file', '/pdb25-6767-valid.release.contactFeatures.pkl',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'test_file', 'Mems400.2015-E1.contactFeatures.pkl',
    'Directory where checkpoints and event logs are written to.')

# dir paths
tf.app.flags.DEFINE_string(
    'train_dir', './output/residual_network/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_string(
    'data_dir', './tfrecords/',
    'Directory of tfrecords database.')

# SELF INSERTED CODE
# tf.app.flags.DEFINE_string(
#     'dataset_dir', '/', 'Directory of dataset_dir for addtotfrecord.'
# )
# SELF INSERTED CODE ENDS


# network building params
tf.app.flags.DEFINE_integer(
    'filter_size_1d', 17,
    'filter size for 1D conv.')

tf.app.flags.DEFINE_integer(
    'filter_size_2d', 3,
    'filter size for 2D conv.')

tf.app.flags.DEFINE_integer(
    'block_num_1d', 1,
    'num of residual block for 1D conv.')

tf.app.flags.DEFINE_integer(
    'block_num_2d', 20,
    'num of residual block for 2D conv.')

# net training params
tf.app.flags.DEFINE_integer(
    'max_iters', 6000,
    'maximum iteration times')

# restore model
tf.app.flags.DEFINE_bool(
    'restore_previous_if_exists', True,
    'restore models trained previous')

FLAGS = tf.app.flags.FLAGS

# ************************************************************************************************************************************************************

# ****************************************************************************************************************************************

# ***************************************DATA_PREPROCESSING.PY************************************************************************************************

def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def to_tfexample_raw(name, seqLen, seq_feature, pair_feature, label_data):
    return tf.train.Example(features=tf.train.Features(feature={
        'name': _bytes_feature(name),
        'seqLen': _int64_feature(seqLen),
        'seq_feature': _bytes_feature(seq_feature),         # of shape (L, L, 26)
        'pair_feature': _bytes_feature(pair_feature),       # of shape (L, L, 5)
        'label_matrix': _bytes_feature(label_data),         # of shape (L, L)
    }))

def get_dataset_filename(dataset_dir, split_name, shard_id, num_shards):
    output_filename = 'F:\\New\\%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, num_shards)
    return os.path.join(dataset_dir, output_filename)

def extract_single(info):
    print("info:",info)

    name = info[b'name']
    seq = info[b'sequence']
    seqLen = len(seq)
    acc = info[b'ACC']
    ss3 = info[b'SS3']
    pssm = info[b'PSSM']
    psfm = info[b'PSFM']

    f = codecs.open("F:/New/feature/sequence/"+name.decode()+".txt",'w','utf-8')
    f.write(seq.decode())

    f = codecs.open("F:/New/feature/accuracy/"+name.decode()+".txt",'w','utf-8')
    f.write(str(acc))

    f = codecs.open("F:/New/feature/ss3/"+name.decode()+".txt",'w','utf-8')
    f.write(str(ss3))

    f = codecs.open("F:/New/feature/pssm/"+name.decode()+".txt",'w','utf-8')
    f.write(str(pssm))

    f = codecs.open("F:/New/feature/psfm/"+name.decode()+".txt",'w','utf-8')
    f.write(str(psfm))


    #diso = info[b'DISO']
    print("psfm:",psfm.shape)
    #print("diso:",diso.shape)
    sequence_profile = np.concatenate((pssm, ss3, acc, psfm), axis = 1)
    ccmpred = info[b'ccmpredZ']

    f = codecs.open("F:/New/feature/ccmpred/"+name.decode()+".txt",'w','utf-8')
    f.write(str(ccmpred))

    #psicov = info[b'psicovZ']
    other = info[b'OtherPairs']
    #pairwise_profile = np.dstack((ccmpred, psicov))
    #pairwise_profile = np.concatenate((pairwise_profile, other), axis = 2) #shape = (L, L, 5)

    pairwise_profile = np.dstack((ccmpred, ccmpred))
    print("pairwise_profile:",pairwise_profile.shape)
    pairwise_profile = np.concatenate((pairwise_profile, other), axis = 2) #shape = (L, L, 5)

    #datafile = "data/76CAMEO/"+name.decode()+".txt"
    # datafile = "data/mems400/"+name.decode()+".txt"
  
    #datafile = "data/CASP11/"+name.decode()+".txt"

    # datafile = "experiment/"+name.decode('ANSI')+".gcnn"
    datafile = "pdb25-6767-train.release.contactFeatures.pkl"
    # datafile = "Mems400.2016.contactFeatures.pkl"

    f = open(datafile,'rb')
    # SELF INSERTED CODE-----------------------------------------------
    # f = open('pdb25.pkl','rb')

    # SELF INSERTED CODE ENDS
    data = pickle.load(f,encoding='bytes')
    f.close()
    true_contact = info[b'contactMatrix']
    #changed from true_contact = data[b'contactMatrix']
    #acc = info[b'ACC']
    #true_contact = [[-1] * pairwise_profile.shape[0] for _ in range(pairwise_profile.shape[1])]
    # true_contact = np.array(true_contact)
    #print("true_contact:",true_contact.shape)
    true_contact = np.array(true_contact)
    print("true_contact:",np.shape(true_contact))
    true_contact[true_contact < 0] = 0 # transfer -1 to 0, shape = (L, L)
    true_contact = np.tril(true_contact, k=-6) + np.triu(true_contact, k=6) # remove the diagnol contact
    true_contact = true_contact.astype(np.uint8)

    return name, seqLen, sequence_profile, pairwise_profile, true_contact
    
def add_to_tfrecord(records_dir, split_name, infos):
    """Loads image files and writes files to a TFRecord.
    Note: masks and bboxes will lose shape info after converting to string.
    """
    num_shards = int(len(infos) / 1000)
    num_per_shard = int(math.ceil(len(infos) / float(num_shards)))
      
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        with tf.Session('') as sess:
            for shard_id in range(num_shards):
                record_filename = get_dataset_filename(records_dir, split_name, shard_id, num_shards)
                options = tf.io.TFRecordOptions(TFRecordCompressionType.ZLIB)
                with tf.io.TFRecordWriter(record_filename, options=options) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(infos))
                    print ("processing %s_data from %d to %d..." %(split_name, start_ndx, end_ndx))
                    for i in range(start_ndx, end_ndx):
                        info = infos[i]
                        name, seqLen, seq_feature, pair_feature, label = extract_single(info)
                        if seqLen > 300:
                            continue
                        #print "generate tfrecord for %s" %name
                        seq_feature = seq_feature.astype(np.float32)
                        pair_feature = pair_feature.astype(np.float32)
                        label = label.astype(np.uint8)
                        
                        example = to_tfexample_raw(name, seqLen, seq_feature.tostring(), pair_feature.tostring(), label.tostring())
                        tfrecord_writer.write(example.SerializeToString())


# ************************************************************************************************************************************************************


# **********************************READ_DATA_INTO_TFRECORDS.PY*************************************************************************************************



def read_pkl(name):
    print(name)
    with open(name,'rb') as fin:
        return pickle.load(fin,encoding='bytes')

train_infos = read_pkl(FLAGS.train_file)


print("FLAGS.data_dir:",FLAGS.data_dir)
records_dir = os.path.join(FLAGS.data_dir, 'records/')
print("records_dir:",records_dir)
# add_to_tfrecord(records_dir, 'train', train_infos)

# iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
# def read_pkl(name):
#     with open(name) as fin:
#         return pickle.load(fin)

# train_infos = read_pkl(FLAGS.train_file)
# records_dir = os.path.join(FLAGS.data_dir, 'records/')
# add_to_tfrecord(records_dir, 'train', train_infos)

# /iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
# ************************************************************************************************************************************************************

# ***************************************DATA_FACTORY.PY*****************************************************************************************

def get_dataset(split_name, dataset_dir, file_pattern=None):
    print("split_name:",split_name)
    print("dataset_dir:",dataset_dir)
    if file_pattern is None:
        file_pattern = split_name + '*.tfrecord'
    print("file_pattern:",file_pattern)
    tfrecords = glob.glob('F:/New/tfrecords/*.tfrecord')
    name,seqLen, seq_feature, pair_feature, label = read_tfrecord(tfrecords)

    return name, seqLen, seq_feature, pair_feature, label 


def read_tfrecord(tfrecords_filename):
    if not isinstance(tfrecords_filename, list):
        tfrecords_filename = [tfrecords_filename]
    filename_queue = tf.train.string_input_producer(tfrecords_filename, num_epochs=100)

    options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
    reader = tf.TFRecordReader(options=options)
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
      'name': tf.FixedLenFeature([], tf.string),
      'seqLen': tf.FixedLenFeature([], tf.int64),
      'seq_feature': tf.FixedLenFeature([], tf.string),
      'pair_feature': tf.FixedLenFeature([], tf.string),
      'label_matrix': tf.FixedLenFeature([], tf.string),
      })
    name = features['name']
    seqLen = tf.cast(features['seqLen'], tf.int32)
    seq_feature = tf.decode_raw(features['seq_feature'], tf.float32)
    seq_feature = tf.reshape(seq_feature, [seqLen, -1])             # reshape seq feature to shape = (L, feature_maps)

    print("11seq_feature:",seq_feature.shape)

    pair_feature = tf.decode_raw(features['pair_feature'], tf.float32)
    pair_feature = tf.reshape(pair_feature, [seqLen, seqLen, -1])   # reshape pair feature to shape = (L, L, feature_maps)
    label = tf.decode_raw(features['label_matrix'], tf.uint8)       
    label = tf.reshape(label, [seqLen, seqLen, 1])                  # reshape label to shape = (L, L, 1)

    return name, seqLen, seq_feature, pair_feature, label

def datafactorytest():
    dataset_dir = "data/"
    split_name = "train"
    name, seqLen, seq_feature, pair_feature, label = get_dataset(split_name, dataset_dir)

    init = tf.initialize_local_variables()
    sess = tf.Session()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    name, seqLen, seq, pair, label = sess.run([name,seqLen, seq_feature, pair_feature, label])
    print (name)
    print (seqLen)
    print (seq.shape)
    print (pair.shape)
    for l in label:
        print (''.join([str(i) for i in l]))

datafactorytest()

# **************************************************************************************


# ************************************************DENSENET_TENSORFLOW.PY********************************************************************


def conv_layer(input, filter, kernel, stride=1, layer_name='conv'):
    with tf.name_scope(layer_name):
        net = tf.layers.conv2d(inputs=input,
                               use_bias=False,
                               filters=filter,
                               kernel_size=kernel,
                               strides=stride,
                               padding='SAME')
        return net

def Global_average_pooling(x, stride=1):
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter

def Batch_norm(x, training, scope):
    return tf.cond(
        training,
        lambda : tf.layers.batch_normalization(
            inputs=x,
            trainable=True,
            reuse=None,
            name=scope),
        lambda : tf.layers.batch_normalization(
            inputs=x,
            trainable=False,
            reuse=True,
            name=scope)
    )

def Drop_out(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2, 2], stride=2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Max_pooling(x, pool_size=[3, 3], stride=2, padding='SAME'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers):
    return tf.concat(layers, axis=3)

def Linear(x, class_num):
    return tf.layers.dense(inputs=x, units=class_num, name='linear')


class DenseNet():
    def __init__(self, x, nb_blocks, filters, training, dropout_rate):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        #self.n_class = n_class
        self.dropout_rate = dropout_rate
        self.logits = self.densenet(x)

    def bottle_neck_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_norm(x, training=self.training, scope='%s_batch1' % scope)
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1, 1], layer_name= '%s_conv1' % scope)
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)

            x = Batch_norm(x, training=self.training, scope='%s_batch2' % scope)
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3, 3], layer_name='%s_batch2' % scope)
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_norm(x, training=self.training, scope='%s_batch1' % scope)
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1, 1], layer_name='%s_conv1' % scope)
            print("tl-x1:",x)
            x = Drop_out(x, rate=self.dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2, 2], stride=1)
            print("tl-x2:",x)
            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layer_concat = list()
            layer_concat.append(input_x)

            x = self.bottle_neck_layer(input_x, scope='%s_bottleN_%d' % (layer_name, 0))

            layer_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layer_concat)
                print("i:",i)
                #print("layer_concat:",layer_concat)
                x = self.bottle_neck_layer(x, scope='%s_bottleN_%d' % (layer_name, i + 1))
                print("x:",x)
                layer_concat.append(x)
                

            x = Concatenation(layer_concat)
            print("x_1:",x)
            return x

    def densenet(self, input_x):

        def nn_layer(data, weights, bias, activate_non_linearity):
            result = tf.add(tf.matmul(data, weights), bias)
            if activate_non_linearity:
                result = tf.nn.relu(result)
            return result

        print("input_x:",input_x)
        x = conv_layer(input_x, filter= 2 * self.filters, kernel=[7, 7], stride=1, layer_name='conv0')
        print("x:",x)
        x = self.dense_block(input_x=x, nb_layers=1, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')
        print("dense_block1:",x)
        x = self.dense_block(input_x=x, nb_layers=2, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')
        print("dense_block2:",x)
        x = self.dense_block(input_x=x, nb_layers=1, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')
        print("dense_block3:",x)
        x = self.dense_block(input_x=x, nb_layers=2, layer_name='dense_final')
        print("dense_block4:",x)

       # 100 Layer
        print("before-x:",x)
        x = Batch_norm(x, training=self.training, scope='linear_batch')
        print("after-x:",x)
        x = Relu(x)
        #x = Global_average_pooling(x)
        print("x:",x)
        #x = tf.layers.flatten(x)
        #print("x:",x)

        #WFC1 = tf.Variable(tf.truncated_normal([792, 89], stddev=0.1))
        #BFC1 = tf.Variable(tf.zeros(1))
        #x = nn_layer(x, WFC1, BFC1, True)
        #x=tf.reshape(x, [-1, 89])
        print("aaa-x:",x)

        #x = Linear(x, self.n_class)
        print("x:",x)
        return x
# ********************************************************************************************************************************************************



# ***************************************************RESNET-DENSENET.PY**********************************************************************
growth_k = 24
nb_block = 2 # how many (dense block + Transition Layer) ?
dropout_rate = 0.2
class_num = 2

def weight_variable(shape, regularizer, name="W"):
    if regularizer == None:
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name)
    else:
        return tf.get_variable(name, shape, 
                initializer=tf.random_normal_initializer(), regularizer=regularizer)


def bias_variable(shape, name="b"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name)

### Incoming shape (batch_size, L(seqLen), feature_num)
### Output[:, i, j, :] = incoming[:. i, :] + incoming[:, j, :] + incoming[:, (i+j)/2, :]
def seq2pairwise(incoming):
    L = tf.shape(incoming)[1]
    #save the indexes of each position
    v = tf.range(0, L, 1)
    i, j = tf.meshgrid(v, v)
    m = (i+j)/2
    #switch batch dim with L dim to put L at first
    incoming2 = tf.transpose(incoming, perm=[1, 0, 2])
    #full matrix i with element in incomming2 indexed i[i][j]
    print("incoming2:",incoming2)
    print('m:',m)
    print("j:",j)
    m = tf.cast(m, tf.int32)
    out1 = tf.nn.embedding_lookup(incoming2, i)
    out2 = tf.nn.embedding_lookup(incoming2, j)
    out3 = tf.nn.embedding_lookup(incoming2, m)
    #concatante final feature dim together
    out = tf.concat([out1, out2, out3], axis=3)
    #return to original dims
    output = tf.transpose(out, perm=[2, 0, 1, 3])
    return output

def build_block_1d(incoming, out_channels, filter_size, 
        regularizer, batch_norm=False, scope=None, name="ResidualBlock_1d"):

    net = incoming
    in_channels = incoming.get_shape().as_list()[-1]
    ident = net
    with tf.variable_scope(scope, default_name = name, values=[incoming]) as scope:
        # 1st conv layer in residual block
        W1 = weight_variable([filter_size, in_channels, out_channels], regularizer, name="W1")
        #variable_summaries(W1)
        b1 = bias_variable([out_channels], name="b1")
        #variable_summaries(b1)
        net = tf.nn.conv1d(net, W1, stride=1, padding='SAME') + b1
        ### Add batch nomalization
        if batch_norm:
            net = tf.contrib.layers.batch_norm(net)
        net = tf.nn.relu(net)
        # 2nd conv layer in residual block
        W2 = weight_variable([filter_size, out_channels, out_channels], regularizer, name="W2")
        #variable_summaries(W2)
        b2 = bias_variable([out_channels], name="b2")
        #variable_summaries(b2)
        net = tf.nn.conv1d(net, W2, stride=1, padding='SAME') + b2
        ### Add batch nomalization
        if batch_norm:
            net = tf.contrib.layers.batch_norm(net)
        net = tf.nn.relu(net)  
        if in_channels != out_channels:
            ch = (out_channels - in_channels)//2
            remain = out_channels-in_channels-ch
            ident = tf.pad(ident, [[0, 0], [0, 0], [ch, remain]])
            in_channels = out_channels
        # Add the original featrues to result, identify
        net = net + ident
    return net

def build_block_2d(incoming, out_channels, filter_size, 
        regularizer, batch_norm=False, scope=None, name="ResidualBlock_2d"):

    net = incoming
    in_channels = incoming.get_shape().as_list()[-1]
    ident = net
    with tf.variable_scope(scope, default_name = name, values=[incoming]) as scope:
        # 1st conv layer in residual block
        W1 = weight_variable([filter_size, filter_size, in_channels, out_channels], regularizer, name="W1")
        #variable_summaries(W1)
        b1 = bias_variable([out_channels], name="b1")
        #variable_summaries(b1)
        net = tf.nn.conv2d(net, W1, strides=[1,1,1,1], padding='SAME') + b1
        ### Add batch nomalization
        if batch_norm:
            net = tf.contrib.layers.batch_norm(net)
        net = tf.nn.relu(net)
        ### 2nd conv layer in residual block
        W2 = weight_variable([filter_size, filter_size, out_channels, out_channels], regularizer, name="W2")
        #variable_summaries(W2)
        b2 = bias_variable([out_channels], name="b2")
        #variable_summaries(b2)
        net = tf.nn.conv2d(net, W2, strides=[1,1,1,1], padding='SAME') + b2
        ### Add batch nomalization
        if batch_norm:
            net = tf.contrib.layers.batch_norm(net)
        net = tf.nn.relu(net)  
        if in_channels != out_channels:
            ch = (out_channels - in_channels)//2
            remain = out_channels-in_channels-ch
            ident = tf.pad(ident, [[0, 0], [0, 0], [0, 0], [ch, remain]])
            in_channels = out_channels
        ### Add the original featrues to result
        net = net + ident
    return net

def one_hot(contact_map):
    # change the shape to (L, L, 2) 
    tmp = np.where(contact_map > 0, 0, 1)
    true_contact = np.stack((tmp, contact_map), axis=-1)
    return true_contact.astype(np.float32)

def build_loss(output_prob, y, weight=None):
    y = tf.py_func(one_hot, [y], tf.float32)
    los = -tf.reduce_mean(tf.multiply(tf.log(tf.clip_by_value(output_prob,1e-10,1.0)), y))
    return los

def build(is_training, input_1d, input_2d, label, 
        filter_size_1d=17, filter_size_2d=3, block_num_1d=0, block_num_2d=10,
        regulation=True, batch_norm=True):
    print("##build##")
    print("input_1d:",input_1d)
    print("input_2d:",input_2d)
    print("label:",label) 

    regularizer = None
    if regulation:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    net = input_1d

    channel_step = 2
    ######## 1d Residual Network ##########
    out_channels = net.get_shape().as_list()[-1]
    for i in range(block_num_1d):    #1D-residual blocks building
        out_channels += channel_step
        net = build_block_1d(net, out_channels, filter_size_1d, 
                regularizer, batch_norm=batch_norm, name="ResidualBlock_1D_"+str(i))
            
    #######################################
    
    # Conversion of sequential to pairwise feature
    with tf.name_scope('1d_to_2d'):
        net = seq2pairwise(net) 

    # Merge coevolution info(pairwise potential) and above feature
    if block_num_1d == 0:
        net = input_2d
    else:
        net = tf.concat([net, input_2d], axis=3)
    out_channels = net.get_shape().as_list()[-1]
    
    ######## 2d Residual Network ##########
    for i in range(block_num_2d):    #2D-residual blocks building
        out_channels += channel_step
        net = build_block_2d(net, out_channels, filter_size_2d, 
                regularizer, batch_norm=batch_norm, name="ResidualBlock_2D_"+str(i))
    #######################################

    ############## DenseNet################
    print("before DenseNet:",net)
    model = DenseNet(x=net,nb_blocks=nb_block, filters=growth_k,training=is_training,dropout_rate=dropout_rate)
    print("after DenseNet:",net)
    logits = model.logits
    print("logits:",logits)
    net = logits
    print("after DenseNet:",net)
    #######################################

    out_channels = net.get_shape().as_list()[-1]
    print("out_channels:",out_channels)


    # softmax channels of each pair into a score
    with tf.variable_scope('softmax_layer', values=[net]) as scpoe:
        W_out = weight_variable([1, 1, out_channels, 2], regularizer, 'W')
        b_out = bias_variable([2], 'b')
        output_prob = tf.nn.softmax(tf.nn.conv2d(net, W_out, strides=[1,1,1,1], padding='SAME') + b_out)
    
    with tf.name_scope('loss_function'):
        loss = build_loss(output_prob, label)
        if regulation:
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
            loss += reg_term
        tf.summary.scalar('loss', loss)
    output = {}
    output['output_prob'] = output_prob
    output['loss'] = loss

    return output
# ***********************************************************************************************************************************************************







# ****************************************** TRAIN.PY *********************************************************************************

# Commented out IPython magic to ensure Python compatibility.

# using GPU numbered 0
os.environ["CUDA_VISIBLE_DEVICES"]='1'

def restore(sess):
     if FLAGS.restore_previous_if_exists:
        try:
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
            restorer = tf.train.Saver()
            restorer.restore(sess, checkpoint_path)
            print ('restored previous model %s from %s'\
                     %(checkpoint_path, FLAGS.train_dir))
            time.sleep(2)
            return
        except:
            print ('--restore_previous_if_exists is set, but failed to restore in %s %s'\
                     % (FLAGS.train_dir, checkpoint_path))
            time.sleep(2)

def train():
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& FIRST LINE OF TRAIN FUNCTION")
    print(FLAGS.data_dir)
    name, seqLen, seq_feature, pair_feature, label = \
        get_dataset('train', FLAGS.data_dir)

    print("1111111111seq_feature:",seq_feature.shape)
    print("YO VANEKO TRAIN WALA RUN VAKO HO")
    data_queue = tf.RandomShuffleQueue(capacity=32, min_after_dequeue=16,
            dtypes=(name.dtype, seqLen.dtype,
                seq_feature.dtype, pair_feature.dtype, label.dtype))
    enqueue_op = data_queue.enqueue((name, seqLen, seq_feature, pair_feature, label))
    data_queue_runner = tf.train.QueueRunner(data_queue, [enqueue_op] * 4)
    tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, data_queue_runner)
    (name, seqLen, seq_feature, pair_feature, label) = data_queue.dequeue()
    print("1111111111seq_feature:",seq_feature.shape)
    input_1d = tf.reshape(seq_feature, (1, seqLen, 46))
    input_2d = tf.reshape(pair_feature, (1, seqLen, seqLen, 5))
    label = tf.reshape(label, (1, seqLen, seqLen))
    print("input_1d:",input_1d)
    print("input_2d:",input_2d)
    print("label:",label)
    print("FLAGS.filter_size_1d:",FLAGS.filter_size_1d)
    print("FLAGS.filter_size_2d:",FLAGS.filter_size_2d)
    print("FLAGS.block_num_1d:",FLAGS.block_num_1d)
    print("FLAGS.block_num_2d:",FLAGS.block_num_2d)
    is_training = tf.placeholder(tf.bool)
    output =build(is_training,input_1d, input_2d, label,
            FLAGS.filter_size_1d, FLAGS.filter_size_2d,
            FLAGS.block_num_1d, FLAGS.block_num_2d,
            regulation=True, batch_norm=True)
    print("output:",output)

    prob = output['output_prob']
    loss = output['loss']
    print("prob:",prob)
    print("loss:",loss)

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.80)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    init_op = tf.group(tf.global_variables_initializer(),
            tf.local_variables_initializer())
    sess.run(init_op)
    
    # save log
    summary_op = tf.summary.merge_all()
    logdir = os.path.join(FLAGS.train_dir, strftime('%Y%m%d%H%M%S', gmtime()))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)

    #restore model
    restore(sess)

    # main loop
    coord = tf.train.Coordinator()
    threads = []
    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
    tf.train.start_queue_runners(sess=sess, coord=coord)
    
    feed_dict = {is_training:True}

    saver = tf.train.Saver(max_to_keep=20)
    # train iteration
    for step in range(FLAGS.max_iters):
        _, ids, L, los, output_prob = \
                sess.run([train_step, name, seqLen, loss, prob], feed_dict={is_training:True})
        print ("iter %d: id = %s, seqLen = %3d, loss = %.4f" %(step, ids, L, los))

        if step % 100 == 0:
            summary_str = sess.run(summary_op, feed_dict={is_training:True})
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
        
        if (step % 10000 == 0 or step + 1 == FLAGS.max_iters) and step != 0:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)

def traintest():
    input_1d = tf.constant(np.random.rand(1,10,26).astype(np.float32))
    input_2d = tf.constant(np.random.rand(1,10,10,5).astype(np.float32))
    label = tf.constant(np.random.randint(2, size=(1,10,10)))

    output =build(input_1d, input_2d, label)
    prob = output['output_prob']
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    print (sess.run(prob))

train()

# ************************************************************************************************************************************************************

# *************************************************************ACC_CAL_V2.PY***********************************************************************************************

def topKaccuracy(y_out, y, k):
    L = y.shape[0]

    m = np.ones_like(y, dtype=np.int8)
    lm = np.triu(m, 24)
    mm = np.triu(m, 12)
    sm = np.triu(m, 6)
    
    print("lm:",lm)
    print("mm:",mm)
    print("sm:",sm)
    print("lm:",lm.shape)
    print("mm:",mm.shape)
    print("sm:",sm.shape)

    sm = sm - mm
    mm = mm - lm

    avg_pred = (y_out + y_out.transpose((1, 0))) / 2.0
    truth = np.concatenate((avg_pred[..., np.newaxis], y[..., np.newaxis]), axis=-1)

    print("y_out:",y_out)
    print("y_out:",y_out.shape)

    print("y:",y)
    print("y:",y.shape)

    print("avg_pred:",avg_pred)
    print("avg_pred:",avg_pred.shape)

    print("truth:",truth)
    print("truth:",truth.shape)

    accs = []
    recalls = []
    for x in [lm, mm, sm]:
        selected_truth = truth[x.nonzero()]
        selected_false = truth[np.where( x == 0 )]
        print("selected_truth:",selected_truth)
        print("selected_truth:",selected_truth.shape)
        selected_truth_sorted = selected_truth[(selected_truth[:, 0]).argsort()[::-1]]
        selected_false_sorted = selected_false[(selected_false[:, 0]).argsort()[::-1]]
        print("selected_truth_sorted:",selected_truth_sorted)
        print("selected_truth_sorted:",selected_truth_sorted.shape)

        print("selected_truth_sorted[:, 1]:",selected_truth_sorted[:, 1])
        print("selected_truth_sorted[:, 1]:",selected_truth_sorted[:, 1].shape)

        tops_num = min(selected_truth_sorted.shape[0], L//k)
        tops_num1 = min(selected_false_sorted.shape[0], L//k)
        print("tops_num:",tops_num)

        truth_in_pred = selected_truth_sorted[:, 1].astype(np.int8)
        false_in_pred = selected_false_sorted[:, 1].astype(np.int8)
        print("truth_in_pred:",truth_in_pred)
        print("truth_in_pred:",truth_in_pred.shape)

        corrects_num = np.bincount(truth_in_pred[0: tops_num], minlength=2)
        corrects_num1 = np.bincount(false_in_pred[0: tops_num1], minlength=2)
        print("corrects_num:",corrects_num)
        print("corrects_num:",corrects_num.shape)
        print("corrects_num1:",corrects_num1)
        print("corrects_num1:",corrects_num1.shape)

        acc = 1.0 * corrects_num[1] / (tops_num + 0.0001)
        recall = 1.0 * corrects_num[1] / (1.0 * corrects_num[1] + 1.0 * corrects_num1[1] + 0.0001)
        print("acc:",acc)
        print("acc:",acc.shape)
        print("recall:",recall)
        print("recall:",recall.shape)

        accs.append(acc)
        recalls.append(recall)
    print("accs:",accs)
    print("recalls:",recalls)

    return accs, recalls

def evaluate(predict_matrix, contact_matrix):
    acc_k_1, recall_k_1 = topKaccuracy(predict_matrix, contact_matrix, 1)
    acc_k_2, recall_k_2 = topKaccuracy(predict_matrix, contact_matrix, 2)
    acc_k_5, recall_k_5 = topKaccuracy(predict_matrix, contact_matrix, 5)
    acc_k_10, recall_k_10 = topKaccuracy(predict_matrix, contact_matrix, 10)
    tmp = []
    tmp1 =[]
    tmp.append(acc_k_1)
    tmp.append(acc_k_2)
    tmp.append(acc_k_5)
    tmp.append(acc_k_10)

    tmp1.append(recall_k_1)
    tmp1.append(recall_k_2)
    tmp1.append(recall_k_5)
    tmp1.append(recall_k_10)

    print("tmp:",tmp)
    print("tmp1:",tmp1)

    return tmp,tmp1


def output_result(avg_acc):
    print ("Long Range(> 24):")
    print ("Method    L/10         L/5          L/2        L")
    print ("Acc :     %.3f        %.3f        %.3f      %.3f" \
            %(avg_acc[3][0], avg_acc[2][0], avg_acc[1][0], avg_acc[0][0]))
    print ("Medium Range(12 - 24):")
    print ("Method    L/10         L/5          L/2        L")
    print ("Acc :     %.3f        %.3f        %.3f      %.3f" \
            %(avg_acc[3][1], avg_acc[2][1], avg_acc[1][1], avg_acc[0][1]))
    print ("Short Range(6 - 12):")
    print ("Method    L/10         L/5          L/2        L")
    print ("Acc :     %.3f        %.3f        %.3f      %.3f" \
            %(avg_acc[3][2], avg_acc[2][2], avg_acc[1][2], avg_acc[0][2]))

def output_result1(avg_acc):
    print ("Long Range(> 24):")
    print ("Method    L/10         L/5          L/2        L")
    print ("Recall :     %.3f        %.3f        %.3f      %.3f" \
            %(avg_acc[3][0], avg_acc[2][0], avg_acc[1][0], avg_acc[0][0]))
    print ("Medium Range(12 - 24):")
    print ("Method    L/10         L/5          L/2        L")
    print ("Recall :     %.3f        %.3f        %.3f      %.3f" \
            %(avg_acc[3][1], avg_acc[2][1], avg_acc[1][1], avg_acc[0][1]))
    print ("Short Range(6 - 12):")
    print ("Method    L/10         L/5          L/2        L")
    print ("Recall :     %.3f        %.3f        %.3f      %.3f" \
            %(avg_acc[3][2], avg_acc[2][2], avg_acc[1][2], avg_acc[0][2]))

def acccalv2test():
    with open("data/PSICOV/psicov.list", "r") as fin:
        names = [line.rstrip("\n") for line in fin]

    accs = []
    for i in range(len(names)):
        name = names[i]
        print ("processing in %d: %s" %(i+1, name))
        
        #prediction_path = "data/PSICOV/clm/"
        prediction_path = "data/PSICOV/new_psicov/"
        #prediction_path = "data/PSICOV/psicov_matrix"
        #prediction_path = "data/PSICOV/mf_matrix"
        #prediction_path = "psicov_result"
        f = os.path.join(prediction_path, name + ".ccmpred")
        if not os.path.exists(f):
            print ("not exist...")
            continue
        y_out = np.loadtxt(f)

        dist_path = "data/PSICOV/dis/"
        y = np.loadtxt(os.path.join(dist_path, name + ".dis"))
        y[y > 8] = 0 
        y[y != 0] = 1
        y = y.astype(np.int8)
        y = np.tril(y, k=-6) + np.triu(y, k=6) 

        acc = evaluate(y_out, y)
        accs.append(acc)
    accs = np.array(accs)
    avg_acc = np.mean(accs, axis=0)
    output_result(avg_acc)
# ************************************************************************************************************************************************************



# ************************************************************TEST.PY************************************************************************************************

# using GPU numbered 0
os.environ["CUDA_VISIBLE_DEVICES"]='1'

def load_test_data():
    #datafile = "data/pdb25-test-500.release.contactFeatures.pkl"
    datafile = "data/Mems400.2016.contactFeatures.pkl"

    f = open(datafile,'rb')
    data = pickle.load(f,encoding='bytes')
    f.close()
    return data

def test():
    # restore graph
    input_1d = tf.placeholder("float", shape=[None, None, 46], name="input_x1")
    input_2d = tf.placeholder("float", shape=[None, None, None, 5], name="input_x2")
    label = tf.placeholder("float", shape=None, name="input_y")
    is_training = tf.placeholder(tf.bool)
    output = build(is_training, input_1d, input_2d, label,
            FLAGS.filter_size_1d, FLAGS.filter_size_2d,
            FLAGS.block_num_1d, FLAGS.block_num_2d,
            regulation=True, batch_norm=True)
    prob = output['output_prob']

    # restore model
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
    #checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt-90000")
    print ("Loading model from %s" %checkpoint_path)
    restorer = tf.train.Saver()
    restorer.restore(sess, checkpoint_path)
    
    # prediction
    #predict single one
    #output_prob = sess.run(prob, feed_dict={input_1d: f_1d, input_2d: f_2d})
    data = load_test_data()
    input_acc = []
    output_acc = []
    output_recall = []
    input_recall = []
    for i in range(len(data)):
        d = data[i]
        name, seqLen, sequence_profile, pairwise_profile, true_contact = \
                extract_single(d)
        print ("processing %d %s" %(i+1, name))
        sequence_profile = sequence_profile[np.newaxis, ...]
        print (sequence_profile.shape)
        pairwise_profile = pairwise_profile[np.newaxis, ...]
        print (pairwise_profile.shape)
        y_out = sess.run(prob, \
                feed_dict = {input_1d: sequence_profile, input_2d: pairwise_profile, is_training:False})
        np.savetxt("results/"+name.decode()+".deepmat", y_out[0,:,:,1])
        np.savetxt("contacts/"+name.decode()+".contacts", true_contact[0])
        input_temp_acc,input_temp_recall = evaluate(pairwise_profile[0,:,:,0], true_contact)
        input_acc.append(input_temp_acc)
        input_recall.append(input_temp_recall)
        print("y_out1:",y_out[0,:,:,1])
        print("true_contact:",true_contact)
        print("y_out1:",y_out[0,:,:,1].shape)
        print("true_contact:",true_contact.shape)
        output_temp_acc,output_temp_recall = evaluate(y_out[0,:,:,1], true_contact)
        output_acc.append(output_temp_acc)
        output_recall.append(output_temp_recall)

    print("output_acc:",output_acc)
    print("output_recall:",output_recall)

    print ("Input result:")
    output_result(np.mean(np.array(input_acc), axis=0))
    print ("\nOutput result:")
    output_result(np.mean(np.array(output_acc), axis=0))
    output_result1(np.mean(np.array(output_recall), axis=0))
    
# if __name__ == "__main__":
    # test()
# ************************************************************************************************************************************************************
