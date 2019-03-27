import os
import time
import sys
import numpy as np
import tensorflow as tf
import argparse
#from PIL import Image
from google.protobuf import text_format
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import timeline
from tensorflow.python.platform import gfile
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
os.environ["OMP_NUM_THREADS"] = "28"
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  print(im_width)
  print(im_height)
  return np.array(image.getdata()).reshape(
      (im_width, im_height, 3)).astype(np.uint8)

with tf.Graph().as_default() as graph:
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.intra_op_parallelism_threads = 28
  config.inter_op_parallelism_threads = 1
  with tf.Session(config=config) as sess: 
    #with tf.gfile.FastGFile("/nfs/site/home/sriniva2/int8-repo/int8_faster_rcnn_no_intel.pb",'rb') as f: 
    #with tf.gfile.FastGFile("/nfs/site/home/sriniva2/int8-repo/int8_faster_rcnn_intel_conv_bias_sum_relu_pool.pb",'rb') as f:  # Load pb as graphdef
    #with tf.gfile.FastGFile("/nfs/site/home/sriniva2/int8-repo/golden_final_test.pb",'rb') as f:  # Load pb as graphdef
    #with tf.gfile.FastGFile("/nfs/site/home/sriniva2/int8-repo/golder_than_golden_freezed_test.pb",'rb') as f:  # Load pb as graphdef
    with tf.gfile.FastGFile("/localdisk/niroop/repo/fasterrcnnfpn/nirooptensorpack/tensorpack/examples/FasterRCNN/temp/built_graph/old/Fasterrcnnfpn_graph_def_freezed.pb",'rb') as f:  # Load pb as graphdef
    #with tf.gfile.FastGFile("/nfs/site/home/sriniva2/int8-repo/test_pad.pb",'rb') as f:  # Load pb as graphdef
      graphdef = tf.GraphDef() 
      graphdef.ParseFromString(f.read()) 
      #text_format.Merge(f.read(),graphdef) 
      sess.graph.as_default()  
      tf.import_graph_def(graphdef, name='')
      # Definite input and output Tensors for detection_graph
      image_tensor = graph.get_tensor_by_name('image:0')

      # Each box represents a part of the image where a particular object was detected.
      detection_boxes = graph.get_tensor_by_name('tower0/output/boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      #detection_scores = graph.get_tensor_by_name('output/scores:0')
      #detection_classes = graph.get_tensor_by_name('output/labels:0')
      #num_detections = graph.get_tensor_by_name('num_detections:0') # Output Tensor
      # # INFERENCE
      tf.global_variables_initializer()    
      i = 0
      avg=0
      for _ in range(500):
       i+=1    
       #image = Image.open("/nfs/site/home/jinghua2/tensorflow-FasterRCNN/tensorflow-models/research/object_detection/test_images/image2.jpg")
       # the array based representation of the image will be used later in order to prepare the
       # result image with boxes and labels on it.
       #image_np = load_image_into_numpy_array(image)
       #image_np = tf.image.resize_images(image_np, [600, 600])
       # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
       #image_np_expanded = np.expand_dims(image_np, axis=0)
       image_np_expanded=np.random.rand(800, 1202, 3).astype(np.uint8)
       #options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
       #run_metadata = tf.RunMetadata()
       start_time = time.time() 

       (boxes) = sess.run([detection_boxes],feed_dict = {image_tensor : image_np_expanded})#,options=options, run_metadata=run_metadata ) 
       #trace = timeline.Timeline(step_stats=run_metadata.step_stats)
       #with gfile.Open('fastrcnn_time', 'w') as trace_file:
       #    trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
       if(i!=1):
        avg+=(time.time()-start_time)
        print(time.time() - start_time) 
      print('%.3f sec'%(float(avg)/float(i)))
