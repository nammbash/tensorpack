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
#os.environ["KMP_BLOCKTIME"] = "0"
#os.environ["KMP_SETTINGS"] = "1"
#os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
#os.environ["OMP_NUM_THREADS"] = "28"

def DetectOneImageModelFuncReadfromFrozenGraph(input_image_np=None):
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.intra_op_parallelism_threads = 28
  config.inter_op_parallelism_threads = 1
  with tf.Session(config=config) as sess:     
    with tf.gfile.FastGFile("/localdisk/niroop/repo/fasterrcnnfpn/nirooptensorpack/tensorpack/examples/FasterRCNN/temp/built_graph/Fasterrcnnfpn_graph_def_freezed.pb",'rb') as f:  # Load pb as graphdef
      graphdef = tf.GraphDef() 
      graphdef.ParseFromString(f.read()) 
      sess.graph.as_default()  
      tf.import_graph_def(graphdef, name='')

      # Definite input and output Tensors for detection_graph
      image_tensor = graph.get_tensor_by_name('image:0')
      detection_boxes = graph.get_tensor_by_name('tower0/output/boxes:0')
      detection_scores = graph.get_tensor_by_name('tower0/output/scores:0')
      detection_classes = graph.get_tensor_by_name('tower0/output/labels:0')
      
      ## Add data
      tf.global_variables_initializer()    
      if not input_image_np:
        # Load Dummy data
        image_np_expanded=np.random.rand(800, 1202, 3).astype(np.uint8)
      else:
        # you can do all the preprocessing for an image here.
        # and send the image_np_expanded
        # input_image_np=image.preprocessstepsfromtrain.py(input_image)
        print("add actual image np here")
      
      ## INFERENCE
      # Choose type of inference 1. single or 2.average
      #inference_type = "single"
      inference_type = "average"
      # Do actual inference
      if inference_type == "single":
        (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes],feed_dict = {image_tensor : image_np_expanded})#,options=options, run_metadata=run_metadata )
        #for _ in range(500):
        #  (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes],feed_dict = {image_tensor : image_np_expanded})#,options=options, run_metadata=run_metadata )          
      else :      
        i = 0
        avg=0
        for _ in range(100):
          i+=1          
          #image_np_expanded=np.random.rand(800, 1202, 3).astype(np.uint8)
          start_time = time.time() 
          (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes],feed_dict = {image_tensor : image_np_expanded})#,options=options, run_metadata=run_metadata ) 
          #trace = timeline.Timeline(step_stats=run_metadata.step_stats)
          #with gfile.Open('fastrcnn_time', 'w') as trace_file:
          #    trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
          if(i>=1):
          #if(i!=1):
            avg+=(time.time()-start_time)
            print("current inference time {} ".format (time.time() - start_time)) 
            print('current inference time%.3f sec'%(float(avg)/float(i)))

      # return some thing
      return (boxes, scores, classes)

with tf.Graph().as_default() as graph:
  output=DetectOneImageModelFuncReadfromFrozenGraph()
  print(output)
