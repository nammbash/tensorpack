# -*- coding: utf-8 -*-
# File: eval.py

import tqdm
import os
from collections import namedtuple
from contextlib import ExitStack
import itertools
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

from tensorpack.utils.utils import get_tqdm_kwargs

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as cocomask

from coco import COCOMeta
from common import CustomResize, clip_boxes
from config import config as cfg

import os
import time
import sys
import numpy as np
import tensorflow as tf
import argparse

DetectionResult = namedtuple(
    'DetectionResult',
    ['box', 'score', 'class_id', 'mask'])
"""
box: 4 float
score: float
class_id: int, 1~NUM_CLASS
mask: None, or a binary image of the original image shape
"""


def paste_mask(box, mask, shape):
    """
    Args:
        box: 4 float
        mask: MxM floats
        shape: h,w
    Returns:
        A uint8 binary image of hxw.
    """
    # int() is floor
    # box fpcoor=0.0 -> intcoor=0.0
    x0, y0 = list(map(int, box[:2] + 0.5))
    # box fpcoor=h -> intcoor=h-1, inclusive
    x1, y1 = list(map(int, box[2:] - 0.5))    # inclusive
    x1 = max(x0, x1)    # require at least 1x1
    y1 = max(y0, y1)

    w = x1 + 1 - x0
    h = y1 + 1 - y0

    # rounding errors could happen here, because masks were not originally computed for this shape.
    # but it's hard to do better, because the network does not know the "original" scale
    mask = (cv2.resize(mask, (w, h)) > 0.5).astype('uint8')
    ret = np.zeros(shape, dtype='uint8')
    ret[y0:y1 + 1, x0:x1 + 1] = mask
    return ret

def DetectOneImageFromFrozenGraph(input_image_np):

    # Each box represents a part of the image where a particular object was detected.
    graph = DetectFromFrozenGraph.sessionvalues[0]
    #config = DetectFromFrozenGraph.sessionvalues[1]
    pbsession = DetectFromFrozenGraph.sessionvalues[1]    
    image_tensor = DetectFromFrozenGraph.sessionvalues[2]
    detection_boxes = DetectFromFrozenGraph.sessionvalues[3]
    detection_scores = DetectFromFrozenGraph.sessionvalues[4]
    detection_labels = DetectFromFrozenGraph.sessionvalues[5]

    # Run real inference from the frozen graph.
    (boxes, scores, labels) = pbsession.run([detection_boxes, detection_scores, detection_labels],feed_dict = {image_tensor : input_image_np})#,options=options, run_metadata=run_metadata )
    return (boxes, scores, labels)

class DetectFromFrozenGraph:
    sessionvalues = []

    def SetupDetectFromFrozenGraph(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = dir_path + "/temp/built_graph"
        frozen_model_path = dir_path + "/Fasterrcnnfpn_graph_def_freezed.pb"

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.intra_op_parallelism_threads = 28
        config.inter_op_parallelism_threads = 1
        #with tf.Session(config=config) as pbsession:
        with tf.Graph().as_default() as tfgraph:
            with tf.gfile.FastGFile(frozen_model_path,'rb') as f:  # Load pb as graphdef
                #graph = tf.Graph()
                graphdef = tf.GraphDef() 
                graphdef.ParseFromString(f.read()) 
                #text_format.Merge(f.read(),graphdef) 
                #pbsession.graph.as_default()
                with tfgraph.as_default() :
                    tf.import_graph_def(graphdef, name='')
                # Definite input and output Tensors for detection_graph
                image_tensor = tfgraph.get_tensor_by_name('image:0')
                detection_boxes = tfgraph.get_tensor_by_name('tower0/output/boxes:0')
                detection_scores = tfgraph.get_tensor_by_name('tower0/output/scores:0')
                detection_labels = tfgraph.get_tensor_by_name('tower0/output/labels:0')
                
                # Get a permanent session object
                pbsession = tf.Session(graph=tfgraph, config=config)
                # initialize the session
                tf.global_variables_initializer()

                # Store all the global variables in the class list.
                self.sessionvalues.append(tfgraph)
                #self.sessionvalues.append(config)
                self.sessionvalues.append(pbsession)                                
                self.sessionvalues.append(image_tensor)
                self.sessionvalues.append(detection_boxes)
                self.sessionvalues.append(detection_scores)
                self.sessionvalues.append(detection_labels)

def detect_one_image(img, model_func, loadfrozenpb=False):
    """
    Run detection on one image, using the TF callable.
    This function should handle the preprocessing internally.

    Args:
        img: an image
        model_func: a callable from TF model,
            takes image and returns (boxes, probs, labels, [masks])

    Returns:
        [DetectionResult]
    """

    orig_shape = img.shape[:2]
    resizer = CustomResize(cfg.PREPROC.TEST_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE)
    resized_img = resizer.augment(img)
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
    
    if not loadfrozenpb:
        boxes, probs, labels, *masks = model_func(resized_img)
    else:
        boxes, probs, labels, *masks = DetectOneImageFromFrozenGraph(resized_img)
    
    boxes = boxes / scale
    # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
    boxes = clip_boxes(boxes, orig_shape)

    if masks:
        # has mask
        full_masks = [paste_mask(box, mask, orig_shape)
                      for box, mask in zip(boxes, masks[0])]
        masks = full_masks
    else:
        # fill with none
        masks = [None] * len(boxes)

    results = [DetectionResult(*args) for args in zip(boxes, probs, labels, masks)]
    return results


def eval_coco(df, detect_func, tqdm_bar=None, loadfrozenpb=False):
    """
    Args:
        df: a DataFlow which produces (image, image_id)
        detect_func: a callable, takes [image] and returns [DetectionResult]
        tqdm_bar: a tqdm object to be shared among multiple evaluation instances. If None,
            will create a new one.

    Returns:
        list of dict, to be dumped to COCO json format
    """
    all_results = []
    if loadfrozenpb:
        detectfrozen = DetectFromFrozenGraph()
        detectfrozen.SetupDetectFromFrozenGraph()

    # tqdm is not quite thread-safe: https://github.com/tqdm/tqdm/issues/323    

    with ExitStack() as stack:
        if tqdm_bar is None:
            tqdm_bar = stack.enter_context(
                tqdm.tqdm(total=df.size(), **get_tqdm_kwargs()))
        for img, img_id in df:
            results = detect_func(img)
            for r in results:
                box = r.box
                cat_id = COCOMeta.class_id_to_category_id[r.class_id]
                box[2] -= box[0]
                box[3] -= box[1]

                res = {
                    'image_id': img_id,
                    'category_id': cat_id,
                    'bbox': list(map(lambda x: round(float(x), 3), box)),
                    'score': round(float(r.score), 4),
                }

                # also append segmentation to results
                if r.mask is not None:
                    rle = cocomask.encode(
                        np.array(r.mask[:, :, None], order='F'))[0]
                    rle['counts'] = rle['counts'].decode('ascii')
                    res['segmentation'] = rle
                all_results.append(res)
            tqdm_bar.update(1)
    return all_results


def multithread_eval_coco(dataflows, detect_funcs):
    """
    Running multiple `eval_coco` in multiple threads, and aggregate the results.

    Args:
        dataflows: a list of DataFlow to be used in :func:`eval_coco`
        detect_funcs: a list of callable to be used in :func:`eval_coco`

    Returns:
        list of dict, to be dumped to COCO json format
    """
    num_worker = len(dataflows)
    assert len(dataflows) == len(detect_funcs)
    with ThreadPoolExecutor(max_workers=num_worker, thread_name_prefix='EvalWorker') as executor, \
            tqdm.tqdm(total=sum([df.size() for df in dataflows])) as pbar:
        futures = []
        for dataflow, pred in zip(dataflows, detect_funcs):
            futures.append(executor.submit(eval_coco, dataflow, pred, pbar))
        all_results = list(itertools.chain(*[fut.result() for fut in futures]))
        return all_results


# https://github.com/pdollar/coco/blob/master/PythonAPI/pycocoEvalDemo.ipynb
def print_coco_metrics(json_file):
    ret = {}
    assert cfg.DATA.BASEDIR and os.path.isdir(cfg.DATA.BASEDIR)
    annofile = os.path.join(
        cfg.DATA.BASEDIR, 'annotations',
        'instances_{}.json'.format(cfg.DATA.VAL))
    coco = COCO(annofile)
    cocoDt = coco.loadRes(json_file)
    cocoEval = COCOeval(coco, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    fields = ['IoU=0.5:0.95', 'IoU=0.5', 'IoU=0.75', 'small', 'medium', 'large']
    for k in range(6):
        ret['mAP(bbox)/' + fields[k]] = cocoEval.stats[k]

    if cfg.MODE_MASK:
        cocoEval = COCOeval(coco, cocoDt, 'segm')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        for k in range(6):
            ret['mAP(segm)/' + fields[k]] = cocoEval.stats[k]
    return ret
