import time
import cv2
import mss
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import torch
import time
from multiprocessing import Queue
import multiprocessing
import LogInTinder as log

from pynput import mouse
from pynput.mouse import Button, Controller

torch.cuda.empty_cache() 
title = "FPS benchmark"

mouse = Controller()
sct = mss.mss()

monitor = {"top": 250, "left": 630, "width": 700, "height": 640}
fps = 0
start_time = time.time()
display_time = 2


PATH_TO_FROZEN_GRAPH = 'frozen_inference_graph.pb'

PATH_TO_LABELS = 'labelmap.pbtxt'
NUM_CLASSES = 2

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.compat.v1.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.compat.v1.import_graph_def(od_graph_def, name='')

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True

def Detection(queue):
  global fps
  global start_time
  global title
  with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph,config=config) as sess:
      while True:
        image_np = np.array(sct.grab(monitor))
        
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        image_np_expanded = np.expand_dims(image_np, axis=0)

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=3)
        for i,b in enumerate(boxes[0]):
          if classes[0][i]==1:
            if scores[0][i]>=0.5:
              queue.put(1)
          elif classes[0][i]==2:
            if scores[0][i]>=0.5:
              queue.put(2)
     

        #im = Image.fromarray(image_np) 
        #im.save("your_file.jpeg")
        cv2.imshow(title,cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        
        if cv2.waitKey(25) & 0xFF == ord("q"):
          cv2.destroyAllWindows()
          sess.close()
          break

        
def like():
  mouse.position =(1218, 791)
  mouse.press(Button.left)
  mouse.release(Button.left)
  time.sleep(10)


def dislike():
  mouse.position =(1083, 787) 
  mouse.press(Button.left)
  mouse.release(Button.left)
  time.sleep(10)


def MouseMovement(queue):
  while True:
    num=queue.get()
    if num==1:
      like()
    elif num==2:
      dislike()      
    
    while not queue.empty():
      queue.get()
          

def main ():
  pqueue = Queue()

  p1 = multiprocessing.Process(target=Detection, args=(pqueue,))
  p2 = multiprocessing.Process(target=MouseMovement, args=(pqueue,))

  # starting our processes
  p1.start()
  p2.start()   

if __name__ == "__main__":
  log.login()
  main()