#! /usr/bin/env python
# coding=utf-8


import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
import time

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_coco.pb"
image_path_prefix      = '/localhome/sxu/Desktop/MA/frustum-pointnets-master/dataset/KITTI/object/training/image_2/'
#file_name = "000001"
num_classes     = 80
input_size      = 416
graph           = tf.Graph()

def yoloPredict(sess, file_name):
    t1 = time.time()
    original_image = cv2.imread(image_path_prefix + file_name + '.png')
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...]
    return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)
    print("Image loaded in {:.3f}s".format(time.time() - t1))
    t1 = time.time()
    pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
        [return_tensors[1], return_tensors[2], return_tensors[3]],
                feed_dict={ return_tensors[0]: image_data})
    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
    print("YOLO Inferenced in {:.3f}s".format(time.time() - t1))
    t1 = time.time()
    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.2)
    bboxes = utils.nms(bboxes, 0.45, method='nms')
    print("NMS in {:.3f}s".format(time.time() - t1))
    roi_box = []
    for i, bbox in enumerate(bboxes):
        if (bbox[4] >= 0.5): # 0: person, 2: car , 1: bicycle (NOT cyclist!)
            roi_box.append(bboxes[i])
            x_min = int(bbox[0])
            x_max = int(bbox[2])
            y_min = int(bbox[1])
            y_max = int(bbox[3])
#            print (x_min, y_min, x_max, y_max)
    image = utils.draw_bbox(original_image, roi_box)
    image = Image.fromarray(image)
    image.show()
    input ("ENTER")
    return roi_box


def write_2d_detection(fName, roi_box):
    resultStr = ""
    for i in range(len(roi_box)):
        if roi_box[i][5] == 0:
            typeName = "Pedestrian"
        elif roi_box[i][5] == 2:
            typeName = "Car"
        else:
            typeName = "Unknown"
        out_str = typeName + " -1 -1 -10 "
        out_str += "%f %f %f %f " % (roi_box[i][0],roi_box[i][1],roi_box[i][2],roi_box[i][3])
        # add 3D info here
        resultStr += out_str + "1.408879 1.580820 3.690379 3.298013 2.223699 34.423156 1.592142 0.953033" +"\n"
#    print (resultStr)
    fout = open(fName, 'w')
    fout.write(resultStr)
    fout.close() 
    


#    roi_box = yoloPredict(sess, "000004")
#    write_2d_detection("./detectedTXT/"+ "000004" +".txt", roi_box)
config = tf.ConfigProto(allow_soft_placement=True,device_count={'gpu':0}) 
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction=1
valIdxPath = "/localhome/sxu/Desktop/MA/frustum-pointnets-master/kitti/image_sets/val.txt"
ff = np.loadtxt(valIdxPath, dtype=np.str)
with tf.Session(graph=graph, config=config) as sess:
    with tf.device('/gpu:0'):
        for i in range(360, len(ff)):
        #    if i % 10 == 0:
        #        sess.close()
        #        sess = tf.Session(graph=graph, config=config)
            fName = ff[i]
            roi_box = yoloPredict(sess, fName)
            write_2d_detection("./detectedTXT/"+ fName +".txt", roi_box)
            print ("Process: " + str(i) + " / " + str(len(ff)))
        #    print ("2D INFERENCE DONE!")
                





