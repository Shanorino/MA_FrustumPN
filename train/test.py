''' Evaluating Frustum PointNets.
Write evaluation results to KITTI format labels.
and [optionally] write results to pickle files.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
#import cPickle as pickle
import pickle
import time

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, g_type_mean_size
import provider
from train_util import get_batch

# Initialize the flags
def get_config(cliParams=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
    parser.add_argument('--model_path', default='train/log_v1/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for inference [default: 32]')
    parser.add_argument('--output', default='test_results', help='output file/folder name [default: test_results]')
    parser.add_argument('--data_path', default='kitti/frustum_carpedcyc_val_rgb_detection.pickle', help='frustum dataset pickle filepath [default: None]')
    parser.add_argument('--from_rgb_detection', action='store_true', help='test from dataset files from rgb detection.')
    parser.add_argument('--idx_path', default=None, help='filename of txt where each line is a data idx, used for rgb detection -- write <id>.txt for all frames. [default: None]')
    parser.add_argument('--dump_result', action='store_true', help='If true, also dump results to .pickle file')
    parser.add_argument('--only_seg', action='store_true', help='Only take PointCloud as Input and output nx1')
    if not cliParams:
        FLAGS = parser.parse_args()
    else:
        FLAGS = parser.parse_args(cliParams.split())
    return FLAGS

FLAGS = get_config(None)

# Set training configurations
BATCH_SIZE = FLAGS.batch_size
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
MODEL = importlib.import_module(FLAGS.model)
NUM_CLASSES = 2
NUM_CHANNEL = 3
global FRAME_TIME

# Load Frustum Datasets.
TEST_DATASET = provider.FrustumDataset(npoints=NUM_POINT, split='val',
    rotate_to_center=True, overwritten_data_path=FLAGS.data_path,
    from_rgb_detection=FLAGS.from_rgb_detection, one_hot=True)


def get_session_and_ops(batch_size, num_point):
    ''' Define model graph, load model parameters,
    create session and return session handle and tensors
    '''
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl, size_box2d_pl = \
                MODEL.placeholder_inputs(batch_size, num_point)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            end_points = MODEL.get_model(pointclouds_pl, size_box2d_pl, one_hot_vec_pl,
                is_training_pl)
            loss = MODEL.get_loss(labels_pl, centers_pl,
                heading_class_label_pl, heading_residual_label_pl,
                size_class_label_pl, size_residual_label_pl, end_points)
            saver = tf.train.Saver()

        # Create a session
#        print ("TEST if running on GPU")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'one_hot_vec_pl': one_hot_vec_pl,
               'size_box2d_pl': size_box2d_pl,
               'labels_pl': labels_pl,
               'centers_pl': centers_pl,
               'heading_class_label_pl': heading_class_label_pl,
               'heading_residual_label_pl': heading_residual_label_pl,
               'size_class_label_pl': size_class_label_pl,
               'size_residual_label_pl': size_residual_label_pl,
               'is_training_pl': is_training_pl,
               'logits': end_points['mask_logits'],
               'center': end_points['center'],
               'end_points': end_points,
               'loss': loss}
        return sess, ops
    
    
def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def inference(sess, ops, pc, one_hot_vec, box2d, batch_size):
    ''' Run inference for frustum pointnets in batch mode '''
    global FRAME_TIME
#    assert pc.shape[0]%batch_size == 0
    num_batches = pc.shape[0]/batch_size
    logits = np.zeros((pc.shape[0], pc.shape[1], NUM_CLASSES))
    centers = np.zeros((pc.shape[0], 3))
    heading_logits = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    heading_residuals = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    size_logits = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER))
    size_residuals = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER, 3))
    scores = np.zeros((pc.shape[0],)) # 3D box score 
   
    ep = ops['end_points'] 
    

    for i in range(int(num_batches)):
        feed_dict = {\
            ops['pointclouds_pl']: pc[i*batch_size:(i+1)*batch_size,...],
            ops['one_hot_vec_pl']: one_hot_vec[i*batch_size:(i+1)*batch_size,:],
            ops['size_box2d_pl']: box2d[i*batch_size:(i+1)*batch_size,:],
            ops['is_training_pl']: False}
        
        t1 = time.time()
        batch_logits, batch_centers, \
        batch_heading_scores, batch_heading_residuals, \
        batch_size_scores, batch_size_residuals, stage1 = \
            sess.run([ops['logits'], ops['center'],
                ep['heading_scores'], ep['heading_residuals'],
                ep['size_scores'], ep['size_residuals'], ep['stage1_center']],
                feed_dict=feed_dict)
        FRAME_TIME += time.time() - t1
        print("PointNets inference for 1 batch (frame) in {:.3f}s".format(time.time() - t1))
#        print ("T-Net Stage1 Center: ", stage1)
        logits[i*batch_size:(i+1)*batch_size,...] = batch_logits
        centers[i*batch_size:(i+1)*batch_size,...] = batch_centers
        heading_logits[i*batch_size:(i+1)*batch_size,...] = batch_heading_scores
        heading_residuals[i*batch_size:(i+1)*batch_size,...] = batch_heading_residuals
        size_logits[i*batch_size:(i+1)*batch_size,...] = batch_size_scores
        size_residuals[i*batch_size:(i+1)*batch_size,...] = batch_size_residuals
    
        # Compute scores
        batch_seg_prob = softmax(batch_logits)[:,:,1] # BxN
        batch_seg_mask = np.argmax(batch_logits, 2) # BxN
        mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1) # B,
        mask_mean_prob = mask_mean_prob / np.sum(batch_seg_mask,1) # B,
        heading_prob = np.max(softmax(batch_heading_scores),1) # B
        size_prob = np.max(softmax(batch_size_scores),1) # B,
        batch_scores = np.log(mask_mean_prob) + np.log(heading_prob) + np.log(size_prob)
        scores[i*batch_size:(i+1)*batch_size] = batch_scores 
    # Finished computing scores
    heading_cls = np.argmax(heading_logits, 1) # B
    size_cls = np.argmax(size_logits, 1) # B
    heading_res = np.array([heading_residuals[i,heading_cls[i]] \
        for i in range(pc.shape[0])])
    size_res = np.vstack([size_residuals[i,size_cls[i],:] \
        for i in range(pc.shape[0])])

    return np.argmax(logits, 2), centers, heading_cls, heading_res, size_cls, size_res, scores

def write_detection_results(result_dir, id_list, type_list, box2d_list, center_list, \
                            heading_cls_list, heading_res_list, \
                            size_cls_list, size_res_list, \
                            rot_angle_list, score_list):
    ''' Write frustum pointnets results to KITTI format label files. '''
    if result_dir is None: return
    results = {} # map from idx to list of strings, each string is a line (without \n)
    for i in range(len(center_list)):
        idx = id_list[i]
        output_str = type_list[i] + " -1 -1 -10 "
        box2d = box2d_list[i]
        output_str += "%f %f %f %f " % (box2d[0],box2d[1],box2d[2],box2d[3])
        h,w,l,tx,ty,tz,ry = provider.from_prediction_to_label_format(center_list[i],
            heading_cls_list[i], heading_res_list[i],
            size_cls_list[i], size_res_list[i], rot_angle_list[i])
        score = score_list[i]
        output_str += "%f %f %f %f %f %f %f %f" % (h,w,l,tx,ty,tz,ry,score)
        if idx not in results: results[idx] = []
        results[idx].append(output_str)

    # Write TXT files
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt'%(idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line+'\n')
        fout.close() 

def fill_files(output_dir, to_fill_filename_list):
    ''' Create empty files if not exist for the filelist. '''
    for filename in to_fill_filename_list:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            fout = open(filepath, 'w')
            fout.close()

def test_from_rgb_detection(output_filename, result_dir=None):
    ''' Test frustum pointents with 2D boxes from a RGB detector.
    Write test results to KITTI format label files.
    todo (rqi): support variable number of points.
    '''
    ps_list = []
    segp_list = []
    center_list = []
    heading_cls_list = []
    heading_res_list = []
    size_cls_list = []
    size_res_list = []
    rot_angle_list = []
    score_list = []
    onehot_list = []
    
    global FRAME_TIME
    FRAME_TIME = 0
    
    test_idxs = np.arange(0, len(TEST_DATASET))
    print(len(TEST_DATASET))
#    print ("id list: ", TEST_DATASET.id_list)
    
    batch_size = BATCH_SIZE
    num_batches = int((len(TEST_DATASET)+batch_size-1)/batch_size)
    
    batch_data_to_feed = np.zeros((batch_size, NUM_POINT, NUM_CHANNEL))
    batch_one_hot_to_feed = np.zeros((batch_size, 3))
    sess, ops = get_session_and_ops(batch_size=batch_size, num_point=NUM_POINT)

    for batch_idx in range(num_batches):
        print('batch idx: %d' % (batch_idx))
        start_idx = batch_idx * batch_size
        end_idx = min(len(TEST_DATASET), (batch_idx+1) * batch_size)
        cur_batch_size = end_idx - start_idx

        batch_data, batch_rot_angle, batch_rgb_prob, batch_one_hot_vec = \
            get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL, from_rgb_detection=True)
        batch_data_to_feed[0:cur_batch_size,...] = batch_data
        batch_one_hot_to_feed[0:cur_batch_size,:] = batch_one_hot_vec

        # BE CAREFUL HERE WHILE TESTING MyPointNet
        box2d_listFeed = np.asarray(TEST_DATASET.box2d_list[start_idx:end_idx])
#        print ("org 2dbb: ", box2d_listFeed)
        batch_yolo_to_feed = np.array((box2d_listFeed[:,2]-box2d_listFeed[:,0], box2d_listFeed[:,3]-box2d_listFeed[:,1])).transpose()
#        print ("FEEDING YOLO INFO: ", batch_yolo_to_feed)
#        batch_yolo_to_feed = np.zeros((cur_batch_size,2))
        
        batch_output, batch_center_pred, \
        batch_hclass_pred, batch_hres_pred, \
        batch_sclass_pred, batch_sres_pred, batch_scores = \
            inference(sess, ops, batch_data,
                batch_one_hot_vec, batch_yolo_to_feed, batch_size=batch_size)
            
	
        for i in range(cur_batch_size):
            ps_list.append(batch_data[i,...])
            segp_list.append(batch_output[i,...])
            center_list.append(batch_center_pred[i,:])
            heading_cls_list.append(batch_hclass_pred[i])
            heading_res_list.append(batch_hres_pred[i])
            size_cls_list.append(batch_sclass_pred[i])
            size_res_list.append(batch_sres_pred[i,:])
            rot_angle_list.append(batch_rot_angle[i])
            #score_list.append(batch_scores[i])
            score_list.append(batch_rgb_prob[i]) # 2D RGB detection score
            onehot_list.append(batch_one_hot_vec[i])

    if FLAGS.dump_result:
        with open(output_filename, 'wp') as fp:
            pickle.dump(ps_list, fp)
            pickle.dump(segp_list, fp)
            pickle.dump(center_list, fp)
            pickle.dump(heading_cls_list, fp)
            pickle.dump(heading_res_list, fp)
            pickle.dump(size_cls_list, fp)
            pickle.dump(size_res_list, fp)
            pickle.dump(rot_angle_list, fp)
            pickle.dump(score_list, fp)
            pickle.dump(onehot_list, fp)

    # Write detection results for KITTI evaluation
    print('Number of point clouds: %d' % (len(ps_list)))
    write_detection_results(result_dir, TEST_DATASET.id_list,
        TEST_DATASET.type_list, TEST_DATASET.box2d_list,
        center_list, heading_cls_list, heading_res_list,
        size_cls_list, size_res_list, rot_angle_list, score_list)
    # Make sure for each frame (no matter if we have measurment for that frame),
    # there is a TXT file
    output_dir = os.path.join(result_dir, 'data')
    if FLAGS.idx_path is not None:
        to_fill_filename_list = [line.rstrip()+'.txt' \
            for line in open(FLAGS.idx_path)]
        fill_files(output_dir, to_fill_filename_list)

def test(output_filename, result_dir=None):
    ''' Test frustum pointnets with GT 2D boxes.
    Write test results to KITTI format label files.
    todo (rqi): support variable number of points.
    '''
    ps_list = []
    seg_list = []
    segp_list = []
    center_list = []
    heading_cls_list = []
    heading_res_list = []
    size_cls_list = []
    size_res_list = []
    rot_angle_list = []
    score_list = []

    global FRAME_TIME
    FRAME_TIME = 0
    
    test_idxs = np.arange(0, len(TEST_DATASET))
    batch_size = BATCH_SIZE
    num_batches = len(TEST_DATASET)/batch_size

    sess, ops = get_session_and_ops(batch_size=batch_size, num_point=NUM_POINT)
    correct_cnt = 0
    for batch_idx in range(int(num_batches)): #num_batches
        print('batch idx: %d' % (batch_idx))
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx+1) * batch_size

        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec, batch_box2d = \
            get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,
                NUM_POINT, NUM_CHANNEL)
            

        # BE CAREFUL HERE WHILE TESTING MyPointNet
        batch_yolo_to_feed = np.zeros((batch_size,2))
        
        batch_output, batch_center_pred, \
        batch_hclass_pred, batch_hres_pred, \
        batch_sclass_pred, batch_sres_pred, batch_scores = \
            inference(sess, ops, batch_data,
                batch_one_hot_vec, batch_yolo_to_feed, batch_size=batch_size)


        correct_cnt += np.sum(batch_output==batch_label)
	
        for i in range(batch_output.shape[0]):
            ps_list.append(batch_data[i,...])
            seg_list.append(batch_label[i,...])
            segp_list.append(batch_output[i,...])
            center_list.append(batch_center_pred[i,:])
            heading_cls_list.append(batch_hclass_pred[i])
            heading_res_list.append(batch_hres_pred[i])
            size_cls_list.append(batch_sclass_pred[i])
            size_res_list.append(batch_sres_pred[i,:])
            rot_angle_list.append(batch_rot_angle[i])
            score_list.append(batch_scores[i])

    print("Segmentation accuracy: %f" % \
        (correct_cnt / float(batch_size*num_batches*NUM_POINT)))

    if FLAGS.dump_result:
        with open(output_filename, 'wp') as fp:
            pickle.dump(ps_list, fp)
            pickle.dump(seg_list, fp)
            pickle.dump(segp_list, fp)
            pickle.dump(center_list, fp)
            pickle.dump(heading_cls_list, fp)
            pickle.dump(heading_res_list, fp)
            pickle.dump(size_cls_list, fp)
            pickle.dump(size_res_list, fp)
            pickle.dump(rot_angle_list, fp)
            pickle.dump(score_list, fp)

    # Write detection results for KITTI evaluation
    write_detection_results(result_dir, TEST_DATASET.id_list,
        TEST_DATASET.type_list, TEST_DATASET.box2d_list, center_list,
        heading_cls_list, heading_res_list,
        size_cls_list, size_res_list, rot_angle_list, score_list)

def test_segmentation(input_pc, input_centroid, result_dir=None):
    ''' Test segmentation pointents with frustum point clouds.
        load PointCloud from a VTK file
    '''
    ps_list = []
    segp_list = []
    center_list = []
    heading_cls_list = []
    heading_res_list = []
    size_cls_list = []
    size_res_list = []
    rot_angle_list = []
    score_list = []
    onehot_list = []

    # load PointCloud here
#    import vtk
#    reader = vtk.vtkPolyDataReader()
#    reader.SetFileName("/localhome/sxu/Desktop/MA/frustum-pointnets-master/dataset/xtion/output.vtk")
#    reader.Update()
#    polydata = reader.GetOutput()
#    np_pts = []
#    for i in range(polydata.GetNumberOfCells()):
#       pts = polydata.GetCell(i).GetPoints()    
#       np_pts.append( np.array([pts.GetPoint(i) for i in range(pts.GetNumberOfPoints())]) )
#    pc = np.asarray(np_pts).squeeze() # numPoints x 3
    t0 = time.time()
    if input_pc is None:
        pc = np.fromfile("/localhome/sxu/Desktop/MA/frustum-pointnets-master/dataset/xtion/kinect.bin", dtype=np.float).reshape(-1, 3)
    else:
        pc = input_pc
    # Down sample the PointCloud
    pc = pc[np.random.randint(pc.shape[0], size=int(pc.shape[0]/10)), 0:3]
    # Get 3d centroid from 2d detection
    if input_centroid is None:
        pc_centroid = np.fromfile("/localhome/sxu/Desktop/MA/frustum-pointnets-master/dataset/xtion/kinect_centroid.bin", dtype=np.float).reshape(-1, 3)
    else:
        pc_centroid = input_centroid
    # Manual calibration
    import importlib.util
    def module_from_file(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    utils = module_from_file("utils", "/localhome/sxu/Desktop/MA/frustum-pointnets-master/kitti/kitti_util.py")
#    viz_util = module_from_file("viz_util", "/localhome/sxu/Desktop/MA/frustum-pointnets-master/mayavi/viz_util.py")
    cali = utils.Calibration('/localhome/sxu/Desktop/MA/frustum-pointnets-master/dataset/KITTI/object/training/calib/000002.txt')
    pc_rect = cali.project_velo_to_rect(pc[:,0:3])
    pc_centroid_rect = cali.project_velo_to_rect(pc_centroid[:,0:3])
    
    # calculate heading angle (to rotate) 

    frustum_angle = -1 * np.arctan2(pc_centroid_rect[0, 2], pc_centroid_rect[0, 0])  +  np.pi/2
#    frustum_angle = -1 * np.arctan2(np.average(pc_rect[:,2]), np.average(pc_rect[:,0])) 
    
    # rotate along y
    cosval = np.cos(frustum_angle)
    sinval = np.sin(frustum_angle)
    rotmat = np.array([[cosval, -sinval],[sinval, cosval]])
    pc_rect[:,[0,2]] = np.dot(pc_rect[:,[0,2]], np.transpose(rotmat))
    #pc[:, 0] = -pc[:, 0]
    #pc[:, 1] = -pc[:, 1]
    #pc_rect[:, 2] = pc_rect[:, 2] + 10
    pc_rect = np.expand_dims(pc_rect, 0)
    
    if (pc.shape[1] > 0):
        print("VTK successfully loaded! dim: ", pc.shape)
    else:
        print ("Failed to load VTK file!")
    #test_idxs = np.arange(0, len(pc))
    print("points: ", pc_rect.shape)
    batch_size = 1 #pc.shape[0] 
    num_batches = 1 #int((pc.shape[0]+batch_size-1)/batch_size)
    print ("batchsize: ", batch_size, "num_batches: ", num_batches)
    
    # hand-made data
    batch_data_to_feed = np.zeros((1, pc_rect.shape[1], 3))
    batch_one_hot_to_feed = np.zeros((batch_size, 3))
    sess, ops = get_session_and_ops(batch_size=batch_size, num_point=pc_rect.shape[1])
    
    for batch_idx in range(1):
        print('batch idx: %d' % (batch_idx))
        start_idx = batch_idx * batch_size
        end_idx = min(len(TEST_DATASET), (batch_idx+1) * batch_size)
        cur_batch_size = end_idx - start_idx

        batch_data = pc_rect  # problems: 1.centroid 2.scale
        batch_one_hot_vec = np.asarray([0,1,0]) #pedestrian
        batch_data_to_feed[0,...] = batch_data
        batch_one_hot_to_feed[0,:] = batch_one_hot_vec
        
#        t1 = time.time()
        
        # Run one batch inference
        batch_output, batch_center_pred, \
            batch_hclass_pred, batch_hres_pred, \
            batch_sclass_pred, batch_sres_pred, batch_scores = \
                inference(sess, ops, batch_data_to_feed,
                    batch_one_hot_to_feed, batch_size=batch_size)
    print("Inference done;")
    
    print("Batch segmentation logits: ", batch_output)
    if (1 not in batch_output):
        print("Segmentation failed: no pedestrain found!")
        return
    else:
        print("Pedestrain found! Number of points: ", (batch_output==1).sum()) #(pc[0,:,2]<4).sum())
        print("Batch predicted center (in rect cam): ", batch_center_pred)
        print("Batch predicted center (in kinect): ", cali.project_rect_to_velo(batch_center_pred))
        boxParams = g_type_mean_size['Pedestrian'] + batch_sres_pred[0]  # could be wrong here?
        print(boxParams)
        idx = np.where(batch_output==1)[1]
        fig = pyplot.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        bx = fig.add_subplot(1, 2, 2, projection='3d')
        # Generate the values
        x_vals = pc[ idx, 0]
        y_vals = pc[ idx, 1]
        z_vals = pc[ idx, 2]

        # Plot the values (masked)
        ax.scatter(x_vals, y_vals, z_vals, c = 'b', marker='o')
        # Draw 3D box
        vertices = get3dBoxVertices(cali.project_rect_to_velo(batch_center_pred), boxParams)

        ax.plot([vertices[0][0], vertices[1][0]], [vertices[0][1], vertices[1][1]], [vertices[0][2], vertices[1][2]], c='r')
        ax.plot([vertices[1][0], vertices[2][0]], [vertices[1][1], vertices[2][1]], [vertices[1][2], vertices[2][2]], c='r')
        ax.plot([vertices[2][0], vertices[3][0]], [vertices[2][1], vertices[3][1]], [vertices[2][2], vertices[3][2]], c='r')
        ax.plot([vertices[3][0], vertices[0][0]], [vertices[3][1], vertices[0][1]], [vertices[3][2], vertices[0][2]], c='r')
        
        ax.plot([vertices[4][0], vertices[5][0]], [vertices[4][1], vertices[5][1]], [vertices[4][2], vertices[5][2]], c='r')
        ax.plot([vertices[5][0], vertices[6][0]], [vertices[5][1], vertices[6][1]], [vertices[5][2], vertices[6][2]], c='r')
        ax.plot([vertices[6][0], vertices[7][0]], [vertices[6][1], vertices[7][1]], [vertices[6][2], vertices[7][2]], c='r')
        ax.plot([vertices[7][0], vertices[4][0]], [vertices[7][1], vertices[4][1]], [vertices[7][2], vertices[4][2]], c='r')
        
        ax.plot([vertices[0][0], vertices[4][0]], [vertices[0][1], vertices[4][1]], [vertices[0][2], vertices[4][2]], c='r')
        ax.plot([vertices[1][0], vertices[5][0]], [vertices[1][1], vertices[5][1]], [vertices[1][2], vertices[5][2]], c='r')
        ax.plot([vertices[2][0], vertices[6][0]], [vertices[2][1], vertices[6][1]], [vertices[2][2], vertices[6][2]], c='r')
        ax.plot([vertices[3][0], vertices[7][0]], [vertices[3][1], vertices[7][1]], [vertices[3][2], vertices[7][2]], c='r')
        
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        # Plot the original point cloud
        bx.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c = 'b', marker='o')
        bx.set_xlabel('X-axis')
        bx.set_ylabel('Y-axis')
        bx.set_zlabel('Z-axis')
        pyplot.show()
        input("PRESS ENTER TO CONTINUE")
    print("Batch predicted center (in rect cam): ", batch_center_pred)
    print("Batch predicted center (in kinect): ", cali.project_rect_to_velo(batch_center_pred))
    # write segmented PointCloud into a txt file
    pc = pc.squeeze()
    for i in range(pc.shape[1]):
        for j in range(3):
            pc[i, j] = pc[i, j] * batch_output[0, i]
    np.savetxt('/localhome/sxu/Desktop/MA/frustum-pointnets-master/SegmentedPC.out', pc, delimiter=' ')
    print("Segmented PointCloud into a txt file written!")

def get3dBoxVertices(center, size):
    '''
    center: (n,3)
    size:   (n,3)
    output: (n, 8,3)
    '''
    x_off = size[0]/2
    y_off = size[1]/2
    z_off = size[2]/2
    c_x = center[0][0]
    c_y = center[0][1]
    c_z = center[0][2]
    vertices = []
    vertices.append([c_x-x_off, c_y-y_off, c_z-z_off])
    vertices.append([c_x+x_off, c_y-y_off, c_z-z_off])
    vertices.append([c_x+x_off, c_y+y_off, c_z-z_off])
    vertices.append([c_x-x_off, c_y+y_off, c_z-z_off])
    vertices.append([c_x-x_off, c_y-y_off, c_z+z_off])
    vertices.append([c_x+x_off, c_y-y_off, c_z+z_off])
    vertices.append([c_x+x_off, c_y+y_off, c_z+z_off])
    vertices.append([c_x-x_off, c_y+y_off, c_z+z_off])
    return vertices


if __name__=='__main__':
    if FLAGS.only_seg:
        test_segmentation(None, None, FLAGS.output)
    elif FLAGS.from_rgb_detection:
        test_from_rgb_detection(FLAGS.output+'.pickle', FLAGS.output)
    else:
        test(FLAGS.output+'.pickle', FLAGS.output)
