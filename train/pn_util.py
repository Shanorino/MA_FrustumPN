'''
Utility class for 3D detection (Segmentation + T-net + Amodal)

Author: Shane Xu
Date: June 2019
'''

from __future__ import print_function

import os
import sys
#import argparse
import importlib
import numpy as np
import tensorflow as tf
#import cPickle as pickle
import pickle
import time
from mpl_toolkits.mplot3d import Axes3D
import importlib.util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from matplotlib import pyplot
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, g_type_mean_size

# Set training configurations
BATCH_SIZE = 32
MODEL_PATH = '/localhome/sxu/Desktop/MA/frustum-pointnets-master/train/log_v1/model.ckpt'
GPU_INDEX = 0
NUM_POINT = 1024
MODEL = importlib.import_module('frustum_pointnets_v1')
NUM_CLASSES = 2
NUM_CHANNEL = 3

def get_session_and_ops(batch_size, num_point):
    ''' Define model graph, load model parameters,
    create session and return session handle and tensors
    '''
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl = \
                MODEL.placeholder_inputs(batch_size, num_point)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl,
                is_training_pl)
            loss = MODEL.get_loss(labels_pl, centers_pl,
                heading_class_label_pl, heading_residual_label_pl,
                size_class_label_pl, size_residual_label_pl, end_points)
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'one_hot_vec_pl': one_hot_vec_pl,
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


def inference(sess, ops, pc, one_hot_vec, batch_size):
    ''' Run inference for frustum pointnets in batch mode '''
    assert pc.shape[0]%batch_size == 0
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
            ops['is_training_pl']: False}
        
        t1 = time.time()
        batch_logits, batch_centers, \
        batch_heading_scores, batch_heading_residuals, \
        batch_size_scores, batch_size_residuals = \
            sess.run([ops['logits'], ops['center'],
                ep['heading_scores'], ep['heading_residuals'],
                ep['size_scores'], ep['size_residuals']],
                feed_dict=feed_dict)
            
        print("PointNets inference for 1 batch (frame) in {:.2f}s".format(time.time() - t1))
        
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


def fill_files(output_dir, to_fill_filename_list):
    ''' Create empty files if not exist for the filelist. '''
    for filename in to_fill_filename_list:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            fout = open(filepath, 'w')
            fout.close()
    

def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module    
    
def test_segmentation(input_pc, input_centroid, sess, ops):
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
    
    if input_pc is None:
        pc = np.fromfile("/localhome/sxu/Desktop/MA/frustum-pointnets-master/dataset/xtion/kinect3.bin", dtype=np.float).reshape(-1, 3)
    else:
        pc = input_pc
    # Down sample the PointCloud
#    pc = pc[np.where(pc[:,0]<3.5)]
#    pc = pc[np.where(pc[:,2]>-1.6)]
    pc = pc[np.random.randint(pc.shape[0], size=2048), 0:3] #10%: size=int(pc.shape[0]/10)
    # Get 3d centroid from 2d detection
    if input_centroid is None:
        pc_centroid = np.fromfile("/localhome/sxu/Desktop/MA/frustum-pointnets-master/dataset/xtion/kinect_centroid3.bin", dtype=np.float).reshape(-1, 3)
    else:
        pc_centroid = input_centroid
    # Manual calibration
    
    utils = module_from_file("utils", "/localhome/sxu/Desktop/MA/frustum-pointnets-master/kitti/kitti_util.py")
#    viz_util = module_from_file("viz_util", "/localhome/sxu/Desktop/MA/frustum-pointnets-master/mayavi/viz_util.py")
    cali = utils.Calibration('/localhome/sxu/Desktop/MA/frustum-pointnets-master/dataset/KITTI/object/training/calib/000002.txt')
    t0 = time.time()
    pc_rect = cali.project_velo_to_rect(pc[:,0:3])
    pc_centroid_rect = cali.project_velo_to_rect(pc_centroid[:,0:3])
    # rotate pc along z
    pc = frustumRotation(pc, pc_centroid, "z")
    pc_rect = frustumRotation(pc_rect, pc_centroid_rect, "y")
    
    # calculate heading angle (to rotate) (right-handed axis)
    frustum_angle = (-1 * np.arctan2(pc_centroid_rect[0, 2], pc_centroid_rect[0, 0])  +  np.pi/2)
    
#    # For the rect coord: rotate along y
#    cosval = np.cos(frustum_angle)
#    sinval = np.sin(frustum_angle)
#    rotmat = np.array([[cosval, -sinval],[sinval, cosval]])
#    pc_rect[:,[0,2]] = np.dot(pc_rect[:,[0,2]], np.transpose(rotmat))
    
    # For the Velodyne coord: rotate the original PC too (along z)
    
    
    #pc[:, 0] = -pc[:, 0]
    #pc[:, 1] = -pc[:, 1]
#    pc_rect[:, 2] = pc_rect[:, 2]
    
    # to feed the tf model
    pc_rect = np.expand_dims(pc_rect, 0)
    
    print("points: ", pc_rect.shape)
    batch_size = 1 #pc.shape[0] 
    num_batches = 1 #int((pc.shape[0]+batch_size-1)/batch_size)
    print ("batchsize: ", batch_size, "num_batches: ", num_batches)
    
    # hand-made data
    batch_data_to_feed = np.zeros((1, pc_rect.shape[1], 3))
    batch_one_hot_to_feed = np.zeros((batch_size, 3))
#    sess, ops = get_session_and_ops(batch_size=batch_size, num_point=pc_rect.shape[1])
    
    for batch_idx in range(1):
        print('batch idx: %d' % (batch_idx))
        start_idx = batch_idx * batch_size
        end_idx = min(1, (batch_idx+1) * batch_size)
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
    print("Inference + rectifying in {:.2f}s".format(time.time() - t0))
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
        vertices = get3dBoxVertices(cali.project_rect_to_velo(batch_center_pred), boxParams, -frustum_angle)
        print ("Frustum angle in rect: ", frustum_angle)
#        vertices = get3dBoxVertices(cali.project_rect_to_velo(batch_center_pred), boxParams, np.pi/6 * batch_hclass_pred[0] + batch_hres_pred[0])

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
        pyplot.close()
#        input("PRESS ENTER TO CONTINUE")
#    print("Batch predicted center (in rect cam): ", batch_center_pred)
#    print("Batch predicted center (in kinect): ", cali.project_rect_to_velo(batch_center_pred))
        
        
#    # write segmented PointCloud into a txt file   ---- only in debug mode ----
#    pc = pc.squeeze()
#    for i in range(pc.shape[1]):
#        for j in range(3):
#            pc[i, j] = pc[i, j] * batch_output[0, i]
#    np.savetxt('/localhome/sxu/Desktop/MA/frustum-pointnets-master/SegmentedPC.out', pc, delimiter=' ')
#    print("Segmented PointCloud into a txt file written!")
        
def frustumRotation(frustum, frustum_centroid, axis): #along z
    if (axis == "z"):
        frustum_angle = -1 * np.arctan2(frustum_centroid[0, 0], frustum_centroid[0, 1])  +  np.pi/2
        print ("Frustum angle in Velo: ", frustum_angle)
    #    frustum_angle = -1 * np.arctan2(np.average(pc_rect[:,2]), np.average(pc_rect[:,0])) 
    #    frustum_angle = 0.8
        # For the rect coord: rotate along z
        cosval = np.cos(frustum_angle)
        sinval = np.sin(frustum_angle)
        rotmat = np.array([[cosval, sinval],[-sinval, cosval]])
        frustum[:,[0,1]] = np.dot(frustum[:,[0,1]], np.transpose(rotmat))
    elif (axis == "y"):
        frustum_angle = -1 * np.arctan2(frustum_centroid[0, 2], frustum_centroid[0, 0])  +  np.pi/2
        print ("Frustum angle in Velo: ", frustum_angle)
        cosval = np.cos(frustum_angle)
        sinval = np.sin(frustum_angle)
        rotmat = np.array([[cosval, -sinval],[sinval, cosval]])
        frustum[:,[0,2]] = np.dot(frustum[:,[0,2]], np.transpose(rotmat))
    return frustum
def get3dBoxVertices(center, size, heading):
    '''
    center: (n,3)
    size:   (n,3)
    output: (n, 8,3)
    '''
    r_matrix = np.array([[np.cos(heading), -np.sin(heading), 0], [np.sin(heading), np.cos(heading), 0], [0 , 0, 1]])
    # rotate
#    size = np.dot(size, r_matrix)
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
#    vertices = np.dot(vertices, r_matrix)
    return vertices

    