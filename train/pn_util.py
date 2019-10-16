'''
Utility class for 3D detection (Segmentation + T-net + Amodal)

Author: Shu Xu
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
#import pickle
import time
from mpl_toolkits.mplot3d import Axes3D
import importlib.util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
FPN_DIR = '/localhome/sxu/Desktop/MA/frustum-pointnets-master/'
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from matplotlib import pyplot
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, g_type_mean_size
import mayavi.mlab as mlab
from scipy.spatial import ConvexHull

# Set training configurations
BATCH_SIZE = 32
#MODEL_PATH = '/localhome/sxu/Desktop/MA/frustum-pointnets-master/train/log_tiny2_100/model.ckpt'
MODEL_PATH = FPN_DIR + 'train/log_tiny1_100/model.ckpt'
GPU_INDEX = 0
NUM_POINT = 1024
#MODEL = importlib.import_module('frustum_pointnets_v1_tiny2')
MODEL = importlib.import_module('frustum_pointnets_v1_tiny1')
NUM_CLASSES = 2
NUM_CHANNEL = 3
ONE_HOT_TEMPLATE = {'Pedestrian': [0, 1, 0], 'Cyclist': [0, 0, 1], 'Car': [1, 0, 0]} # do not change the order
LOG_FOUT = open(os.path.join(ROOT_DIR, 'speed_test.txt'), 'w')
global FRAME_TIME

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

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
    
def test_segmentation(input_pc, input_centroid, sess, ops, objectBoxes=None, objectTypes=None):
    ''' Test segmentation pointents with frustum point clouds.
        
        input_pc: the input point cloud, if is None then reads offline data
        input_centroid: the input centroid of the point cloud (from 2D detection + 3D projection), if is None then reads offline data
        sess: the Tensorflow session
        ops: the placeholders, loss and endpoints
        objectTypes: the classification result from 2D detector
    '''
    vertices_list = []
    global FRAME_TIME
    FRAME_TIME = 0
    CAMERA_SCALE = 3
#    print (objectBoxes)
    # load PointCloud here    
    if input_pc is None:
        pc = np.fromfile(FPN_DIR + "dataset/xtion/rsPC3rrrrrr.bin", dtype=np.float).reshape(-1, 2048, 3)
    else:
        pc = input_pc#[0, :, :]
    # Cropping-trick
#    pc = pc[np.where(pc[:,1]<0)]
#    pc = pc[np.where(pc[:,2]>-1.6)]
#    pc[:,0] += 10
    
#    pc = pc[np.random.randint(pc.shape[0], size=1024), 0:3] #10%: size=int(pc.shape[0]/10)
    # Get 3d centroid from 2d detection
    if input_centroid is None:
        pc_centroid = np.fromfile(FPN_DIR + "dataset/xtion/rsCentroid3rrrrrr.bin", dtype=np.float).reshape(-1, 3)
    else:
        pc_centroid = input_centroid
        
    # Manual calibration
    utils = module_from_file("utils", FPN_DIR + "kitti/kitti_util.py")
    cali = utils.Calibration(FPN_DIR + 'dataset/KITTI/object/training/calib/000002.txt')
    
    t0 = time.time()
    objNum = pc.shape[0] # How many objects are detected by 2D detection
    
    # Frustum rotation
    pc_rect = np.zeros_like(pc)
    pc_centroid_rect = np.zeros_like(pc_centroid)
    frustum_angle = []
    for i in range(objNum):
        pc_rect[i, :, :] = cali.project_velo_to_rect(pc[i, :, :])
        pc_centroid_rect[i, :] = cali.project_velo_to_rect(pc_centroid[i, :].reshape(-1,3))
        # rotate pc along z
#        pc[i, :, :] = frustumRotation(pc[i, :, :], pc_centroid[i, :].reshape(-1,3), "z")
        pc_rect[i, :, :] = frustumRotation(pc_rect[i, :, :], pc_centroid_rect[i, :].reshape(-1,3), "y")
        
        # calculate heading angle (to rotate) (for the 3d bounding box) (right-handed axis)
        frustum_angle.append(1 * np.arctan2(pc_centroid[i, 0], pc_centroid[i, 1])  -  np.pi/2) 
    
    #pc[:, 0] = -pc[:, 0]
    #pc[:, 1] = -pc[:, 1]
#    pc_rect[:, 2] = pc_rect[:, 2] + 10
    
    # to feed the tf model
#    pc_rect = np.expand_dims(pc_rect, 0)
    
    print("Input points: ", pc_rect.shape)
#    if np.array_equal(pc_rect[0, :, :], pc_rect[1, :, :]):
#        print ("IDENTICAL!")
#    else:
#        print ("NOT IDENTICAL!")
    batch_size = 1 #pc.shape[0] # how many pictures
    num_batches = 1 #int((pc.shape[0]+batch_size-1)/batch_size)
#    print ("batchsize: ", batch_size, "num_batches: ", num_batches)
    
    # hand-made data
    batch_data_to_feed = np.zeros((objNum, pc_rect.shape[1], 3))
    batch_one_hot_to_feed = np.zeros((objNum, 3))
    batch_yolo_to_feed = np.zeros((objNum,2))
#    sess, ops = get_session_and_ops(batch_size=batch_size, num_point=pc_rect.shape[1])
    FRAME_TIME += time.time() - t0
    print("Pre-processing in {:.3f}s".format(time.time() - t0))
#    t0 = time.time()
    for batch_idx in range(objNum):
        print('Item idx: %d' % (batch_idx))

        batch_data = pc_rect[batch_idx, :, :]  # problems: 1.centroid 2.scale
        if (objectTypes is not None):
            batch_one_hot_vec = cocoIndexToOnehot(objectTypes[batch_idx]) # for multiple instances
        else:
            batch_one_hot_vec = np.asarray(ONE_HOT_TEMPLATE['Pedestrian']) #pedestrian 
#        batch_yolo = np.asarray([190,240])*CAMERA_SCALE
        batch_yolo = np.asarray(objectBoxes[batch_idx])*CAMERA_SCALE
        
        batch_data_to_feed[batch_idx,...] = batch_data
        batch_one_hot_to_feed[batch_idx,:] = batch_one_hot_vec
        batch_yolo_to_feed[batch_idx,:] = batch_yolo
#        t1 = time.time()
        
    # Run one batch inference
    batch_output, batch_center_pred, \
        batch_hclass_pred, batch_hres_pred, \
        batch_sclass_pred, batch_sres_pred, batch_scores = \
            inference(sess, ops, batch_data_to_feed,
                batch_one_hot_to_feed, batch_yolo_to_feed, batch_size=batch_size)
#    print("Inference + rectifying done in {:.2f}s".format(time.time() - t0))
#    print("Batch segmentation logits: ", batch_output)
    if (1 not in batch_output):
        print("PointNet: I did not see anyone!")
        return
    else:
#        print("Pedestrain found! Number of points: ", (batch_output==1).sum()) #(pc[0,:,2]<4).sum())
#        print("Shape: ", batch_output.shape)
#        print("Batch predicted center (in rect cam): ", batch_center_pred) #Bx3
        print("Batch predicted center (in Velo): ", cali.project_rect_to_velo(batch_center_pred))
#        print("Batch heading angle(degree): ", batch_hclass_pred*30, "+", batch_hres_pred*57)
#        print("Batch heading angle(rad): ", batch_hclass_pred*np.pi/6, "+", batch_hres_pred)
        print ("classification from PN: ", batch_sclass_pred)
        boxParams = np.asarray(list(g_type_mean_size.values()))[batch_sclass_pred] + batch_sres_pred
#        boxParams = np.asarray(list(g_type_mean_size.values()))[[4,3]] + batch_sres_pred
#        boxParams = g_type_mean_size['Pedestrian'] + batch_sres_pred  # could be wrong here?
        print("3D box sizes: ", boxParams)
    
        # Mask the points from Seg-PN
#        idx = np.where(batch_output==1)[1]
#        x_vals = pc[0, idx, 0]
#        y_vals = pc[0, idx, 1]
#        z_vals = pc[0, idx, 2]

        # Draw 3D box
        
        vertices = get3dBoxVertices(cali.project_rect_to_velo(batch_center_pred), boxParams, frustum_angle, -np.pi/6 * batch_hclass_pred - batch_hres_pred)

#        print ("vertices: ", vertices)
#        print ("Frustum angle in rect: ", frustum_angle)
#        vertices = get3dBoxVertices(cali.project_rect_to_velo(batch_center_pred), boxParams, np.pi/6 * batch_hclass_pred[0] + batch_hres_pred[0])
        
#        visualizePNwithBox(x_vals, y_vals, z_vals, pc[0, :, :], vertices, True)
        
        if (False): # Offline Testing
            fig = mlab.figure(figure=None, bgcolor=(0,0,0),fgcolor=None, engine=None, size=(800, 500))
            verts = np.fromfile(FPN_DIR + "dataset/xtion/rsVerts3rrrrrr.bin", dtype=np.float).reshape(-1, 3)
            draw_lidar_with_boxes(verts, vertices, fig)
            draw_boxes3d(vertices, fig, draw_text=False)
            print ("FRAME_TIME: ", FRAME_TIME)
            log_string("FRAME_TIME: " + str(FRAME_TIME))
            input("ENTER TO QUIT TESTING")
        log_string("FPS: " + str(1/(FRAME_TIME+0.00001)))
        return vertices

def visualizePNwithBox(masked_x, masked_y, masked_z, pc, verticess, draw=False):
    fig = pyplot.figure()
    pyplot.ion()
    pyplot.show()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    bx = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(masked_x, masked_y, masked_z, c = 'r', marker='o')
    vertices = verticess[0,...]
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
    
#    vertices = verticess[1,...]
#    ax.plot([vertices[0][0], vertices[1][0]], [vertices[0][1], vertices[1][1]], [vertices[0][2], vertices[1][2]], c='r')
#    ax.plot([vertices[1][0], vertices[2][0]], [vertices[1][1], vertices[2][1]], [vertices[1][2], vertices[2][2]], c='r')
#    ax.plot([vertices[2][0], vertices[3][0]], [vertices[2][1], vertices[3][1]], [vertices[2][2], vertices[3][2]], c='r')
#    ax.plot([vertices[3][0], vertices[0][0]], [vertices[3][1], vertices[0][1]], [vertices[3][2], vertices[0][2]], c='r')
#    
#    ax.plot([vertices[4][0], vertices[5][0]], [vertices[4][1], vertices[5][1]], [vertices[4][2], vertices[5][2]], c='r')
#    ax.plot([vertices[5][0], vertices[6][0]], [vertices[5][1], vertices[6][1]], [vertices[5][2], vertices[6][2]], c='r')
#    ax.plot([vertices[6][0], vertices[7][0]], [vertices[6][1], vertices[7][1]], [vertices[6][2], vertices[7][2]], c='r')
#    ax.plot([vertices[7][0], vertices[4][0]], [vertices[7][1], vertices[4][1]], [vertices[7][2], vertices[4][2]], c='r')
#    
#    ax.plot([vertices[0][0], vertices[4][0]], [vertices[0][1], vertices[4][1]], [vertices[0][2], vertices[4][2]], c='r')
#    ax.plot([vertices[1][0], vertices[5][0]], [vertices[1][1], vertices[5][1]], [vertices[1][2], vertices[5][2]], c='r')
#    ax.plot([vertices[2][0], vertices[6][0]], [vertices[2][1], vertices[6][1]], [vertices[2][2], vertices[6][2]], c='r')
#    ax.plot([vertices[3][0], vertices[7][0]], [vertices[3][1], vertices[7][1]], [vertices[3][2], vertices[7][2]], c='r')
    
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    
    bx.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c = 'b', marker='o')
        
    bx.plot([vertices[0][0], vertices[1][0]], [vertices[0][1], vertices[1][1]], [vertices[0][2], vertices[1][2]], c='r')
    bx.plot([vertices[1][0], vertices[2][0]], [vertices[1][1], vertices[2][1]], [vertices[1][2], vertices[2][2]], c='r')
    bx.plot([vertices[2][0], vertices[3][0]], [vertices[2][1], vertices[3][1]], [vertices[2][2], vertices[3][2]], c='r')
    bx.plot([vertices[3][0], vertices[0][0]], [vertices[3][1], vertices[0][1]], [vertices[3][2], vertices[0][2]], c='r')
    
    bx.plot([vertices[4][0], vertices[5][0]], [vertices[4][1], vertices[5][1]], [vertices[4][2], vertices[5][2]], c='r')
    bx.plot([vertices[5][0], vertices[6][0]], [vertices[5][1], vertices[6][1]], [vertices[5][2], vertices[6][2]], c='r')
    bx.plot([vertices[6][0], vertices[7][0]], [vertices[6][1], vertices[7][1]], [vertices[6][2], vertices[7][2]], c='r')
    bx.plot([vertices[7][0], vertices[4][0]], [vertices[7][1], vertices[4][1]], [vertices[7][2], vertices[4][2]], c='r')
    
    bx.plot([vertices[0][0], vertices[4][0]], [vertices[0][1], vertices[4][1]], [vertices[0][2], vertices[4][2]], c='r')
    bx.plot([vertices[1][0], vertices[5][0]], [vertices[1][1], vertices[5][1]], [vertices[1][2], vertices[5][2]], c='r')
    bx.plot([vertices[2][0], vertices[6][0]], [vertices[2][1], vertices[6][1]], [vertices[2][2], vertices[6][2]], c='r')
    bx.plot([vertices[3][0], vertices[7][0]], [vertices[3][1], vertices[7][1]], [vertices[3][2], vertices[7][2]], c='r')
    bx.set_xlabel('X-axis')
    bx.set_ylabel('Y-axis')
    bx.set_zlabel('Z-axis')
    
    ax.view_init(elev=20., azim=200)
    bx.view_init(elev=20., azim=200)
    
    if (draw):
        pyplot.draw()
        pyplot.pause(1)
        input ("press ENTER to continue")
        pyplot.close()
    
def frustumRotation(frustum, frustum_centroid, axis): #along z
    if (axis == "z"):
        frustum_angle = -1 * np.arctan2(frustum_centroid[0, 0], frustum_centroid[0, 1])  +  np.pi/2 #- 0.1
#        print ("Frustum angle in Velo: ", frustum_angle, "From centroid: ", frustum_centroid[0, :])
        # For the rect coord: rotate along z
        cosval = np.cos(frustum_angle)
        sinval = np.sin(frustum_angle)
        rotmat = np.array([[cosval, sinval],[-sinval, cosval]])
        frustum[:,[0,1]] = np.dot(frustum[:,[0,1]], np.transpose(rotmat))
    elif (axis == "y"):
        frustum_angle = -1 * np.arctan2(frustum_centroid[0, 2], frustum_centroid[0, 0])  +  np.pi/2  #+ 0.2
#        print ("Frustum angle in Rect: ", frustum_angle, "From centroid: ", frustum_centroid[0, :])
        cosval = np.cos(frustum_angle)
        sinval = np.sin(frustum_angle)
        rotmat = np.array([[cosval, -sinval],[sinval, cosval]])
        frustum[:,[0,2]] = np.dot(frustum[:,[0,2]], np.transpose(rotmat))
    return frustum

def get3dBoxVertices(center, size, rotation, heading):
    '''
    center: (B,n,3)
    size:   (B,n,3)
    rotation: angle along Z (B,3)
    heading: angle along its own center (B,3)
    output: (B,8,3)
    '''
    objNum = center.shape[0]
    vertices = np.zeros((objNum, 8, 3))
    
    for i in range(objNum):
        r_matrix = np.array([[np.cos(rotation[i]), -np.sin(rotation[i]), 0], [np.sin(rotation[i]), np.cos(rotation[i]), 0], [0 , 0, 1]])
        h_matrix = np.array([[np.cos(heading[i]), -np.sin(heading[i]), 0], [np.sin(heading[i]), np.cos(heading[i]), 0], [0 , 0, 1]])
        vertice = []
        
        x_off = size[i][0]/2
        y_off = size[i][1]/2
        z_off = size[i][2]/2
        c_x = center[i][0]
        c_y = center[i][1]
        c_z = center[i][2]
        
        vertice.append([-x_off, -y_off, -z_off])
        vertice.append([x_off, -y_off, -z_off])
        vertice.append([x_off, y_off, -z_off])
        vertice.append([-x_off, y_off, -z_off])
        vertice.append([-x_off, -y_off, z_off])
        vertice.append([x_off, -y_off, z_off])
        vertice.append([x_off, y_off, z_off])
        vertice.append([-x_off, y_off, z_off])
        vertice = np.dot(vertice, h_matrix)
        vertice = vertice + [c_x, c_y, c_z]
        vertice = np.dot(vertice, r_matrix)
        
        vertices[i, :, :] = vertice
    return vertices

def draw_lidar_with_boxes(pc, vertices, fig=None, color=None):
    ''' Draw lidar points. simplest set up. '''
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(800, 500))
    if color is None: 
        color = crop_from_3dbox(pc, vertices)
#        color = pc[:,0]
    #draw points
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color, color=None, mode='point', colormap = 'gnuplot', scale_factor=1, figure=fig)
    #draw origin
    mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)
    #draw axis
    axes=np.array([
        [2.,0.,0.,0.],
        [0.,2.,0.,0.],
        [0.,0.,2.,0.],
    ],dtype=np.float64)
    mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
    mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
#    draw_boxes3d(vertices, fig, draw_text=False)
    return fig

def draw_boxes3d(gt_boxes3d, fig, color=(1,1,1), line_width=1, draw_text=True, text_scale=(1,1,1), color_list=None):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    ''' 
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n] 
        if draw_text: mlab.text3d(b[4,0], b[4,1], b[4,2], '%d'%n, scale=text_scale, color=color, figure=fig)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    return fig

def crop_from_3dbox(pc, vertices):
    color = np.copy(pc[:, 0])
    pointNum = len(pc) #2048
    for box in vertices:
        hull = ConvexHull(box)
        for i in range(pointNum):
            new_pts = np.vstack((box, pc[i]))
            new_hull = ConvexHull(new_pts)
            if np.array_equal(new_hull.vertices, hull.vertices):
                color[i] = 9
    return color

# Python 3.6 required for this function!
def cocoIndexToOnehot(index):
    onehotArray = np.asarray(list(ONE_HOT_TEMPLATE.values()))
    if index == 0:
        return onehotArray[0]
    if index == 2:
        return onehotArray[2]
    else:
        return onehotArray[1]
