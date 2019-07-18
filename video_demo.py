#! /usr/bin/env python
# coding=utf-8
'''
Demo class for YOLO + Frumstum PointNet detection

Author: Shu Xu
Date: July 2019
'''
import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
import pyrealsense2 as rs


import sys
sys.path.append('/localhome/sxu/Desktop/MA/frustum-pointnets-master/train')
from pn_util import *
import mayavi.mlab as mlab
from scipy.spatial import ConvexHull

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_coco.pb"
video_path      = "./docs/images/road.mp4"
# video_path      = 0
num_classes     = 80
input_size      = 416
graph           = tf.Graph()
return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)

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
    draw_boxes3d(vertices, fig)
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
                color[i] = 5
    return color

# Realsense system into Velodyne
def rsToVelo(pc):
    t_matrix = np.array([[0,    -1.1,    0], 
                         [0,     0,    -1.1], 
                         [1.1,   0,    0]])
    
    tl_matrix = np.array([0, 0, -1.1])
    return (np.dot(pc, t_matrix) + tl_matrix)

# Randomly collect (1024) points
def collectPoints(pc, numPoints):
    # clear empty points first
    pointCloud = pc[~np.all(pc == 0, axis=1)]
    # filter out the ridiculous points (e.g. height=20m)
    pointCloud = pointCloud[np.where(pointCloud[:,1]<3)]
    return pointCloud[np.random.randint(pointCloud.shape[0], size=numPoints), 0:3]








pipeline = rs.pipeline()
    
#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.depth
align = rs.align(align_to)
pc = rs.pointcloud()

with tf.Session(graph=graph) as sess:
    vid = cv2.VideoCapture(video_path)
    sess_3d, ops_3d = get_session_and_ops(batch_size=1, num_point=2048)
    while True:
#        return_value, frame = vid.read()
#        if return_value:
#            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#            image = Image.fromarray(frame)
#        else:
#            raise ValueError("No image!")
        
        
        #wait for frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
            
        color_aligned_to_depth = aligned_frames.first(rs.stream.color)
        color_image = np.asanyarray(color_aligned_to_depth.get_data())
        
        depth_frame = frames.first(rs.stream.depth)
        points = pc.calculate(depth_frame) 
        # Feed into YOLO
        image = Image.fromarray(color_image)
        frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        w = rs.video_frame(depth_frame).width
        h = rs.video_frame(depth_frame).height
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(h, w, 3) # Uncropped PC
        
        frame_size = frame.shape[:2]
        image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...]
        prev_time = time.time()

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
        
        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')
        # only draw "person"
        SAMPLE_NUM = 2048
        VISUALIZE_ROI = False
        SAVE_POINTCLOUD = False
        human_box = []
        interestedObjects = 0
        for i, bbox in enumerate(bboxes):
            if (bbox[5] == 0 and bbox[4] >= 0.5):
                interestedObjects += 1
        print ("YOLO: Found", interestedObjects, "people")
        rois = np.zeros((interestedObjects, SAMPLE_NUM, 3)) # BxNx3
        centroids = np.zeros((interestedObjects, 3))  # Bx3
        for i, bbox in enumerate(bboxes):
            if (bbox[5] == 0 and bbox[4] >= 0.5):
                interestedObjects -= 1
                human_box.append(bboxes[i])
                x_min = int(bbox[0])
                x_max = int(bbox[2])
                y_min = int(bbox[1])
                y_max = int(bbox[3])
                center = [int((x_min+x_max)/2), int((y_min+y_max)/2)]
                roi = verts[y_min:y_max, x_min:x_max, :].reshape(-1, 3)
                roi = rsToVelo(collectPoints(roi, SAMPLE_NUM)) # 2048x3
                rois[interestedObjects, :, :] = roi
                centroid = verts[center[1], center[0], :].reshape(-1, 3)
                centroid = rsToVelo(centroid) # 1x3
                centroids[interestedObjects, :] = centroid
        # Test the cropped rois
        if (VISUALIZE_ROI):
            fig = mlab.figure(figure=None, bgcolor=(0,0,0),fgcolor=None, engine=None, size=(800, 500))
            if rois.shape[0] == 1:    
                draw_lidar_simple(rois[0,...], fig)
            elif rois.shape[0] == 2:
                draw_lidar_simple(rois[0,...], fig)
                draw_lidar_simple(rois[1,...], fig)
        # Draw 2D boxes for the 2D image(from YOLO)
        image = utils.draw_bbox(frame, human_box)
        # 2D Visualize
        if (True):
            curr_time = time.time()
            exec_time = curr_time - prev_time
            result = np.asarray(image)
            info = "time: %.2f ms" %(1000*exec_time)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        # 3D Segmentation
        box_vertices = test_segmentation(rois, centroids, sess_3d, ops_3d)
        if (box_vertices is not None):
            time.sleep(1)
            fig = mlab.figure(figure=None, bgcolor=(0,0,0),fgcolor=None, engine=None, size=(800, 500))
            org_pc = rsToVelo(collectPoints(verts.reshape(-1,3), 5000))
            draw_lidar_with_boxes(org_pc, box_vertices, fig)
#            draw_lidar_simple(rois[0,...], fig)
#            draw_lidar_simple(rsToVelo(collectPoints(verts.reshape(-1,3), fig)
#            draw_lidar_simple(rois[0,...], fig)
#            draw_lidar_simple(rois[1,...], fig)
#            draw_boxes3d(box_vertices, fig)
            input ("ENTER")
            mlab.close()
            if (SAVE_POINTCLOUD):
                org_pc.tofile("rsVerts3.bin")
                rois.tofile("rsPC3.bin")
                centroids.tofile("rsCentroid3.bin")


