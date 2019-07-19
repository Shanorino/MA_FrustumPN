#! /usr/bin/env python
# coding=utf-8
'''
Video visualizer for the saved PointClouds + Boxes

Author: Shu Xu
Date: July 2019
'''

import numpy as np
import mayavi.mlab as mlab
from scipy.spatial import ConvexHull


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
    
    draw_boxes3d(vertices, fig)
    
    mlab.view(azimuth=180, elevation=80, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=19.0, figure=fig)
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
                color[i] = 10
    return color

################################ Start Rendering ##################################

for i in range(50):
    pc = np.fromfile("/localhome/sxu/Desktop/tensorflow-yolov3/Video_Verts_" +str(i)+ ".bin", dtype=np.float).reshape(-1, 3)
    box = np.fromfile("/localhome/sxu/Desktop/tensorflow-yolov3/Video_Boxes_" +str(i)+ ".bin", dtype=np.float).reshape(-1, 8, 3)
    fig = mlab.figure(figure=None, bgcolor=(0,0,0),fgcolor=None, engine=None, size=(800, 500))
    draw_lidar_with_boxes(pc, box, fig)
    mlab.savefig(filename=str(i)+".png")
    mlab.close()
    