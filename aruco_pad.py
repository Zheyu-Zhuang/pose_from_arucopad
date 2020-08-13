#!/usr/bin/env python

import cv2
import os
import copy
import cv2.aruco as aruco
# ROS Sys Pkg
import message_filters
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
import json
# from cfg import aruco_dict
from spatial_transform import *

import argparse



bridge = CvBridge()
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)
parameters = aruco.DetectorParameters_create()
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
parser = argparse.ArgumentParser()
#
parser.add_argument('--intrinsics_file',
                    type=str,
                    default='system_calib.json',
                    help='file for storing the camera intrinsics')
parser.add_argument('--camera_topic', type=str,
                    default='/camera/color/image_raw')
parser.add_argument('--target', type=str,
                    default='tuna_can')

ARGS = parser.parse_args()
CAM_CFG = json.load(open(ARGS.intrinsics_file, 'r'))
PAD_CFG = json.load(open('aruco_pad_config.json', 'r'))
K = np.array(CAM_CFG["K"]).reshape(3, 3)
DIST = np.array(CAM_CFG['distortion'])
X_Cr = np.array(CAM_CFG['pose']).reshape(4, 4)
CAM_RES = CAM_CFG['resolution']
Cr_X_Ccv = np.array([[0, 0, 1, 0],
                     [-1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, 0, 1]])


class PoseFromMarker:
    def __init__(self):
        self.joint_sub = message_filters.Subscriber(
            "/joint_states", JointState)
        self.img_sub = message_filters.Subscriber(ARGS.camera_topic, Image)
        self.synced_msgs = message_filters.ApproximateTimeSynchronizer(
            [self.img_sub, self.joint_sub], 10, 0.1)
        self.synced_msgs.registerCallback(self.sync_callback)
        self.img_pub = rospy.Publisher('aruco_pad', Image, queue_size=10)
        self.cv2_img = None
        self.joint_config = None
        self.pad_ids = {'0': range(0, 12), '1': range(12, 24),
                        '2': range(24, 36), '3': range(36, 48)}
        self.pose_correction = np.array(
            PAD_CFG[ARGS.target]['correction']).reshape(4, 4)

    def sync_callback(self, image_msg, joint_msg):
        self.cv2_img = bridge.imgmsg_to_cv2(image_msg)
        self.joint_config = self.msg_to_joint_config(joint_msg)
        img_msg = self.est_marker_pose(self.cv2_img)
        if img_msg:
            self.img_pub.publish(img_msg)

    def est_marker_pose(self, cv2_img):
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict,
                                              parameters=parameters)
        vis_img = aruco.drawDetectedMarkers(cv2_img, corners, ids,
                                            borderColor=(0, 0, 255))
        if ids is not None and len(ids) > 3:
            n_ids = len(ids)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, PAD_CFG[ARGS.target]["marker_size"], K, DIST)
            rvecs, tvecs = rvecs.reshape((n_ids, 3)), tvecs.reshape((n_ids, 3))
            Ccv_X_Ts = np.zeros((4, 4, 4))
            X_Ts = np.zeros((4, 4, 4))
            n_pads = 0
            for i in range(4):
                Ccv_X_T_temp = self.cmpt_target_pose(rvecs, tvecs,
                                                  ids, self.pad_ids[str(i)])
                if Ccv_X_T_temp is not None:
                    n_pads += 1
                    X_T_temp = X_Cr.dot(Cr_X_Ccv).dot(Ccv_X_T_temp)
                    # print(X_T_temp[2, 3])
                    vis_img = aruco.drawAxis(
                        vis_img, K, DIST,
                        cv2.Rodrigues(Ccv_X_T_temp[0:3, 0:3])[0],
                        Ccv_X_T_temp[0:3, 3], 0.04)
                    vis_img = draw_BB(vis_img, X_T_temp)
                    #
                    Ccv_X_Ts[i, :, :] = Ccv_X_T_temp
                    X_Ts[i, :, :] = X_T_temp
            rospy.sleep(0.05)
            # Display our image
            return bridge.cv2_to_imgmsg(vis_img, 'rgb8')
            # # X_Ts = X_Ts[~np.isnan(X_Ts)]
            # # print(X_Ts)
            # return X_Ts, n_pads

    def cmpt_target_pose(self, rvecs, tvecs, ids, pad_ids):
        ids = np.array(ids).flatten()
        ids, filtered_idx = np.unique(ids, return_index=True)
        rvecs = rvecs[filtered_idx, :]
        tvecs = tvecs[filtered_idx, :]
        visible_ids = ids[np.in1d(ids, pad_ids)]
        n_markers = visible_ids.size
        if n_markers == 0:
            return None
        else:
            offset_idxs = np.arange(12)[np.in1d(pad_ids, visible_ids)]
            struct_idxs = np.arange(len(ids))[np.in1d(ids, pad_ids)]
            Ccv_X_T_stack = np.zeros((n_markers, 4, 4))
            Ccv_X_T_stack[:, 3, 3] = np.ones(n_markers)
            rvecs = rvecs[struct_idxs, :]
            tvecs = tvecs[struct_idxs, :]
            offset = self.marker_offset()[offset_idxs, :]
            for i in range(n_markers):
                rot_temp = cv2.Rodrigues(rvecs[i, :])[0]
                trans_temp = rot_temp.dot(offset[i, :].reshape(3, 1)) \
                             + tvecs[i, :].reshape(3, 1)
                Ccv_X_T_stack[i, :3, :3] = rot_temp
                Ccv_X_T_stack[i, :3, 3] = trans_temp.reshape(3)
            out = SE3_avg(Ccv_X_T_stack).dot(self.pose_correction)
            return out

    @staticmethod
    def marker_offset():
        marker_size = PAD_CFG[ARGS.target]["marker_size"]
        gap = PAD_CFG[ARGS.target]["gap"]
        grid_size = (marker_size+gap)/2
        marker_coor = np.zeros((12, 3))
        marker_coor[:, :2] = np.array(PAD_CFG[ARGS.target]["marker_coor"])
        return -grid_size*marker_coor

    @staticmethod
    def msg_to_joint_config(joint_msg):
        order = [2, 1, 0, 3, 4, 5]
        q = np.asarray([joint_msg.position[i] for i in order])
        q = np.around(q, decimals=4)
        q = q.reshape((6, 1))
        return q

    @staticmethod
    def check_corners(cv2_img):
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(image=gray,
                                              dictionary=aruco_dict)
        if ids is not None and len(ids) > 5:
            print("{} markers detected".format(len(ids)))
        else:
            raise Exception('Need to detect more corners')


def obj_in_FOV(p):
    p_tilde = np.ones((4, 1))
    p_tilde[0:3, 0] = p
    K_3x4 = np.zeros((3, 4))
    K_3x4[:, :3] = K
    Ccv_X = np.linalg.inv(X_Cr.dot(Cr_X_Ccv))
    img_coor_tilde = K_3x4.dot(Ccv_X).dot(p_tilde)
    img_coor = img_coor_tilde / img_coor_tilde[-1]
    img_coor = img_coor[0:2]
    u, v = img_coor[0], img_coor[1]
    in_range = 0 <= u <= CAM_RES[1] and 0 <= v <= CAM_RES[0]
    return in_range, img_coor.astype(int)


def get_3D_BB_corners(X_T):
    x_len, y_len, z_len = PAD_CFG[ARGS.target]["obj_size"]
    x_coor = 0.5*np.array([-x_len, x_len])
    y_coor = 0.5*np.array([-y_len, y_len])
    z_coor = 0.5*np.array([-z_len, z_len])
    bb_corners = np.array([[x_coor[1], y_coor[1], z_coor[1]],
                           [x_coor[1], y_coor[0], z_coor[1]],
                           [x_coor[0], y_coor[0], z_coor[1]],
                           [x_coor[0], y_coor[1], z_coor[1]],
                           [x_coor[1], y_coor[1], z_coor[0]],
                           [x_coor[1], y_coor[0], z_coor[0]],
                           [x_coor[0], y_coor[0], z_coor[0]],
                           [x_coor[0], y_coor[1], z_coor[0]]])
    R0, t0 = X_T[0:3, 0:3], X_T[0:3, 3].reshape(3, 1)
    bb_corners = R0.dot(bb_corners.T) + t0
    # proj to img plane
    n_points = bb_corners.shape[1]
    corners_uv = np.zeros((2, n_points))
    for i in range(n_points):
        p = bb_corners[:, i]
        _, uv_temp = obj_in_FOV(p)
        corners_uv[:, i] = uv_temp.reshape(2)
    return corners_uv


def get_2d_bb_corners(X_i):
    corners = get_3D_BB_corners(X_i)
    pt0 = np.min(corners, axis=1)
    pt0[0], pt0[1] = min(pt0[0], CAM_RES[1]), min(pt0[1], CAM_RES[0])
    pt0 = tuple(pt0.astype(int))
    pt1 = np.max(corners, axis=1)
    pt1[0], pt1[1] = min(pt1[0], CAM_RES[1]), min(pt1[1], CAM_RES[0])
    pt1 = tuple(pt1.astype(int))
    return pt0, pt1


def draw_BB(cv2_img, X_T, mode='3d'):
    corners = get_3D_BB_corners(X_T)
    if mode == '3d':
        for i in range(4):
            cv2.circle(cv2_img, tuple(corners[:, i].astype(int)), 3,
                       (255, 0, 0), -1)
            cv2.line(cv2_img, tuple(corners[:, i % 4].astype(int)),
                     tuple(corners[:, (i + 1) % 4].astype(int)),
                     (0, 255, 0), 2)
            cv2.line(cv2_img, tuple(corners[:, i % 4 + 4].astype(int)),
                     tuple(corners[:, (i + 1) % 4 + 4].astype(int)),
                     (0, 255, 0), 2)
            cv2.line(cv2_img, tuple(corners[:, i % 4].astype(int)),
                     tuple(corners[:, i % 4 + 4].astype(int)),
                     (0, 255, 0), 2)
    elif mode == '2d':
        pt0 = tuple(np.min(corners, axis=1).astype(int))
        pt1 = tuple(np.max(corners, axis=1).astype(int))
        cv2.rectangle(cv2_img, pt0, pt1, (255, 255, 0), 2)
    return cv2_img


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    rospy.init_node("aruco_calib", anonymous=True)
    collector = PoseFromMarker()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting Down')
