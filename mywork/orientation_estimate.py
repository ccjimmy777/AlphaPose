# created by ccj at 24/3/24
import numpy as np
import cv2
import colorsys
from third_party.MEBOW.tools.demo_api import predict_orientation
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib
# matplotlib.use('TkAgg')

def orientation_estimate_medow(input_tensor):
    ori = predict_orientation(input_tensor)
    return degree_to_vector(ori)

def orientation_estimate_mywork2d(pose_keypoints_2d, focal_length, cx, cy, format='halpe26', isDebug=False):
    pose_keypoints_2d = np.asarray(pose_keypoints_2d)
    if format == 'halpe26':
        assert pose_keypoints_2d.shape[0] == 26, 'halpe26 format should have 26 keypoints'
        
        # extract left shoulder, right shoulder, left hip, right hip
        ls_2d = pose_keypoints_2d[5, :]          # left shoulder
        rs_2d = pose_keypoints_2d[6, :]          # right shoulder
        neck_2d = pose_keypoints_2d[18, :]       # neck
        root_2d = pose_keypoints_2d[19, :]       # hip
    elif format == 'h36m':
        assert pose_keypoints_2d.shape[0] == 17, 'h36m format should have 17 keypoints'
        ls_2d = pose_keypoints_2d[11, :]         # left shoulder
        rs_2d = pose_keypoints_2d[14, :]         # right shoulder
        neck_2d = pose_keypoints_2d[8, :]        # thorax
        root_2d = pose_keypoints_2d[0, :]        # root (pelvis)

    K_inv = np.array([[1.0/focal_length, 0, -cx/focal_length], 
                      [0, 1.0/focal_length, -cy/focal_length],
                      [0, 0, 1]])

    ls_neck = np.linalg.norm(ls_2d-neck_2d)
    rs_neck = np.linalg.norm(rs_2d-neck_2d)
    theta = ls_neck/(ls_neck + rs_neck)

    ls_2d_homo = np.append(ls_2d, 1)
    rs_2d_homo = np.append(rs_2d, 1)
    neck_2d_homo = np.append(neck_2d, 1)
    root_2d_homo = np.append(root_2d, 1)

    ls_3d = 2 * (1-theta) * K_inv @ ls_2d_homo
    rs_3d = 2 * theta * K_inv @ rs_2d_homo
    neck_3d = K_inv @ neck_2d_homo
    should_homo = (1-theta) * K_inv @ ls_2d_homo - theta * K_inv @ rs_2d_homo
    root_3d = np.abs(np.dot(neck_2d_homo, K_inv.T @ should_homo)) \
                / np.maximum(np.abs(np.dot(root_2d_homo, K_inv.T @ should_homo)), 1e-4) \
                    * K_inv @ root_2d_homo

    ori_vec3d = np.cross(root_3d-neck_3d, ls_3d-rs_3d)
    ori_vec3d = ori_vec3d / np.linalg.norm(ori_vec3d)
    ori_vec2d = ori_vec3d[[2, 0]]  # 取 z、x 坐标
    # 暂时不用。不用归一化使得对于朝向为向上的情形更为鲁棒
    # ori_vec2d = ori_vec2d / np.linalg.norm(ori_vec2d)
    ori_score = np.linalg.norm(ori_vec2d)

    ''' vis for debug
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(ls_3d[0], ls_3d[1], ls_3d[2], c='r')
    # ax.scatter(rs_3d[0], rs_3d[1], rs_3d[2], c='y')
    # ax.scatter(neck_3d[0], neck_3d[1], neck_3d[2], c='k')
    # ax.scatter(root_3d[0], root_3d[1], root_3d[2], c='g')
    # xs = [neck_3d[0], neck_3d[0] + ori_vec3d[0]]
    # ys = [neck_3d[1], neck_3d[1] + ori_vec3d[1]]
    # zs = [neck_3d[2], neck_3d[2] + ori_vec3d[2]]
    # ax.plot(xs, ys, zs)
    # ax.xaxis.set_rotate_label(True) # 设置标签旋转
    # ax.yaxis.set_rotate_label(True)
    # ax.zaxis.set_rotate_label(False)
    # ax.set_xlabel('X轴', rotation=20) # 设置标签角度
    # ax.set_ylabel('Y轴', rotation=-45)
    # ax.set_zlabel('Z轴', rotation=0)
    # plt.show()
    '''
    return ori_vec2d, ori_score

# 暂时不用了
def orientation_estimate_mywork(pose_keypoints_2d, focal_length, cx, cy):
    pose_keypoints_2d = np.asarray(pose_keypoints_2d)
    # pose_keypoints_2d = pose_keypoints_2d.reshape(-1, 17, 2)
    # extract left shoulder, right shoulder, left hip, right hip
    ls_2d = pose_keypoints_2d[:, 5, :]          # left shoulder
    rs_2d = pose_keypoints_2d[:, 6, :]          # right shoulder
    lh_2d = pose_keypoints_2d[:, 11, :]         # left hip
    rh_2d = pose_keypoints_2d[:, 12, :]         # right hip
    neck_2d = (ls_2d + rs_2d) / 2
    root_2d = (lh_2d + rh_2d) / 2

    K_inv = np.array([[1.0/focal_length, 0, -cx/focal_length], 
                      [0, 1.0/focal_length, -cy/focal_length],
                      [0, 0, 1]])

    ori_vec3d_list = []; theta_list = []; ls_3d_list = []; rs_3d_list = []; neck_3d_list = []; root_3d_list = []
    for i in range(len(pose_keypoints_2d)):
        lh_neck = np.linalg.norm(lh_2d[i]-neck_2d[i])
        rh_neck = np.linalg.norm(rh_2d[i]-neck_2d[i])
        theta = lh_neck/(lh_neck + rh_neck)

        ls_2d_homo = np.append(ls_2d[i], 1)
        rs_2d_homo = np.append(rs_2d[i], 1)
        neck_2d_homo = np.append(neck_2d[i], 1)
        root_2d_homo = np.append(root_2d[i], 1)

        ls_3d = 2 * (1-theta) * K_inv @ ls_2d_homo
        rs_3d = 2 * theta * K_inv @ rs_2d_homo
        neck_3d = K_inv @ neck_2d_homo
        should_homo = (1-theta) * K_inv @ ls_2d_homo - theta * K_inv @ rs_2d_homo
        root_3d = np.abs(np.dot(neck_2d_homo, K_inv.T @ should_homo)) \
                    / np.maximum(np.abs(np.dot(root_2d_homo, K_inv.T @ should_homo)), 1e-4) \
                        * K_inv @ root_2d_homo

        ori_vec3d = np.cross(root_3d-neck_3d, ls_3d-rs_3d)
        ori_vec3d = ori_vec3d / np.linalg.norm(ori_vec3d)
        ori_vec3d_list.append(ori_vec3d)
        theta_list.append(theta)
        ls_3d_list.append(ls_3d)
        rs_3d_list.append(rs_3d)
        neck_3d_list.append(neck_3d)
        root_3d_list.append(root_3d)
    
    return ori_vec3d_list, theta_list, ls_3d_list, rs_3d_list, neck_3d_list, root_3d_list


# transform degree to vector
def degree_to_vector(degree):
    rad = degree * np.pi / 180.  # degree to radian
    return np.array([np.cos(rad), np.sin(rad)])

def vector_to_degree(vector):
    degree = np.arctan2(vector[1], vector[0]) * 180. / np.pi
    degree = degree if degree >= 0 else degree + 360
    return degree

# def vector3d_to_degree(vector3d):
#     degree = np.arctan2(vector3d[1], vector3d[0]) * 180. / np.pi
#     degree = degree if degree >= 0 else degree + 360
#     return degree_x, degree_y

def degree_to_color(degree, saturation=1, value=1, mode='bgr'):
    # transform degree to color in hsv
    hue = degree / 360. * 1.
    saturation = saturation
    value = value
    
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    # convert for opencv
    bgr = np.uint8(np.array(rgb)*255)[::-1]
    color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    return color

def draw_orientation(image, center, degree, color=None, thickness=2):
    if color is None:
        color = degree_to_color(degree)
    # color = (int(color[0]), int(color[1]), int(color[2]))

    scale = 200
    vector = degree_to_vector(degree)
    start = tuple(center)
    end = (int(center[0] + vector[1] * scale), int(center[1]))
    cv2.arrowedLine(image, start, end, color, thickness)

# 暂时不用了
def draw_orientation_vector(image, neck_2d, orient_vec3d, focal_length, cx=960, cy=540, color=(0,0,255), thickness=3):
    K = np.array([[focal_length, 0, cx],
                  [0, focal_length, cy],
                  [0, 0, 1]])
    K_inv = np.array([[1.0/focal_length, 0, -cx/focal_length], 
                    [0, 1.0/focal_length, -cy/focal_length],
                    [0, 0, 1]])
    
    neck_3d = 2.0 * K_inv @ np.append(neck_2d, 1)
    end = neck_3d + orient_vec3d

    start_2d = K @ neck_3d
    start_2d = start_2d[:2] / start_2d[2]
    end_2d = K @ end
    end_2d = end_2d[:2] / end_2d[2]

    # scale = 20
    start = (int(start_2d[0]), int(start_2d[1]))
    end = (int(end_2d[0]), int(end_2d[1]))
    cv2.arrowedLine(image, start, end, color, thickness)