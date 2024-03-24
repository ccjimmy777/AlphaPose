import numpy as np
import torch
import cv2
import colorsys
from torchvision import transforms
from third_party.WildCamera.tools.calibrator import MonocularCalibrator
from third_party.WildCamera.WildCamera.newcrfs.newcrf_incidencefield import NEWCRFIF

# 四分位数法清除异常值
def clear_abnormal(array):
    if len(array) == 0: return array, []

    outlier_indices = []
    Q1 = np.percentile(array, 25)
    Q3 = np.percentile(array, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    for i in range(len(array)):
        if array[i] < lower_bound or array[i] > upper_bound:
            outlier_indices.append(i)
    
    return np.delete(array, outlier_indices), outlier_indices

# test for clear_abnormal
# a = [-133, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1331, 12, 13, 14, 15, 16, 17, 1338, 19, 20]
# b = clear_abnormal(a)

def load_wild_camera_model():
    model = NEWCRFIF(version='large07', pretrained=None)
    model.eval()
    # model.cuda()

    ckpt_path = '/home/jimmy/projects/AlphaPose/third_party/WildCamera/model_zoo/wild_camera_all.pth'
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)

    return model



def predict_focal(model, orig_img):
    if isinstance(orig_img, np.ndarray):
        w = orig_img.shape[1]
        h = orig_img.shape[0]
    
    input_wt, input_ht = 640, 480

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    totensor = transforms.ToTensor()
    resize = transforms.Resize((input_ht, input_wt), interpolation=transforms.InterpolationMode.BILINEAR)
    rgb = normalize(resize(totensor(orig_img)))
    rgb = rgb.unsqueeze(0).to('cpu')

    scaleM = np.eye(3)
    scaleM[0, 0] = input_wt / w
    scaleM[1, 1] = input_ht / h
    scaleM = torch.from_numpy(scaleM).float().to('cpu')

    monocalibrator = MonocularCalibrator()
    incidence = model.forward(rgb)
    Kest = monocalibrator.calibrate_camera_4DoF(incidence)
    Kest = torch.inverse(scaleM) @ Kest

    focal = Kest[0, 0].item()

    return focal


@staticmethod
def pixel_coordinates(point_3d,
                width_image, height_image,
                focal_length_x, aspect_ratio=1):
    """ Compute image coordinates of a 3d point positioned in camera frame

    Args:
        - point (float, float, float): a point/array of points in space
                expressed in camera frame coordinates

    return:
        - (int, int): coordinate of point in image in pix
    """
    pt = np.array(point_3d)
    x, y, z = pt.T
    focal_length_y = aspect_ratio * focal_length_x

    u = x / z * focal_length_x + width_image / 2.0
    v = y / z * focal_length_y + height_image / 2.0

    if len(pt.shape) > 1:
        return np.column_stack((u, v))
    else:
        return u, v


# transform degree to vector
def degree_to_vector(degree):
    rad = degree * np.pi / 180.  # degree to radian
    return np.array([np.cos(rad), np.sin(rad)])

def vector_to_degree(vector):
    degree = np.arctan2(vector[1], vector[0]) * 180. / np.pi
    degree = degree if degree >= 0 else degree + 360
    return degree

# test vector_to_degree
# assert(abs(vector_to_degree(np.array([1, 0])) - 0) < 1e-5)
# assert(abs(vector_to_degree(np.array([0, 3])) - 90) < 1e-5)
# assert(abs(vector_to_degree(np.array([-2, 0])) - 180) < 1e-5)
# assert(abs(vector_to_degree(np.array([0, -7])) - 270) < 1e-5)

# transform degree to color in bgr (used in opencv)
def degree_to_color(degree, saturation=1, value=1, mode='bgr'):
    # transform degree to color in hsv
    hue = degree / 360. * 1.
    saturation = saturation
    value = value
    
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    # convert for opencv
    bgr = np.uint8(np.array(rgb)*255)[::-1]

    return bgr if mode == 'bgr' else rgb

# test for degree_to_color
# print(degree_to_color(120))
# assert(np.array_equal(degree_to_color(0), np.array([0, 0, 255])))
# assert(np.array_equal(degree_to_color(120), np.array([0, 255, 0])))
# assert(np.array_equal(degree_to_color(240), np.array([255, 0, 0])))

def draw_orientation(image, center, degree, color=None, thickness=1):
    if color is None:
        color = degree_to_color(degree)
    color = (int(color[0]), int(color[1]), int(color[2]))

    scale = 20
    vector = degree_to_vector(degree)
    start = tuple(center)
    end = (int(center[0] + vector[1] * scale), int(center[1]))
    cv2.arrowedLine(image, start, end, color, thickness)

# test for draw_orientation
# image = np.zeros((100, 100, 3), np.uint8)
# center = (50, 50)
# draw_orientation(image, center, 0)
# draw_orientation(image, center, 90)
# draw_orientation(image, center, 120)
# draw_orientation(image, center, 300)
# cv2.imshow('image', image)
# cv2.waitKey()