import cv2
import numpy as np
import scipy
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from cython_bbox import bbox_overlaps as bbox_ious
from trackers.utils import kalman_filter
import time

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    """
    Simple linear assignment
    :type cost_matrix: np.ndarray
    :type thresh: float
    :return: matches, unmatched_a, unmatched_b
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    cost_matrix[cost_matrix > thresh] = thresh + 1e-4
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    indices = np.column_stack((row_ind, col_ind))

    return _indices_to_matches(cost_matrix, indices, thresh)


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def ious_ours(atlbrs, btlbrs):
    boxes = np.ascontiguousarray(atlbrs, dtype=np.float32)
    query_boxes = np.ascontiguousarray(btlbrs, dtype=np.float32)
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    ious = np.zeros((N, K), dtype=boxes.dtype)
    if N * K == 0:
        return ious

    for k in range(K):
        query_box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + 1) *
                            (query_boxes[k, 3] - query_boxes[k, 1] + 1))
        
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(
                boxes[n, 0], query_boxes[k, 0]) + 1)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(
                    boxes[n, 1], query_boxes[k, 1]) + 1)
                if ih > 0:
                    box_area = ((boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1))
                    # ua = float((boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1) + query_box_area - iw * ih)
                    area_min = min(box_area, query_box_area)
                    ious[n, k] = iw * ih / area_min
    return ious

def iou_distance_ours(atracks, btracks):
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious_ours(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    for i, track in enumerate(tracks):
        cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    return cost_matrix

def orientation_distance_onetrack(track, detections, metric='cosine'):
    cost_matrix = np.zeros(len(detections), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_orientations = np.asarray([track.curr_orient for track in detections], dtype=np.float32)
    cost_matrix = np.maximum(0.0, cdist(track.smooth_orient.reshape(1,-1), det_orientations, metric)).reshape(-1)
    return cost_matrix

def orientation_distance(tracks, detections, metric='cosine'):
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_orientations = np.asarray([track.curr_orient for track in detections], dtype=np.float)
    for i, track in enumerate(tracks):
        cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_orient.reshape(1,-1), det_orientations, metric))
    return cost_matrix

def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix

def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):  # 0.98
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])

    dscores = np.asarray([det.detscore for det in detections])

    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        
        ori_distance = orientation_distance_onetrack(track, detections, metric='cosine')

        unmatched_detect_using_motion = gating_distance > gating_threshold
        match_num_using_motion = np.sum(~unmatched_detect_using_motion)

        # 对检测做最小面积限制
        bbox_size_mask = np.asarray([(det.tlwh[2] * det.tlwh[3]) > 1024 for det in detections])
        if np.sum(bbox_size_mask) < len(detections):
            print('np.sum(bbox_size_mask) < len(detections) !!!')
        unmatched_detect_using_orient = (ori_distance > 1.0) & bbox_size_mask & (dscores > 0.5)
        if match_num_using_motion <= 1:
            unmatched_detect = unmatched_detect_using_motion
        else:
            unmatched_detect = unmatched_detect_using_motion # | unmatched_detect_using_orient

        # unmatched_detect_for_sure = np.where((gating_distance > gating_threshold) & (ori_distance > 0.5))[0]

        gating_distance_normalize = gating_distance / gating_threshold
        # ori_distance_normalize = ori_distance / 2
        # motion_orient_fuse_dist = gating_distance_normalize + ori_distance_normalize
        cost_matrix[row, unmatched_detect] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1-lambda_)* gating_distance_normalize
        # cost_matrix[row] = np.multiply(cost_matrix[row], dscores) + np.multiply(gating_distance_normalize, (1-dscores))
    return cost_matrix

def fuse_orientation(cost_matrix, tracks, detections, lambda_, iou_thresh=0.8):  # lambda_ = 0.98
    if cost_matrix.size == 0:
        return cost_matrix

    for row, track in enumerate(tracks):
        ori_distance = orientation_distance_onetrack(track, detections, metric='cosine')
        # unmatched_detect_using_orient = ori_distance > 1.0
        unmatched_detect_using_iou = cost_matrix[row] > iou_thresh
        # unmatched_detect = unmatched_detect_using_orient | unmatched_detect_using_iou
        unmatched_detect = unmatched_detect_using_iou
        ori_distance_normalize = ori_distance
        cost_matrix[row, unmatched_detect] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1-lambda_)* ori_distance_normalize
    
    return cost_matrix