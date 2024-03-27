# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# 
# -----------------------------------------------------

"""API of tracker"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from abc import ABC, abstractmethod
import platform
import numpy as np
from collections import deque
import itertools
import os.path as osp
import time
import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.utils import *
from utils.log import logger
from utils.kalman_filter import KalmanFilter
from tracking.matching import *
from tracking.basetrack import BaseTrack, TrackState
from utils.transform import build_transforms
from ReidModels.ResBnLin import ResModel
from ReidModels.osnet import *
from ReidModels.osnet_ain import osnet_ain_x1_0
from ReidModels.resnet_fc import resnet50_fc512

from alphapose.utils import vis
from alphapose.utils.transforms import heatmap_to_coord_simple
from mywork.orientation_estimate import orientation_estimate_mywork2d

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, pose,crop_box,file_name,ds,buffer_size=30,cfg_camera=dict()):  # buffer_size=30

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9 
        self.pose = pose
        self.detscore = ds.item()
        self.crop_box = crop_box
        self.file_name = file_name
        
        # added by ccj at 24/3/24
        self.pose_recover = None
        self.pose_recover_score = 0
        self.focal = cfg_camera.get('focal', 1000.0)
        self.cx = cfg_camera.get('cx', 960.0)
        self.cy = cfg_camera.get('cy', 540.0)
        self.curr_orient = None
        self.curr_orient_score = 0.8 
        self.smooth_orient = None
        self.smooth_orient_score = 0.8
        self.orientations = deque([], maxlen=5)
        self.update_orientations(pose)
    
    def update_orientations(self, pose):
        pose_coord, pose_score = heatmap_to_coord_simple(pose, self.crop_box)
        ori_vec2d, ori_score = orientation_estimate_mywork2d(pose_coord, self.focal, self.cx, self.cy)
        self.pose_recover = pose_coord
        self.pose_recover_score = pose_score

        self.curr_orient = ori_vec2d
        self.curr_orient_score = ori_score
        if self.smooth_orient is None:
            self.smooth_orient = ori_vec2d
            self.smooth_orient_score = ori_score
        else:
            weight = self.smooth_orient_score / (ori_score + self.smooth_orient_score)
            self.smooth_orient = weight * self.smooth_orient + (1-weight) * ori_vec2d
            self.smooth_orient_score = np.linalg.norm(self.smooth_orient)
        ####
        self.orientations.append(ori_vec2d)
        self.smooth_orient /= np.linalg.norm(self.smooth_orient)

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat 
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha *self.smooth_feat + (1-self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0  # For the lost tracks, we assume the changing of height (vh) is 0.
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i,st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0  # For the lost tracks, we assume the changing of height (vh) is 0.
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov


    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.pose = new_track.pose
        self.detscore = new_track.detscore
        self.crop_box = new_track.crop_box
        self.file_name = new_track.file_name
        self.update_orientations(new_track.pose)

    def update(self, new_track, frame_id, update_feature=True, update_orient=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.pose = new_track.pose
        self.detscore = new_track.detscore
        self.crop_box = new_track.crop_box
        self.file_name = new_track.file_name
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)
        if update_orient:
            self.update_orientations(new_track.pose)

    @property
    #@jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    #@jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    #@jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    #@jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    #@jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)



class Tracker(object):
    def __init__(self, opt, args):
        self.opt = opt
        self.num_joints = 17
        self.frame_rate = opt.frame_rate
        #m = ResModel(n_ID=opt.nid)
        if self.opt.arch == "res50-fc512":
            m = resnet50_fc512(num_classes=1,pretrained=False)
        elif self.opt.arch == "osnet_ain":
            m = osnet_ain_x1_0(num_classes=1,pretrained=False)
        
        self.model = nn.DataParallel(m,device_ids=args.gpus).to(args.device).eval()
        
        load_pretrained_weights(self.model,self.opt.loadmodel)
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(self.frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size

        self.kalman_filter = KalmanFilter()

    def update(self,img0,inps=None,bboxs=None,pose=None,cropped_boxes=None,file_name='',dscores=None, camera_cfg=None, _debug=False,_debug_frame_id=67):  # _debug_frame_id=58
        #bboxs:[x1,y1.x2,y2]
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # if _debug:
        #     logger.debug('self.frame_id: {}'.format(self.frame_id))

        ''' Step 0: Network forward, get human identity embedding''' 
        assert len(inps)==len(bboxs),'Unmatched Length Between Inps and Bboxs'
        assert len(inps)==len(pose),'Unmatched Length Between Inps and Heatmaps'  
        with torch.no_grad():
            feats = self.model(inps).cpu().numpy()
        bboxs = np.asarray(bboxs)
        if len(bboxs)>0:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:]), 0.9, f,p,c,file_name,ds,30,camera_cfg) for
                          (tlbrs, f,p,c,ds) in zip(bboxs, feats,pose,cropped_boxes,dscores)]
        else:
            detections = []

        if _debug and self.frame_id >= _debug_frame_id:
            img = np.array(img0, dtype=np.uint8)[:, :, ::-1].copy()
            win_name = 'frame_id:'+str(self.frame_id)
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            vis.vis_frame_debug(img, win_name, track_list=detections, isDetect=True, stage=0)

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
                strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        ###joint track with bbox-iou
        # Step 1: First association, with re-id feature + bbox motion
        # TODO: self.lost_stracks 缺乏有效利用，仅有重识别的一次机会（即当前步骤）
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        dists_emb = embedding_distance(strack_pool, detections)
        dists_emb = fuse_motion(self.kalman_filter, dists_emb, strack_pool, detections)
        matches, u_track, u_detection = linear_assignment(dists_emb, thresh=0.7)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        if _debug and self.frame_id >= _debug_frame_id:
            logger.debug('========== Frame {0} === Stage 1: {1} ======'.format(self.frame_id, 'dists_emb'))
            logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
            logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
            logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
            logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
            vis.vis_frame_debug(img, win_name, track_list=activated_starcks, isDetect=False, stage=1)

        #Step 1-2 利用朝向信息做重识别
        detections = [detections[i] for i in u_detection]
        tracks = [strack_pool[i] for i in u_track]
        dists_orient = orientation_distance(tracks, detections)
        matches, u_track, u_detection =linear_assignment(dists_orient, thresh=0.7)
        for itracked, idet in matches:
            track = tracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        if _debug and self.frame_id >= _debug_frame_id:
            logger.debug('========== Frame {0} === Stage 1-2: {1} ======'.format(self.frame_id, 'dists_orient'))
            logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
            logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
            logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
            logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
            vis.vis_frame_debug(img, win_name, track_list=activated_starcks, isDetect=False, stage=1.2)

        #Step 2: Second association, with IOU
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state==TrackState.Tracked]
        dists_iou = iou_distance(r_tracked_stracks, detections) 
        matches, u_track, u_detection =linear_assignment(dists_iou, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        
        if _debug and self.frame_id >= _debug_frame_id:
            logger.debug('========== Frame {0} === Stage 2: {1} ======'.format(self.frame_id, 'dists_iou'))
            logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
            logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
            logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
            logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
            vis.vis_frame_debug(img, win_name, track_list=activated_starcks, isDetect=False, stage=2)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        
        if _debug and self.frame_id >= _debug_frame_id:
            logger.debug('========== Frame {0} === Stage 3: {1} ======'.format(self.frame_id, 'dists_iou_unconfirmed'))
            logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
            logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
            logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
            logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
            vis.vis_frame_debug(img, win_name, track_list=activated_starcks, isDetect=False, stage=3)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.detscore < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        
        if _debug and self.frame_id >= _debug_frame_id:
            logger.debug('========== Frame {0} === Stage 4: {1} ======'.format(self.frame_id, 'Init new stracks'))
            logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
            logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
            logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
            logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
            vis.vis_frame_debug(img, win_name, track_list=activated_starcks, isDetect=False, stage=4)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        
        if _debug and self.frame_id >= _debug_frame_id:
            logger.debug('========== Frame {0} === Stage 5: {1} ======'.format(self.frame_id, 'Remove old stracks'))
            logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
            logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
            logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
            logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
            vis.vis_frame_debug(img, win_name, track_list=activated_starcks, isDetect=False, stage=5)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks]
        if _debug and self.frame_id >= _debug_frame_id:
            logger.debug('===========Frame {} === Summary =========='.format(self.frame_id))
            logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
            logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
            logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
            logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
            vis.vis_frame_debug(img, win_name, track_list=activated_starcks, isDetect=False, stage=6)

        return output_stracks

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist<0.15)
    dupa, dupb = list(), list()
    for p,q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i,t in enumerate(stracksa) if not i in dupa]
    resb = [t for i,t in enumerate(stracksb) if not i in dupb]
    return resa, resb
            

