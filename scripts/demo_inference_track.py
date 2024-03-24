"""Script for single-gpu/multi-gpu demo."""
import argparse
import os
import platform
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
import natsort

from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter

from demo_visualization import load_wild_camera_model, clear_abnormal
from PIL import Image
from mmengine.fileio import dump, load

"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg', type=str, required=True,
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='checkpoint file name')
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--detfile', dest='detfile',
                    help='detection result file', default="")
parser.add_argument('--indir', dest='inputpath',
                    help='image-directory', default="")
parser.add_argument('--list', dest='inputlist',
                    help='image-list', default="")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="examples/res/")
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--detbatch', type=int, default=5,
                    help='detection batch size PER GPU')
parser.add_argument('--posebatch', type=int, default=64,  # 64
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=32,  # 1024
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=int,
                    help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)
"""----------------------------- My work options -----------------------------"""
parser.add_argument('--focal_est', dest='focal_estimation',
                    help='use wild camera model for focal estimation', action='store_true', default=False)

args = parser.parse_args()
cfg = update_config(args.cfg)

if platform.system() == 'Windows':
    args.sp = True

args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)
args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'

if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')


def print_finish_info():
    print('===========================> Finish Model Running.')
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')


def loop():
    n = 0
    while True:
        yield n
        n += 1


if __name__ == "__main__":
    # Check input
    inputpath = args.inputpath
    if len(inputpath) and inputpath != '/':
        root, dirs, _ = next(os.walk(inputpath))
        mode = 'image'
    else:
        print('An error occurs on arg "inputpath", please check it')
        sys.exit(-1)

    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    # Load pose model
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print('Loading pose model from %s...' % (args.checkpoint,))
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
    if args.pose_track:
        tracker = Tracker(tcfg, args)
    if len(args.gpus) > 1:
        pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
    else:
        pose_model.to(args.device)
    pose_model.eval()

    # 预加载detector模型存在bug！
    # Load detector
    # detector = get_detector(args)
    # detector.load_model()

    if len(dirs) > 0:
        dirs_list = tqdm(dirs, dynamic_ncols=True, desc="val seqs(dirs)") 
    else:
        dirs_list = ['']
    
    for dir in dirs_list:
        args.inputpath = os.path.join(inputpath, dir)
        _, _, files = next(os.walk(args.inputpath))
        input_source = natsort.natsorted(files)

        seq_name = dir if dir != '' else root.split('/')[-1]

        # Load wild camera model for focal estimation
        focal_result_path = os.path.join(args.outputpath, 'focal_estimation', seq_name + '.json')
        if os.path.exists(focal_result_path):
            focal = load(focal_result_path)['focal']
        else:
            wild_camera_model = load_wild_camera_model()
            focal_list = []
            for im_name in tqdm(input_source, dynamic_ncols=True, desc=seq_name+'\'s focal estimation'):
                intrinsic, _ = wild_camera_model.inference(Image.open(os.path.join(args.inputpath, im_name)), wtassumption=False)
                focal = intrinsic[0, 0].item()
                focal_list.append(focal)
        
            focal_list_clear, abnormal_indexes = clear_abnormal(focal_list)
            focal = np.mean(focal_list_clear)
            focal_info = {}
            focal_info['focal'] = focal
            focal_info['focal_list_clear'] = focal_list_clear
            focal_info['abnormal_indexes'] = abnormal_indexes
            focal_info['focal_list'] = focal_list
            dump(focal_info, focal_result_path, sort_keys=True, indent=4)

        if args.focal_estimation:
            continue
        
        # 结果汇总
        seq_result_path = os.path.join(args.outputpath, seq_name + '.json')
        if os.path.exists(seq_result_path) and len(dirs) > 0:
            continue

        # Load detection loader
        det_loader = DetectionLoader(input_source, get_detector(args), cfg, args, batchSize=args.detbatch, mode=mode, queueSize=args.qsize)
        det_worker = det_loader.start()

        runtime_profile = {
            'dt': [],  # detect time
            'pt': [],  # pose time
            'pn': []   # post process time
        }

        # Init data writer
        queueSize = args.qsize
        writer = DataWriter(cfg, args, save_video=False, queueSize=queueSize, seq_name=seq_name).start()

        data_len = det_loader.length
        im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

        batchSize = args.posebatch
        if args.flip:
            batchSize = int(batchSize / 2)
        try:
            for i in im_names_desc:
                start_time = getTime()
                with torch.no_grad():
                    # i. Detection and data transform (i.e. crop and resize)
                    (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                    if orig_img is None:
                        break
                    if boxes is None or boxes.nelement() == 0:
                        writer.save(None, None, None, None, None, orig_img, im_name)
                        continue
                    if args.profile:
                        ckpt_time, det_time = getTime(start_time)
                        runtime_profile['dt'].append(det_time)
                    # ii. Pose Estimation
                    inps = inps.to(args.device)
                    datalen = inps.size(0)
                    leftover = 0
                    if (datalen) % batchSize:
                        leftover = 1
                    num_batches = datalen // batchSize + leftover
                    hm = []
                    for j in range(num_batches):
                        inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                        if args.flip:
                            inps_j = torch.cat((inps_j, flip(inps_j)))
                        hm_j = pose_model(inps_j)
                        if args.flip:
                            hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], pose_dataset.joint_pairs, shift=True)
                            hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                        hm.append(hm_j)
                    hm = torch.cat(hm)
                    if args.profile:
                        ckpt_time, pose_time = getTime(ckpt_time)
                        runtime_profile['pt'].append(pose_time)
                    # iii. Pose Tracking
                    if args.pose_track:
                        boxes,scores,ids,hm,cropped_boxes = track(tracker,args,orig_img,inps,boxes,hm,cropped_boxes,im_name,scores)
                    hm = hm.cpu()
                    writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                    if args.profile:
                        ckpt_time, post_time = getTime(ckpt_time)
                        runtime_profile['pn'].append(post_time)

                if args.profile:
                    # TQDM
                    dt_mean = np.mean(runtime_profile['dt'])
                    pt_mean = np.mean(runtime_profile['pt'])
                    pn_mean = np.mean(runtime_profile['pn'])
                    im_names_desc.set_description(
                        'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                            dt=dt_mean, pt=pt_mean, pn=pn_mean)
                    )

            if args.profile:
                writer.save_profile(dt_mean, pt_mean, pn_mean)
            print_finish_info()
            while(writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
            writer.stop()
            det_loader.stop()
        except Exception as e:
            print(repr(e))
            print('An error as above occurs when processing the images, please check it')
            pass
        except KeyboardInterrupt:
            print_finish_info()
            # Thread won't be killed when press Ctrl+C
            if args.sp:
                det_loader.terminate()
                while(writer.running()):
                    time.sleep(1)
                    print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...', end='\r')
                writer.stop()
            else:
                # subprocesses are killed, manually clear queues

                det_loader.terminate()
                writer.terminate()
                writer.clear_queues()
                det_loader.clear_queues()
