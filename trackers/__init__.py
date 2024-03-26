import numpy as np
import torch
def track(tracker,args,orig_img,inps,boxes,hm,cropped_boxes,im_name,scores,camera_cfg):
    hm = hm.cpu().data.numpy()
    online_targets = tracker.update(orig_img,inps,boxes,hm,cropped_boxes,im_name,scores,camera_cfg)
    new_boxes,new_scores,new_ids,new_hm,new_crop = [],[],[],[],[]
    for t in online_targets:
        tlbr = t.tlbr
        tid = t.track_id
        thm = t.pose
        tcrop = t.crop_box
        tscore = t.detscore
        new_boxes.append(tlbr)
        new_crop.append(tcrop)
        new_hm.append(thm)
        new_ids.append(tid)
        new_scores.append(tscore)

    # 在转换为张量之前，将列表转换为一个单一的 numpy 数组，然后再进行张量转换
    new_hm = torch.Tensor(np.array(new_hm)).to(args.device)  
    return new_boxes,new_scores,new_ids,new_hm,new_crop
