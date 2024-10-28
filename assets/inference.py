# This script is modified from https://github.com/caizhongang/SMPLer-X/blob/main/main/inference.py
# Licensed under:
"""
S-Lab License 1.0

Copyright 2022 S-Lab
Redistribution and use for non-commercial purpose in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
4. In the event that redistribution and/or use for commercial purpose in source or binary forms, with or without modification is required, please contact the contributor(s) of the work.
"""

import os
import sys
import os.path as osp
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch

CUR_DIR = osp.dirname(os.path.abspath(__file__))
sys.path.insert(0, osp.join(CUR_DIR, "..", "main"))
sys.path.insert(0, osp.join(CUR_DIR, "..", "common"))
from config import cfg
from mmdet.apis import init_detector, inference_detector
from utils.inference_utils import process_mmdet_results


class Inferer:
    def __init__(self, pretrained_model, num_gpus, output_folder):
        self.output_folder = output_folder
        self.device = torch.device("cuda") if (num_gpus > 0) else torch.device("cpu")
        config_path = osp.join(CUR_DIR, "./config", f"config_{pretrained_model}.py")
        ckpt_path = osp.join(CUR_DIR, "../pretrained_models", f"{pretrained_model}.pth.tar")
        cfg.get_config_fromfile(config_path)
        cfg.update_config(num_gpus, ckpt_path, output_folder, self.device)
        self.cfg = cfg
        cudnn.benchmark = True

        # load model
        from base import Demoer

        demoer = Demoer()
        demoer._make_model()
        demoer.model.eval()
        self.demoer = demoer
        checkpoint_file = osp.join(CUR_DIR, "../pretrained_models/mmdet/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth")
        config_file = osp.join(CUR_DIR, "../pretrained_models/mmdet/mmdet_faster_rcnn_r50_fpn_coco.py")
        model = init_detector(config_file, checkpoint_file, device=self.device)  # or device='cuda:0'
        self.model = model

    def infer(self, original_img, iou_thr, frame, multi_person=False, mesh_as_vertices=False):
        from utils.preprocessing import process_bbox, generate_patch_image

        mesh_paths = []
        smplx_paths = []
        # prepare input image
        transform = transforms.ToTensor()
        vis_img = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]

        ## mmdet inference
        mmdet_results = inference_detector(self.model, original_img)

        pred_instance = mmdet_results.pred_instances.cpu().numpy()
        bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[pred_instance.labels == 0]
        bboxes = np.expand_dims(bboxes, axis=0)
        mmdet_box = process_mmdet_results(bboxes, cat_id=0, multi_person=True)

        # save original image if no bbox
        if len(mmdet_box[0]) < 1:
            return original_img, [], []

        num_bbox = 1
        mmdet_box = mmdet_box[0]

        ## loop all detected bboxes
        for bbox_id in range(num_bbox):
            mmdet_box_xywh = np.zeros((4))
            mmdet_box_xywh[0] = mmdet_box[bbox_id][0]
            mmdet_box_xywh[1] = mmdet_box[bbox_id][1]
            mmdet_box_xywh[2] = abs(mmdet_box[bbox_id][2] - mmdet_box[bbox_id][0])
            mmdet_box_xywh[3] = abs(mmdet_box[bbox_id][3] - mmdet_box[bbox_id][1])

            # skip small bboxes by bbox_thr in pixel
            if mmdet_box_xywh[2] < 50 or mmdet_box_xywh[3] < 150:
                continue

            bbox = process_bbox(mmdet_box_xywh, original_img_width, original_img_height)
            img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, self.cfg.input_img_shape)
            img = transform(img.astype(np.float32)) / 255
            img = img.to(cfg.device)[None, :, :, :]
            inputs = {"img": img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = self.demoer.model(inputs, targets, meta_info, "test")

            ## save single person param
            smplx_pred = {}
            smplx_pred["global_orient"] = out["smplx_root_pose"].reshape(-1, 3).cpu().numpy()
            smplx_pred["body_pose"] = out["smplx_body_pose"].reshape(-1, 3).cpu().numpy()
            smplx_pred["left_hand_pose"] = out["smplx_lhand_pose"].reshape(-1, 3).cpu().numpy()
            smplx_pred["right_hand_pose"] = out["smplx_rhand_pose"].reshape(-1, 3).cpu().numpy()
            smplx_pred["jaw_pose"] = out["smplx_jaw_pose"].reshape(-1, 3).cpu().numpy()
            smplx_pred["leye_pose"] = np.zeros((1, 3))
            smplx_pred["reye_pose"] = np.zeros((1, 3))
            smplx_pred["betas"] = out["smplx_shape"].reshape(-1, 10).cpu().numpy()
            smplx_pred["expression"] = out["smplx_expr"].reshape(-1, 10).cpu().numpy()
            smplx_pred["transl"] = out["cam_trans"].reshape(-1, 3).cpu().numpy()
            save_path_smplx = os.path.join(self.output_folder, "smplx")
            os.makedirs(save_path_smplx, exist_ok=True)

            npz_path = os.path.join(save_path_smplx, f"{frame:05}_{bbox_id}.npz")
            np.savez(npz_path, **smplx_pred)
            smplx_paths.append(npz_path)

            vis_img = None
            mesh_paths = None
        return vis_img, mesh_paths, smplx_paths
