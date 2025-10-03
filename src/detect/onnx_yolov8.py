import logging
import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _nms_xyxy(boxes, scores, iou_thres=0.45, top_k=300):
    if boxes.size == 0:
        return np.array([], dtype=np.int32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < top_k:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)

class OnnxYoloV8:
    """
    Handles Ultralytics YOLOv8 ONNX exported heads.
    Auto-detects (1,84,N) vs (1,N,84) and whether xywh appears normalized.
    """
    def __init__(self, model_path: str, input_size: int = 640, apply_sigmoid: bool = False):
        self.model_path = model_path
        self.input_size = int(input_size)
        self.apply_sigmoid = bool(apply_sigmoid)

        so = ort.SessionOptions()
        so.enable_mem_pattern = False
        so.enable_cpu_mem_arena = True
        providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        self.iname = self.sess.get_inputs()[0].name
        self.onames = [o.name for o in self.sess.get_outputs()]

        # Probe output layout
        dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
        out = self.sess.run(self.onames, {self.iname: dummy})[0]
        if out.ndim != 3:
            raise RuntimeError(f"Unexpected YOLOv8 ONNX output shape: {out.shape}")
        self.layout_N_first = out.shape[1] > out.shape[2]  # True if (1, N, 84)
        log.info(f"Loaded ONNX: {model_path}")
        log.info(f"Output shape example: {out.shape}, layout_N_first={self.layout_N_first}, apply_sigmoid={self.apply_sigmoid}")

    def infer_rgb_square(self, rgb: np.ndarray, conf_thres=0.15, iou_thres=0.45, top_k=300):
        """
        rgb: HxWx3 uint8, already resized to (input_size,input_size), RGB
        Returns dict with xyxy boxes in that square pixel space.
        """
        H, W, C = rgb.shape
        log.debug(f"[infer] frame_in: {W}x{H}x{C}, expected={self.input_size}")
        img = rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]  # (1,3,H,W)

        out = self.sess.run(self.onames, {self.iname: img})[0]  # (1,N,84) or (1,84,N)
        if self.layout_N_first:
            preds = out[0]              # (N, 84)
            xywh = preds[:, :4]
            scores_all = preds[:, 4:]
        else:
            preds = out[0]              # (84, N)
            xywh = preds[:4, :].T
            scores_all = preds[4:, :].T

        if self.apply_sigmoid:
            scores_all = _sigmoid(scores_all)

        class_ids = np.argmax(scores_all, axis=1)
        confs = scores_all[np.arange(scores_all.shape[0]), class_ids]

        total_preds = confs.shape[0]
        keep_mask = confs >= conf_thres
        kept = int(np.count_nonzero(keep_mask))
        log.debug(f"[infer] raw_preds={total_preds}, kept@{conf_thres}={kept}")

        if kept == 0:
            return {"boxes": [], "scores": [], "classes": []}

        xywh = xywh[keep_mask]
        confs = confs[keep_mask]
        class_ids = class_ids[keep_mask]

        # Heuristic: if xywh looks normalized (≤ ~2), scale to pixel coords
        max_val = float(np.max(xywh)) if xywh.size else 0.0
        if max_val <= 2.0:
            xywh = xywh * float(self.input_size)
            log.debug("[infer] xywh normalized -> scaled to pixel space")

        # xywh -> xyxy
        x, y, w, h = xywh.T
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Sanity clamp to model square
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, self.input_size - 1)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, self.input_size - 1)

        # NMS
        keep_idx = _nms_xyxy(boxes_xyxy, confs, iou_thres=iou_thres, top_k=top_k)
        boxes_xyxy = boxes_xyxy[keep_idx]
        confs = confs[keep_idx]
        class_ids = class_ids[keep_idx]

        # Log first few boxes
        sample = boxes_xyxy[:3].tolist()
        log.debug(f"[infer] after_nms: n={len(boxes_xyxy)}, sample_boxes(model_space)={sample}")

        return {
            "boxes": boxes_xyxy.tolist(),   # pixel coords in model square (0..input_size)
            "scores": confs.tolist(),
            "classes": class_ids.tolist(),
        }
