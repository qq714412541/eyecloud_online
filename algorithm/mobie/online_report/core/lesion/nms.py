import numpy as np


def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    inter_area = max(inter_rect_x2 - inter_rect_x1, 0) * max(inter_rect_y2 - inter_rect_y1, 0)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


# scores should be sorted
# this function is only used to suppress different label in one area
def non_max_suppression(boxes, scores, labels, exclude_lbs, nms_thres=0.1):
    args = list(range(len(boxes)))
    i = 0
    while i < len(args):
        arg = args[i]
        box = boxes[arg]
        for k in reversed(range(i+1, len(args))):
            other_arg = args[k]
            if labels[arg] in exclude_lbs and labels[other_arg] in exclude_lbs:
                other_box = boxes[other_arg]
                iou = bbox_iou(box, other_box)
                if iou > nms_thres:
                    args.remove(other_arg)
        i += 1

    return args
