from al.al_algos import *
from al.al_data_loader import QueryRepo
from al.train_test_func import *
import datetime
import psutil
from queue import PriorityQueue

POS_BOX = 25.5
NEG_BOX = 1.6
OUT_BOX = 1.6
WATCH_IMAGE = 7.8
CENTER_POINT = 3


class HeapItem:
    def __init__(self, p, t):
        self.p = p
        self.t = t

    def __lt__(self, other):
        return self.p < other.p

    def __getitem__(self, idx):
        if idx == 0:
            return self.p
        else:
            return self.t

    def __getstate__(self):
        return self.p, self.t

    def __setstate__(self, state):
        self.p, self.t = state

def labeling_and_calc_partial_cost(gt_boxes, queried_box, label_map, img_size=416):
    """partially labeling examples.

    :param gt_boxes: targets from source gt dataloader. index must not be mapped to target class space
    :param queried_box: queried_boxes. After NMS. (x1, y1, x2, y2, al_score, class_conf, class)
    :param label_map: dict
    :return:
    label, cost. The class has not been mapped to target classes.
    """
    queried_gt = None
    cost = 0
    btype = []
    assert gt_boxes is not None and len(gt_boxes) > 0
    # convert to gt format
    rel_coord_qb = xyxy2xywh(ops.expand_dims(queried_box[:4], 0)) / img_size
    # validity checking
    q_coord = rel_coord_qb[0][:4]
    q_coord[q_coord < 0] = 0
    q_coord[q_coord > 1] = 1
    rel_coord_qb[0][:4] = q_coord

    iou = bbox_iou(rel_coord_qb.squeeze(0), gt_boxes[:, 2:], x1y1x2y2=False)
    overlap_iou = bbox_overlap_iou(rel_coord_qb.squeeze(0), gt_boxes[:, 2:], x1y1x2y2=False)
    # part area quering
    queried_gt = gt_boxes[(iou > 0.3) | (overlap_iou > 0.7)]

    has_pos = False
    if len(queried_gt) > 0:
        # fg
        tgt_pos_mask = [True] * queried_gt.shape[0]
        for ti, target in enumerate(queried_gt):
            if int(target[1]) in label_map.keys():
                cost += POS_BOX
                btype.append(1)
                has_pos = True
            else:
                tgt_pos_mask[ti] = False
        queried_gt = queried_gt[tgt_pos_mask]
        if not has_pos:
            queried_gt = ops.concat((ms.Tensor([[0.0, -2.0]]), ms.Tensor(rel_coord_qb)), 1)
            cost += OUT_BOX
            btype.append(-2)
    else:
        # bg, class is -1
        queried_gt = ops.concat((ms.Tensor([[0.0, -1.0]]), ms.Tensor(rel_coord_qb)), 1)
        cost += NEG_BOX
        btype.append(-1)

    return queried_gt, cost, np.asarray(btype)


def qbox_al_mainloop(scoring_arr, src_gt_ds, queried_repo, budget, iteration,
                     src2tgt_label_map, save_suffix='', save_res=True,
                     save_root="/data/saved_al/", fold=0):
    """AL main loop"""
    assert src2tgt_label_map is not None
    start_time = time.clock()

    # combine sort by value, descending order
    pq = PriorityQueue()  # element: (img_name, box) maxsize=budget, low means higher priority
    for k in scoring_arr.keys():
        for box in scoring_arr[k]:
            pq.put((-float(box[4]), HeapItem(k, box)), block=False)

    total_cost = 0
    total_pos, total_neg, total_out = 0, 0, 0

    # labeling, calc cost
    for i in range(pq.qsize()):  # (img_path, score)
        # get ground truth
        pair = pq.get(block=False)
        item = pair[1]
        if item[0] in queried_repo.fs_database:
            # has already fully labeled
            continue
        _, targets, _, _ = src_gt_ds.get_item_by_path(item[0])

        if len(targets) == 0:   # missing GT
            continue

        # check if the query times is more than 3
        if item[0] in queried_repo:
            al_queried_boxes = queried_repo[item[0]][0]
            ar_q_num = len(al_queried_boxes)
            if ar_q_num > 3:
                # fully labeling and continue
                queried_repo.fs_database.append(item[0])
                queried_repo.database.pop(item[0])
                # sum number of queried gt
                queried_tgt_box_num = torch.sum(al_queried_boxes[:, 1] > 0)
                # static the number of target GT
                tgt_cls_num = 0
                for tid in targets:
                    if int(tid[1]) in src2tgt_label_map.keys():
                        # target class
                        tgt_cls_num += 1
                assert tgt_cls_num >= queried_tgt_box_num
                # calc cost
                total_cost += WATCH_IMAGE
                total_cost += POS_BOX * (tgt_cls_num-queried_tgt_box_num)
                continue

        # labeling
        queried_gt, cost, btype = labeling_and_calc_partial_cost(gt_boxes=targets, queried_box=item[1],
                                                                 label_map=src2tgt_label_map)

        # check if the gt is in the repo
        if item[0] in queried_repo:
            qgt_mask = [True]*len(queried_gt)
            for iq, qgt_box in enumerate(queried_gt):
                iou = bbox_iou(qgt_box[2:], al_queried_boxes[:, 2:], x1y1x2y2=False)
                if torch.max(iou) > 0.9:
                    if qgt_box[1] > 0:
                        cost -= POS_BOX
                    else:
                        cost -= NEG_BOX
                    qgt_mask[iq] = False
            if sum(qgt_mask) == 0:
                continue
            queried_gt = queried_gt[qgt_mask]
            btype = btype[qgt_mask]

        assert len(queried_gt) > 0, f'empty query is found.'

        # add cost
        total_cost += cost

        num_pos = np.sum(btype == 1)
        num_neg = np.sum(btype == -1)
        num_outlier = np.sum(btype == -2)
        total_pos += num_pos
        total_neg += num_neg
        total_out += num_outlier

        # update repo
        queried_repo.update(img_path=item[0], im_info=0, gt_boxes=queried_gt,
                            num_pos=num_pos, num_neg=num_neg, num_outlier=num_outlier, cost=cost,
                            domain_label=1, iteration=iteration, method_name='qbox', score=-pair[0])

        if total_cost > budget:
            break

        print(
            f"\rlabeling: {total_cost:0.1f}/{budget:.1f}\t\tpos num: {total_pos}\t\tneg num: {total_neg}\t\toutlier num: {total_out}",
            end='')

    print()
    print(f"QBox labeling end. Total cost: {total_cost}\tannotated examples num: {len(queried_repo)}\tfully anno:{len(queried_repo.fs_database)}")
    print("\nLabeling instances end in %s " % datetime.timedelta(seconds=(time.clock() - start_time)))

    # return query_arr, cost
    return queried_repo, total_cost, (total_pos, total_neg, total_out)


def qbox_al_scoring(unlab_arr, model, s_gt_ds,
                    cocoid2vocid=None, queried_repo=None,
                    pos_ins_weight=0.05,
                    da_tradeoff=10,
                    min_area=15):
    start_time = time.clock()
    total_cand_boxes_num = 0
    scoring = dict()
    model.set_train(False)

    for i, img_path in enumerate(unlab_arr):
        imgs, targets, img_path2, _ = s_gt_ds.get_item_by_path(img_path)
        imgs = ops.expand_dims(imgs, 0)
        domain_label = msnp.zeros(imgs.shape[0])

        # Run model
        inf_out, train_out, da_out = model(imgs, domain_label=domain_label)  # inference and training outputs
        incons_score, trans_score, total_scores = our_scoring_anchor_boxes(train_out, da_out[1:],
                                                                           img_da_output=da_out[0],
                                                                           pos_ins_weight=pos_ins_weight,
                                                                           da_tradeoff=da_tradeoff)
        # get queried boxes for NMS
        if img_path in queried_repo:
            targets, metainfo = queried_repo[img_path]
            for ti, target in enumerate(targets):
                if int(target[1]) in cocoid2vocid.keys():
                    target[1] = cocoid2vocid[int(target[1])]
                else:
                    pass
        else:
            targets = None

        # filter out low score, overlapped boxes
        final_cand_boxes = get_candidate_boxes(train_out, total_scores, model, gt_boxes=targets,
                                               score_thres=0.1, store_part=True,
                                               part_array=[torch.sigmoid(train_out[i][..., 4]) for i in range(3)],
                                               min_area=min_area)
        # update candidate set
        if final_cand_boxes[0] is not None and len(final_cand_boxes[0]) > 0:
            scoring[img_path] = final_cand_boxes[0]
            total_cand_boxes_num += len(final_cand_boxes[0])
    # end query
    end_time = time.clock()
    # datetime.timedelta
    print("\nScoring examples end in %s " % datetime.timedelta(seconds=(end_time - start_time)))
    return scoring
