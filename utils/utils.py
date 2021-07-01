import cv2
import numpy as np
import math
import torch
import torch.nn.functional as F
import nn_cuda
import ransac_voting_gpu
import transforms3d.quaternions as txq
import transforms3d.euler as txe

def b_inv(b_mat):
    '''
    code from
    https://stackoverflow.com/questions/46595157/how-to-apply-the-torch-inverse-function-of-pytorch-to-every-sample-in-the-batc
    :param b_mat:
    :return:
    '''
    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    if torch.__version__ >= '1.0.0':
        b_inv, _ = torch.solve(eye, b_mat)
    else:
        b_inv, _ = torch.gesv(eye, b_mat)
    # b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv

def ransac_voting_vertex(mask, vertex, round_hyp_num, inlier_thresh=0.999, confidence=0.99, max_iter=20,
                           min_num=5, max_num=30000):
    '''
    :param mask:      [b,h,w]
    :param vertex:    [b,h,w,vn,2]
    :param round_hyp_num:
    :param inlier_thresh:
    :return: [b,vn,2]
    '''
    b, h, w, vn, _ = vertex.shape
    batch_win_pts = []
    for bi in range(b):
        hyp_num = 0
        cur_mask = (mask[bi]).byte()
        foreground_num = torch.sum(cur_mask)

        # if too few points, just skip it
        if foreground_num < min_num:
            win_pts = torch.zeros([1, vn, 2], dtype=torch.float32, device=mask.device)
            batch_win_pts.append(win_pts)  # [1,vn,2]
            continue

        # if too many inliers, we randomly down sample it
        if foreground_num > max_num:
            selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=mask.device).uniform_(0, 1)
            selected_mask = (selection < (max_num / foreground_num.float()))
            cur_mask *= selected_mask

        coords = torch.nonzero(cur_mask).float()  # [tn,2]
#         print(coords.shape)
        coords = coords[:, [1, 0]]
        direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask, 2), 3))  # [tn,vn,2]
        direct = direct.view([coords.shape[0], vn, 2])
        tn = coords.shape[0]
        idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])
        all_win_ratio = torch.zeros([vn], dtype=torch.float32, device=mask.device)
        all_win_pts = torch.zeros([vn, 2], dtype=torch.float32, device=mask.device)

        cur_iter = 0
        while True:
            # generate hypothesis
            cur_hyp_pts = ransac_voting_gpu.generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]

            # voting for hypothesis
            cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
            ransac_voting_gpu.voting_for_hypothesis(direct, coords, cur_hyp_pts, cur_inlier, inlier_thresh)  # [hn,vn,tn]

            # find max
            cur_inlier_counts = torch.sum(cur_inlier, 2)                   # [hn,vn]
            cur_win_counts, cur_win_idx = torch.max(cur_inlier_counts, 0)  # [vn]
            cur_win_pts = cur_hyp_pts[cur_win_idx, torch.arange(vn)]
            cur_win_ratio = cur_win_counts.float() / tn

            # update best point
            larger_mask = all_win_ratio < cur_win_ratio
            all_win_pts[larger_mask, :] = cur_win_pts[larger_mask, :]
            all_win_ratio[larger_mask] = cur_win_ratio[larger_mask]

            # check confidence
            hyp_num += round_hyp_num
            cur_iter += 1
            cur_min_ratio = torch.min(all_win_ratio)
            if (1 - (1 - cur_min_ratio ** 2) ** hyp_num) > confidence or cur_iter > max_iter:
                break

        # compute mean intersection again
        normal = torch.zeros_like(direct)   # [tn,vn,2]
        normal[:, :, 0] = direct[:, :, 1]
        normal[:, :, 1] = -direct[:, :, 0]
        all_inlier = torch.zeros([1, vn, tn], dtype=torch.uint8, device=mask.device)
        all_win_pts = torch.unsqueeze(all_win_pts, 0)  # [1,vn,2]
        ransac_voting_gpu.voting_for_hypothesis(direct, coords, all_win_pts, all_inlier, inlier_thresh)  # [1,vn,tn]

        # coords [tn,2] normal [vn,tn,2]
        all_inlier=torch.squeeze(all_inlier.float(),0)              # [vn,tn]
        normal=normal.permute(1,0,2)                                # [vn,tn,2]
        normal=normal*torch.unsqueeze(all_inlier,2)                 # [vn,tn,2] outlier is all zero
        if torch.norm(normal, p=2) > 1e-6:
            b=torch.sum(normal*torch.unsqueeze(coords,0),2)             # [vn,tn]
            ATA=torch.matmul(normal.permute(0,2,1),normal)              # [vn,2,2]
            ATb=torch.sum(normal*torch.unsqueeze(b,2),1)                # [vn,2]
            all_win_pts=torch.matmul(b_inv(ATA),torch.unsqueeze(ATb,2)) # [vn,2,1]
            batch_win_pts.append(all_win_pts[None,:,:,0])
    
    if len(batch_win_pts) > 0:
        return 'success', torch.cat(batch_win_pts)
    else:
        return 'failed', None

def ransac_voting_vertex_v2(mask, vertex, round_hyp_num, inlier_thresh=0.999, confidence=0.99, max_iter=20,
                           min_num=30, max_num=30000, min_inlier_count=5, neighbor_radius=6.0):
    '''
    :param mask:      [b,h,w]
    :param vertex:    [b,h,w,vn,2]
    :param round_hyp_num:
    :param inlier_thresh:
    :return: [b,vn,2]
    '''
    b, h, w, vn, _ = vertex.shape
    batch_win_pts = []
    for bi in range(b):
        hyp_num = 0
        cur_mask = (mask[bi]).byte()
        foreground_num = torch.sum(cur_mask)

        # if too few points, just skip it
        if foreground_num < min_num:
            win_pts = torch.zeros([1, vn, 2], dtype=torch.float32, device=mask.device)
            batch_win_pts.append(win_pts)  # [1,vn,2]
            continue

        # if too many inliers, we randomly down sample it
        if foreground_num > max_num:
            selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=mask.device).uniform_(0, 1)
            selected_mask = (selection < (max_num / foreground_num.float()))
            cur_mask *= selected_mask

        coords = torch.nonzero(cur_mask).float()  # [tn,2]

        coords = coords[:, [1, 0]]
        direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask, 2), 3))  # [tn,vn,2]
        direct = direct.view([coords.shape[0], vn, 2])
        tn = coords.shape[0]
        idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])
        all_win_ratio = torch.zeros([vn], dtype=torch.float32, device=mask.device)
        all_win_pts = torch.zeros([vn, 2], dtype=torch.float32, device=mask.device)

        cur_iter = 0
        while True:
            # generate hypothesis
            cur_hyp_pts = ransac_voting_gpu.generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]

            # voting for hypothesis
            cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
            ransac_voting_gpu.voting_for_hypothesis(direct, coords, cur_hyp_pts, cur_inlier, inlier_thresh)  # [hn,vn,tn]

            # find max
            cur_inlier_counts = torch.sum(cur_inlier, 2)                   # [hn,vn]
            cur_win_counts, cur_win_idx = torch.max(cur_inlier_counts, 0)  # [vn]
            cur_win_pts = cur_hyp_pts[cur_win_idx, torch.arange(vn)]
            cur_win_ratio = cur_win_counts.float() / tn

            # update best point
            larger_mask = all_win_ratio < cur_win_ratio
            all_win_pts[larger_mask, :] = cur_win_pts[larger_mask, :]
            all_win_ratio[larger_mask] = cur_win_ratio[larger_mask]

            # check confidence
            hyp_num += round_hyp_num
            cur_iter += 1
            cur_min_ratio = torch.min(all_win_ratio)
            if (1 - (1 - cur_min_ratio ** 2) ** hyp_num) > confidence:
#                 curr_confidence = (1 - (1 - cur_min_ratio ** 2) ** hyp_num)
#                 print('stop by confidence', curr_confidence, 'cur_min_ratio', cur_min_ratio, 'final iter', cur_iter)
                break
            if cur_iter > max_iter:
#                 print('stop by max_iter, final iter', cur_iter)
                break
                
            
        # compute mean intersection again
        normal = torch.zeros_like(direct)   # [tn,vn,2]
        normal[:, :, 0] = direct[:, :, 1]
        normal[:, :, 1] = -direct[:, :, 0]
        all_inlier = torch.zeros([1, vn, tn], dtype=torch.uint8, device=mask.device)
        all_win_pts = torch.unsqueeze(all_win_pts, 0)  # [1,vn,2]
        ransac_voting_gpu.voting_for_hypothesis(direct, coords, all_win_pts, all_inlier, inlier_thresh)  # [1,vn,tn]
        # coords [tn,2] normal [vn,tn,2]
        all_inlier=torch.squeeze(all_inlier.float(),0)              # [vn,tn]
        normal=normal.permute(1,0,2)                                # [vn,tn,2]
        normal=normal*torch.unsqueeze(all_inlier,2)                 # [vn,tn,2] outlier is all zero
        if torch.norm(normal, p=2) > 1e-6:
            b=torch.sum(normal*torch.unsqueeze(coords,0),2)             # [vn,tn]
            ATA=torch.matmul(normal.permute(0,2,1),normal)              # [vn,2,2]
            ATb=torch.sum(normal*torch.unsqueeze(b,2),1)                # [vn,2]
            all_win_pts=torch.matmul(b_inv(ATA),torch.unsqueeze(ATb,2)) # [vn,2,1]
            batch_win_pts.append(all_win_pts[None,:,:,0])

        # iterative recompute new vertex from neighbor points
        # neighbor_radius = 5.0
        iter_time = 0
        while True:
            iter_time += 1
            if iter_time > 20:
                break
            if len(batch_win_pts) <= 0: break
            last_vertex = batch_win_pts[0].reshape(1, 2)
            # last_vertex = torch.from_numpy(np.array(batch_win_pts[0].cpu()).reshape(1, 2)).to(coords.device)
#             print(iter_time, last_vertex)
            # compute distance
            dist = torch.norm(coords - last_vertex, p=2, dim=1)
            # we only use points near last vertex to compute new vertex
            neighbor_idx = (dist < neighbor_radius).nonzero().squeeze()
            if neighbor_idx.nelement() < min_inlier_count:
                return 'failed', None
            neighbor_coords = coords[neighbor_idx, :]
            neighbor_direct = direct[neighbor_idx, :, :]

            tn = neighbor_coords.shape[0]
            idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])
            all_win_ratio = torch.zeros([vn], dtype=torch.float32, device=mask.device)
            all_win_pts = torch.zeros([vn, 2], dtype=torch.float32, device=mask.device)

            # compute mean intersection again
            normal = torch.zeros_like(neighbor_direct)   # [tn,vn,2]
            normal[:, :, 0] = neighbor_direct[:, :, 1]
            normal[:, :, 1] = -neighbor_direct[:, :, 0]
            # torch.ones! all neighbors will be regarded as inliers
            all_inlier = torch.ones([1, vn, tn], dtype=torch.uint8, device=mask.device)

            # coords [tn,2] normal [vn,tn,2]
            batch_win_pts = []
            all_inlier=torch.squeeze(all_inlier.float(),0)              # [vn,tn]
            normal=normal.permute(1,0,2)                                # [vn,tn,2]
            normal=normal*torch.unsqueeze(all_inlier,2)                 # [vn,tn,2] outlier is all zero
            if torch.norm(normal, p=2) > 1e-6:
                b=torch.sum(normal*torch.unsqueeze(neighbor_coords,0),2)             # [vn,tn]
                ATA=torch.matmul(normal.permute(0,2,1),normal)              # [vn,2,2]
                ATb=torch.sum(normal*torch.unsqueeze(b,2),1)                # [vn,2]
                all_win_pts=torch.matmul(b_inv(ATA),torch.unsqueeze(ATb,2)) # [vn,2,1]
                batch_win_pts.append(all_win_pts[None,:,:,0])
            if len(batch_win_pts) > 0:
                curr_vertex = batch_win_pts[0].reshape(1, 2)
                # curr_vertex = torch.from_numpy(np.array(batch_win_pts[0].cpu()).reshape(1, 2)).to(coords.device)
                iter_step = torch.norm(curr_vertex - last_vertex, p=2, dim=1)
                if iter_step < 1e-3:
#                     print('iter stop at %d with step %.5e' % (iter_time, iter_step))
                    break
            
    if len(batch_win_pts) > 0:
        return 'success', torch.cat(batch_win_pts)
    else:
        return 'failed', None


def evaluate_segmentation(seg_pred, coding_book, size=None, use_own_nn=False):
    # evaluate seg_pred
    epsilon = 1e-8
    seg_pred = seg_pred / (torch.norm(seg_pred, dim=1) + epsilon)[:, None, :, :].expand_as(seg_pred)
    coding_book = coding_book / (torch.norm(coding_book, dim=1) + epsilon)[:, None].expand_as(coding_book)
    n, c, h, w = seg_pred.shape
    
    if use_own_nn == True:
        seg_mask = torch.zeros(n, h, w).cuda().float()
        nn_cuda.NearestNeighbor(seg_pred.data.permute(0, 2, 3, 1).contiguous(), coding_book.data, seg_mask)        
        seg_pred = seg_mask.detach().squeeze().float()
    else:
        assert n == 1
        e, _ = coding_book.shape
        coding_book = coding_book.detach().unsqueeze(2).unsqueeze(3).expand(e, c, h, w)
        seg_pred = seg_pred.detach(0).expand(e, c, h, w)
        seg_pred = torch.argmin((seg_pred - coding_book).pow(2).sum(1), dim=0).float()

    seg_pred = F.interpolate(seg_pred.unsqueeze(0).unsqueeze(0), size=size, mode="nearest").squeeze().long()
    return seg_pred, None

def evaluate_vertex(vertex_pred, seg_pred, id2center, round_hyp_num=256, inlier_thresh=0.999, max_num=10000, max_iter=30, min_mask_num=20):
    vertex_pred = vertex_pred.permute(0, 2, 3, 1)
    b, h, w, vn_2 = vertex_pred.shape
    vertex_pred = vertex_pred.view(b, h, w, vn_2//2, 2)
    
    unique_labels = torch.unique(seg_pred)

    keypoint_preds = []
    for label in unique_labels:
        if label == 0: continue
        mask = (seg_pred == label)
        mask = mask.unsqueeze(0)
        if mask.sum() < min_mask_num: continue
        keypoint_preds.append((ransac_voting_vertex(mask, vertex_pred, round_hyp_num, 
                            inlier_thresh=inlier_thresh, max_num=max_num, max_iter=max_iter), label))
    pt3d_filter = []
    pt2d_filter = []
    idx_filter = []
    for (status, pt2d_pred), idx in keypoint_preds:
        if status == 'failed': continue
        pt2d_pred = pt2d_pred.cpu().numpy()
        if True in np.isnan(pt2d_pred): continue
        pt3d_filter.append(id2center[idx])
        pt2d_filter.append(pt2d_pred[0][0])
        idx_filter.append(idx.data.item())
    
    if len(pt3d_filter) > 0:
        pt3d_filter = np.concatenate(pt3d_filter).reshape(-1, 3)
        pt2d_filter = np.concatenate(pt2d_filter).reshape(-1, 2)
    else:
        pt3d_filter = np.array(pt3d_filter)
        pt2d_filter = np.array(pt2d_filter)
    
    idx_filter = np.array(idx_filter)
    
    return pt3d_filter, pt2d_filter, idx_filter


def evaluate_vertex_v2(vertex_pred, seg_pred, id2center, round_hyp_num=256, inlier_thresh=0.999, max_num=10000, max_iter=30, min_mask_num=20, min_inlier_count=5, neighbor_radius=6.0):
    vertex_pred = vertex_pred.permute(0, 2, 3, 1)
    b, h, w, vn_2 = vertex_pred.shape
    vertex_pred = vertex_pred.view(b, h, w, vn_2//2, 2)
    
    unique_labels = torch.unique(seg_pred)

    keypoint_preds = []
    for label in unique_labels:
        if label == 0: continue
        mask = (seg_pred == label)
        mask = mask.unsqueeze(0)
        if mask.sum() < min_mask_num: continue
        keypoint_preds.append((ransac_voting_vertex_v2(mask, vertex_pred, round_hyp_num, 
                            inlier_thresh=inlier_thresh, max_num=max_num, max_iter=max_iter,
                            min_inlier_count=min_inlier_count, neighbor_radius=neighbor_radius), label))
    pt3d_filter = []
    pt2d_filter = []
    idx_filter = []
    for (status, pt2d_pred), idx in keypoint_preds:
        if status == 'failed': continue
        pt2d_pred = pt2d_pred.cpu().numpy()
        if True in np.isnan(pt2d_pred): continue
        pt3d_filter.append(id2center[idx])
        pt2d_filter.append(pt2d_pred[0][0])
        idx_filter.append(idx.data.item())
    
    if len(pt3d_filter) > 0:
        pt3d_filter = np.concatenate(pt3d_filter).reshape(-1, 3)
        pt2d_filter = np.concatenate(pt2d_filter).reshape(-1, 2)
    else:
        pt3d_filter = np.array(pt3d_filter)
        pt2d_filter = np.array(pt2d_filter)
    
    idx_filter = np.array(idx_filter)
    
    return pt3d_filter, pt2d_filter, idx_filter

def reproject_error(pt3d, pt2d, pose_pred, k_matrix):
    k_matrix = torch.from_numpy(k_matrix).double()
    f, ppx, ppy = k_matrix[0, 0], k_matrix[0, 2], k_matrix[1, 2]
    pt3d = torch.from_numpy(pt3d).permute(1, 0).double()
    pt2d = torch.from_numpy(pt2d).double()
    pose_pred = torch.from_numpy(pose_pred)
    R = pose_pred[:3, :3]
    t = pose_pred[:3, 3]
    objMat = torch.matmul(R, pt3d).permute(1, 0) + t
    obj_pt = torch.zeros_like(objMat[..., :2]).double()
    obj_pt[..., 0] = objMat[..., 0] * f / objMat[..., 2] + ppx
    obj_pt[..., 1] = objMat[..., 1] * f / objMat[..., 2] + ppy
    error = torch.norm(pt2d - obj_pt, p=2, dim=1).clamp(0, 100).mean()
    return error.float()

def quaternion_angular_error(q1, q2):
    """
    angular error between two quaternions
    :param q1: (4, )
    :param q2: (4, )
    :return:
    """
    d = abs(np.dot(q1, q2))
    d = min(1.0, max(-1.0, d))
    theta = 2 * np.arccos(d) * 180 / np.pi
    return theta
    
def cm_degree_metric(pose_pred, pose_target):
    """
    1. pose_pred is considered correct if the translation and rotation errors are below 5 cm and 5 degree respectively
    """
    pose_target = pose_target.squeeze()[:3, :].cpu().numpy()

    pose_target[:3, :3] = np.linalg.inv(pose_target[:3, :3])
    pose_target[:3, 3] = np.matmul(pose_target[:3, :3], - pose_target[:3, 3])

    pose_pred[:3, :3] = np.linalg.inv(pose_pred[:3, :3])
    pose_pred[:3, 3] = np.matmul(pose_pred[:3, :3], - pose_pred[:3, 3])

    translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_target[:, 3]) * 100
    q1 = txq.mat2quat(pose_target[:3, :3])
    q2 = txq.mat2quat(pose_pred[:3, :3])
    q1 = q1 / txq.qnorm(q1)
    q2 = q2 / txq.qnorm(q2)
    angular_distance = quaternion_angular_error(q1, q2)
    return translation_distance, angular_distance
        

def pnp(points_3d, points_2d, camera_matrix, method=cv2.SOLVEPNP_ITERATIVE):
    dist_coeffs = np.zeros(shape=[8, 1], dtype='float64')

    assert points_3d.shape[0] == points_2d.shape[0], 'points 3D and points 2D must have same number of vertices'
    if method == cv2.SOLVEPNP_EPNP:
        points_3d = np.expand_dims(points_3d, 0)
        points_2d = np.expand_dims(points_2d, 0)

    points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
    points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
    camera_matrix = camera_matrix.astype(np.float64)

    ret, R_exp, t, inliers = cv2.solvePnPRansac(points_3d,
                                                points_2d,
                                                camera_matrix,
                                                None,
                                                iterationsCount=2000,
                                                reprojectionError=10,
                                                flags=cv2.SOLVEPNP_EPNP)

    if ret and inliers.shape[0] > 6:
        if method == cv2.SOLVEPNP_EPNP:
            inliers = np.squeeze(inliers)
        _, R_exp, t = cv2.solvePnP(points_3d[inliers],
                                   points_2d[inliers],
                                   camera_matrix,
                                   None,
                                   rvec=R_exp,
                                   tvec=t,
                                   useExtrinsicGuess=True,
                                   flags=method)

        R, _ = cv2.Rodrigues(R_exp)

        for _ in range(8):
            f, ppx, ppy = camera_matrix[0,
                                        0], camera_matrix[0, 2], camera_matrix[1, 2]
            pt3d = points_3d.transpose(1, 0)
            pt2d = points_2d
            objMat = np.matmul(R, pt3d).transpose(1, 0) + t.transpose(1, 0)
            obj_pt = np.zeros_like(objMat[..., :2]).astype(np.float64)
            obj_pt[..., 0] = objMat[..., 0] * f / objMat[..., 2] + ppx
            obj_pt[..., 1] = objMat[..., 1] * f / objMat[..., 2] + ppy
            error = np.linalg.norm(pt2d - obj_pt, axis=1)
            inliers = error < 10
            if np.sum(inliers) < 6:
                break
            _, R_exp, t = cv2.solvePnP(points_3d[inliers],
                                       points_2d[inliers],
                                       camera_matrix,
                                       None,
                                       rvec=R_exp,
                                       tvec=t,
                                       useExtrinsicGuess=True,
                                       flags=method)
            R, _ = cv2.Rodrigues(R_exp)

    R, _ = cv2.Rodrigues(R_exp)

    return ret, np.concatenate([R, t], axis=-1)
