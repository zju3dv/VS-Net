import torch
from torch import nn, einsum


# top k contrastive loss
def embeddingLoss_v3(seg_pred,
                     seg_target,
                     coding_book,
                     margin=0.5,
                     seg_k=3,
                     valid_mask=None):
    # seg_pred B x E x W x H
    # seg_target B x W x H
    # coding_book E x L
    start_idx = 1000
    stop_idx = start_idx + 10
    seg_pred = seg_pred.transpose(
        0, 1).contiguous().reshape(seg_pred.shape[1], -1)
    seg_target = seg_target.reshape(-1)
    embedding_length = seg_pred.shape[0]
    sample_number = seg_target.shape[0]

    seg_target = seg_target.reshape(-1)
    # print("seg target = {}".format(seg_target[start_idx:stop_idx]))

    unique_labels = torch.unique(seg_target)
    # print("unique_labels = {}".format(unique_labels[:30]))
    label_number = unique_labels.shape[0]
    seg_pred = seg_pred / (torch.norm(seg_pred, dim=0) + 1e-8)
    coding_book = coding_book / \
        (torch.norm(coding_book, dim=1) +
         1e-8)[:, None].expand_as(coding_book)

    mean_embedding = torch.zeros(
        (embedding_length, label_number), dtype=torch.float32).cuda()
    # label_count = torch.zeros((label_number), dtype=torch.int32).cuda()
    for idx in range(label_number):
        # mask = seg_target == unique_labels[idx]
        # label_count[idx] = torch.sum(mask)
        # assert label_count[idx] != 0
        # mean_embedding[:, idx] = torch.sum(seg_pred[:, mask], dim=1) / label_count[idx].float()
        mean_embedding[:, idx] = torch.mean(
            seg_pred[:, seg_target == unique_labels[idx]], dim=1).detach()

    scores = torch.mm(mean_embedding.transpose(0, 1),
                      coding_book.transpose(0, 1))
    for idx in range(label_number):
        scores[idx, unique_labels[idx]] = -1
    # print("scores shape = {}".format(scores.shape))
    # print("scores = {}".format(scores[:10, :10]))
    # print("scores: {} / {} / {}".format(torch.min(scores), torch.median(scores), torch.max(scores)))
    # print("embedding[1000] = {}".format(coding_book[1000, :5]))
    # print("embedding[2000] = {}".format(coding_book[2000, :5]))
    sorted_scores, indices = torch.topk(
        scores, k=seg_k, dim=1, largest=True, sorted=False)
    # print("sorted scores: {}".format(sorted_scores))

    local_to_global = unique_labels
    # print("local to global = {}".format(local_to_global[:20]))
    index_table = torch.range(
        0, unique_labels.shape[0], dtype=torch.int64).cuda()
    # print("index table = {}".format(index_table[:20]))
    global_to_local = torch.zeros(
        (coding_book.shape[0]), dtype=torch.int64).cuda()
    # print("global to local shape = {}".format(global_to_local.shape))
    # print("max index in local_to_global = {}".format(torch.max(local_to_global)))
    global_to_local = global_to_local.scatter(
        0, local_to_global, index_table)
    # print("local to global = {}".format(global_to_local[:20]))
    local_seg_target = torch.gather(global_to_local, 0, seg_target)
    # print("local seg target = {}".format(local_seg_target[:20]))
    eye = torch.eye(seg_k, dtype=torch.uint8)
    # print("indices = {}".format(indices[start_idx:stop_idx]))
    negative_indices = torch.gather(
        indices, 0, local_seg_target[:, None].expand(sample_number, seg_k))
    # print("negative indices shape= {}".format(negative_indices.shape))
    negative_indices = negative_indices[eye[(
        torch.rand(sample_number).cuda() * seg_k).long()]]
    negative_embeddings = coding_book.transpose(0, 1)[:, negative_indices]
    coding_book_norm = torch.norm(coding_book, dim=1)
    negative_activations = torch.sum(
        torch.mul(negative_embeddings, seg_pred[:, :]), dim=0)
    positive_activations = torch.sum(
        torch.mul(coding_book.transpose(0, 1)[:, seg_target], seg_pred), dim=0)
    # loss = - torch.log(torch.exp(positive_activations) / norm_denominator)
    loss = torch.relu(negative_activations - positive_activations + margin)
    # print("loss = {}".format(loss[start_idx:stop_idx]))
    # print("loss: {} / {} / {}".format(torch.min(loss), torch.median(loss), torch.max(loss)))

    # print("not use weight", "="* 50)
    if valid_mask is not None:
        # print("use mask", "="* 50)
        mask = valid_mask.reshape(-1)
        loss *= mask
        loss = torch.sum(loss) / (torch.sum(mask) + 1e-8)
        # print(torch.sum(mask), loss)
    else:
        loss = torch.mean(loss)

    return loss


def cross_entropy(seg_pred, target, coding_book=None, margin=None):
    criterion = nn.CrossEntropyLoss(weight=None,
                                    ignore_index=-1,
                                    size_average=True)

    loss = criterion(seg_pred, target.long())

    return loss.mean()


def smooth_l1_loss(vertex_pred,
                   vertex_targets,
                   vertex_weights,
                   sigma=1.0,
                   normalize=True,
                   reduce=True):
    """
    :param vertex_pred:     [b,vn*2,h,w]
    :param vertex_targets:  [b,vn*2,h,w]
    :param vertex_weights:  [b,1,h,w]
    :param sigma:
    :param normalize:
    :param reduce:
    :return:
    """
    b, ver_dim, _, _ = vertex_pred.shape
    sigma_2 = sigma**2
    vertex_diff = vertex_pred - vertex_targets
    diff = vertex_weights * vertex_diff
    abs_diff = torch.abs(diff)
    smoothL1_sign = (abs_diff < 1. / sigma_2).detach().float()
    in_loss = torch.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
        + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)

    if normalize:
        in_loss = torch.sum(in_loss.view(b, -1), 1) / (
            ver_dim * torch.sum(vertex_weights.view(b, -1), 1) + 1e-3)

    if reduce:
        in_loss = torch.mean(in_loss)

    return in_loss


def l1_loss(vertex_pred, vertex_targets, vertex_weights):
    """
    :param vertex_pred:     [b,vn*2,h,w]
    :param vertex_targets:  [b,vn*2,h,w]
    :param vertex_weights:  [b,1,h,w]
    :return:
    """
    if len(vertex_weights.shape) < 4:
        vertex_weights = vertex_weights.unsqueeze(1)
    b, ver_dim, _, _ = vertex_pred.shape
    vertex_pred = vertex_pred * vertex_weights
    vertex_targets = vertex_targets * vertex_weights
    vertex_diff = vertex_pred - vertex_targets
    diff = vertex_weights * vertex_diff
    in_loss = torch.abs(diff)
    in_loss = torch.sum(in_loss.view(b, -1), 1) / (
        ver_dim * torch.sum(vertex_weights.view(b, -1), 1) + 1e-3)

    in_loss = torch.mean(in_loss)

    return in_loss


loss_dict = {
    'l1_loss': l1_loss,
    'smooth_l1_loss': smooth_l1_loss,
    'ce': cross_entropy,
    'embedding_v3': embeddingLoss_v3,
}
