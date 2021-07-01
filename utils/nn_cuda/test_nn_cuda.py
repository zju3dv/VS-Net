import torch
import nn_cuda

n = 1
h = 1
w = 1
c = 8
coding_book = torch.randn(10, c).cuda()

seg_pred = torch.randn(n, c, h, w).cuda()

seg_mask = torch.zeros(n, h, w).cuda().float()

nn_cuda.NearestNeighbor(seg_pred.data.permute(0, 2, 3, 1).contiguous(), coding_book.data, seg_mask)

seg_pred = seg_pred / (torch.norm(seg_pred, dim=1) + 1e-8)[:, None, :, :].expand_as(seg_pred)
coding_book = coding_book / (torch.norm(coding_book, dim=1) + 1e-8)[:, None].expand_as(coding_book)

print(seg_pred.squeeze())

score = torch.matmul(seg_pred.squeeze(), coding_book.permute(1, 0))
print(score)
print(score.max())

print(seg_mask)
