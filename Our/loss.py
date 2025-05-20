import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=255, use_top_k=False,
                 top_k_ratio=1.0, future_discount=1.0,weight=1.0):

        super().__init__()

        self.class_weights = torch.tensor(class_weights).float()
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.future_discount = future_discount
        self.weight = weight


    def forward(self, prediction, target):
        b,c, h, w = prediction.shape
        loss = F.cross_entropy(
            prediction,
            target,
            reduction='none',
            weight=self.class_weights.to(target.device).float(),
        )

        # ce_loss = self.ce_criterion(prediction, target)
        # pred_logsoftmax = F.log_softmax(prediction)
        # loss = self.nll_criterion(pred_logsoftmax, target)

        loss = loss.view(b,  -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss, _ = torch.sort(loss, dim=2, descending=True)
            loss = loss[:, :, :k]
        m = loss.mean()
        return self.weight*m