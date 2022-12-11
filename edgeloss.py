import torch
import torch.nn.functional as F


class EdgeLoss:
    def __init__(self, KSIZE=7):
        self.KSIZE = KSIZE
        self.MASK = torch.zeros([KSIZE, KSIZE])
        self.cal_mask()

    def cal_mask(self):
        num = 0
        ksize = self.MASK
        MASK = self.MASK
        for x in range(0, ksize):
            for y in range(0, ksize):
                if (x + 0.5 - ksize / 2) ** 2 + (y + 0.5 - ksize / 2) ** 2 <= (
                    (ksize - 1) / 2
                ) ** 2:
                    MASK[x][y] = 1
                    num += 1
        MASK = MASK.reshape(1, 1, 1, 1, -1).float() / num
        MASK = MASK.to(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.MASK = MASK

    def tensor_average(self, bin_img, ksize):
        B, C, H, W = bin_img.shape
        pad = (ksize - 1) // 2
        bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode="constant", value=0)

        patches = bin_img.unfold(dimension=2, size=ksize, step=1)
        patches = patches.unfold(dimension=3, size=ksize, step=1)

        eroded = torch.sum(patches.reshape(B, C, H, W, -1).float() * self.MASK, dim=-1)
        return eroded

    def edgeLoss(self, input, target):
        targets = target.unsqueeze(dim=1)
        targetAve = self.tensor_average(targets, ksize=self.KSIZE)
        at = torch.abs(targets.float() - targetAve)
        at = at.view(-1)

        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))

        target = target.view(-1, 1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)

        loss = -1 * logpt * at
        return loss.mean()


def combine_loss(prediction, target):
    """Calculating the loss"""
    loss = 0
    EL = EdgeLoss(KSIZE=7)

    el = EL.edgeLoss(prediction, target)

    # loss += Focal Loss
    # loss += Dice Loss
    loss += el

    return loss
