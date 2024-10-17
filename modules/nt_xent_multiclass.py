from .gather import GatherLayer
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils1


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, device, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.world_size = world_size
    
    def forward(self, z_i, z_j, dist_labels):
        
        N = 2 * z_i.shape[0] * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        dist_labels = torch.cat((dist_labels, dist_labels),dim=0)
        
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)
            dist_labels = torch.cat(GatherLayer.apply(dist_labels), dim=0)
        
        # calculate similarity and divide by temperature parameter
        z = nn.functional.normalize(z, p=2, dim=1)
        sim = torch.mm(z, z.T) / self.temperature
        dist_labels = dist_labels.cpu()
        
        positive_mask = torch.mm(dist_labels.to_sparse(), dist_labels.T)
        positive_mask = positive_mask.fill_diagonal_(0).to(sim.device)
        zero_diag = torch.ones((N, N)).fill_diagonal_(0).to(sim.device)
        
        # calculate normalized cross entropy value
        positive_sum = torch.sum(positive_mask, dim=1)
        denominator = torch.sum(torch.exp(sim)*zero_diag,dim=1)
        loss = torch.mean(torch.log(denominator) - \
                          (torch.sum(sim * positive_mask, dim=1)/positive_sum))
        
        return loss


class SIMCLRLoss(nn.Module):
    """
    This is the SimCLR loss in https://arxiv.org/abs/2002.05709
    The embedding vectors are assumed to have size (2 x batch_size, embedding_dim) and
    the memory layout that can be reshaped into shape (2, batch_size, embedding_dim).
    This memory layout is consistent with the SimCLR collator in
    https://github.com/facebookresearch/vissl/blob/master/vissl/data/collators/simclr_collator.py
    Config params:
        temperature (float): the temperature to be applied on the logits
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature
        self.labels = None
        self.masks = None
        self.last_local_batch_size = None

    def forward(self, z_i, z_j, dist_labels):
        q_a = z_i
        q_b = z_j

        q_a = F.normalize(q_a, dim=-1, p=2)
        q_b = F.normalize(q_b, dim=-1, p=2)

        local_batch_size = q_a.size(0)

        k_a, k_b = utils1.all_gather_batch_with_grad([q_a, q_b])

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils1.get_rank() + torch.arange(
                local_batch_size, device=q_a.device
            )
            total_batch_size = local_batch_size * utils1.get_world_size()
            self.masks = F.one_hot(self.labels, total_batch_size) * 1e9
            self.last_local_batch_size = local_batch_size

        logits_aa = torch.matmul(q_a, k_a.transpose(0, 1)) / self.tau
        logits_aa = logits_aa - self.masks
        logits_bb = torch.matmul(q_b, k_b.transpose(0, 1)) / self.tau
        logits_bb = logits_bb - self.masks
        logits_ab = torch.matmul(q_a, k_b.transpose(0, 1)) / self.tau
        logits_ba = torch.matmul(q_b, k_a.transpose(0, 1)) / self.tau

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), dist_labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), dist_labels)
        loss = (loss_a + loss_b) / 2  # divide by 2 to average over all samples

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(torch.cat([logits_ab, logits_aa], dim=1), dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {'loss': loss, 'ssl_loss': loss, 'ssl_acc': acc}

