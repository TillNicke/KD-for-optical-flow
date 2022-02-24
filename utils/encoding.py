import torch
from skimage.metrics import hausdorff_distance

def dice_coeff(outputs, labels, max_label):
    """
    estimating the dice similarity coefficient between predicted and reference labels

    outputs: predicted segmentation
    labels: reference seg
    max_label: int. maximum number of labels
    """
    dice = torch.FloatTensor(max_label-1).fill_(0)
    for label_num in range(1, max_label):
        iflat = (outputs==label_num).reshape(-1).float()
        tflat = (labels==label_num).reshape(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num-1] = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
    return dice

def labelMatrixOneHot(segmentation, label_num):
    """
    generate a oneHot representation of a segmenation
    """
    B, H, W = segmentation.size()
    values = segmentation.view(B,1,H,W).expand(B,label_num,H,W).to(segmentation.device)
    linspace = torch.linspace(0, label_num-1, label_num).long().view(1,label_num,1,1).expand(B,label_num,H,W).to(segmentation.device)
    matrix = (values.float()==linspace.float()).float().to(segmentation.device)
    for j in range(2,matrix.shape[1]):
        matrix[0,j,:,:] = matrix[0,j,:,:]
    return matrix


def hausdorff_dist(outputs, labels, max_label):
    """
    outputs: prediction segmentation [HxW]
    labels: ground truth segmentation [HxW]
    max_label: int; label of classes including Background 
    
    return: torch.Tensor [num_labels -1] containing the HD between the labels Excluding BG
    """
    dist = torch.FloatTensor(max_label-1).fill_(0)
    B,H,W = outputs.shape
    
    for label in range(1, max_label):
        out_flat = (outputs==label).view(B,H,W).float()
        gt_flat = (labels==label).view(B,H,W).float()
        hd_dist = hausdorff_distance(out_flat.numpy(), gt_flat.numpy())
        dist[label-1] = hd_dist
    
    return dist