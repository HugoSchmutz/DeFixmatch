from __future__ import print_function, division
import os

import torch

from utils import net_builder
from datasets.ssl_dataset import SSL_Dataset
from datasets.data_utils import get_data_loader
from train_utils import ce_loss
import torch.nn.functional as F

def accuracy(output, target, topk=(1,5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        print(correct[:5].shape)
        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def accuracy_per_class(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        res = [output.cpu().max(1)[1].eq(target)[target==i].sum().numpy()/len(target[target==i]) for i in range(10)]
        
        return res

def compute_expected_calibration_error(logits, labels, num_bins: int = 15):
  """Calculates the Expected Calibration Error of a model.
  The input to this metric is the logits of a model, NOT the softmax scores.
  This divides the confidence outputs into equally-sized interval bins.
  In each bin, we compute the confidence gap:
  bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
  We then return a weighted average of the gaps, based on the number
  of samples in each bin
  See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
  "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
  2015. See https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
  Args:
      logits : logits of a model, NOT the softmax scores.
      labels : True labels. Same dimension as logits
      num_bins: Number of bins ending with a dot (but it's not too important).
  Returns:
      float: ece metric of the model
  """
  bin_boundaries = torch.linspace(0, 1, num_bins + 1)
  bin_lowers = bin_boundaries[:-1]
  bin_uppers = bin_boundaries[1:]

  softmaxes = F.softmax(logits, dim=1)
  confidences, predictions = torch.max(softmaxes, 1)
  accuracies = predictions.eq(labels)

  ece = torch.zeros(1, device=logits.device)
  for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
      # Calculated |confidence - accuracy| in each bin
      in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
      prop_in_bin = in_bin.float().mean()
      if prop_in_bin.item() > 0:
          accuracy_in_bin = accuracies[in_bin].float().mean()
          avg_confidence_in_bin = confidences[in_bin].mean()
          ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
  return ece

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, default='./saved_models/fixmatch/model_best.pth')
    parser.add_argument('--use_train_model', action='store_true')

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    Data Configurations
    '''
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_classes', type=int, default=10)
    args = parser.parse_args()
    
    checkpoint_path = os.path.join(args.load_path)
    checkpoint = torch.load(checkpoint_path)
    load_model = checkpoint['train_model'] if args.use_train_model else checkpoint['eval_model']
    
    _net_builder = net_builder(args.net, 
                               args.net_from_name,
                               {'depth': args.depth, 
                                'widen_factor': args.widen_factor,
                                'leaky_slope': args.leaky_slope,
                                'dropRate': args.dropout})
    
    net = _net_builder(num_classes=args.num_classes)
    net.load_state_dict(load_model)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    _eval_dset = SSL_Dataset(name=args.dataset, train=False, data_dir=args.data_dir)
    eval_dset = _eval_dset.get_dset()
    
    eval_loader = get_data_loader(eval_dset,
                                  args.batch_size, 
                                  num_workers=1)
 
    acc = 0.0
    loss = 0.0
    logits_list, labels_list = [], []

    with torch.no_grad():
        for image, target in eval_loader:
            image = image.type(torch.FloatTensor).cuda()
            logit = net(image)
            
            acc += logit.cpu().max(1)[1].eq(target).sum().numpy()
            loss += ce_loss(logit.cpu(), target, reduction='sum').numpy()
            logits_list.append(logit.cpu())
            labels_list.append(target)
        
        logits = torch.cat(logits_list).cpu()
        labels = torch.cat(labels_list).cpu()
        
        ece = compute_expected_calibration_error(logits, labels).cpu().item()
        top5_acc = accuracy(logits, labels, topk=(1,5))[1].detach().item()
        acc_per_class = accuracy_per_class(logits, labels)
    print(f"Test Accuracy: {acc/len(eval_dset)} \tLoss: {loss/len(eval_dset)} \tECE: {ece} \tTop5 Accuracy {top5_acc} \tACC/class: {acc_per_class}")