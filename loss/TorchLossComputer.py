'''
  Adapted from here: https://github.com/ZitongYu/PhysFormer/TorchLossComputer.py
  Modifed based on the HR-CNN here: https://github.com/radimspetlik/hr-cnn
'''
import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import scipy.stats
import pdb
import torch.nn as nn
from post_process import calculate_hr , calculate_psd
import torch.fft as fft

EPSILON = 1e-10
BP_LOW=2/3
BP_HIGH=3.0
BP_DELTA=0.1

def normal_sampling(mean, label_k, std):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)

def kl_loss(inputs, labels):
    criterion = nn.KLDivLoss(reduce=False)
    outputs = torch.log(inputs)
    loss = criterion(outputs, labels)
    #loss = loss.sum()/loss.shape[0]
    loss = loss.sum()
    return loss

class MMDLoss(nn.Module):

    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)
        
        
        if total.dim() == 1:
            total = total.unsqueeze(1)  # if 1D, transfer to 2D

        total0 = total.unsqueeze(0).expand(
            total.size(0), total.size(0), total.size(1)
        )
        total1 = total.unsqueeze(1).expand(
            total.size(0), total.size(0), total.size(1)
        )
        
        L2_distance = ((total0 - total1) ** 2).sum(2)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss


class Neg_Pearson(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson,self).__init__()

    def forward(self, preds, labels):       # all variable operation
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])                # x
            sum_y = torch.sum(labels[i])               # y
            sum_xy = torch.sum(preds[i]*labels[i])        # xy
            sum_x2 = torch.sum(torch.pow(preds[i],2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i],2)) # y^2
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
            loss += 1 - pearson
            
        loss = loss/preds.shape[0]
        return loss
    
class Weak_Supervised(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Weak_Supervised,self).__init__()

    def forward(self, preds, labels):       # all variable operation
        loss = 0
        max_preds = torch.sum(torch.max(preds, dim=0))
        max_labels = torch.sum(torch.max(labels, dim=0))
        
        sum_xy = torch.sum(max_preds*max_labels) 
        sum_x2 = torch.sum(torch.pow(max_preds, 2)) 
        sum_y2 = torch.sum(torch.pow(max_labels,2))
        
        mean_preds, std_preds =  torch.std_mean(preds)
        mean_labels, std_labels =  torch.std(labels)
        
        
        sum_xy1 = torch.sum(std_preds*std_labels) 
        sum_x21 = torch.sum(torch.pow(std_preds, 2)) 
        sum_y21 = torch.sum(torch.pow(std_labels,2))
        
        
        
        N = preds.shape[1]
        pearson = (N*sum_xy - max_preds*max_labels)/(torch.sqrt((N*sum_x2 - torch.pow(max_preds,2))*(N*sum_y2 - torch.pow(max_labels,2))))
        loss += 1 - pearson
        
        pearson1 = (N*sum_xy1 - std_preds*std_labels)/(torch.sqrt((N*sum_x21 - torch.pow(std_preds,2))*(N*sum_y21 - torch.pow(std_labels,2))))
        loss += 1 - pearson1
    
        return loss   
     
def _IPR_SSL(freqs, psd, low_hz=BP_LOW, high_hz=BP_HIGH, device=None):
    use_freqs = torch.logical_and(freqs >= low_hz, freqs <= high_hz)
    zero_freqs = torch.logical_not(use_freqs)
    use_energy = torch.sum(psd[:,use_freqs], dim=1)
    zero_energy = torch.sum(psd[:,zero_freqs], dim=1)
    denom = use_energy + zero_energy + EPSILON
    ipr_loss = torch.mean(zero_energy / denom)
    return ipr_loss


def IPR_SSL(freqs, psd, speed=None, low_hz=BP_LOW, high_hz=BP_HIGH, device=None):
    if speed is None:
        ipr_loss = _IPR_SSL(freqs, psd, low_hz=low_hz, high_hz=high_hz, device=device)
    else:
        batch_size = psd.shape[0]
        ipr_losses = torch.ones((batch_size,1)).to(device)
        for b in range(batch_size):
            low_hz_b = low_hz * speed[b]
            high_hz_b = high_hz * speed[b]
            psd_b = psd[b].view(1,-1)
            ipr_losses[b] = _IPR_SSL(freqs, psd_b, low_hz=low_hz_b, high_hz=high_hz_b, device=device)
        ipr_loss = torch.mean(ipr_losses)
    return ipr_loss

def ideal_bandpass(freqs, psd, low_hz, high_hz):
    freq_idcs = torch.logical_and(freqs >= low_hz, freqs <= high_hz)
    freqs = freqs[freq_idcs]
    psd = psd[:,freq_idcs]
    return freqs, psd

def normalize_psd(psd):
    return psd / torch.sum(psd, keepdim=True, dim=1) ## treat as probabilities

def torch_power_spectral_density(x, nfft=5400, fps=90, low_hz=BP_LOW, high_hz=BP_HIGH, return_angle=False, radians=True, normalize=True, bandpass=True):
    centered = x - torch.mean(x, keepdim=True, dim=1)
    rfft_out = fft.rfft(centered, n=nfft, dim=1)
    psd = torch.abs(rfft_out)**2
    N = psd.shape[1]
    freqs = fft.rfftfreq(2*N-1, 1/fps)
    if return_angle:
        angle = torch.angle(rfft_out)
        if not radians:
            angle = torch.rad2deg(angle)
        if bandpass:
            freqs, psd, angle = ideal_bandpass(freqs, psd, low_hz, high_hz, angle=angle)
        if normalize:
            psd = normalize_psd(psd)
        return freqs, psd, angle
    else:
        if bandpass:
            freqs, psd = ideal_bandpass(freqs, psd, low_hz, high_hz)
        if normalize:
            psd = normalize_psd(psd)
        return freqs, psd  

# class Hybrid_Loss(nn.Module): 
#     def __init__(self):
#         super(Hybrid_Loss,self).__init__()
#         self.criterion_WSL = Neg_Pearson()

#     def forward(self, pred_ppg, labels, epoch, FS, diff_flag):    
#         loss_time = self.criterion_WSL(pred_ppg.view(1,-1) , labels.view(1,-1))    
#         # loss_Fre , _ = TorchLossComputer.Frequency_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1), diff_flag=diff_flag, Fs=FS, std=3.0) 
#         freqs, psd = torch_power_spectral_density(self, fps=90, normalize=False, bandpass=False)
#         loss_Fre_self = IPR_SSL(freqs, psd, low_hz=BP_LOW, high_hz=BP_HIGH, device=torch.cuda.device)
#         # if torch.isnan(loss_time) : 
#         #    loss_time = 0
#         loss = 1.0 *loss_Fre_self + 0.5 * loss_time
#         return loss
    
class Hybrid_Loss(nn.Module): 
    def __init__(self):
        super(Hybrid_Loss, self).__init__()
        self.criterion_WSL = Neg_Pearson()

    def forward(self, pred_ppg, labels, epoch, FS, diff_flag):    
        loss_time = self.criterion_WSL(pred_ppg.view(1, -1), labels.view(1, -1))  
        
        loss_CE , loss_distribution_kl = TorchLossComputer.Frequency_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1), diff_flag=diff_flag, Fs=FS, std=3.0)
        
        loss_hr = TorchLossComputer.HR_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1), diff_flag=diff_flag, Fs=FS, std=3.0)  
        
         
        pred_ppg = pred_ppg.squeeze(-1)  
        if pred_ppg.dim() == 1:  
            pred_ppg = pred_ppg.unsqueeze(0)  # [1, C]

        # frequency density
        freqs, psd = torch_power_spectral_density(pred_ppg, nfft=5400, fps=FS, low_hz=BP_LOW, high_hz=BP_HIGH, normalize=False, bandpass=False)
        
        if torch.isnan(loss_time) : 
           loss_time = 0
        # IPR_SSL 
        loss_Fre_self = IPR_SSL(freqs, psd, low_hz=BP_LOW, high_hz=BP_HIGH, device=torch.cuda.current_device())
        
        # alternative
        # if epoch > 2:
        #     loss = 0.5 * loss_hr + 0.5 * loss_time
        # else:  
        # loss = 1.0 * loss_hr + 0.5 * loss_time + 0.5 * loss_Fre_self  
        loss = 1.0 * loss_CE + 1.0 * loss_hr + 0.2*loss_time
        
        return loss
        
class RhythmFormer_Loss(nn.Module): 
    def __init__(self):
        super(RhythmFormer_Loss,self).__init__()
        self.criterion_Pearson = Neg_Pearson()
    def forward(self, pred_ppg, labels, epoch, FS, diff_flag):    
        loss_time = self.criterion_Pearson(pred_ppg.view(1,-1) , labels.view(1,-1))    
        loss_CE , loss_distribution_kl = TorchLossComputer.Frequency_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1), diff_flag=diff_flag, Fs=FS, std=3.0)
        loss_hr = TorchLossComputer.HR_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1), diff_flag=diff_flag, Fs=FS, std=3.0)
        if torch.isnan(loss_time) : 
           loss_time = 0
        loss = 0.2 * loss_time + 1.0 * loss_CE + 1.0 * loss_hr
        return loss

class PhysFormer_Loss(nn.Module): 
    def __init__(self):
        super(PhysFormer_Loss,self).__init__()
        self.criterion_Pearson = Neg_Pearson()

    def forward(self, pred_ppg, labels , epoch , FS , diff_flag):       
        loss_rPPG = self.criterion_Pearson(pred_ppg.view(1,-1) , labels.view(1,-1))
        loss_CE , loss_distribution_kl = TorchLossComputer.Frequency_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1) , diff_flag = diff_flag , Fs = FS, std=1.0)
        if torch.isnan(loss_rPPG) : 
           loss_rPPG = 0
        if epoch >30:
            a = 1.0
            b = 5.0
        else:
            a = 1.0
            b = 1.0*math.pow(5.0, epoch/30.0)

        loss = a * loss_rPPG + b * (loss_distribution_kl + loss_CE)
        return loss
    
class TorchLossComputer(object):
    @staticmethod
    def compute_complex_absolute_given_k(output, k, N):
        two_pi_n_over_N = Variable(2 * math.pi * torch.arange(0, N, dtype=torch.float), requires_grad=True) / N
        hanning = Variable(torch.from_numpy(np.hanning(N)).type(torch.FloatTensor), requires_grad=True).view(1, -1)

        k = k.type(torch.FloatTensor).cuda()
        two_pi_n_over_N = two_pi_n_over_N.cuda()
        hanning = hanning.cuda()
            
        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).type(torch.cuda.FloatTensor)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                           + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2

        return complex_absolute

    @staticmethod
    def complex_absolute(output, Fs, bpm_range=None):
        output = output.view(1, -1)

        N = output.size()[1]

        unit_per_hz = Fs / N
        feasible_bpm = bpm_range / 60.0
        k = feasible_bpm / unit_per_hz
        
        # only calculate feasible PSD range [0.7,4]Hz
        complex_absolute = TorchLossComputer.compute_complex_absolute_given_k(output, k, N)

        return (1.0 / complex_absolute.sum()) * complex_absolute	# Analogous Softmax operator
        
        
    @staticmethod
    def cross_entropy_power_spectrum_loss(inputs, target, Fs):
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        #pdb.set_trace()

        #return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)).view(1),  (target.item() - whole_max_idx.item()) ** 2
        return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)

    # @staticmethod
    # def cross_entropy_power_spectrum_focal_loss(inputs, target, Fs, gamma):
    #     inputs = inputs.view(1, -1)
    #     target = target.view(1, -1)
    #     bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
    #     #bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

    #     complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

    #     whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
    #     whole_max_idx = whole_max_idx.type(torch.float)
        
    #     #pdb.set_trace()
    #     criterion = FocalLoss(gamma=gamma)

    #     #return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)).view(1),  (target.item() - whole_max_idx.item()) ** 2
    #     return criterion(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)

        
    @staticmethod
    def cross_entropy_power_spectrum_forward_pred(inputs, Fs):
        inputs = inputs.view(1, -1)
        bpm_range = torch.arange(40, 190, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)

        return whole_max_idx
    
    @staticmethod
    def Frequency_loss(inputs, target, diff_flag , Fs, std):
        hr_pred, hr_gt = calculate_hr(inputs.detach().cpu(), target.detach().cpu() , diff_flag = diff_flag , fs=Fs)
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(45, 150, dtype=torch.float).to(torch.device('cuda'))
        ca = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)
        sa = ca/torch.sum(ca)

        target_distribution = [normal_sampling(int(hr_gt), i, std) for i in range(45, 150)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).to(torch.device('cuda'))

        hr_gt = torch.tensor(hr_gt-45).view(1).type(torch.long).to(torch.device('cuda'))
        return F.cross_entropy(ca, hr_gt) , kl_loss(sa , target_distribution)
    
    @staticmethod
    def HR_loss(inputs, target,  diff_flag , Fs, std):
        # inputs_cpu = inputs.cpu()
        # target_cpu = target.cpu()
        # pred_distribution = [normal_sampling(np.argmax(inputs_cpu.detach().numpy()), i, std) for i in range(inputs.shape[0])]
        # pred_distribution = [i if i > 1e-15 else 1e-15 for i in pred_distribution]
        # pred_distribution = torch.Tensor(pred_distribution).to(torch.device('cuda'))
        # target_distribution = [normal_sampling(np.argmax(target_cpu.detach().numpy()), i, std) for i in range(target.shape[0])]
        # target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        # target_distribution = torch.Tensor(target_distribution).to(torch.device('cuda'))
        # return kl_loss(pred_distribution , target_distribution)
        
        # psd_pred, psd_gt = calculate_psd(inputs.detach().cpu(), target.detach().cpu() , diff_flag = diff_flag , fs=Fs)
        # pred_distribution = [normal_sampling(np.argmax(psd_pred), i, std) for i in range(psd_pred.size)]
        # pred_distribution = [i if i > 1e-15 else 1e-15 for i in pred_distribution]
        # pred_distribution = torch.Tensor(pred_distribution).to(torch.device('cuda'))
        # target_distribution = [normal_sampling(np.argmax(psd_gt), i, std) for i in range(psd_gt.size)]
        # target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        # target_distribution = torch.Tensor(target_distribution).to(torch.device('cuda'))
        # loss = MMDLoss()
        
        # return kl_loss(pred_distribution , target_distribution)
        psd_pred, psd_gt = calculate_psd(inputs.detach().cpu(), target.detach().cpu(), diff_flag=diff_flag, fs=Fs)

        # 生成预测分布
        pred_distribution = []
        for i in range(psd_pred.size):
            sample = normal_sampling(np.argmax(psd_pred), i, std)  # generate more samples
            sample = max(sample, 1e-15)  # > 1e-15
            pred_distribution.append(sample)

        pred_distribution = torch.Tensor(pred_distribution).to(torch.device('cuda'))

        # 生成目标分布
        target_distribution = []
        for i in range(psd_gt.size):
            sample = normal_sampling(np.argmax(psd_gt), i, std)  
            sample = max(sample, 1e-15)  # > 1e-15
            target_distribution.append(sample)

        target_distribution = torch.Tensor(target_distribution).to(torch.device('cuda'))

        # MMDLoss
        loss_func = MMDLoss()

        # loss = scipy.stats.wasserstein_distance(pred_distribution.cpu().detach().numpy(), target_distribution.cpu().detach().numpy())
        loss = loss_func.forward(pred_distribution, target_distribution)

        return loss
        
    
    