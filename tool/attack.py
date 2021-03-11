'''
Description: 
Autor: Jiachen Sun
Date: 2021-03-10 13:48:38
LastEditors: Jiachen Sun
LastEditTime: 2021-03-10 21:37:36
'''
import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
# from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import cv2
import kornia

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

def pgd_t(model, image, label, mean, std, target_mask, patch_init, patch_orig, step_size = 1, eps=10, iters=10, alpha = 1, beta = 2., restarts=1, target_label=None, rap=False, init_tf_pts=None, patch_mask = None):
    
    images = image.cuda()

    t_labels = torch.ones_like(label)
    labels = t_labels.cuda(async=True)
    patches = patch_init.cuda()

    u_labels = label.cuda(async=True)
    u_labels = torch.autograd.Variable(u_labels)

    target_mask = torch.from_numpy(target_mask).cuda()

    mean = torch.FloatTensor(mean).cuda().unsqueeze(0)
    mean = mean[..., None, None]
    std = torch.FloatTensor(std).cuda().unsqueeze(0)
    std = std[..., None, None]

    # loss = nn.CrossEntropyLoss()
    loss = nn.NLLLoss2d(ignore_index=255)

    tv_loss = TVLoss()

    best_adv_img = [torch.zeros_like(images.data), -1e8]

    # init transformation matrix
    h, w = images.shape[-2:]  # destination size
    points_src = torch.FloatTensor(init_tf_pts[0]).unsqueeze(0)
    # the destination points are the image vertexes
    points_dst = torch.FloatTensor(init_tf_pts[1]).unsqueeze(0)
    M: torch.tensor = kornia.get_perspective_transform(points_dst, points_src).cuda()

    if patch_mask is None:
        patch_mask_var = torch.ones_like(patches)
    else:
        patch_mask_var = patch_mask
    t_patch_mask_var = kornia.warp_perspective(patch_mask_var.float(), M, dsize=(h, w))

    # print(t_patch_mask_var)
    print(t_patch_mask_var.clone().cpu().data.numpy().shape)
    # cv2.imwrite('mask.png', np.int8(t_patch_mask_var.clone().cpu().data.numpy()*255))

    ori_patches = patch_orig.data

    best_adv_patches = [torch.zeros_like(patches),-1e8]

    for j in range(restarts):
        # delta = torch.rand_like(patches, requires_grad=True)
        delta = torch.zeros_like(patches, requires_grad=True)
        # delta.data = (delta.data * 2 * eps - eps) * perturb_mask

        for i in range(iters) :

            step_size  = np.max([1e-3, step_size * 0.99])
            images.requires_grad = False
            patches.requires_grad = False
            delta.requires_grad = True
            patch_mask_var.requires_grad = False

            t_patch: torch.tensor = kornia.warp_perspective((patches+delta).float(), M, dsize=(h, w))

            adv_images = (torch.clamp(t_patch*t_patch_mask_var+(1-t_patch_mask_var)*(images),min=0, max=255) - mean) / std

            outputs = model(adv_images)[0]

            model.zero_grad()
        
            # remove attack
            # cost = - loss(outputs*target_mask*upper_mask, labels*2*target_mask*upper_mask) - alpha * loss(outputs*perturb_mask[:,0,:,:], u_labels*perturb_mask[:,0,:,:])

            # rap attack
            if rap:
                if target_label != None:
                    # target attack
                    # print(outputs.shape,labels.shape)
                    obj_loss_value = - loss(outputs.unsqueeze(0)*target_mask, labels.long()*target_label*target_mask)
                    tv_loss_value = tv_loss(ori_patches + delta)
                    cost = alpha * obj_loss_value + (1-alpha) * tv_loss_value
                else:
                    # untargeted attack
                    obj_loss_value = loss(outputs.unsqueeze(0)*target_mask, u_labels.long()*target_mask)
                    tv_loss_value = tv_loss(ori_patches + delta)
                    cost = alpha * obj_loss_value + (1-alpha) * tv_loss_value

            cost.backward()
            print(i,cost.data, obj_loss_value.data, tv_loss_value.data)

            adv_patches = patches + delta + step_size*eps*delta.grad.sign()
            eta = torch.clamp(adv_patches - ori_patches, min=-eps, max=eps)
            delta = torch.clamp(ori_patches + eta, min=0, max=255).detach_() - ori_patches

            if cost.cpu().data.numpy() > best_adv_patches[1]:
                best_adv_patches = [delta.data, cost.cpu().data.numpy()]

    t_patch: torch.tensor = kornia.warp_perspective((ori_patches+best_adv_patches[0]).float(), M, dsize=(h, w))

    cv2.imwrite('./test.png',np.uint8(torch.clamp(t_patch*t_patch_mask_var+(1-t_patch_mask_var)*(images),min=0, max=255).clone().squeeze(0).cpu().numpy().transpose((1,2,0))))

    adv_images = (torch.clamp(t_patch*t_patch_mask_var+(1-t_patch_mask_var)*(images),min=0, max=255)- mean)/std

    return adv_images, best_adv_patches[0]+ori_patches, t_patch_mask_var.cpu().data.numpy()