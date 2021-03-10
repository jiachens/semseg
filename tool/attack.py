'''
Description: 
Autor: Jiachen Sun
Date: 2021-03-10 13:48:38
LastEditors: Jiachen Sun
LastEditTime: 2021-03-10 16:59:38
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

def pgd_t(model, image, label, mean, std, target_mask, patch_init, patch_orig, step_size = 0.1, eps=10/255., iters=10, alpha = 1e-1, beta = 2., restarts=1, target_label=None, rap=False, init_tf_pts=None, patch_mask = None):
    
    images = image.cuda()
    t_labels = torch.ones_like(label)
    labels = t_labels.cuda(async=True)
    patches = patch_init.cuda()

    u_labels = label.cuda(async=True)
    u_labels = torch.autograd.Variable(u_labels)

    target_mask = torch.from_numpy(target_mask).cuda()

    # mean = torch.from_numpy(NORM_MEAN).float().cuda().unsqueeze(0)
    # mean = mean[..., None, None]
    # std = torch.from_numpy(NORM_STD).float().cuda().unsqueeze(0)
    # std = std[..., None, None]

    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)

    # loss = nn.CrossEntropyLoss()
    loss = nn.NLLLoss2d(ignore_index=255)

    tv_loss = nn.TVLoss()

    h_loss = nn.houdini_loss()

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

    ori_patches = patch_orig.data

    best_adv_patches = [torch.zeros_like(patches),-1e8]

    for j in range(restarts):
        delta = torch.rand_like(patches, requires_grad=True)
        delta = torch.zeros_like(patches, requires_grad=True)
        # delta.data = (delta.data * 2 * eps - eps) * perturb_mask

        for i in range(iters) :

            start = time.time()
            step_size  = np.max([1e-3, step_size * 0.99])
            images.requires_grad = False
            patches.requires_grad = False
            delta.requires_grad = True
            patch_mask_var.requires_grad = False

            t_patch: torch.tensor = kornia.warp_perspective((patches+delta).float(), M, dsize=(h, w))

            adv_images = (torch.clamp(t_patch*t_patch_mask_var+(1-t_patch_mask_var)*(images*std+mean),min=0, max=1)- mean)/std

            outputs = model(adv_images)[0]

            model.zero_grad()
        
            # remove attack
            # cost = - loss(outputs*target_mask*upper_mask, labels*2*target_mask*upper_mask) - alpha * loss(outputs*perturb_mask[:,0,:,:], u_labels*perturb_mask[:,0,:,:])

            # rap attack
            if rap:
                if target_label != None:
                    # target attack
                    obj_loss_value = - loss(outputs*target_mask, labels*target_label*target_mask)
                    tv_loss_value = tv_loss(ori_patches + delta)
                    cost = alpha * obj_loss_value + (1-alpha) * tv_loss_value
                else:
                    # untargeted attack
                    obj_loss_value = loss(outputs*target_mask, u_labels*target_mask)
                    tv_loss_value = tv_loss(ori_patches + delta)
                    cost = alpha * obj_loss_value + (1-alpha) * tv_loss_value

            cost.backward()
            print(i,cost.data, obj_loss_value.data, tv_loss_value.data)

            adv_patches = patches + delta + step_size*eps*delta.grad.sign()
            eta = torch.clamp(adv_patches - ori_patches, min=-eps, max=eps)
            delta = torch.clamp(ori_patches + eta, min=0, max=1).detach_() - ori_patches

            if cost.cpu().data.numpy() > best_adv_patches[1]:
                best_adv_patches = [delta.data, cost.cpu().data.numpy()]

    t_patch: torch.tensor = kornia.warp_perspective((ori_patches+best_adv_patches[0]).float(), M, dsize=(h, w))

    adv_images = (torch.clamp(t_patch*t_patch_mask_var+(1-t_patch_mask_var)*(images*std+mean),min=0, max=1)- mean)/std

    return adv_images, best_adv_patches[0]+ori_patches, t_patch_mask_var.cpu().data.numpy()