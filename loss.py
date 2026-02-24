from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import itertools as it
import random
import logging

import torch
from torch import nn
import torch.nn.functional as F


class CorrespondingLoss(nn.Module):

    def __init__(self, exp, batch_size=6, margin=2.0, temp=0.5):
        super(CorrespondingLoss, self).__init__()
        self.exp = exp
        self.margin = margin
        self.batch_size = batch_size
        self.temp = temp

    def forward(self, zero_output, one_output, labels):
        """
        exp：
        1：对比学习
        2：需要验证的对比学习方案
        """
        zero_output = zero_output.view(zero_output.size(0), -1)
        one_output = one_output.view(one_output.size(0), -1)
        
        batch_size, _ = zero_output.size()
        zero = torch.zeros_like(zero_output)
        one = torch.zeros_like(one_output)

        for index, label_number in enumerate(labels[:, 0]):
            if int(label_number) == 0:
                zero[index] = zero_output[index]
                one[index] = one_output[index]
            elif int(label_number) == 1:
                zero[index] = one_output[index]
                one[index] = zero_output[index]
        
        zero_output = zero
        one_output = one
        del zero, one

        if self.exp == 1:
            com_lis = min_combinations(zero_output.size(0))
            loss_contrastive = triple_con_loss(zero_output, one_output, com_lis, temp=int(self.temp))
            # loss_contrastive = all_loss(zero_output, one_output, temp=int(self.temp))

        elif self.exp == 2:
            com_lis = min_combinations(zero_output.size(0))
            cover1_abs = zero_output.norm(dim=1)
            stego1_abs = one_output.norm(dim=1)

            for i, number in enumerate(cover1_abs):
                if number == 0:
                    cover1_abs[i] = 1
            for i, number in enumerate(stego1_abs):
                if number == 0:
                    stego1_abs[i] = 1

            mask = torch.zeros(batch_size, batch_size).cuda()
            for i in com_lis:
                mask[i[0], i[1]] = 1
            posc_matrix = torch.einsum('ik,jk->ij', zero_output, zero_output) / torch.einsum('i,j->ij', cover1_abs, cover1_abs)
            posc_matrix = torch.mul(mask, posc_matrix)
            poss_matrix = torch.einsum('ik,jk->ij', one_output, one_output) / torch.einsum('i,j->ij', stego1_abs, stego1_abs)
            poss_matrix = torch.mul(mask, poss_matrix)
            sim_matrix = torch.einsum('ik,jk->ij', zero_output, one_output) / torch.einsum('i,j->ij', cover1_abs, stego1_abs)
            loss_contrastive = torch.clamp(-torch.mean(0.5*posc_matrix.sum(dim=1) + 0.5*poss_matrix.sum(dim=1) - sim_matrix[range(batch_size), range(batch_size)]) + self.temp, min=0)


        return loss_contrastive


def all_combinations(size):
    com = it.combinations(list(range(size)), 2)
    com_list = []
    for i in com:
        com_list.append(i)

    return com_list


def min_combinations(size):
    # 顺123456最小组合
    index_list = list(range(size))
    com_list = []
    for i in range(size - 1):
        del index_list[0]
        random_choice = random.sample(index_list, 1)
        com_list.append((i, random_choice[0]))

    return com_list


def new_combinations(size):
    pass


def all_loss(cover1, stego1, temp=2):
    batch_size, _ = cover1.size()
    cover1_abs = cover1.norm(dim=1)
    stego1_abs = stego1.norm(dim=1)

    for i, number in enumerate(cover1_abs):
        if number == 0:
            cover1_abs[i] = 1
    for i, number in enumerate(stego1_abs):
        if number == 0:
            stego1_abs[i] = 1

    posc_matrix = torch.einsum('ik,jk->ij', cover1, cover1) / torch.einsum('i,j->ij', cover1_abs, cover1_abs)
    poss_matrix = torch.einsum('ik,jk->ij', stego1, stego1) / torch.einsum('i,j->ij', stego1_abs, stego1_abs)
    sim_matrix = torch.einsum('ik,jk->ij', cover1, stego1) / torch.einsum('i,j->ij', cover1_abs, stego1_abs)
    loss_contrastive = torch.clamp(-torch.mean(0.5*posc_matrix.sum(dim=1) + 0.5*poss_matrix.sum(dim=1) - sim_matrix[range(batch_size), range(batch_size)]) + temp, min=0)


    return loss_contrastive
    
    
    



def triple_con_loss(cover1, stego1, com, temp=2, sim_all=1):
    batch_size, _ = cover1.size()
    cover1_abs = cover1.norm(dim=1)
    stego1_abs = stego1.norm(dim=1)
    for i, number in enumerate(cover1_abs):
        if number == 0:
            cover1_abs[i] = 1
    for i, number in enumerate(stego1_abs):
        if number == 0:
            stego1_abs[i] = 1

    # 分子
    mask = torch.zeros(batch_size, batch_size).cuda()
    for i in com:
        mask[i[0], i[1]] = 1
    # logging.info(mask)
    pos_matrix = torch.einsum('ik,jk->ij', cover1, cover1) / torch.einsum('i,j->ij', cover1_abs, cover1_abs)
    pos_matrix = torch.mul(mask, pos_matrix)
    pos_matrix = torch.exp(pos_matrix / temp)

    # 分母
    sim_matrix = torch.einsum('ik,jk->ij', cover1, stego1) / torch.einsum('i,j->ij', cover1_abs, stego1_abs)
    sim_matrix = torch.exp(sim_matrix / temp)

    if sim_all == 0:
        loss = pos_matrix.sum(dim=1) / sim_matrix[range(batch_size), range(batch_size)]  # [range(batch_size), range(batch_size)]
    else:
        loss = pos_matrix.sum(dim=1) / sim_matrix.sum(dim=1)  # [range(batch_size), range(batch_size)]
    loss = - torch.log(loss).mean()
    return loss


def four_con_loss(cover1, stego1, com, temp=2, sim_all=1):
    batch_size, _ = cover1.size()
    cover1_abs = cover1.norm(dim=1)
    stego1_abs = stego1.norm(dim=1)
    for i, number in enumerate(cover1_abs):
        if number == 0:
            cover1_abs[i] = 1
    for i, number in enumerate(stego1_abs):
        if number == 0:
            stego1_abs[i] = 1

    # 分子
    mask = torch.zeros(batch_size, batch_size).cuda()
    for i in com:
        mask[i[0], i[1]] = 1
    pos_matrix_cover = torch.einsum('ik,jk->ij', cover1, cover1) / torch.einsum('i,j->ij', cover1_abs, cover1_abs)
    pos_matrix_cover = torch.mul(mask, pos_matrix_cover)
    pos_matrix_cover = torch.exp(pos_matrix_cover / temp)

    pos_matrix_stego = torch.einsum('ik,jk->ij', stego1, stego1) / torch.einsum('i,j->ij', stego1_abs, stego1_abs)
    pos_matrix_stego = torch.mul(mask, pos_matrix_stego)
    pos_matrix_stego = torch.exp(pos_matrix_stego / temp)

    # 分母
    sim_matrix = torch.einsum('ik,jk->ij', cover1, stego1) / torch.einsum('i,j->ij', cover1_abs, stego1_abs)
    sim_matrix = torch.exp(sim_matrix / temp)

    if sim_all == 0:
        loss = (pos_matrix_cover.sum(dim=1) + pos_matrix_stego.sum(dim=1)) / sim_matrix[range(batch_size), range(batch_size)]
    else:
        loss = (pos_matrix_cover.sum(dim=1) + pos_matrix_stego.sum(dim=1)) / sim_matrix.sum(dim=1)
    loss = - torch.log(loss).mean()

    return loss


class CosineSimilarity(nn.Module):
    # cos(x1, x2)
    def forward(self, tensor_1, tensor_2):
        if tensor_1.ndim == 1:
            norm_tensor_1 = tensor_1.norm(dim=-1, keepdim=True)
            norm_tensor_2 = tensor_2.norm(dim=-1, keepdim=True)

            if norm_tensor_1[0] == 0:
                norm_tensor_1[0] = 1
            if norm_tensor_2[0] == 0:
                norm_tensor_2[0] = 1

            normalized_tensor_1 = tensor_1 / norm_tensor_1
            normalized_tensor_2 = tensor_2 / norm_tensor_2
            result = (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)

        elif tensor_1.ndim == 2:
            batch_size, _ = tensor_1.size()
            tensor1_abs = tensor_1.norm(dim=1)
            tensor2_abs = tensor_2.norm(dim=1)
            for i, number in enumerate(tensor1_abs):
                if number == 0:
                    tensor1_abs[i] = 1
            for i, number in enumerate(tensor2_abs):
                if number == 0:
                    tensor2_abs[i] = 1

            result = torch.einsum('ik,jk->ij', tensor_1, tensor_2) / torch.einsum('i,j->ij', tensor1_abs, tensor2_abs)
        else:
            assert ValueError("求cos时候的tensor维度有问题")

        return result


class DotProductSimilarity(nn.Module):
    # x1 * x2
    def __init__(self, scale_output=False):
        super(DotProductSimilarity, self).__init__()
        self.scale_output = scale_output
    def forward(self, tensor_1, tensor_2):
        result=(tensor_1 * tensor_2).sum(dim=-1)
        if(self.scale_output):
            result /= math.sqrt(tensor_1.size(-1))
        return result


class BiLinearSimilarity(nn.Module):
    # 双线性相似度，W和b是需要训练的
    # b=x1^T W x2 + b
    def __init__(self,tensor_1_dim,tensor_2_dim,activation=None):
        super(BiLinearSimilarity,self).__init__()
        self.weight_matrix=nn.Parameter(torch.Tensor(tensor_1_dim,tensor_2_dim))
        self.bias=nn.Parameter(torch.Tensor(1))
        self.activation=activation
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_matrix)
        self.bias.data.fill_(0)
    def forward(self, tensor_1,tensor_2):
        intermediate=torch.matmul(tensor_1,self.weight_matrix)
        result=(intermediate*tensor_2).sum(dim=-1)+self.bias
        if self.activation is not None:
            result=self.activation(result)
        return result


class PearsonCorrelation(nn.Module):
    # 皮尔逊相关系数
    def forward(self, tensor_1, tensor_2):
        if tensor_1.ndim == 1:
            vx = tensor_1 - torch.mean(tensor_1)
            vy = tensor_2 - torch.mean(tensor_2)
            cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        elif tensor_1.ndim == 2:
            vx = tensor_1 - torch.mean(tensor_1, dim=1)
            vy = tensor_2 - torch.mean(tensor_2, dim=1)
            cost = torch.einsum('ik,jk->ij', vx, vy) / torch.einsum('i,j->ij', torch.sqrt(torch.sum(vx ** 2, dim=1)), torch.sqrt(torch.sum(vy ** 2, dim=1)))

        return cost


