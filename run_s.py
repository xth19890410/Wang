# -*- coding: utf-8 -*-
# @Time : 2024/6/1
# @Author : Wang Wei Jun

from __future__ import division
from __future__ import print_function
from sklearn.metrics import f1_score
import random
import argparse
import scipy.sparse as sp
import numpy as np
import hashlib
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import accuracy,get_balls
from utils import load_citation
from model import *
from utils import visual1
from dataLoader import dataloader
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
# 计算两点距离
from matplotlib.widgets import RectangleSelector
from sklearn.cluster import k_means
from model import MAUGCN
from time import perf_counter, time
from LoderData import dataprocess
from  Kmeans import kmeans_adj
from utils_1 import load_data,load_adjs
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--layer', type=int, default=2, help='Number of layers.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd3', type=float, default=0.005, help='weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128, help='hidden dimensions.')
parser.add_argument('--hidden1', type=int, default=32, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--lamda1',nargs='+', type=float, default=[0.8], help='weight for the attention.')
# parser.add_argument('--lamda1', type=float, default=0.5,help='weight')
# parser.add_argument('--lamda1',nargs='+', type=float, default=[0.01,0.6,0.7,0.9,1], help='weight for the attention.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=True, help='evaluation on test set.')
parser.add_argument('--wd11', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd22', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--nb_heads', type=int, default=3, help='Number of head attentions.')
parser.add_argument('--alpha1', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--batch_size2', type=int, default=10)
parser.add_argument('--batchsize', type=int, default=10, help='batchsize for train')
parser.add_argument('--test_gap', type=int, default=10,help='the train epochs between two test')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument("--dataset-name", nargs="?", default="ALOI")
parser.add_argument("--k", type=int, default=15, help="k of KNN graph.")
parser.add_argument("--beta", type=float, default=1.0, help="beta. Default is 1.0")
parser.add_argument("--rho", type=float, default=0.05, help="rho. Default is 0.05")
parser.add_argument("--ratio", type=float, default=0.1, help="Ratio of labeled samples")
parser.add_argument("--sf_seed", type=int, default=2042, help="Random seed for train-test split. Default is 42.")
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
label = 5

def draw_point(data):
    N = data.shape[0]
    plt.figure()
    plt.axis()
    for i in range(N):
        plt.scatter(data[i][0],data[i][1],s=16.,c='k')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('origin graph')
    plt.show()

# 判断粒球的标签和纯度
def get_num(gb):
    # 矩阵的行数
    num = gb.shape[0]
    return num

# 返回粒球中心和半径
def calculate_center_and_radius(gb):
    data_no_label = gb[:,:]#取坐标
    center = data_no_label.mean(axis=0)#压缩行，对列取均值  取出平均的 x,y
    data_no_label = data_no_label.numpy()
    center = center.numpy()
    radius = np.max((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))  #（x1-x1）**2 + (y1-y2)**2   所有点到中心的距离平均
    return center, radius

def gb_plot(gb_list, plt_type=0):
    plt.figure()
    plt.axis()
    for gb in gb_list:
        center, radius = calculate_center_and_radius(gb)  # 返回中心和半径
        if plt_type == 0:  # 绘制所有点
            plt.plot(gb[:, 0], gb[:, 1], '.', c='k', markersize=5)
        if plt_type == 0 or plt_type == 1:  # 绘制粒球
            theta = np.arange(0, 2 * np.pi, 0.01)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            plt.plot(x, y, c='r', linewidth=0.8)
        plt.plot(center[0], center[1], 'x' if plt_type == 0 else '.', color='r')  # 绘制粒球中心
    plt.show()


def splits(gb_list, num, splitting_method):
    gb_list_new = []
    for gb in gb_list:
        p = get_num(gb)
        if p < num:
            gb_list_new.append(gb)#该粒球包含的点数小于等于num，那
        else:
            gb_list_new.extend(splits_ball(gb, splitting_method))#反之，进行划分，本来是[[1],[2],[3]]  变成[...,[1],[2],[3]]
    return gb_list_new

def splits_ball(gb, splitting_method):
    splits_k = 2
    ball_list = []

    # 数组去重
    len_no_label = np.unique(gb, axis=0)
    if splitting_method == '2-means':
        if len_no_label.shape[0] < splits_k:
            splits_k = len_no_label.shape[0]
        # n_init:用不同聚类中心初始化运行算法的次数
        #random_state，通过固定它的值，每次可以分割得到同样的训练集和测试集
        label = k_means(X=gb, n_clusters=splits_k, n_init=1, random_state=8)[1]  # 返回标签
    elif splitting_method == 'center_split':
        # 采用正、负类中心直接划分
        p_left = gb[gb[:, 0] == 1, 1:].mean(0)#求坐标平均值
        p_right = gb[gb[:, 0] == 0, 1:].mean(0)
        distances_to_p_left = distances(gb, p_left)#求出各点到平均点的距离
        distances_to_p_right = distances(gb, p_right)

        relative_distances = distances_to_p_left - distances_to_p_right
        label = np.array(list(map(lambda x: 0 if x <= 0 else 1, relative_distances)))

    elif splitting_method == 'center_means':
        # 采用正负类中心作为 2-means 的初始中心点
        p_left = gb[gb[:, 0] == 1, 1:].mean(0)
        p_right = gb[gb[:, 0] == 0, 1:].mean(0)
        centers = np.vstack([p_left, p_right])#[[],[]]
        label = k_means(X=gb, n_clusters=2, init=centers, n_init=10)[1]#以centers为中心进行聚类
    else:
        return gb
    for single_label in range(0, splits_k):
        ball_list.append(gb[label == single_label, :])#按照新打的标签分类
    return ball_list


# 距离
def distances(data, p):
    return ((data - p) ** 2).sum(axis=1) ** 0.5


#计算所有点到粒球中心的平均距离：
def get_ball_quality(gb, center):
    N = gb.shape[0]
    ball_quality =  N
    gb = gb.numpy()
    mean_r = np.mean(((gb - center) **2)**0.5)
    return ball_quality, mean_r


#计算粒球的密度---计算密度的方法二：粒球的密度=粒球的质量/粒球的体积
#粒球的质量=所有点到中心点的平均距离  体积=粒球半径的维数次方radiusA, dimen, ball_qualitysA
def ball_density2(radiusAD, ball_qualitysA, mean_rs):
    N = radiusAD.shape[0]
    ball_dens2 = np.zeros(shape=N)
    for i in range(N):
        if radiusAD[i] == 0:
            ball_dens2[i] = 0
        else:
            ball_dens2[i] = ball_qualitysA[i] / (radiusAD[i] * radiusAD[i] * mean_rs[i])
    return ball_dens2


#计算粒球的相对距离
def ball_distance(centersAD):
    Y1 = pdist(centersAD)
    ball_distAD = squareform(Y1)
    return ball_distAD

#计算最小密度峰距离以及该点ball_min_dist3
def ball_min_dist(ball_distS, ball_densS):
    N3 = ball_distS.shape[0]
    ball_min_distAD = np.zeros(shape=N3)
    ball_nearestAD = np.zeros(shape=N3)
    #按密度从大到小排号
    index_ball_dens = np.argsort(-ball_densS)
    for i3, index in enumerate(index_ball_dens):
        if i3 == 0:
            continue
        index_ball_higher_dens = index_ball_dens[:i3]
        ball_min_distAD[index] = np.min([ball_distS[index, j]for j in index_ball_higher_dens])
        ball_index_near = np.argmin([ball_distS[index, j]for j in index_ball_higher_dens])
        ball_nearestAD[index] = int(index_ball_higher_dens[ball_index_near])
    ball_min_distAD[index_ball_dens[0]] = np.max(ball_min_distAD)
    if np.max(ball_min_distAD) < 1:
        ball_min_distAD = ball_min_distAD * 10
    return ball_min_distAD, ball_nearestAD

#画图
def ball_draw_decision(ball_densS, ball_min_distS):
    # Bval1_start = time.time()
    fig, ax = plt.subplots()
    N = ball_densS.shape[0]
    lst = []
    for i4 in range(N):
        ax.plot(ball_densS[i4], ball_min_distS[i4], marker='o', markersize=4.0, c='k')
        plt.xlabel('density')
        plt.ylabel('min_dist')
        ax.set_title('decision graph')
        # 矩形选区选择时的回调函数
    def select_callback(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        lst.append([x1, y1])

    RS = RectangleSelector(ax, select_callback,
                           drawtype='box', useblit=True,
                           button=[1, 3],  # disable middle button
                           minspanx=0, minspany=0,
                           spancoords='data',
                           interactive=True)
    # a = Annotate()
    plt.show()
    # Bval1_end = time.time()
    # Bval1 = Bval1_end - Bval1_start
    return lst


#找粒球中心点
def ball_find_centers(ball_densS, ball_min_distS, lst):
    ball_density_threshold = lst[0][0]
    ball_min_dist_threshold = lst[0][1]
    centers = []
    N4 = ball_densS.shape[0]
    for i4 in range(N4):
        if ball_densS[i4] >= ball_density_threshold and ball_min_distS[i4] >= ball_min_dist_threshold:
            centers.append(i4)
    return np.array(centers)


def ball_cluster(ball_densS, ball_centers, ball_nearest, ball_min_distS):
    K1 = len(ball_centers)
    if K1 == 0:
        print('no centers')
        return
    N5 = ball_densS.shape[0]
    ball_labs = -1 * np.ones(N5).astype(int)
    for i5, cen1 in enumerate(ball_centers):
        ball_labs[cen1] = int(i5+1)
    ball_index_density = np.argsort(-ball_densS)
    for i5, index2 in enumerate(ball_index_density):
        if ball_labs[index2] == -1:
            ball_labs[index2] = ball_labs[int(ball_nearest[index2])]
    return ball_labs

def  ball_draw_cluster(centersA, radiusA, ball_labs, dic_colors, gb_list, ball_centers):
    plt.figure()
    N6 = centersA.shape[0]
    for i6 in range(N6):
        for j6, point in enumerate(gb_list[i6]):
            plt.plot(point[0], point[1], marker='o', markersize=4.0, color=dic_colors[ball_labs[i6]])
    plt.show()
# features, gnd, p_labeled, p_unlabeled, adjs, adj_hats = dataprocess(args, args.dataset_name, args.k)

def construct_adjacency_hat(adj):
    """
    :param adj: original adjacency matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))  # <class 'numpy.ndarray'> (n_samples, 1)
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).toarray()
    return adj_normalized
def tensor_to_index(tensor):
    # 使用hash函数将张量转化为唯一的索引
    return int(hashlib.md5(tensor.numpy().tobytes()).hexdigest(), 16)
################################GB-DP#######################################
############################################################################

# features, gnd, p_labeled, p_unlabeled, adjs, adj_hats = dataloader(args.dataset_name, args.k, args.ratio)
features, gnd, p_labeled, p_unlabeled, adjs = dataprocess(args, args.dataset_name, args.k)

# print("ok")
llun = 1


features_sum = []
nfeat_sum =[]
adjtensor=[]
b = features.shape[1]
for jj in range(features.shape[1]):
    features_g = features[0][jj]
    features_g = torch.FloatTensor(features_g).float()
    nfeat_a = features_g.shape[1]
    # features_g = features_g.to(device)
    nfeat_sum.append(nfeat_a )
    features_sum.append(features_g)
##############################排序#####################################3
# 此处特征由大到小，由小到大排序则删除reverse=True
sorted_indices = sorted(range(len(nfeat_sum)), key=lambda k: nfeat_sum[k], reverse=True)
# sorted_indices = sorted(range(len(nfeat_sum)), key=lambda k: nfeat_sum[k])
features_sum_sorted = [features_sum[i] for i in sorted_indices]
nfeat_sum_sorted = [nfeat_sum[i] for i in sorted_indices]

# 更新 features_sum 和 nfeat_sum
features_sum = features_sum_sorted
nfeat_sum = nfeat_sum_sorted
##########################################################################
adj_nor = []


#######################################GB-DP###########################
for dd in range(len(features_sum)):
    feature_to_index = {}
    index_counter = 0
    data = features_sum[dd]
    num = np.ceil(np.sqrt(data.shape[0]))
    # num = num/2
    # print(max_radius)
    gb_list = [data]

    while True:
        ball_number_1 = len(gb_list)  # 点数
        gb_list = splits(gb_list, num=num, splitting_method='2-means')
        ball_number_2 = len(gb_list)  # 被划分成了几个
        # gb_plot(gb_list)
        if ball_number_1 == ball_number_2:  # 没有划分出新的粒球
            break

    centers = []
    radiuss = []
    ball_num = []  # 粒球里面的元素个数
    ball_qualitys = []  # 每个粒球的质量
    mean_rs = []
    i = 0
    for gb in gb_list:
        center, radius = calculate_center_and_radius(gb)
        ball_quality, mean_r = get_ball_quality(gb, center)
        ball_qualitys.append(ball_quality)
        mean_rs.append(mean_r)
        centers.append(center)
        radiuss.append(radius)
        ball_num.append(gb.shape[0])
    centersA = np.array(centers)
    radiusA = np.array(radiuss)
    ball_numA = np.array(ball_num)
    ball_qualitysA = np.array(ball_qualitys)  # 每一个粒球的半径和中心

    ball_densS = ball_density2(radiusA, ball_qualitysA, mean_rs)

    # 计算每个粒球的相对距离
    ball_distS = ball_distance(centersA)
    # 计算最小密度峰距离以及该点ball_min_dist  ball_min_distAD, ball_nearestAD
    ball_min_distS, ball_nearest = ball_min_dist(ball_distS, ball_densS)

    features_all = np.vstack(gb_list)
    current_index = 0
    indices = []
    for ball in gb_list:
        num_nodes = ball.shape[0]  # 当前粒球的节点数
        indices.extend(range(current_index, current_index + num_nodes))  # 为每个节点分配索引
        current_index += num_nodes # 为每个粒球的节点分配一个索引值
    # 将索引列转换为 numpy 数组
    indices = np.array(indices)

    # 3. 将索引列添加到特征矩阵的最后一列
    # 使用 np.column_stack 将特征矩阵与索引列合并
    final_matrix = np.column_stack((features_all, indices))
    # 初始化邻接矩阵大小
    num_total_points = data.shape[0]  # 所有粒球中所有点的总数
    adj_matrix = np.zeros((num_total_points, num_total_points))  # 创建一个空的邻接矩阵

    # 为每个粒球中的点添加边，并更新邻接矩阵
    for cc in range(len(gb_list)):
        if ball_nearest[cc] == 0:
            continue
        point1_gb = gb_list[cc]
        point2_gb =  gb_list[int(ball_nearest[cc])]
        for point1 in point1_gb:
            for point2 in point2_gb:
                point1_np = point1.numpy() if hasattr(point1, 'numpy') else point1
                point2_np = point2.numpy() if hasattr(point2, 'numpy') else point2

                # exists = np.any(np.all(final_matrix[:, :-1] == point1_np, axis=1))
                #
                # if exists:
                #     print("point1_np 存在于 data_with_indices 中")
                # else:
                #     print("point1_np 不存在于 data_with_indices 中")
                # 获取与 point1 特征匹配的行索引
                idx1 = np.where(np.all(final_matrix[:, :-1] == point1_np, axis=1))[0]
                # 获取与 point2 特征匹配的行索引
                idx2 = np.where(np.all(final_matrix[:, :-1] == point2_np, axis=1))[0]
                adj_matrix[idx1, idx2] = 1
                adj_matrix[idx2, idx1] = 1
    adj_matrix = construct_adjacency_hat(adj_matrix)
    adj_nor.append(adj_matrix)

###################################GB-DP####################################################

    # # Bval1选中中心所花的时间
    # lst = ball_draw_decision(ball_densS, ball_min_distS)
    # ball_centers = ball_find_centers(ball_densS, ball_min_distS, lst)
    #
    # ball_labs = ball_cluster(ball_densS, ball_centers, ball_nearest, ball_min_distS)

    # print('Please wait for drawing clustering results......')
    # # 最后的聚类结果
    # ball_draw_cluster(centersA, radiusA, ball_labs, gb_list, ball_centers)
    # print('Complete!')
######################################GB#############################

features_sum1 = []
for ki in range(len(features_sum)):
    features_f = features_sum[ki]
    features_f = torch.FloatTensor(features_f).float()
    features_f = features_f.to(device)
    features_sum1.append(features_f)

for j in range(len(adj_nor)):
    adja = adj_nor[j].astype(np.float32)
    adja = torch.tensor(adja, dtype=torch.float32)
    # adja = adj_nor[j].float()
    adja = adja.to(device)
    adjtensor.append(adja)
#
gnd = torch.from_numpy(gnd).long().to(args.device)
# for lamda1 in args.lamda1:
#     model = MAUGCN(nfeat=nfeat_sum,
#                    nlayers=args.layer,
#                    nhidden=args.hidden,
#                    nhidden1=args.hidden1,
#                    nclass=int(gnd.max()) + 1,
#                    dropout=args.dropout,
#                    lamda=args.lamda,
#                    lamda1=lamda1,
#                    alpha=args.alpha,
#                    variant=args.variant,
#                    ).to(device)
# # optimizer = optim.Adam(model.parameters(),#  优化器Adam
# #                        lr=args.lr, weight_decay=args.wd11)
#     optimizer = optim.Adam([
#         {'params': model.params1, 'weight_decay': args.wd1},
#         {'params': model.params2, 'weight_decay': args.wd2},
#         # {'params': model.params3, 'weight_decay': args.wd3}
#     ], lr=args.lr)


def train():
    model.train()
    optimizer.zero_grad()
    output,outputs,outputts,w  = model(features_sum1, adjtensor)
    acc_train = accuracy(output[p_labeled], gnd[p_labeled].to(device))
    loss_train = F.nll_loss(output[p_labeled],gnd[p_labeled].to(device))
    loss_train.backward()
    # loss_train3_sum.backward()
    optimizer.step()

    return loss_train.item(), acc_train.item()


def test():
    model.eval()
    with torch.no_grad():
        output,outputs,outputts,w = model(features_sum1, adjtensor)
        label_pre = []
        for idx in p_unlabeled:
            label_pre.append(torch.argmax(output[idx]).item())
        label_true = gnd[p_unlabeled].data.cpu()
        macro_f1 = f1_score(label_true, label_pre, average='macro')
        loss_test = F.nll_loss(output[p_unlabeled], gnd[p_unlabeled].to(device))  # .to(device)
        acc_test = accuracy(output[p_unlabeled], gnd[p_unlabeled].to(device))  # .to(device)

        # visual1(output, gnd, epoch, lamda, args.ratio,args.dataset_name)
        return loss_test.item(), acc_test.item(), macro_f1, output,w

acc_test_value = []
f1_test_value = []
loss_train_value = []
lamda_outputs = []
for lamda in args.lamda1:
    output_values = []
    for j in range(llun):
        accGCN = np.zeros((1, llun))
        timesGCN = np.zeros((1, llun))
        time = perf_counter()
        for epoch in range(args.epochs):
            loss_train, acc_train= train()
            loss_test, acc_test, f1 ,output,w= test()
            loss_train_value.append(loss_train)
            acc_test_value.append(acc_test)
            f1_test_value.append(f1)
            print(f"Epoch {epoch} - acc_train(GOC_GCN): {round(acc_train * 100, 1)}%, loss_train(GOC_GCN): {loss_train}, acc_test(GOC_GCN): {acc_test}, F1_test(GOC_GCN): {f1}, w:{w}")
        timesGCN[0, j] = perf_counter() - time
        lose_test, acc_test, macro_f1, output,w= test()
        # visual1(output, gnd, epoch, lamda, args.ratio, args.dataset_name)
        accGCN[0, j] = acc_test
        print("acc_test(MAUGCN):", round(acc_test * 100, 1), "%")
        print(f" Output: {acc_test},f1_score:{macro_f1},w:{w}")
    #     output_values.append((lamda, acc_test,macro_f1))
    # lamda_outputs.extend(output_values)

#########################loss###################
# fig, ax1 = plt.subplots(figsize=(10, 6))
# ax1.plot(range(args.epochs), loss_train_value, label="Loss Train", color='red')
# ax1.set_xlabel("The number of epochs",fontsize = '14')
# ax1.set_ylabel("Loss", color='black',fontsize = '14')
# ax1.tick_params(axis='y', labelcolor='blue')
# ax1.set_ylim(0, 2.5)
# ax1.set_xlim(-10, 210)
# # plt.legend(loc='best', bbox_to_anchor=(0.5,-0.15), fancybox=True, shadow=True, ncol=6, frameon=False, fontsize="10 ")
# ax2 = ax1.twinx()
# ax2.plot(range(args.epochs), acc_test_value, label="Acc Test", color='green')
# ax2.plot(range(args.epochs), f1_test_value, label="Acc Test", color='blue')
# ax2.set_ylabel("Accuracy and F1-score", color='black',fontsize = '14')
# ax2.set_ylim(0, 1)  # 设置右侧纵轴范围为 0 到 1
# ax2.tick_params(axis='y', labelcolor='black')
#   # 显示上面的label
# # 显示网格
# # plt.grid(True)
# # plt.legend(loc='best', bbox_to_anchor=(0.5,-0.15), fancybox=True, shadow=True, ncol=6, frameon=False, fontsize="10 ")
# plt.savefig('./result/loss/'+args.dataset_name+'_'+str(epoch)+'.png', dpi=1000)
# # 显示图像
# plt.show()
########################################################################################

np.save('./result/maugcn'  + args.dataset_name +'.npy', accGCN)
np.save('./result/maugcn'  + args.dataset_name +'.npy', timesGCN)
print(args.dataset_name)



