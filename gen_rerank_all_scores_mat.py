import scipy.io
import torch
import numpy as np
import time
import argparse
import os
import math

parser = argparse.ArgumentParser(description='evaluate')
parser.add_argument('--name',default='ft_ResNet50_market_pcb_r', type=str, help='0,1,2,3...or last')
parser.add_argument('--alpha', default=5, type=float, help='alpha')
parser.add_argument('--smooth', default=50, type=float, help='smooth')
opt = parser.parse_args()
name = opt.name
alpha=opt.alpha
smooth=opt.smooth



#######################################################################
# Evaluate
def evaluate(qf,ql,qc,qfr,gf,gl,gc,gfr,distribution):
    query = qf
    score = np.dot(gf,query)

    # spatial temporal scores: qfr,gfr, qc, gc
    # TODO
    interval = 100
    score_st = np.zeros(len(gc))
    for i in range(len(gc)):
        if qfr>gfr[i]:
            diff = qfr-gfr[i]
            hist_ = int(diff/interval)
            pr = distribution[qc-1][gc[i]-1][hist_]
        else:
            diff = gfr[i]-qfr
            hist_ = int(diff/interval)
            pr = distribution[gc[i]-1][qc-1][hist_]
        score_st[i] = pr
    # ========================
    score  = 1/(1+np.exp(-alpha*score))*1/(1+2*np.exp(-alpha*score_st))
    ###############################################################################################

    index = np.argsort(-score)  #from large to small

    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp
def evaluate2(qf,ql,qc,qfr,gf,gl,gc,gfr,distribution):
    query = qf
    score = np.dot(gf,query)

    # spatial temporal scores: qfr,gfr, qc, gc
    # TODO
    interval = 100
    score_st = np.zeros(len(gc))
    for i in range(len(gc)):
        if qfr>gfr[i]:
            diff = qfr-gfr[i]
            hist_ = int(diff/interval)
            # print('debug:',qc-1,gc[i]-1,hist_)
            pr = distribution[qc-1][gc[i]-1][hist_]
        else:
            diff = gfr[i]-qfr
            hist_ = int(diff/interval)
            # print('debug:',qc-1,gc[i]-1,hist_)
            pr = distribution[gc[i]-1][qc-1][hist_]
        score_st[i] = pr
    # ========================
    score  = 1/(1+np.exp(-alpha*score))*1/(1+2*np.exp(-alpha*score_st))
    return score

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

def gaussian_func(x, u, o=50):
    if (o == 0):
        print("In gaussian, o shouldn't equel to zero")
        return 0
    temp1 = 1.0 / (o * math.sqrt(2 * math.pi))
    temp2 = -(math.pow(x - u, 2)) / (2 * math.pow(o, 2))
    return temp1 * math.exp(temp2)

def gaussian_func2(x, u, o=50):
    temp1 = 1.0 / (o * math.sqrt(2 * math.pi))
    temp2 = -(np.power(x - u, 2)) / (2 * np.power(o, 2))
    return temp1 * np.exp(temp2)


def gauss_smooth(arr):
    hist_num = len(arr)
    vect= np.zeros((hist_num,1))
    for i in range(hist_num):
        vect[i,0]=i
    # gaussian_vect= gaussian_func2(vect,0,1)
    gaussian_vect= gaussian_func2(vect,0,50)
    matrix = np.zeros((hist_num,hist_num))
    # matrix = np.eye(hist_num)
    for i in range(hist_num):
        for j in range(i,hist_num):
            matrix[i][j]=gaussian_vect[j-i]    
    matrix = matrix+matrix.transpose()
    for i in range(hist_num):
        matrix[i][i]=matrix[i][i]/2    
    xxx = np.dot(matrix,arr)
    return xxx

# faster gauss_smooth
def gauss_smooth2(arr,o):
    hist_num = len(arr)
    vect= np.zeros((hist_num,1))
    for i in range(hist_num):
        vect[i,0]=i
    approximate_delta = 3*o     #  when x-u>approximate_delta, e.g., 6*o, the gaussian value is approximately equal to 0.
    gaussian_vect= gaussian_func2(vect,0,o)
    matrix = np.zeros((hist_num,hist_num))
    for i in range(hist_num):
        k=0
        for j in range(i,hist_num):
            if k>approximate_delta:
                continue
            matrix[i][j]=gaussian_vect[j-i] 
            k=k+1  
    matrix = matrix+matrix.transpose()
    for i in range(hist_num):
        matrix[i][i]=matrix[i][i]/2   
    xxx = np.dot(matrix,arr)
    return xxx

######################################################################
result = scipy.io.loadmat('model/'+name+'/'+'pytorch_result.mat')
query_feature = result['query_f']
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
query_frames = result['query_frames'][0]


gallery_feature = result['gallery_f']
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]
gallery_frames = result['gallery_frames'][0]

query_feature=query_feature.transpose()/np.power(np.sum(np.power(query_feature,2),axis=1),0.5)
query_feature=query_feature.transpose()
print('query_feature:',query_feature.shape)
gallery_feature=gallery_feature.transpose()/np.power(np.sum(np.power(gallery_feature,2),axis=1),0.5)
gallery_feature=gallery_feature.transpose()
print('gallery_feature:',gallery_feature.shape)


#############################################################

result2 = scipy.io.loadmat('model/'+name+'/'+'pytorch_result2.mat')
distribution = result2['distribution']

#############################################################
for i in range(0,8):
    for j in range(0,8):
        print("gauss "+str(i)+"->"+str(j))
        # gauss_smooth(distribution[i][j])
        distribution[i][j][:]=gauss_smooth2(distribution[i][j][:],smooth)


eps = 0.0000001
sum_ = np.sum(distribution,axis=2)
for i in range(8):
    for j in range(8):
        distribution[i][j][:]=distribution[i][j][:]/(sum_[i][j]+eps)        
#############################################################

all_features = np.concatenate([query_feature,gallery_feature],axis=0)
all_labels = np.concatenate([query_label,gallery_label],axis=0)
all_cams = np.concatenate([query_cam,gallery_cam],axis=0)
all_frames = np.concatenate([query_frames,gallery_frames],axis=0)

all_scores = np.zeros((len(all_labels),len(all_labels)))

print('all_features shape:',all_features.shape)
print('all_labels shape:',all_labels.shape)
print('all_cams shape:',all_cams.shape)
print('all_frames shape:',all_frames.shape)
print('all_scores shape:',all_scores.shape)


CMC = torch.IntTensor(len(all_labels)).zero_()
ap = 0.0
for i in range(len(all_labels)):
    scores_new = evaluate2(all_features[i],all_labels[i],all_cams[i],all_frames[i], all_features,all_labels,all_cams,all_frames,distribution)
    print('scores_new shape:',scores_new.shape)
    all_scores[i,:] = scores_new
    print(i)

print('type(all_scores):',type(all_scores))
all_scores = {'all_scores':all_scores}
scipy.io.savemat('model/'+name+'/'+'all_scores'+'.mat',all_scores)
###############################################################################################

