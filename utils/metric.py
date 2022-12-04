import numpy as np
import pylab

# def evalExp(gtBin, cur_prob, thres):

    # # thresInf = np.concatenate(([-np.Inf], thres, [np.Inf]))

    # # 求FN
    # fnArray = cur_prob[(gtBin == True)]
    # FN = np.array([sum(fnArray <= t) for t in thres])
    # # fnHist = np.histogram(fnArray, bins=thresInf)[0]
    # # fnCum = np.cumsum(fnHist)
    # # FN = fnCum[0: 0+len(thres)]

    # # 求FP
    # fpArray = cur_prob[(gtBin == False)]
    # FP = np.array([sum(fpArray > t) for t in thres])
    # # fpHist = np.histogram(fpArray, bins=thresInf)[0]
    # # fpCum = np.flipud(np.cumsum(np.flipud(fpHist)))
    # # FP = fpCum[1: 1+len(thres)]

    # posNum = np.sum(gtBin == True)    # 实际为changed的数量(TP+FN)
    # negNum = np.sum(gtBin == False)   # 实际为unchanged的数量(TN+FP)

    # return FN, FP, posNum, negNum


def evalExp(gtBin, cur_prob, thres):

    thresInf = np.concatenate(([-np.Inf], thres, [np.Inf])) # 便于后续进行直方图统计

    fnArray = cur_prob[(gtBin == True)]
    fnHist = np.histogram(fnArray, bins=thresInf)[0]
    fnCum = np.cumsum(fnHist)
    FN = fnCum[0: 0 + len(thres)]

    fpArray = cur_prob[(gtBin == False)]
    fpHist = np.histogram(fpArray, bins=thresInf)[0] # 左闭右开区间
    # 倒置求累计和再倒置回去
    fpCum = np.flipud(np.cumsum(np.flipud(fpHist)))
    FP = fpCum[1: 1 + len(thres)]  # 从0或1开始取数取决于开闭区间

    posNum = np.sum(gtBin == True)
    negNum = np.sum(gtBin == False)

    return FN, FP, posNum, negNum


def eval_image(gt_image, prob, cl_index, thresh):
    # 设定阈值
    thresh = thresh
    # true/false map of ground truth
    cl_gt = gt_image[:, :] == cl_index
    FN, FP, posNum, negNum = evalExp(cl_gt, prob, thresh)
    return FN, FP, posNum, negNum


def pxEval_maximizeFMeasure(totalPosNum, totalNegNum, totalFN, totalFP, thresh=None):
    '''
    @param totalPosNum: scalar
    @param totalNegNum: scalar
    @param totalFN: vector
    @param totalFP: vector
    @param thresh: vector
    '''
    # TP TN
    totalTP = totalPosNum - totalFN
    totalTN = totalNegNum - totalFP

    valid = (totalTP >= 0) & (totalTN >= 0)  # 检测有效值
    assert valid.all(), 'Detected invalid elements in eval'

    recall = totalTP / float(totalPosNum)               # 分母一定非0
    precision = totalTP / (totalTP + totalFP + 1e-10)   # 防止出现分母为0
    selector_invalid = (recall == 0) & (precision == 0) # 找到TP=0的情况
    recall = recall[~selector_invalid]                  # 将其排除
    precision = precision[~selector_invalid]

    # F-measure
    beta = 1.0
    betasq = beta ** 2
    F = (1 + betasq) * (precision * recall) / ((betasq * precision) + recall + 1e-10) # 防止出现分母为0
    index = F.argmax()
    MaxF = F[index]  # 求maxf

    # recall_bst = recall[index]
    # precision_bst = precision[index]
    # TP = totalTP[index]
    # TN = totalTN[index]
    # FP = totalFP[index]
    # FN = totalFN[index]
    # valuesMaxF = np.zeros((1, 4), 'u4')
    # valuesMaxF[0, 0] = TP
    # valuesMaxF[0, 1] = TN
    # valuesMaxF[0, 2] = FP
    # valuesMaxF[0, 3] = FN

    # ACC = (totalTP+ totalTN)/(totalPosNum+totalNegNum)

    prob_eval_scores = {}
    prob_eval_scores['MaxF'] = MaxF
    prob_eval_scores['totalPosNum'] = totalPosNum
    prob_eval_scores['totalNegNum'] = totalNegNum
    prob_eval_scores['precision'] = precision
    prob_eval_scores['recall'] = recall
    prob_eval_scores['thresh'] = thresh

    if np.any(thresh) != None:
        BestThresh = thresh[index]
        prob_eval_scores['BestThresh'] = BestThresh
        print('cur_best_thresh: ', BestThresh)  

    # return a dict
    return prob_eval_scores


def setFigLinesBW(fig):
    """
    Take each axes in the figure, and for each line in the axes, make the
    line viewable in black and white.
    """
    for ax in fig.get_axes():
        setAxLinesBW(ax)


def setAxLinesBW(ax):
    """
    Take each Line2D in the axes, ax, and convert the line style to be
    suitable for black and white viewing.
    """
    MARKERSIZE = 3

    #     COLORMAP = {
    #         'r': {'marker': None, 'dash': (None,None)},
    #         'g': {'marker': None, 'dash': [5,2]},
    #         'm': {'marker': None, 'dash': [11,3]},
    #         'b': {'marker': None, 'dash': [6,3,2,3]},
    #         'c': {'marker': None, 'dash': [1,3]},
    #         'y': {'marker': None, 'dash': [5,3,1,2,1,10]},
    #         'k': {'marker': 'o', 'dash': (None,None)} #[1,2,1,10]}
    #         }
    '''''''''
    COLORMAP = {
        'r': {'marker': "None", 'dash': (None,None)},
        'g': {'marker': "None", 'dash': [5,2]},
        'm': {'marker': "None", 'dash': [11,3]},
        'b': {'marker': "None", 'dash': [6,3,2,3]},
        'c': {'marker': "None", 'dash': [1,3]},
        'y': {'marker': "None", 'dash': [5,3,1,2,1,10]},
        'k': {'marker': 'o', 'dash': (None,None)} #[1,2,1,10]}
        }
    '''''''''

    COLORMAP = {
        'r': {'marker': "None", 'dash': (None, None)},
        'g': {'marker': "None", 'dash': (None, None)},
        'm': {'marker': "None", 'dash': (None, None)},
        'b': {'marker': "None", 'dash': (None, None)},
        'c': {'marker': "None", 'dash': (None, None)},
        'y': {'marker': "None", 'dash': (None, None)},
        'k': {'marker': 'o', 'dash': (None, None)}  # [1,2,1,10]}
    }

    for line in ax.get_lines():
        origColor = line.get_color()
        # line.set_color('black')

        line.set_dashes(COLORMAP[origColor]['dash'])
        line.set_marker(COLORMAP[origColor]['marker'])
        line.set_markersize(MARKERSIZE)


def plotPrecisionRecall(precision, recall, outFileName, benchmark_pr=None, Fig=None, drawCol=0, textLabel=None,
                        title=None, fontsize1=14, fontsize2=10, linewidth=3):
    '''
    :param precision:
    :param recall:
    :param outFileName:
    :param Fig:
    :param drawCol:
    :param textLabel:
    :param fontsize1:
    :param fontsize2:
    :param linewidth:
    '''
    clearFig = False

    if Fig == None:
        Fig = pylab.figure()
        clearFig = True

    linecol = ['r', 'm', 'b', 'c']

    if benchmark_pr != None:

        benchmark_recall = np.array(benchmark_pr['recall'])
        benchmark_precision = np.array(benchmark_pr['precision'])
        pylab.plot(100 * benchmark_recall, 100 * benchmark_precision, linewidth=linewidth, color=linecol[drawCol],
                   label=textLabel)
    else:
        pylab.plot(100 * recall, 100 * precision, linewidth=2, color=linecol[drawCol], label=textLabel)

    # writing out PrecRecall curves as graphic
    setFigLinesBW(Fig)
    if textLabel != None:
        pylab.legend(loc='lower left', prop={'size': fontsize2})

    if title != None:
        pylab.title(title, fontsize=fontsize1)

    # pylab.title(title,fontsize=24)
    pylab.ylabel('Precision [%]', fontsize=fontsize1)
    pylab.xlabel('Recall [%]', fontsize=fontsize1)

    pylab.xlim(0, 100)
    pylab.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                 ('0', '', '0.20', '', '0.40', '', '0.60', '', '0.80', '', '1.0'), fontsize=fontsize2)
    pylab.ylim(0, 100)
    pylab.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                 ('0', '', '0.20', '', '0.40', '', '0.60', '', '0.80', '', '1.0'), fontsize=fontsize2)

    # pylab.grid(True)
    #
    if type(outFileName) != list:
        pylab.savefig(outFileName)
    else:
        for outFn in outFileName:
            pylab.savefig(outFn)
    if clearFig:
        pylab.close()
        Fig.clear()


def save_metric_json(metrics, save_path, epoch, batch_idx):
    import json
    # metric_dict= {}
    recall_ = list(metrics['metric']['recall'])
    precision_ = list(metrics['metric']['precision'])
    f_score = metrics['metric']['MaxF']
    # cont_conv5 = metrics['contrast_conv5']
    # cont_embedding = metrics['contrast_embedding']
    metric_ = {'recall': recall_, 'precision': precision_, 'f-score': f_score}
    # metric_ = {'recall': recall_, 'precision': precision_, 'f-score': f_score,
    #            'contrast_embedding': cont_embedding,'contrast_conv5':cont_conv5}
    file_ = open(save_path + '/' + str(epoch) + '_' + str(batch_idx) + '_metric.json', 'w')
    file_.write(json.dumps(metric_, ensure_ascii=False, indent=2))
    file_.close()


def RMS_Contrast(dist_map):
    n, c, h, w = dist_map.shape
    dist_map_l = np.resize(dist_map, (n * c * h * w))
    mean = np.mean(dist_map_l, axis=0)
    std = np.std(dist_map_l, axis=0, ddof=1)
    contrast = std / mean
    return contrast