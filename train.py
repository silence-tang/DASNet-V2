import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.nn import functional as F
import utils.transforms as trans
import utils.utils as util
import layer.loss as ls
import utils.metric as mc
import cv2
import model.siameseNet.dares as models
import cfg.CDD as cfg
import dataset.rs as dates
import time
from datetime import datetime
import logging
import configargparse
from torch.cuda.amp import autocast, GradScaler


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def various_distance(out_vec_t0, out_vec_t1,dist_flag):
    if dist_flag == 'l2':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=2)
    if dist_flag == 'l1':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=1)
    if dist_flag == 'cos':
        distance = 1 - F.cosine_similarity(out_vec_t0, out_vec_t1)
    return distance

def single_layer_similar_heatmap_visual(idx,output_t0,output_t1,save_change_map_dir,epoch,batch_idx,filename,layer_flag,dist_flag):
    n, c, h, w = output_t0.data.shape # 1,512,32,32 -> 1,64,256,256
    # 拉成1024x512 -> 65536x64的向量
    out_t0_rz = torch.transpose(output_t0.view(c, h * w), 1, 0)
    out_t1_rz = torch.transpose(output_t1.view(c, h * w), 1, 0)
    # 计算像素对在通道上的距离，比如0,0处的两个通道的距离，这个距离应该是都小于2的
    distance = various_distance(out_t0_rz,out_t1_rz,dist_flag=dist_flag)
    similar_distance_map = distance.view(h,w).data.cpu().numpy() # 256x256
    # 插值到256x256
    similar_distance_map_rz = nn.functional.interpolate(torch.from_numpy(similar_distance_map[np.newaxis, np.newaxis, :]),size=[cfg.TRANSFROM_SCALES[1],cfg.TRANSFROM_SCALES[0]], mode='bilinear',align_corners=True)
    # 渲染热力图并保存
    save_change_map_dir_ = os.path.join(save_change_map_dir, 'epoch_' + str(epoch))
    check_dir(save_change_map_dir_)
    save_change_map_dir_layer = os.path.join(save_change_map_dir_,layer_flag)
    check_dir(save_change_map_dir_layer)
    save_weight_fig_dir = os.path.join(save_change_map_dir_layer, str(batch_idx) + '_' + filename[0].split('/')[2])
    if idx % 20 == 0:
        similar_dis_map_colorize = cv2.applyColorMap(np.uint8(255 * similar_distance_map_rz.data.cpu().numpy()[0][0]), cv2.COLORMAP_JET)
        cv2.imwrite(save_weight_fig_dir, similar_dis_map_colorize)
    # 返回距离map
    return similar_distance_map_rz.data.cpu().numpy()

def validate(net, val_dataloader, epoch, batch_idx, save_change_map_dir, save_pr_dir, best_metric, best_epoch, best_batch_idx):
    net.eval()
    with torch.no_grad():   
        # 初始化
        num = 0.0
        # 减小阈值个数可以加速
        num_thresh = 96
        thresh = np.linspace(0.0, 2.2, num_thresh)
        # metric_dict = {'total_fp': [0,...,0], 'total_fn':[0,...,0], 'total_posnum':0, 'total_negnum':0}
        metric_dict = util.init_metric_dict(thresh=thresh)
        for idx, batch in enumerate(val_dataloader):
            input1, input2, targets, filename, height, width = batch
            input1, input2, targets = input1.cuda(), input2.cuda(), targets.cuda()
            out_conv5, out_fc, out_embedding = net(input1, input2)
            out_embedding_t0, out_embedding_t1 = out_embedding # 已经被标准化为[0,1]向量
            embedding_distance_map = single_layer_similar_heatmap_visual(idx,out_embedding_t0,out_embedding_t1,save_change_map_dir,epoch,batch_idx,filename,'embedding','l2')
            num += 1
            prob_change = embedding_distance_map[0][0]   # 256x256
            gt = targets.data.cpu().numpy()
            # 求单个batch的FN, FP等, 这里的FN是在不同阈值下算出来的, posNum=TP+FN, negNum=TN+FP
            FN, FP, posNum, negNum = mc.eval_image(gt[0], prob_change, cl_index=1, thresh=thresh)
            # 循环结束后metric_dict存的是所有batch在各度量指标上的的度量数值之和
            metric_dict['total_fp'] += FP
            metric_dict['total_fn'] += FN
            metric_dict['total_posnum'] += posNum
            metric_dict['total_negnum'] += negNum
        
        # 拿到整个val set在各指标上的数值之和
        total_fp = metric_dict['total_fp']
        total_fn = metric_dict['total_fn']
        total_posnum = metric_dict['total_posnum']
        total_negnum = metric_dict['total_negnum']
        # mc.pxEval_maximizeFMeasure计算一个batch中最大的F_score值
        res_dict = mc.pxEval_maximizeFMeasure(total_posnum, total_negnum, total_fn, total_fp, thresh=thresh)
        metric_dict.setdefault('metric', res_dict)
        # 拿到f_score
        pr, recall, f_score = metric_dict['metric']['precision'], metric_dict['metric']['recall'], metric_dict['metric']['MaxF']
        pr_save_epoch_dir = os.path.join(save_pr_dir)
        check_dir(pr_save_epoch_dir)
        pr_save_epoch_cat_dir = os.path.join(pr_save_epoch_dir)
        check_dir(pr_save_epoch_cat_dir)
        # 保存metric日志
        mc.save_metric_json(metric_dict, pr_save_epoch_cat_dir, epoch, batch_idx)
        pr_save_dir = os.path.join(pr_save_epoch_cat_dir, str(epoch) + '_' + str(batch_idx) + '_pr.png')
        # 保存P-R曲线
        mc.plotPrecisionRecall(pr, recall, pr_save_dir, benchmark_pr=None)
        print('f_max: ', f_score)
        print('best_f_max: ', best_metric)
        print('best_epoch: ', best_epoch)
        print('best_batch_idx: ',best_batch_idx)
        return f_score


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--resume", type=int, default=0, help='0: no resume, 100: load the best model, others: load trained model_i.pth after i_th epoch')
    parser.add_argument("--start_epoch", type=int, default=0, help='from which epoch to continue training')
    parser.add_argument("--datatime", type=str, default=None, help='used for resume, for example: 10.28_09')
    return parser

def main():
    # parse args
    parser = config_parser()
    args = parser.parse_args()
    # logs
    DateTime = datetime.now().strftime("%D-%H")
    logname = DateTime.split('/')[0] + '.' + DateTime.split('/')[1] + '_' + DateTime.split('/')[2].split('-')[1]
    logging.basicConfig(filename=logname + '.txt', level=logging.DEBUG)
    # configs
    best_metric = 0
    best_epoch = 0
    best_batch_idx = 0
    # load datasets
    train_transform_det = trans.Compose([trans.Scale(cfg.TRANSFROM_SCALES)])
    val_transform_det = trans.Compose([trans.Scale(cfg.TRANSFROM_SCALES)])
    train_data = dates.Dataset(cfg.TRAIN_DATA_PATH, cfg.TRAIN_LABEL_PATH, cfg.TRAIN_TXT_PATH, 'train', transform=True, transform_med=train_transform_det)
    val_data = dates.Dataset(cfg.VAL_DATA_PATH, cfg.VAL_LABEL_PATH, cfg.VAL_TXT_PATH, 'val', transform=True, transform_med=val_transform_det)
    train_loader = Data.DataLoader(train_data, batch_size=cfg.TRAIN_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=False)
    val_loader = Data.DataLoader(val_data, batch_size=cfg.VAL_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    
    # build models
    model = models.SiameseNet(norm_flag='l2')
    # init training configs
    model = model.cuda()
    MaskLoss = ls.CLNew()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.INIT_LEARNING_RATE, weight_decay=cfg.DECAY)
    
    if args.resume == 100:
        checkpoint = torch.load(cfg.TRAINED_BEST_PERFORMANCE_CKPT)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('The best model has been loaded.')

    elif args.resume != 0:
        checkpoint = torch.load(os.path.join(cfg.TRAINED_RESUME_PATH, args.datatime, 'model_'+str(args.resume)+'.pth'))
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(args.datatime+'/'+'model_'+str(args.resume)+' has been loaded.')

    else:
        print('ResNet50 backbone has been loaded.')

    # check directories
    ab_test_dir = os.path.join(cfg.SAVE_PRED_PATH, logname)
    check_dir(ab_test_dir)
    save_change_map_dir = os.path.join(ab_test_dir, 'distance_maps/')
    save_pr_dir = os.path.join(ab_test_dir,'pr_curves')
    check_dir(save_change_map_dir)
    check_dir(save_pr_dir)

    # train loop
    time_start = time.time()
    start = args.start_epoch
    print('Start training from epoch {}.'.format(start))
    for epoch in range(start, cfg.MAX_EPOCH):
        for batch_idx, batch in enumerate(train_loader):
            # -----lr needs to be adjusted for different batch_size-----
            step = epoch * 10000 + batch_idx  # 30 * 10000 + 1000
            util.adjust_learning_rate(cfg.INIT_LEARNING_RATE, optimizer, step)
            # AMP traning
            # with autocast():
            model.train()
            # img1, img2: (8, 3, 256, 256), label: (8, 256, 256)
            img1, img2, label, filename, height, width = batch
            img1, img2, label = img1.cuda(), img2.cuda(), label.cuda().float()
            out_conv5, out_fc, out_embedding = model(img1, img2)  # forward
            out_conv5_t0, out_conv5_t1 = out_conv5                # (1, 512, 32, 32)  --> (1, 64, 256, 256)
            out_fc_t0, out_fc_t1 = out_fc                         # (1, 512, 32, 32)  --> (1, 64, 256, 256)
            out_embedding_t0, out_embedding_t1 = out_embedding    # (1, 512, 32, 32)  --> (1, 64, 256, 256)
            # 这三个rz_label相等 32x32  --> 256x256
            label_rz_conv5 = util.rz_label(label, size=out_conv5_t0.data.cpu().numpy().shape[2:]).cuda()
            label_rz_fc = util.rz_label(label, size=out_fc_t0.data.cpu().numpy().shape[2:]).cuda()
            label_rz_embedding = util.rz_label(label, size=out_embedding_t0.data.cpu().numpy().shape[2:]).cuda()
            # 求3个loss
            contrastive_loss_conv5 = MaskLoss(out_conv5_t0, out_conv5_t1, label_rz_conv5) # 一个实数
            contrastive_loss_fc = MaskLoss(out_fc_t0, out_fc_t1, label_rz_fc)
            contrastive_loss_embedding = MaskLoss(out_embedding_t0, out_embedding_t1, label_rz_embedding)
            loss = contrastive_loss_conv5 + contrastive_loss_fc + contrastive_loss_embedding
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 20 == 0:
                logging.info("epoch/batch_idx: [%d/%d] lr: %.6f best_f_max: %.4f best_epoch: %d best_batch_idx: %d loss: %.4f loss_conv5: %.4f loss_fc: %.4f "
                      "loss_embedding: %.4f" % (epoch, batch_idx, optimizer.state_dict()['param_groups'][0]['lr'], best_metric, best_epoch, best_batch_idx, loss.item(), contrastive_loss_conv5.item(),
                                                     contrastive_loss_fc.item(), contrastive_loss_embedding.item()))
                print("epoch/batch_idx: [%d/%d] lr: %.6f loss: %.4f loss_conv5: %.4f loss_fc: %.4f "
                      "loss_embedding: %.4f" % (epoch, batch_idx, optimizer.state_dict()['param_groups'][0]['lr'], loss.item(), contrastive_loss_conv5.item(),
                                                     contrastive_loss_fc.item(), contrastive_loss_embedding.item()))
            # 每1000个batch执行一次f_score验证
            #  if (batch_idx) % 1000 == 0:
            if (batch_idx != 0) & (batch_idx % 1000 == 0):
                model.eval()
                current_metric = validate(model, val_loader, epoch, batch_idx, save_change_map_dir, save_pr_dir, best_metric, best_epoch, best_batch_idx)
                if current_metric > best_metric:
                    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
                    torch.save(state, os.path.join(ab_test_dir, 'model_best_'+logname+'_'+str(best_epoch)+'_'+str(best_batch_idx)+'.pth'))
                    best_metric = current_metric
                    best_epoch = epoch
                    best_batch_idx = batch_idx

        # 训练集已遍历一遍
        model.eval()
        current_metric = validate(model, val_loader, epoch, batch_idx, save_change_map_dir, save_pr_dir, best_metric, best_epoch, best_batch_idx)
        # model_i.pth表示经过第i个epoch的训练后得到的模型
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, os.path.join(ab_test_dir, 'model_' + str(epoch) + '.pth'))
        if current_metric > best_metric:
            state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, os.path.join(ab_test_dir, 'model_best_'+logname+'_'+str(best_epoch)+'_'+str(best_batch_idx)+'.pth'))
            best_metric = current_metric
            best_epoch = epoch
            best_batch_idx = batch_idx
  
    elapsed = round(time.time() - time_start)
    print('Elapsed {}'.format(elapsed))

if __name__ == '__main__':
    main()