from __future__ import print_function
import numpy as np
from sklearn import metrics
from sklearn import metrics
import pandas as pd
import random
import newmetrics
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def parse_args():
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,conflict_handler='resolve')
        parser.add_argument('--inputscore', help='score path')
        parser.add_argument('--blocksize',type=int, help='blocksize')
        parser.add_argument('--blockratio',type=float, help='blockratio')
        args = parser.parse_args()
        return args


def eval(truth,pred,pos_label=1):
        evaluation_dict = {}
        acc = metrics.accuracy_score(truth,pred);evaluation_dict['acc']=acc
        precision = metrics.precision_score(truth,pred,pos_label=pos_label);evaluation_dict['precision']=precision
        recall = metrics.recall_score(truth,pred,pos_label=pos_label);evaluation_dict['recall']=recall
        f1_macro = metrics.f1_score(truth,pred, average='macro',pos_label=pos_label);evaluation_dict['f1_macro']=f1_macro
        f1_micro = metrics.f1_score(truth,pred, average='micro',pos_label=pos_label);evaluation_dict['f1_micro']=f1_micro
        mcc = metrics.matthews_corrcoef(truth,pred);evaluation_dict['mcc']=mcc
        # auc = metrics.roc_auc_score(truth,pred);evaluation_dict['auc']=auc
        return evaluation_dict


def run(args):
#    df_roc=pd.read_csv(args.inputroc,header=0, sep=' ',encoding="utf-8")
#    df_roc['dist']=df_roc['tpr']*df_roc['tpr']+(1-df_roc['fpr'])*(1-df_roc['fpr'])
    blocksize=args.blocksize
    blockratio=args.blockratio
    df_score=pd.read_csv(args.inputscore,header=0, sep=' ',encoding="utf-8")
    individual_label=df_score['true'].values.tolist()
    individual_pred=df_score['pred'].values.tolist()
    pred_block = []
    true_block = []
    testing_block_num = int(np.floor(len(df_score)/blocksize))
    print('block num: ',testing_block_num)
    for i in range(testing_block_num):
        i1=i*blocksize
        i2=i1+blocksize
        pred_block_1 = np.mean(individual_pred[i1:i2])
        true_block_1 = np.sum(individual_label[i1:i2])
        true_block_1 = int(true_block_1 <  blocksize*blockratio)
        pred_block.append(pred_block_1)#by dingfu
        true_block.append(true_block_1)#by dingfu
        #print(true_block_1,pred_block_1)
    auc,fpr, tpr, thresholds=newmetrics.roc(true_block,pred_block,pos_label=1,output_path=None)
    print('auc: ',auc)
    dist=[x[1]**2+(1-x[0])**2 for x in zip(tpr,fpr) ]
    mindist=min(dist)
    ithred=dist.index(mindist)
    thred=thresholds[ithred]
    print('mindist',mindist,'thred',thred,'ithred',ithred)
    truth=np.array(true_block,dtype=int)
    pred=[x > thred for x in pred_block]
    pred=np.array(pred,dtype=int)
    out=eval(truth,pred)
    print(out)
    conf_mat = metrics.confusion_matrix(truth,pred)
    print(conf_mat)
    #for tmp in zip(fpr, tpr, thresholds,dist):
    #    print('\t'.join([str(x) for x in tmp]))
if __name__ == '__main__':
    args = parse_args()
    run(args)
