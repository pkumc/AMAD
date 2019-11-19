from __future__ import print_function
import numpy as np
from sklearn import metrics
from sklearn import metrics
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def parse_args():
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,conflict_handler='resolve')
        parser.add_argument('--inputroc', help='roc path')
        parser.add_argument('--inputscore', help='score path')
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
    df_roc=pd.read_csv(args.inputroc,header=0, sep=' ',encoding="utf-8")
    df_roc['dist']=df_roc['fpr']*df_roc['fpr']+(1-df_roc['tpr'])*(1-df_roc['tpr'])
    df_score=pd.read_csv(args.inputscore,header=0, sep=' ',encoding="utf-8")
    idx = df_roc['dist'].idxmin()
    minrow=df_roc.iloc[idx]
    print('minrow',minrow)
    df_score['pred']=df_score['pred']<minrow['thresholds']
    truth=np.array(df_score['true'].tolist(),dtype=int)
    pred=np.array(df_score['pred'].tolist(),dtype=int)
    out=eval(truth,pred,pos_label=0)
    print(out)
if __name__ == '__main__':
    args = parse_args()
    run(args)
