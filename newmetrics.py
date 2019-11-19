from sklearn import metrics

def roc(label,pred,pos_label=1,output_path=None):
    fpr,tpr,thresholds = metrics.roc_curve(label, pred,pos_label=pos_label)
    auc=metrics.auc(fpr, tpr)
    if output_path is not None:
         path=output_path.split('/')
         path[-1]='roc_'+path[-1]
         path='/'.join(path)
         print(path)
         with open(path,'w') as f:
             tmp='fpr tpr thresholds\n'
             f.write(tmp)
             for i in zip(fpr,tpr,thresholds):
                 tmp=str(i[0])+' '+str(i[1])+' '+ str(i[2])+'\n'
                 f.write(tmp)
    return auc,fpr, tpr, thresholds 

if __name__ == '__main__':
    print(roc([1,0,1,0],[0.9,0.5,0.1,0.4]))
