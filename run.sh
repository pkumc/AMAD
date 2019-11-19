codename=train.py
mkdir /Users/zhenggao/Desktop/alibaba/阿里妈妈/data/synthetic/results
path= /Users/zhenggao/Desktop/alibaba/阿里妈妈/data/synthetic/results/${codename//.py/}
mkdir ${path}
echo ${path}

python ${codename} \
--input /Users/zhenggao/Desktop/alibaba/阿里妈妈/data/synthetic/synthetic.txt \
--instance-output ${path}/instance.txt \
--block-output ${path}/block.txt \
--batch-size 10 \
--block-size 20 \
--epoch 100 \
> ${path}/log
# grep 'instance level auc' ${path}/log

# python newpostproc.py \
# --inputroc ${path}/roc_instance.txt_47 \
# --inputscore ${path}/instance.txt_47

# python newpostprocblock.py \
# --inputscore ${path}/instance.txt_47 \
# --blocksize 20 \
# --blockratio 0.5


CUDA_VISIBLE_DEVICES=0 python train.py --input /Users/zhenggao/Desktop/alibaba/阿里妈妈/data/alimama/alimama.txt \
 --instance-output /Users/zhenggao/Desktop/alibaba/阿里妈妈/data/alimama/results/train/instance.txt \
 --block-output /Users/zhenggao/Desktop/alibaba/阿里妈妈/data/alimama/results/train/block.txt \
 --batch-size 10 --block-size 20 --epoch 100


 
