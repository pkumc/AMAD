codename=train.py
mkdir ~/anomaly_detection/results
path=~/anomaly_detection/results/${codename//.py/}
mkdir ${path}
echo ${path}

python ${codename} \
--input /home/zheng.gz/anomaly_detection/synthetic/synthetic.txt \
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
