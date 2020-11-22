
# run.py arg1 arg2 ...
# arg1: VERSION
# arg2: i-th RUN (ITER)
# arg3: random seed

ver=1
rn=42
nohup python run.py $ver 1 $rn iris &> logiris$ver-$rn &
