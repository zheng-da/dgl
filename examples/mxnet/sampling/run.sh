export OMP_NUM_THREADS=4
DGLBACKEND=mxnet python3 dist_train.py --id 0 --graph-name MAG0 --server 1 --num-parts 8 --model gcn_ns --n-classes 1008 --n-features 768 &
DGLBACKEND=mxnet python3 dist_train.py --id 1 --graph-name MAG0 --server 1 --num-parts 8 --model gcn_ns --n-classes 1008 --n-features 768 &
DGLBACKEND=mxnet python3 dist_train.py --id 2 --graph-name MAG0 --server 1 --num-parts 8 --model gcn_ns --n-classes 1008 --n-features 768 &
DGLBACKEND=mxnet python3 dist_train.py --id 3 --graph-name MAG0 --server 1 --num-parts 8 --model gcn_ns --n-classes 1008 --n-features 768 &
DGLBACKEND=mxnet python3 dist_train.py --id 4 --graph-name MAG0 --server 1 --num-parts 8 --model gcn_ns --n-classes 1008 --n-features 768 &
DGLBACKEND=mxnet python3 dist_train.py --id 5 --graph-name MAG0 --server 1 --num-parts 8 --model gcn_ns --n-classes 1008 --n-features 768 &
DGLBACKEND=mxnet python3 dist_train.py --id 6 --graph-name MAG0 --server 1 --num-parts 8 --model gcn_ns --n-classes 1008 --n-features 768 &
DGLBACKEND=mxnet python3 dist_train.py --id 7 --graph-name MAG0 --server 1 --num-parts 8 --model gcn_ns --n-classes 1008 --n-features 768 &

sleep 120

DGLBACKEND=mxnet python3 dist_train.py --id 0 --graph-name MAG0 --server 0 --num-parts 8 --model gcn_ns --n-classes 1008 --n-features 768 &
DGLBACKEND=mxnet python3 dist_train.py --id 1 --graph-name MAG0 --server 0 --num-parts 8 --model gcn_ns --n-classes 1008 --n-features 768 &
DGLBACKEND=mxnet python3 dist_train.py --id 2 --graph-name MAG0 --server 0 --num-parts 8 --model gcn_ns --n-classes 1008 --n-features 768 &
DGLBACKEND=mxnet python3 dist_train.py --id 3 --graph-name MAG0 --server 0 --num-parts 8 --model gcn_ns --n-classes 1008 --n-features 768 &
DGLBACKEND=mxnet python3 dist_train.py --id 4 --graph-name MAG0 --server 0 --num-parts 8 --model gcn_ns --n-classes 1008 --n-features 768 &
DGLBACKEND=mxnet python3 dist_train.py --id 5 --graph-name MAG0 --server 0 --num-parts 8 --model gcn_ns --n-classes 1008 --n-features 768 &
DGLBACKEND=mxnet python3 dist_train.py --id 6 --graph-name MAG0 --server 0 --num-parts 8 --model gcn_ns --n-classes 1008 --n-features 768 &
DGLBACKEND=mxnet python3 dist_train.py --id 7 --graph-name MAG0 --server 0 --num-parts 8 --model gcn_ns --n-classes 1008 --n-features 768 &

#DGLBACKEND=mxnet python3 dist_train.py --id 0 --graph-name reddit --server 1 --num-parts 1 --model gcn_ns --n-classes 41 --n-features 602 --dataset reddit &
#DGLBACKEND=mxnet python3 dist_train.py --id 0 --graph-name reddit --server 0 --num-parts 1 --model gcn_ns --n-classes 41 --n-features 602 --dataset reddit &
