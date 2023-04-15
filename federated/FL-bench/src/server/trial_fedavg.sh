python fedavg.py --model res18 -d tiny_imagenet -p num_clients-20 -lr 1e-3 -wd 1e-6 --prefix lr-1e-3_wd-1e-6 -ge 50 --visible 0 -jr 0.7
python fedavg.py --model res18 -d tiny_imagenet -p num_clients-20 -lr 1e-4 -wd 1e-6 --prefix lr-1e-4_wd-1e-6 -ge 50 --visible 0 -jr 0.7
