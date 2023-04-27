python fedavg.py --model res18 -d tiny_imagenet --visible 0 -p num_clients-20 --local_epoch 25 -jr 0.2 -vg 1 -tg 1 --global_epoch 16 &> log25.log
python fedavg.py --model res18 -d tiny_imagenet --visible 0 -p num_clients-20 --local_epoch 20 -jr 0.2 -vg 1 -tg 1 --global_epoch 20 &> log20.log
python fedavg.py --model res18 -d tiny_imagenet --visible 0 -p num_clients-20 --local_epoch 10 -jr 0.2 -vg 1 -tg 1 --global_epoch 40 &> log10.log
python fedavg.py --model res18 -d tiny_imagenet --visible 0 -p num_clients-20 --local_epoch 8 -jr 0.2 -vg 1 -tg 1 --global_epoch 50 &> log8.log
python fedavg.py --model res18 -d tiny_imagenet --visible 0 -p num_clients-20 --local_epoch 6 -jr 0.2 -vg 1 -tg 1 --global_epoch 67 &> log6.log
python fedavg.py --model res18 -d tiny_imagenet --visible 0 -p num_clients-20 --local_epoch 4 -jr 0.2 -vg 1 -tg 1 --global_epoch 100 &> log4.log
python fedavg.py --model res18 -d tiny_imagenet --visible 0 -p num_clients-20 --local_epoch 2 -jr 0.2 -vg 1 -tg 1 --global_epoch 200 &> log2.log
python fedavg.py --model res18 -d tiny_imagenet --visible 0 -p num_clients-20 --local_epoch 1 -jr 0.2 -vg 1 -tg 1 --global_epoch 400 &> log1.log
