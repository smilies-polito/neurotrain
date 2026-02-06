

cd ..

for lr in 0.02 0.01 0.005 0.001
do
  python main.py --model_type LIF --dataset_name shd  --nb_epochs 100 --method esd-rtrl --nb_hiddens 1024 --lr $lr --devices 0 --etrace_decay 0.88
  python main.py --model_type RLIF --dataset_name shd  --nb_epochs 100 --method esd-rtrl --nb_hiddens 1024 --lr $lr --devices 0 --etrace_decay 0.91
  python main.py --model_type adLIF --dataset_name shd --nb_epochs 100 --method esd-rtrl --nb_hiddens 1024 --lr $lr --devices 0 --etrace_decay 0.95
  python main.py --model_type RadLIF --dataset_name shd --nb_epochs 100 --method esd-rtrl --nb_hiddens 1024 --lr $lr --devices 0 --etrace_decay 0.98
done
