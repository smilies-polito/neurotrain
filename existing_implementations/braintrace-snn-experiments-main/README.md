# ``BrainTrace`` experiments on spiking neural networks



## Requirements


```bash

pip install BrainX[cuda12]
# or
pip install BrainX[cuda13]
pip install h5py matplotlib msgpack tonic prettytable
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

```




## RSNN long-term dependency evaluation: DMS task

```bash

# BPTT
python task-rsnn-long-term-dependency.py --epochs 2000 --method bptt --dataset dms --t_delay 1000 \
  --tau_I2 1500 --tau_neu 100 --tau_syn 100 --n_rec 200 --lr 0.001 --A2 1 --optimizer adam --devices 3 \
  --t_fixation 10. --spk_fun relu --acc_th 0.95 --n_data_worker 4  --dt 1. --ff_scale 6 --rec_scale 2
  
  
# ES-D-RTRL (IO Dim)  
python task-rsnn-long-term-dependency.py --epochs 2000 --method expsm_diag --etrace_decay 0.99 --vjp_time t --dataset dms --t_delay 1000 \
    --tau_I2 1500 --tau_neu 100 --tau_syn 100 --n_rec 200 --lr 0.001 --A2 1  --optimizer adam --devices 3 \
    --t_fixation 10. --spk_fun relu --acc_th 0.95  --n_data_worker 4  --dt 1. --ff_scale 6 --rec_scale 2
    

# D-RTRL (Param Dim)
python task-rsnn-long-term-dependency.py --epochs 2000   --method diag --dataset dms  --t_delay 1000 \
    --tau_I2 1500 --tau_neu 100 --tau_syn 100 --n_rec 200 --lr 0.001 --A2 1   --optimizer adam --devices 3 \
    --t_fixation 10. --spk_fun relu --acc_th 0.95  --n_data_worker 4  --dt 1. --ff_scale 6 --rec_scale 2

```



## EI network for decision making tasks

ES-D-RTRL training of the EI network for decision making tasks. 

```bash
cd ./ei_coba_net_decision_making
python training.py --tau_neu 200 --tau_syn 10 --tau_I2 2000  --ff_scale 4.0 --rec_scale 2.0  --method esd-rtrl  --n_rec 800  --epoch_per_step 20 --diff_spike  0  --epochs 300 --lr 0.001 --etrace_decay 0.9
```

D-RTRL training of the EI network for decision making tasks. 

```bash
python training.py --tau_neu 200 --tau_syn 10 --tau_I2 2000  --ff_scale 4.0 --rec_scale 2.0  --method d-rtrl  --n_rec 800  --epoch_per_step 30 --diff_spike  0  --epochs 300 --lr 0.001
```


BPTT training of the EI network for decision making tasks. 

```bash
python training.py --tau_neu 200 --tau_syn 10 --tau_I2 2000  --ff_scale 4.0 --rec_scale 2.0  --method bptt  --n_rec 800  --epoch_per_step 30 --diff_spike  0  --epochs 300 --lr 0.001 
```


## Memory and speed evaluation


```bash

python task-memory-and-speed-evaluation-tpu.py

```




## RSNN image classification on Gesture dataset

The code below is used to train a spiking neural network on the Gesture dataset using different methods (BPTT, ES-D-RTRL, D-RTRL).

The codebase is located in `./event_gru_dvs_gesture` di


BPTT

```bash
python main.py --batch-size 64 --units 1024 \
    --num-layers 1 --frame-size 128 --method bptt  \
    --train-epochs 500 --frame-time 25 --rnn-type event-gru \
    --learning-rate 0.001 --lr-gamma 0.9 --lr-decay-epochs 100 \
    --event-agg-method mean --use-cnn --dropout 0.5 --zoneout 0 \
    --pseudo-derivative-width 1.7 --threshold-mean 0.25 --augment-data \
    --devices 0  --data ../data --cache ./cache
```


D-RTRL

```bash
python main.py --batch-size 64 --units 1024 \
    --num-layers 1 --frame-size 128 --method d-rtrl  \
    --train-epochs 500 --frame-time 25 --rnn-type event-gru \
    --learning-rate 0.001 --lr-gamma 0.9 --lr-decay-epochs 50 \
    --event-agg-method mean --use-cnn --dropout 0.5 --zoneout 0 \
    --pseudo-derivative-width 1.7 --threshold-mean 0.25 \
    --augment-data  --data ../data --cache ./cache --devices 1
```

ES-D-RTRL

```bash
python main.py --batch-size 64 --units 1024 \
    --num-layers 1 --frame-size 128 --method es-d-rtrl --etrace-decay 0.2  \
    --train-epochs 500 --frame-time 25 --rnn-type event-gru \
    --learning-rate 0.001 --lr-gamma 0.9 --lr-decay-epochs 100 \
    --event-agg-method mean --use-cnn --dropout 0.5 --zoneout 0 \
    --pseudo-derivative-width 1.7 --threshold-mean 0.25 \
    --augment-data  --data ../data --cache ./cache --devices 6
```



## RSNN classification on SHD dataset


See the codebase in [braintrace-shd-experiments](https://github.com/chaobrain/braintrace-shd-experiments). 




## Citation 

If you use this code or data, please cite:

```bibtex

@Article{Wang2026,
  author={Wang, Chaoming
          and Dong, Xingsi
          and Ji, Zilong
          and Xiao, Mingqing
          and Jiang, Jiedong
          and Liu, Xiao
          and Huan, Yuxiang
          and Wu, Si},
  title={Model-agnostic linear-memory online learning in spiking neural networks},
  journal={Nature Communications},
  year={2026},
  month={Jan},
  day={19},
  abstract={Spiking neural networks (SNNs) offer a promising paradigm for modeling brain dynamics and developing neuromorphic intelligence, yet an online learning system capable of training rich spiking dynamics over long horizons with low memory footprints has been missing. Existing online approaches either incur quadratic memory growth, sacrifice biological fidelity through oversimplified models, or lack end-to-end automated tooling. Here, we introduce BrainTrace, a model-agnostic, linear-memory, and automated online learning system for spiking neural networks. BrainTrace standardizes model specification to encompass diverse neuronal and synaptic dynamics; implements a linear-memory online learning rule by exploiting intrinsic properties of spiking dynamics; and provides a compiler that automatically generates optimized online-learning code for arbitrary user-defined models. Across diverse dynamics and tasks, BrainTrace achieves strong learning performance with a low memory footprint and high computational throughput. Critically, these properties enable online fitting of a whole-brain-scale Drosophila SNN that recapitulates region-level functional activity. By reconciling generality, efficiency, and usability, BrainTrace establishes a foundation for spiking network modeling at scale.},
  issn={2041-1723},
  doi={10.1038/s41467-026-68453-w},
  url={https://doi.org/10.1038/s41467-026-68453-w}
}

```


