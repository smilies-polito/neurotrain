import argparse
from datetime import datetime
import os
import logging

import torch
from bisect import bisect_right
import pandas as pd

from load_dataset import load_dataset
from utils import setup_logging, save_checkpoint, to_one_hot, reproducible_config




def train(epoch, lr):
    model.train()
    loss_total = 0

    # Clear layerwise statistics
    if args.print_stats:
        for m in model.modules():
            if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                m.clear_stats()

    for batch_idx, (data, target) in enumerate(train_loader):
        target_onehot = to_one_hot(target, num_classes)
        if is_cuda:
            data, target = data.cuda(), target.cuda()
            target_onehot = target_onehot.cuda()

        # Clear accumulated gradient
        model.optim_zero_grad()

        error_percent, loss = model(data, target, target_onehot, args.time_window)
        loss_total += loss * target.size(0)

    # Format and print debug string
    loss_average = loss_total / len(train_loader.dataset)

    string_print = 'Train epoch={}, lr={:.2e}, loss_local={:.4f}, error={:.3f}%, mem={:.0f}MiB, max_mem={:.0f}MiB\n'.format(
        epoch,
        lr,
        loss_average,
        error_percent,
        torch.cuda.memory_allocated() / 1e6,
        torch.cuda.max_memory_allocated() / 1e6)

    # To store layer-wise errors
    err_list = []
    if args.print_stats:
        for m in model.modules():
            if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                states, err = m.print_stats()
                string_print += states
                err_list.append(err)

    logging.info(string_print)
    err_list.append(error_percent)
    return loss_average, error_percent, err_list


def test():
    # Change to the evaluation mode
    model.eval()

    # Clear layerwise statistics
    if args.print_stats:
        for m in model.modules():
            if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                m.clear_stats()

    for data, target in test_loader:
        target_onehot = to_one_hot(target, num_classes)
        if is_cuda:
            data, target = data.cuda(), target.cuda()
            target_onehot = target_onehot.cuda()

        with torch.no_grad():
            error_percent, _ = model(data, target, target_onehot, args.time_window)

    # Format and print debug string
    string_print = 'Test error={:.3f}%\n'.format(error_percent)

    # To store layer-wise errors
    err_list = []
    if args.print_stats:
        for m in model.modules():
            if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv):
                states, err = m.print_stats()
                string_print += states
                err_list.append(err)
    logging.info(string_print)
    err_list.append(error_percent)
    return error_percent, err_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Code for 'Deep Spike Learning with Local Classifiers'")
    parser.add_argument('--model', default='CIFARCNN',
                        help='models including MNIST(MNISTDNN, MNISTCNN),FashionMNSIT(FashionDNN, FashionCNN), CIFAR10(CIFARCNN), SVHN(SVHNCNN)')
    parser.add_argument('--dataset', default='CIFAR10', help='datasets including MNIST, FashionMNIST, SVHN and CIFAR10')
    parser.add_argument('--batch-size', type=int, default=100, help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate (default: 5e-4)')
    parser.add_argument('--lr-decay-milestones', nargs='+', type=int, default=[100, 200],
                        help='decay learning rate at these milestone epochs')
    parser.add_argument('--lr-decay-fact', type=float, default=0.1,
                        help='learning rate decay factor to use at milestone epochs (default: 0.1)')
    parser.add_argument('--dim-in-decoder', type=int, default=1024,
                        help='input dimension of decoder_y used in pred loss(default: 1024)')
    parser.add_argument('--thresh', type=float, default=1, help='neuronal threshold (default: 1)')
    parser.add_argument('--time-window', type=int, default=10, help='total time steps (default: 10)')
    parser.add_argument('--tau', type=int, default=1, help='tau factor (default: 1)')
    parser.add_argument('--fa', action='store_true', default=False, help='enable feedback alignment')
    parser.add_argument('--random-matrix', action='store_true', default=False,
                        help='use layer-wise classifiers with random weights.')
    parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1234)')
    parser.add_argument('--save-path', default='', type=str, help='the directory used to save the trained models')
    parser.add_argument('--resume', action='store_true', default=False, help='load checkpoint.')
    parser.add_argument('--print-stats', action='store_true', default=True,
                        help='print layerwise statistics during training with local loss')
    parser.add_argument('--encoding', default='real', help='spike encoding: real, latency (default: real)')
    parser.add_argument('--learning-rule', default='ELL', help='Learning algorithms: FELL, BELL, ELL (default: ELL)')
    args = parser.parse_args()
    # Create dir for saving models
    if args.save_path is '':
        save_path = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    else:
        save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Logging settings
    setup_logging(os.path.join(save_path, 'log.txt'))
    logging.info('args:' + str(args))
    logging.info('saving to:' + str(save_path))

    if args.learning_rule == 'FELL':
        from local_linear_FELL import LocalLossBlockLinear
        from local_conv_FELL import LocalLossBlockConv
    elif args.learning_rule == 'BELL':
        from local_linear_BELL import LocalLossBlockLinear
        from local_conv_BELL import LocalLossBlockConv
    elif args.learning_rule == 'ELL':
        from local_linear_ELL import LocalLossBlockLinear
        from local_conv_ELL import LocalLossBlockConv
    else:
        raise Exception('Unrecognized learning rule.')

    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')

    # Reproducibility
    reproducible_config(seed=args.seed, is_cuda=is_cuda)

    # Load datasets
    train_loader, test_loader, num_classes = load_dataset(dataset=args.dataset, batch_size=args.batch_size,
                                                          dataset_path='../data', is_cuda=is_cuda)

    params = {
        'num_classes': num_classes,
        'threshold': args.thresh,
        'decay': torch.exp(torch.tensor(-1 / args.tau, device=device)),
        'bias': False,
        'random_matrix': args.random_matrix,
        'fa': args.fa,
        'print_stats': args.print_stats,
        'dim_decoder': args.dim_in_decoder,
        'encoding': args.encoding,
        'learning_rule': args.learning_rule
    }

    # Load spiking model
    if args.model == 'MNISTDNN':
        from models.MNISTDNN import MNISTDNN

        model = MNISTDNN(params)
    elif args.model == 'MNISTCNN':
        from models.MNISTCNN import MNISTCNN

        model = MNISTCNN(params)
    elif args.model == 'FashionDNN':
        from models.FashionDNN import FashionDNN

        model = FashionDNN(params)
    elif args.model == 'FashionCNN':
        from models.FashionCNN import FashionCNN

        model = FashionCNN(params)
    elif args.model == 'CIFARCNN':
        from models.CIFARCNN import CIFARCNN

        model = CIFARCNN(params)
    elif args.model == 'SVHNCNN':
        from models.SVHNCNN import SVHNCNN

        model = SVHNCNN(params)
    else:
        raise Exception('No valid model is specified.')

    if is_cuda:
        model.cuda()

    # # Define optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.)

    model.set_learning_rate(args.lr)
    # logging.info('Model has {} parameters influenced by global loss'.format(count_parameters(model)))

    start_epoch = 1
    best_error = 100.
    if args.resume:
        # Load checkpoint.
        logging.info('==> Resuming from checkpoint..')
        checkpoint = torch.load(os.path.join(save_path, 'checkpoint.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
        best_error = checkpoint['best_error']
        start_epoch = checkpoint['epoch']

    # For storing results
    train_res = pd.DataFrame()
    test_res = pd.DataFrame()
    # Train loop
    for epoch in range(start_epoch, args.epochs + 1):
        # Adjust learning rate
        lr = args.lr * args.lr_decay_fact ** bisect_right(args.lr_decay_milestones, (epoch - 1))
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        model.set_learning_rate(lr)

        train_loss, train_error, train_err_list = train(epoch, lr)

        test_error, test_err_list = test()

        train_res[str(epoch)] = train_err_list
        test_res[str(epoch)] = test_err_list

        # Save checkpoints and the best model
        is_best = test_error < best_error
        best_error = min(best_error, test_error)
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_error': best_error,},
                        is_best, save_path)
        train_res.to_csv(os.path.join(save_path, 'train_res.csv'), index=False)
        test_res.to_csv(os.path.join(save_path, 'test_res.csv'), index=False)
    logging.info(f'best acc :{100 - best_error}')
