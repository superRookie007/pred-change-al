'''
Experiments on Cifar10 using resnet18. It shows an example of computing predition changes duing 
training for active learning.
'''
import time
import argparse
import csv

import torch
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

# from utils import utils
from utils.utils import str2bool
from utils import utils
from datasets import cifar10
from torchvision.models import resnet18

from utils.logger import Logger
from sampling_methods.pred_changes import PredictionChange

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='CIFAR10 Example')
    parser.add_argument('--root', type=str, metavar='S',
                        help='Path to the root.')
    parser.add_argument('--init-num-labelled', type=int, default=None, metavar='N',
                        help='Initial number of labelled examples.')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='total batch size for training (default: 100)')
    parser.add_argument('--init-epochs', type=int, metavar='N',
                        help='number of epochs to train for active learning.')
    parser.add_argument('--train-on-updated', default=False, type=str2bool, metavar='BOOL',
                        help='Train on updated data? (default: False)')
    parser.add_argument('--active-learning', default=False, type=str2bool, metavar='BOOL',
                        help='Run proposed active learning? (default: False)')            
    parser.add_argument('--skip', type=int, default=0, metavar='N',
                        help='Skip the first N epochs when computing the accumulated prediction changes.')

    parser.add_argument('--test-batch-size', type=int, default=500, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, metavar='N',
                        help='number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--seed', type=int, metavar='S',
                        help='Seed for random number generator.')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num-workers', type=int, default=1, metavar='N',
                        help='Number of workers for dataloader (default: 1)')
                        
    parser.add_argument('--num-to-sample', type=int, metavar='N',
                        help='Number of unlabelled exmples to be sampled')
    parser.add_argument('--validate', default=False, type=str2bool, metavar='BOOL',
                        help='Use validation set instead of test set? (default: False)')
    parser.add_argument('--output', default='default_ouput.csv', type=str, metavar='S',
                        help='File name for the output.')


    args = parser.parse_args()
    torch.manual_seed(args.seed) # set seed for pytorch
    use_cuda = torch.cuda.is_available()

    args.num_classes = 10

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}
    
##################### Active learning sampling using prediction change (fluctuation) ###############################
    if args.active_learning:
        train_dataset = cifar10.CIFAR10(
            root=args.root,
            dataset='train',
            init_n_labeled=args.init_num_labelled,
            seed=args.seed,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
            target_transform=None,
            indices_name=None) #initialise indices

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=SubsetRandomSampler(train_dataset.l_indices),
            **kwargs)
        
        test_loader = torch.utils.data.DataLoader(
        cifar10.CIFAR10(args.root, 'test', seed=args.seed, transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

        # train on the initial labelled set
        global_step = 0
        model = resnet18(pretrained=False, progress=False, num_classes=args.num_classes).to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        method = PredictionChange(u_indices=train_dataset.u_indices, model=model, dataset=train_dataset, data_name='cifar10')

        logs = {
            "train_losses": [],
            "train_acces": [],
            "test_acces": [],
            "pred_changes": []
        }
        logger = Logger(logs)

        for epoch in range(1, args.init_epochs + 1):
            start = time.time()
            global_step, train_loss, train_acc = utils.train(args, model, device, train_loader, optimizer, epoch, global_step)
            print('Training one epoch took: {:.4f} seconds.\n'.format(time.time()-start))
            test_acc, _ = utils.test(model, device, test_loader)
            
            print('Computing prediction changes...')
            pred_change = method.compute_pred_changes(model)

            logger.append(train_losses=train_loss, train_acces=train_acc, test_acces=test_acc, pred_changes=pred_change)

        # save the logs
        train_dataset.save_logs(logger.logs)


############################### Training on updated indices #########################################
    if args.train_on_updated:
        import os
        # create Dataset object and load initial indices.
        train_dataset = cifar10.CIFAR10(
            root=args.root,
            dataset='train',
            init_n_labeled=args.init_num_labelled,
            seed=args.seed,
            download=False,
            transform=transforms.Compose([transforms.ToTensor()]),
            target_transform=None,
            indices_name="init_indices.npz") #load initial indices from file

        logs_path = os.path.join(train_dataset.init_folder, 'logs.npz')
        print("Updating indices using log file: {}...".format(logs_path))
        start = time.time()
        # sampling using proposed prediction change method
        method = PredictionChange(
            u_indices=train_dataset.u_indices,
            dataset=train_dataset,
            data_name='CIFAR-10')

        sample = method.select_batch_from_logs(
            N=args.num_to_sample,
            skip=args.skip,
            path=logs_path,
            key="pred_changes")
        # update and save updated indices
        filename_updated_indices = "updated_indices_N_{}_skip_{}".format(args.num_to_sample, args.skip)
        method.update_indices(dataset=train_dataset, indices=sample, filename=filename_updated_indices)
        print('Active learning sampling took: {:.4f} seconds.\n'.format(time.time()-start))
    

        print("Training on updated labelled training set...")
        train_dataset = cifar10.CIFAR10(
            root=args.root,
            dataset='train',
            init_n_labeled=args.init_num_labelled,
            seed=args.seed,
            download=False,
            transform=transforms.Compose([transforms.ToTensor()]),
            target_transform=None,
            indices_name=filename_updated_indices+".npz") #load updated indices from file

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=SubsetRandomSampler(train_dataset.l_indices),
            **kwargs)


        if args.validate:
            test_or_validate = 'validation'
        else:
            test_or_validate = 'test'
        test_loader = torch.utils.data.DataLoader(
            cifar10.CIFAR10(args.root, test_or_validate, seed=args.seed, transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

        model = resnet18(pretrained=False, progress=False, num_classes=args.num_classes).to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        global_step = 0

        for epoch in range(1, args.epochs + 1):
            start = time.time()
            ###
            global_step, _, _ = utils.train(args, model, device, train_loader, optimizer, epoch, global_step)
            print('\nTraining one epoch took: {:.4f} seconds.\n'.format(time.time()-start))
            ###
            test_acc, _ = utils.test(model, device, test_loader)

        with open(args.output, 'a') as write_file:
            writer = csv.writer(write_file)
            writer.writerow([args.seed, test_acc])


        # if args.model_name:
        #     import os
        #     model_path = os.path.join(folder, args.model_name)
        #     torch.save(model.state_dict(), model_path+".pt")        



if __name__ == '__main__':
    main()
