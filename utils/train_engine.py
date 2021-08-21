import os
from datetime import datetime

import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import numpy as np

#from datasets.PoseEstimation import PoseEstimationDataset
from utils.loss import JointsMSELoss, JointsOHKMMSELoss
from misc.checkpoint import save_checkpoint, load_checkpoint
from misc.utils import flip_tensor, flip_back
from misc.visualization import save_images
from models.hrnet.hrnet import HRNet


class Train(object):
    """
    Train class 
    """

    def __init__(self,
                 exp_name,
                 ds_train,
                 ds_val,
                 epochs=210,
                 batch_size=16,
                 num_workers=4,
                 loss='JointsMSELoss',
                 lr=0.001,
                 lr_decay=True,
                 lr_decay_steps=(170, 200),
                 lr_decay_gamma=0.1,
                 optimizer='Adam',
                 weight_decay=0.,
                 momentum=0.9,
                 nesterov=False,
                 pretrained_weight_path=None,
                 checkpoint_path=None,
                 log_path='./logs',
                 use_tensorboard=False,
                 model_c=48,
                 model_nof_joints=6,
                 model_bn_momentum=0.1,
                 flip_test_images=False,
                 device=None,
                 train_dataset_dir='not added',
                 val_dataset_dir='not added'
                 ):
        """
        Initializes a new ExcavatorTrain object which extends the parent Train class.
        The initialization function calls the init function of the Train class.

        Args:
            exp_name (str):  experiment name.
            ds_train (HumanPoseEstimationDataset): train dataset.
            ds_val (HumanPoseEstimationDataset): validation dataset.
            epochs (int): number of epochs.
                Default: 210
            batch_size (int): batch size.
                Default: 16
            num_workers (int): number of workers for each DataLoader
                Default: 4
            loss (str): loss function. Valid values are 'JointsMSELoss' and 'JointsOHKMMSELoss'.
                Default: "JointsMSELoss"
            lr (float): learning rate.
                Default: 0.001
            lr_decay (bool): learning rate decay.
                Default: True
            lr_decay_steps (tuple): steps for the learning rate decay scheduler.
                Default: (170, 200)
            lr_decay_gamma (float): scale factor for each learning rate decay step.
                Default: 0.1
            optimizer (str): network optimizer. Valid values are 'Adam' and 'SGD'.
                Default: "Adam"
            weight_decay (float): weight decay.
                Default: 0.
            momentum (float): momentum factor.
                Default: 0.9
            nesterov (bool): Nesterov momentum.
                Default: False
            pretrained_weight_path (str): path to pre-trained weights (such as weights from pre-train on imagenet).
                Default: None
            checkpoint_path (str): path to a previous checkpoint.
                Default: None
            log_path (str): path where tensorboard data and checkpoints will be saved.
                Default: "./logs"
            use_tensorboard (bool): enables tensorboard use.
                Default: True
            model_c (int): hrnet parameters - number of channels.
                Default: 48
            model_nof_joints (int): hrnet parameters - number of joints.
                Default: 6
            model_bn_momentum (float): hrnet parameters - path to the pretrained weights.
                Default: 0.1
            flip_test_images (bool): flip images during validating.
                Default: True
            device (torch.device): device to be used (default: cuda, if available).
                Default: None
        """
        super(Train, self).__init__()
    
        self.exp_name = exp_name
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.epochs = epochs
        self.batch_size=batch_size
        self.num_workers = num_workers
        self.loss = loss
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_gamma = lr_decay_gamma
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.pretrained_weight_path = pretrained_weight_path
        self.checkpoint_path = checkpoint_path
        self.log_path = os.path.join(log_path, self.exp_name)
        self.use_tensorboard = use_tensorboard
        self.model_c=model_c
        self.model_nof_joints = model_nof_joints
        self.model_bn_momentum = model_bn_momentum
        self.flip_test_images = flip_test_images
        self.device = None
        
        self.epoch = 0
        
        # added here so they are printed in parameters.txt
        self.train_dataset_dir = train_dataset_dir 
        self.val_dataset_dir = val_dataset_dir
        
        # torch device
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')

        print(self.device, '\n')

        
        os.makedirs(self.log_path, 0o755, exist_ok=False)  # exist_ok=False to avoid overwriting

        #if self.use_tensorboard:
        #    self.summary_writer = tb.SummaryWriter(self.log_path)

        #
        # write all experiment parameters in parameters.txt and in tensorboard text field
        self.parameters = [x + ': ' + str(y) + '\n' for x, y in locals().items()]
        with open(os.path.join(self.log_path, 'parameters.txt'), 'w') as fd:
            fd.writelines(self.parameters)
        #if self.use_tensorboard:
        #    self.summary_writer.add_text('parameters', '\n'.join(self.parameters))

        #
        # load model
        self.model = HRNet(c=self.model_c, nof_joints=self.model_nof_joints,
                           bn_momentum=self.model_bn_momentum).to(self.device)

        #
        # define loss and optimizers
        if self.loss == 'JointsMSELoss':
            self.loss_fn = JointsMSELoss().to(self.device)
        elif self.loss == 'JointsOHKMMSELoss':
            self.loss_fn = JointsOHKMMSELoss().to(self.device)
        else:
            raise NotImplementedError

        if optimizer == 'SGD':
            self.optim = SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                             momentum=self.momentum, nesterov=self.nesterov)
        elif optimizer == 'Adam':
            self.optim = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

        #
        # load pre-trained weights (such as those pre-trained on imagenet)

        print('\n##\nLoading pre-trained weights\nLoad from: ', os.path.abspath(self.pretrained_weight_path))
        
        if self.pretrained_weight_path is not None:
            # loading pretrained_weights partially (under development)
            pretrained_dict = torch.load(self.pretrained_weight_path, map_location=self.device)
            model_dict = self.model.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
            #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.model.load_state_dict(model_dict)            

            #self.model.load_state_dict(torch.load(self.pretrained_weight_path, map_location=self.device), strict=False)
            print('Pre-trained weights loaded.')

        #
        # load previous checkpoint
        if self.checkpoint_path is not None:
            print('Loading checkpoint %s...' % self.checkpoint_path)
            if os.path.isdir(self.checkpoint_path):
                path = os.path.join(self.checkpoint_path, 'checkpoint_last.pth')
            else:
                path = self.checkpoint_path
            self.starting_epoch, self.model, self.optim, self.params = load_checkpoint(path, self.model, self.optim,
                                                                                       self.device)
        else:
            self.starting_epoch = 0

        if lr_decay:
            self.lr_scheduler = MultiStepLR(self.optim, list(self.lr_decay_steps), gamma=self.lr_decay_gamma,
                                            last_epoch=self.starting_epoch if self.starting_epoch else -1)

        #
        # load train and val datasets
        self.dl_train = DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True,
                                   num_workers=self.num_workers, drop_last=True)
        self.len_dl_train = len(self.dl_train)
        
        print('train dataset length: ', self.len_dl_train*self.batch_size)

        # dl_val = DataLoader(self.ds_val, batch_size=1, shuffle=False, num_workers=num_workers)
        self.dl_val = DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.len_dl_val = len(self.dl_val)

        print('val dataset length: ', self.len_dl_val*self.batch_size)

        #
        # initialize variables
        self.mean_loss_train = 0.
        self.mean_acc_train = 0.
        self.mean_loss_val = 0.
        self.mean_acc_val = 0.
        self.mean_mAP_val = 0.

        self.best_loss = None
        self.best_acc = None
        self.best_mAP = None
        


    def _train(self):

        num_samples = self.len_dl_train * self.batch_size

        self.model.train()
        for step, (image, target, target_weight, joints_data) in enumerate(tqdm(self.dl_train, desc='Training')):
            image = image.to(self.device)
            target = target.to(self.device)
            target_weight = target_weight.to(self.device)

            self.optim.zero_grad()

            output = self.model(image)

            loss = self.loss_fn(output, target, target_weight)

            loss.backward()

            self.optim.step()

            # Evaluate accuracy
            # Get predictions on the resized images (given as input)
            accs, avg_acc, cnt, joints_preds, joints_target = \
                self.ds_train.evaluate_accuracy(output, target)


            self.mean_loss_train += loss.item()
            self.mean_acc_train += avg_acc.item()

            #if self.use_tensorboard:
            #    self.summary_writer.add_scalar('train_loss', loss.item(),
            #                                   global_step=step + self.epoch * self.len_dl_train)
            #    self.summary_writer.add_scalar('train_acc', avg_acc.item(),
            #                                   global_step=step + self.epoch * self.len_dl_train)
            #    if step == 0:
            #        save_images(image, target, joints_target, output, joints_preds, joints_data['joints_visibility'],
            #                    self.summary_writer, step=step + self.epoch * self.len_dl_train, prefix='train_')

        self.mean_loss_train /= len(self.dl_train)
        self.mean_acc_train /= len(self.dl_train)

        # this is the loss and accuracy or each epoch 
        print('\nTrain: Loss %f - Accuracy %f\n' % (self.mean_loss_train, self.mean_acc_train))
        
        return self.mean_loss_train, self.mean_acc_train
    
    
    def _val(self):
        
        num_samples = len(self.ds_val)
        
        #all_preds = np.zeros((num_samples, self.model_nof_joints, 3), dtype=np.float32)
        #all_boxes = np.zeros((num_samples, 6), dtype=np.float32)

        self.model.eval()
        
        with torch.no_grad():
            for step, (image, target, target_weight, joints_data) in enumerate(tqdm(self.dl_val, desc='Validating')):
                image = image.to(self.device)
                target = target.to(self.device)
                target_weight = target_weight.to(self.device)

                output = self.model(image)
                
                if self.flip_test_images:
                    image_flipped = flip_tensor(image, dim=1)
                    output_flipped = self.model(image_flipped)
                    
                    output_flipped = flip_back(output_flipped, self.ds_val.flip_pairs)
                    output = (output + output_flipped) * 0.5

                loss = self.loss_fn(output, target, target_weight)

                # Evaluate accuracy
                accs, avg_acc, cnt, joints_preds, joints_target = \
                    self.ds_train.evaluate_accuracy(output, target)

                self.mean_loss_val += loss.item()
                self.mean_acc_val += avg_acc.item()
                

        self.mean_loss_val /= len(self.dl_val)
        self.mean_acc_val /= len(self.dl_val)
        
        print('\nValidation: Loss %f - Accuracy %f' % (self.mean_loss_val, self.mean_acc_val))
        
        return self.mean_loss_val, self.mean_acc_val


    def _checkpoint(self):

        save_checkpoint(path=os.path.join(self.log_path, 'checkpoint_last.pth'), epoch=self.epoch + 1, model=self.model,
                        optimizer=self.optim, params=self.parameters)
        
        checkpoint_epoch = str(self.epoch + 1)
        save_checkpoint(path=os.path.join(self.log_path, 'checkpoint_epoch_' + checkpoint_epoch + '.pth'), epoch=self.epoch +1, model=self.model, optimizer=self.optim, params=self.parameters)


        if self.best_loss is None or self.best_loss > self.mean_loss_val:
            self.best_loss = self.mean_loss_val
            print('best_loss %f at epoch %d' % (self.best_loss, self.epoch + 1))
            save_checkpoint(path=os.path.join(self.log_path, 'checkpoint_best_loss.pth'), epoch=self.epoch + 1,
                            model=self.model, optimizer=self.optim, params=self.parameters)
        if self.best_acc is None or self.best_acc < self.mean_acc_val:
            self.best_acc = self.mean_acc_val
            print('best_acc %f at epoch %d' % (self.best_acc, self.epoch + 1))
            save_checkpoint(path=os.path.join(self.log_path, 'checkpoint_best_acc.pth'), epoch=self.epoch + 1,
                            model=self.model, optimizer=self.optim, params=self.parameters)
        if self.best_mAP is None or self.best_mAP < self.mean_mAP_val:
            self.best_mAP = self.mean_mAP_val
            print('best_mAP %f at epoch %d' % (self.best_mAP, self.epoch + 1))
            save_checkpoint(path=os.path.join(self.log_path, 'checkpoint_best_mAP.pth'), epoch=self.epoch + 1,
                            model=self.model, optimizer=self.optim, params=self.parameters)

    def run(self):
        """
        Runs the training.
        """

        print('\nTraining started @ %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        with open(self.log_path + "/model.log", "a") as f:
            f.write(f"{self.exp_name},{'epoch'},{'loss_train'},{'acc_train'},{'loss_val'},{'acc_val'}\n")
        
        # start training
        for self.epoch in range(self.starting_epoch, self.epochs):
            print('\n------------------------------------\n')
            print('\nEpoch %d of %d @ %s' % (self.epoch + 1, self.epochs, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            self.mean_loss_train = 0.
            self.mean_loss_val = 0.
            self.mean_acc_train = 0.
            self.mean_acc_val = 0.
            self.mean_mAP_val = 0.

            #
            # Train (run through the dataset once)
            loss_train, acc_train = self._train()
            with open(self.log_path + "/model.log", "a") as f:
                f.write(f"{self.exp_name},{self.epoch + 1},{round(float(loss_train) ,7)},{round(float(acc_train) ,2)},")

            #
            # Val
            loss_val, acc_val = self._val()            
            with open(self.log_path + "/model.log", "a") as f:
                f.write(f"{round(float(loss_val) ,7)},{round(float(acc_val) ,2)}\n")
            
            #
            # LR Update
            if self.lr_decay:
                self.lr_scheduler.step()

            #
            # Checkpoint
            self._checkpoint()

        print('\nTraining ended @ %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))