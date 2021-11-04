import torch
from torch.autograd import Variable 
import utils
from attack_method import Adversarial_methods
import torch.nn as nn
class Adversarial_Trainings(object):
    def __init__(self,  m_repeat,trainloader, use_cuda, optimizer, attack_iters, net, epsilon, alpha, learning_rate_decay_start, learning_rate_decay_every, learning_rate_decay_rate, lr):
        self.trainloader = trainloader
        self.m_repeat = m_repeat
        self.use_cuda = use_cuda
        self.optimizer = optimizer
        self.attack_iters = attack_iters
        self.net = net
        self.epsilon = epsilon
        self.alpha = alpha
        self.learning_rate_decay_start = learning_rate_decay_start
        self.learning_rate_decay_every = learning_rate_decay_every
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.lr = lr


    def fast_free_advTraining(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0

        if epoch > self.learning_rate_decay_start and self.learning_rate_decay_start >= 0:
            frac = (epoch - self.learning_rate_decay_start) // self.learning_rate_decay_every
            decay_factor = self.learning_rate_decay_rate ** frac
            current_lr = self.lr * decay_factor
            utils.set_lr(self.optimizer, current_lr)  # set the decayed rate
        else:
            current_lr = self.lr
        print('learning_rate: %s' % str(current_lr))

        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            self.optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            adversarial_attack = Adversarial_methods(inputs, targets, self.attack_iters, self.net, self.epsilon, self.alpha) 
            delta = adversarial_attack.fgsm()

            outputs = self.net(torch.clamp(inputs + delta, 0, 1))
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            utils.clip_gradient(self.optimizer, 0.1)
            self.optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            utils.progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        Train_acc = 100.*correct/total
        print("The final free_fast accuracy is :", Train_acc)
