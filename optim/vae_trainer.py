from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from utils.misc import binary_cross_entropy
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np


class VAETrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.f = open('./log/vae.txt')
    def train(self, dataset: BaseADDataset, vae: BaseNet):

        # Get train data loader
        train_loader, _ = dataset.loaders(
            batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device
        vae = vae.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(vae.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.lr_milestones, gamma=0.1)
        
        # Training
        print('Starting pretraining...')
        print('Starting pretraining...',file = self.f)
        start_time = time.time()
        vae.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
               print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]),file = self.f)

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)
                inputs = inputs.view(inputs.size(0), -1)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                rec = vae(inputs)

                likelihood = -binary_cross_entropy(rec, inputs)
                elbo = likelihood - vae.kl_divergence

                # Overall loss
                loss = -torch.mean(elbo)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            print(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')
            
            print(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |',file = self.f)
        self.train_time = time.time() - start_time
        print('Pretraining Time: {:.3f}s'.format(self.train_time),file = self.f)
        print('Finished pretraining.',file = self.f)

        return vae

    def test(self, dataset: BaseADDataset, vae: BaseNet):
        f  = open('test_vae.txt','w')

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device
        vae = vae.to(self.device)

        # Testing
        print('Starting testing...')
        print('Starting testing...',file = f)
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        vae.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, _, idx = data
                inputs, labels, idx = inputs.to(self.device), labels.to(self.device), idx.to(self.device)

                inputs = inputs.view(inputs.size(0), -1)

                rec = vae(inputs)
                likelihood = -binary_cross_entropy(rec, inputs)
                scores = -likelihood  # negative likelihood as anomaly score

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                # Overall loss
                elbo = likelihood - vae.kl_divergence
                loss = -torch.mean(elbo)

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)

        # Log results
        print('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        print('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        print('Test Time: {:.3f}s'.format(self.test_time))
        print('Finished testing variational autoencoder.')
        print('Test Loss: {:.6f}'.format(epoch_loss / n_batches),file = f)
        print('Test AUC: {:.2f}%'.format(100. * self.test_auc),file = f)
        print('Test Time: {:.3f}s'.format(self.test_time),file = f)
        print('Finished testing variational autoencoder.',file = f)
