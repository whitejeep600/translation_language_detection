import json
import random

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import LABEL_TO_INT, NUM_EPOCH, NUM_LABELS, \
    MAX_SENTENCE_LENGTH, D, LEARNING_RATE, SAVE_DIR, BATCH_SIZE
from dataset import TranslationDetectionDataset
from model import TranslationDetector
from readers import read_data
from utils import get_target_device, get_number_of_correct


class Trainer:
    def __init__(self, model, loss_function, optimizer, train_loader, validation_loader, num_epoch, save_dir):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.num_epoch = num_epoch
        self.save_path = save_dir + '/model.pt'
        self.best_accuracy = -1
        self.batch_losses = []
        pass

    def train(self):
        for i in range(self.num_epoch):
            print(f'Epoch number {i} out of {self.num_epoch}')
            self.train_iteration()
            self.eval_iteration()
        self.dump_losses()

    def train_iteration(self):
        self.model.train()
        progress = tqdm(total=len(self.train_loader.dataset) // BATCH_SIZE, desc="Processed batch")
        for i, batch in enumerate(self.train_loader):
            sentences = batch['text']
            labels = batch['label']
            predictions = self.model(sentences)
            current_loss = self.loss_function(predictions, labels)
            self.optimizer.zero_grad()
            current_loss.backward()
            self.optimizer.step()
            progress.update(1)
            if i % 512 == 0:
                self.batch_losses.append(current_loss.item())

    def eval_iteration(self):
        print('\nEvaluating')
        all_samples_no = len(self.validation_loader.dataset)
        correct = 0
        batch_losses = []
        self.model.eval()
        with torch.no_grad():
            progress = tqdm(total=len(self.validation_loader.dataset) // BATCH_SIZE, desc="Evaluated batch")
            for batch in iter(self.validation_loader):
                sentences = batch['text']
                labels = batch['label']
                predictions = self.model(sentences)
                correct += get_number_of_correct(predictions, labels)
                batch_losses.append(self.loss_function(predictions, labels))
                progress.update(1)
        average_loss = sum(batch_losses) / len(batch_losses)
        print(f'Average validation loss this epoch (per batch): {average_loss}')
        print(f'correct: {correct} out of {all_samples_no}. Epoch ended.\n')
        if correct > self.best_accuracy:
            print('Saving model to ' + self.save_path)
            torch.save(self.model.state_dict(), self.save_path)
            self.best_accuracy = correct

    def dump_losses(self):
        losses_json = json.dumps(self.batch_losses, indent=4)
        with open(SAVE_DIR + '/batch_losses', 'w') as file:
            file.write(losses_json)


def create_dataloader(split):
    dataset = TranslationDetectionDataset(split, LABEL_TO_INT)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=dataset.collate_fn)


if __name__ == '__main__':
    all_sentences = read_data()
    random.shuffle(all_sentences)
    validation_split = all_sentences[:len(all_sentences) // 10]
    train_split = all_sentences[len(all_sentences) // 10:]
    validation_loader = create_dataloader(validation_split)
    train_loader = create_dataloader(train_split)
    target_device = get_target_device()  # always using GPU if available
    model_no_device = TranslationDetector(D, MAX_SENTENCE_LENGTH, NUM_LABELS)
    model = model_no_device.to(target_device)
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    loss_function = torch.nn.CrossEntropyLoss()
    trainer = Trainer(model, loss_function, optimizer, train_loader, validation_loader, NUM_EPOCH, SAVE_DIR)
    trainer.train()
