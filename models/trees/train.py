import random

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import label_to_int, NUM_EPOCH, NUM_LABELS, \
    MAX_SENTENCE_LENGTH, D, LEARNING_RATE, SAVE_DIR
from dataset import TranslationDetectionDataset
from model import TranslationDetector
from readers import read_data


class Trainer:
    def __init__(self, model, loss_function, optimizer, train_loader, validation_loader, num_epoch, save_dir):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.num_epoch = num_epoch
        self.save_dir = save_dir
        self.best_accuracy = -1
        pass

    def get_number_of_correct(self, predictions, languages):
        return len([i for i in range(len(predictions)) if torch.argmax(predictions[i]) == languages[i]])

    def train(self):
        for i in range(self.num_epoch):
            print(f'Epoch number {i} out of {self.num_epoch}')
            self.train_iteration()
            self.eval_iteration()

    def train_iteration(self):
        self.model.train()
        progress = tqdm(total=len(self.train_loader.dataset), desc="Processed batch")
        for batch in iter(self.train_loader):
            sentences = batch['text']
            labels = batch['label']
            predictions = self.model(sentences)
            current_loss = self.loss_function(predictions, labels)
            self.optimizer.zero_grad()
            current_loss.backward()
            self.optimizer.step()
            progress.update(1)
            print(f'loss: {current_loss.item()}')

    def eval_iteration(self):
        all_samples_no = len(self.validation_loader.dataset)
        correct = 0
        batch_losses = []
        self.model.eval()
        with torch.no_grad():
            for batch in iter(self.validation_loader):
                sentences = batch['text']
                labels = batch['label']
                predictions = self.model(sentences)
                correct += self.get_number_of_correct(predictions, labels)
                batch_losses.append(self.loss_function(predictions, labels))
        average_loss = sum(batch_losses) / len(batch_losses)
        print(f'Average validation loss this epoch (per batch): {average_loss}\n')
        print(f'correct: {correct} out of {all_samples_no}. Epoch ended\n')
        if correct > self.best_accuracy:
            print('Saving model to ' + self.save_dir)
            torch.save(self.model.state_dict(), self.save_dir)
            self.best_accuracy = correct


def create_dataloader(split):
    dataset = TranslationDetectionDataset(split, label_to_int)
    return DataLoader(dataset, batch_size=32, shuffle=True,
                      collate_fn=dataset.collate_fn)


if __name__ == '__main__':
    all_sentences = read_data()[:10]
    random.shuffle(all_sentences)
    validation_split = all_sentences[:len(all_sentences) // 10]
    train_split = all_sentences[len(all_sentences) // 10:]
    validation_loader = create_dataloader(validation_split)
    train_loader = create_dataloader(train_split)
    target_device = "cuda:0" if torch.cuda.is_available() else "cpu"  # always using GPU if available
    model_no_device = TranslationDetector(D, MAX_SENTENCE_LENGTH, NUM_LABELS)
    model = model_no_device.to(target_device)
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    loss_function = torch.nn.CrossEntropyLoss()
    trainer = Trainer(model, loss_function, optimizer, train_loader, validation_loader, NUM_EPOCH, SAVE_DIR)
    trainer.train()
