import torch


def get_target_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def get_number_of_correct(predictions, languages):
    return len([i for i in range(len(predictions)) if torch.argmax(predictions[i]) == languages[i]])