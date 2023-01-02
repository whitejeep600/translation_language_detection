import json
import csv
from argparse import ArgumentParser, Namespace
from typing import List, Dict
import random

def main():
    train_data = []
    test_data = []
    for language in ['arabic','indonesian','chinese','japanese']:
        train_file = f"../../data/translated/from_{language}/train_{language}.json"
        test_file = f"../../data/translated/from_{language}/test_{language}.json"
        print(train_file,test_file)
        with open(train_file,'r') as f1:
            train_data += json.loads(f1.read())
        with open(test_file,'r') as f2:
            test_data += json.loads(f2.read())
    '''
    new_test_file = f"../../data/translated/new_test.json"
    with open(test_file,'r') as f2:
        mew_test_data += json.loads(f2.read())
    '''

    print(len(train_data))
    print(len(test_data))
    # print(len(new_test_data))

    random.shuffle(train_data)
    random.shuffle(test_data)
    # random.shuffle(new_test_data)
    
    csv_train_data = []
    csv_test_data = []
    csv_train_data.append(["paragraph","label"])
    csv_test_data.append(["paragraph","label"])
    for data in train_data:
        csv_train_data.append([data['paragraph'],data['language']])
    for data in test_data:
        csv_test_data.append([data['paragraph'],data['language']])
    with open("./train.csv","w",newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_train_data)
    with open("./test.csv","w",newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_test_data)

if __name__ == "__main__":
    main()
