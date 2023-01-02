import json
import csv
from argparse import ArgumentParser, Namespace
from typing import List, Dict
import random

def translator_name(translator):
    if translator == 'mbart' or translator == 'mBART' or translator=='facebook/mbart-large-50-many-to-one-mmt':
        return 'mbart' 
    elif translator == 'staka':
        return 'staka'
    elif translator == 'helsinki' or translator == 'Helsinki-NLP/opus-mt-ar-en':
        return 'helsinki'
    else:
        return None

def main():
    test_data = []
    for language in ['arabic','indonesian','chinese','japanese']:
        test_file = f"../../data/translated/from_{language}/test_{language}.json"
        print(test_file)
        with open(test_file,'r') as f2:
            test_data += json.loads(f2.read())

    print(len(test_data))
    
    csv_test_data = {}
    for language in ['arabic','chinese','indonesian','japanese']:
        csv_test_data[language] = {}
        if language != 'japanese':
            csv_test_data[language]['mbart'] = []
            csv_test_data[language]['helsinki'] = []
        else:
            csv_test_data[language]['mbart'] = []
            csv_test_data[language]['staka'] = []
    for language in csv_test_data:
        for translator in csv_test_data[language]:
            csv_test_data[language][translator].append(["paragraph","label"])
            for data in test_data:
                if data['language']==language and translator_name(data['translator'])==translator:
                    csv_test_data[language][translator].append([data['paragraph'],data['language']])

            with open(f"./test_{language}_{translator}.csv","w",newline='') as file:
                print(language,translator,len(csv_test_data[language][translator])-1)
                writer = csv.writer(file)
                writer.writerows(csv_test_data[language][translator])

if __name__ == "__main__":
    main()
