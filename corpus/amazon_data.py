import json
import gzip
import argparse
import codecs
import random
import os


def _removeNonAscii(s): return "".join(i for i in s if ord(i) < 128)


def generate_strict_json(zippath, jsonpath):
    def parse(path):
        g = gzip.open(path, 'r')
        for l in g:
            yield json.dumps(eval(l))

    f = open(jsonpath, 'w', encoding='utf-8', errors='ignore')
    for l in parse(zippath):
        f.write(_removeNonAscii(l) + '\n')


def generate_dataset(jsonpath, datadir):
    def parse(path):
        with open(jsonpath, 'r', encoding='utf-8', errors='ignore') as f:
            for l in f:
                yield json.loads(l)

    with open(os.path.join(datadir, 'train.from'), 'w') as qfile:
        with open(os.path.join(datadir, 'train.to'), 'w') as ansfile:
            with open(os.path.join(datadir, 'test.from'), 'w') as qtestfile:
                with open(os.path.join(datadir, 'test.to'), 'w') as anstestfile:
                    with open(os.path.join(datadir, 'dev.from'), 'w') as qdevfile:
                        with open(os.path.join(datadir, 'dev.to'), 'w') as ansdevfile:
                            pred = lambda x: 'dev' if (x < 0.1) else 'test' if (x > 0.9) else 'train'
                            for l in parse(jsonpath):
                                part = pred(random.random())
                                if len(l["question"].split(' ')) < 100:
                                    if len(l["answer"].split(' ')) < 100:
                                        if part == 'train':
                                            qfile.write(l["question"].replace('\n', ' ').strip() + '\n')
                                            ansfile.write(l["answer"].replace('\n', ' ').strip() + '\n')
                                        elif part == 'test':
                                            qtestfile.write(l["question"].replace('\n', ' ').strip() + '\n')
                                            anstestfile.write(l["answer"].replace('\n', ' ').strip() + '\n')
                                        elif part == 'dev':
                                            qdevfile.write(l["question"].replace('\n', ' ').strip() + '\n')
                                            ansdevfile.write(l["answer"].replace('\n', ' ').strip() + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--zippath', help='Path for zip file to process')
    parser.add_argument('--jsonpath', help='Path for json file ')
    parser.add_argument('--datadir', help='Path to store data')
    args = parser.parse_args()

    if args.zippath and args.jsonpath:
        generate_strict_json(args.zippath, args.jsonpath)
    if args.datadir and args.jsonpath:
        generate_dataset(args.jsonpath, args.datadir)

# python corpus\amazon_data.py --zippath raw_data\qa_Electronics.json.gz --jsonpath raw_data\qa_Electronics.json --datadir data\amazon
