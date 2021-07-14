import json
from tqdm import tqdm
import argparse
import random
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True, default='')
    parser.add_argument('--output_dir', required=True, default='')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        print(f'Make dir {args.output_dir}')

    with open(args.input_path, 'r', encoding='utf8') as fr, \
        open(f'{args.output_dir}/train_ged.json', 'w', encoding='utf8') as fw_train, \
        open(f'{args.output_dir}/valid_ged.json', 'w', encoding='utf8') as fw_val, \
        open(f'{args.output_dir}/test_ged.json', 'w', encoding='utf8') as fw_test:

        lines = fr.readlines()
        shuffle_lst = list(range(len(lines)))
        random.shuffle(shuffle_lst)

        for i, shuffle_idx in tqdm(enumerate(shuffle_lst)):
            line = json.loads(lines[shuffle_idx])
            src = line['source']
            tgt = line['tag_lst']
            assert len(src) == len(tgt)

            if i < 90000:
                json.dump(line, fw_train, ensure_ascii=False)
                fw_train.write('\n')
            elif i < 92000:
                json.dump(line, fw_val, ensure_ascii=False)
                fw_val.write('\n')
            else:
                json.dump(line, fw_test, ensure_ascii=False)
                fw_test.write('\n')

        print('Split Finished')

if __name__ == '__main__':
    main()