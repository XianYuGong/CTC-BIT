import argparse
import json
import Levenshtein
from tqdm import tqdm

def read_data(input_path):
    src_lst, tgt_lst = [], []
    suffix = input_path.split('.')[-1]

    assert suffix in ['json', 'txt']

    if suffix == 'json':
        with open(input_path, 'r', encoding='utf8') as fr:
            for line in tqdm(fr.readlines()):
                line = json.loads(line)
                src = line['source'].strip('\n').strip()
                tgt = line['target'].strip('\n').strip()
                src_lst.append(src)
                tgt_lst.append(tgt)
    elif suffix == 'txt':
        with open(input_path, 'r', encoding='utf8') as fr:
            for line in tqdm(fr.readlines()):
                line = line.strip('\n')
                src, tgt = line.split('\t')
                src_lst.append(src)
                tgt_lst.append(tgt)

    assert len(src_lst) == len(tgt_lst)
    return src_lst, tgt_lst

def get_tag(src_lst, tgt_lst):
    tag_all_lst = []
    err_lst = {
        'replace': [],
        'insert': [],
        'delete': []
    }
    '''
    tag scheme: BIES0
    [
        'O',
        'S-REP', 'B-REP', 'I-REP', 'E-REP',
        'S-DEL', 'B-DEL', 'I-DEL', 'E-DEL',
        'F-INS', 'B-INS', 'E-INS', 'L-INS',
    ]
    '''
    for i, (src, tgt) in tqdm(enumerate(zip(src_lst, tgt_lst))):
        edits = Levenshtein.opcodes(src, tgt)
        tag_lst = ['O'] * len(src)

        for edit in edits:
            option, s_index, t_index = edit[0], edit[1], edit[2]
            if option == 'replace':
                if tag_lst[s_index:t_index] == ['O'] * (t_index-s_index):
                    if t_index-s_index == 1:
                        tag_lst[s_index] = 'S-REP'
                    else:
                        tag_lst[s_index] = 'B-REP'
                        tag_lst[t_index-1] = 'E-REP'
                        for idx in range(s_index+1, t_index-1):
                            tag_lst[idx] = 'I-REP'
                else:
                    if i not in err_lst['replace']:
                        err_lst['replace'].append(i)
            elif option == 'delete':
                if tag_lst[s_index:t_index] == ['O'] * (t_index-s_index):
                    if t_index-s_index == 1:
                        tag_lst[s_index] = 'S-DEL'
                    else:
                        tag_lst[s_index] = 'B-DEL'
                        tag_lst[t_index-1] = 'E-DEL'
                        for idx in range(s_index+1, t_index-1):
                            tag_lst[idx] = 'I-DEL'
                else:
                    if i not in err_lst['delete']:
                        err_lst['delete'].append(i)
            elif option == 'insert':
                assert s_index == t_index
                if s_index == 0:
                    if tag_lst[s_index] == 'O':
                        tag_lst[s_index] = 'F-INS'
                    else:
                        if i not in err_lst['insert']:
                            err_lst['insert'].append(i)
                elif s_index == len(src):
                    if tag_lst[s_index-1] == 'O':
                        tag_lst[s_index-1] = 'L-INS'
                    else:
                        if i not in err_lst['insert']:
                            err_lst['insert'].append(i)
                else:
                    if tag_lst[s_index-1:s_index+1] == ['O'] * 2:
                        tag_lst[s_index-1] = 'B-INS'
                        tag_lst[s_index] = 'E-INS'
                    else:
                        if i not in err_lst['insert']:
                            err_lst['insert'].append(i)

        assert len(tag_lst) == len(src)
        tag_all_lst.append(tag_lst)

    return tag_all_lst, err_lst

def filter_tag_conflict(src_lst, tgt_lst, err_lst, tag_all_lst):
    src_filter_lst, tgt_filter_lst, tag_filter_lst = [], [], []
    assert len(src_lst) == len(tgt_lst) == len(tag_all_lst)

    for i, (src, tgt, tag_lst) in tqdm(enumerate(zip(src_lst, tgt_lst, tag_all_lst))):
        if i in err_lst:
            continue
        else:
            src_filter_lst.append(src)
            tgt_filter_lst.append(tgt)
            tag_filter_lst.append(tag_lst)

    return src_filter_lst, tgt_filter_lst, tag_filter_lst

def make_ged_json(src_lst, tgt_lst, tag_all_lst, json_path):
    with open(json_path, 'w', encoding='utf8') as fw:
        for i, (src, tgt, tag_lst) in tqdm(enumerate(zip(src_lst, tgt_lst, tag_all_lst))):
            assert len(src) == len(tag_lst)
            json.dump(
                {
                    'source': src,
                    'target': tgt,
                    'tag_lst': tag_lst,
                },
                fw,
                ensure_ascii=False,
            )
            fw.write('\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True, default='')
    parser.add_argument('--output_path', required=True, default='')
    args = parser.parse_args()

    src_lst, tgt_lst = read_data(args.input_path)
    print(f'Loading about {len(src_lst)} lines')

    print('============Start Tagging============')
    tag_all_lst, err_lst = get_tag(src_lst, tgt_lst)
    print('============End Tagging============')

    del_idx_lst = set(err_lst['replace'] + err_lst['insert'] + err_lst['delete'])
    print(f'{len(del_idx_lst)} lines have tag conflict')

    src_filter_lst, tgt_filter_lst, tag_filter_all_lst = filter_tag_conflict(src_lst, tgt_lst, \
                                                                            del_idx_lst, tag_all_lst)
    print(f'After filter, has {len(src_filter_lst)} lines')
    make_ged_json(src_filter_lst, tgt_filter_lst, tag_filter_all_lst, args.output_path)
    print('Writing new GED json')

if __name__ == '__main__':
    main()