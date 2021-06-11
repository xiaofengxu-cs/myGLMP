from utils.config import *
from models.GLMP import *

from utils.utils_temp import normalize_string

directory = args['path'].split("/")
task = directory[2].split('HDD')[0]
HDD = directory[2].split('HDD')[1].split('BSZ')[0]
L = directory[2].split('L')[1].split('lr')[0].split("-")[0]
decoder = directory[1].split('-')[0]
BSZ = 1
DS = 'babi'

if DS == 'babi':
    from utils.utils_Ent_babi import *
else:
    print("You need to provide the --dataset information")

_, _, _, _, lang, max_resp_len = prepare_data_seq(task, batch_size=BSZ)

model = globals()[decoder](
    int(HDD),
    lang,
    max_resp_len,
    args['path'],
    "",
    lr=0.0,
    n_layers=int(L),
    dropout=0.0)

input_sentence = ''
type_dict = get_type_dict('data/dialog-bAbI-tasks/dialog-babi-kb-all.txt', dstc2=False)
global_ent = entityList('data/dialog-bAbI-tasks/dialog-babi-kb-all.txt', int(task))

turn_num = 1
while input_sentence != 'q':
    dialog_history = []
    print("\nPlease start speaking or type 'q' to exit or type 'r' to reset the dialog.")
    while 1:
        input_sentence = input("User:")
        input_sentence = normalize_string(input_sentence)
        if input_sentence == 'r' or input_sentence == 'q': break

        flag = False
        err_word = []
        for word in input_sentence.split(' '):
            if word not in lang.word2index:
                err_word.append(word)
                flag = True

        if flag:
            print("\nFalse to recognize '{}'.\nPlease input again.".format('\',\''.join(err_word)))
            continue
        # input_sentence = normalize_string(input_sentence)

        dialog_history.append('{} {}\t$$$$\n'.format(turn_num, input_sentence))
        turn_num += 1
        with open('data/myData/dialog_input.csv', 'w') as fin:
            fin.write(''.join(dialog_history))

        pair_sents, sents_max_len = read_langs('data/myData/dialog_input.csv', global_ent, type_dict)
        pair_sents = [pair_sents[-1]]
        sents = get_seq(pair_sents, lang, 1, True)

        for data in sents:
            repn = model.evaluate_input(data)
        dialog_history[turn_num - 2] = dialog_history[turn_num - 2].replace('$$$$', repn, 1)
        print("\nBot:{}\n".format(repn))
