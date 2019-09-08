
def get_vocab_list():
    vocab_dict = {}
    with open('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex6\ex6\\vocab.txt') as f:
        for line in f:
            (val, key) = line.split()
            vocab_dict[int(val)] = key

    print(  vocab_dict   ) 

get_vocab_list()