from gensim.models import KeyedVectors

tc_wv_model = KeyedVectors.load_word2vec_format('./Tencent_AILab_ChineseEmbedding.txt', binary=False)

import json

from collections import OrderedDict

# build word index
word_index = OrderedDict()

for counter, key in enumerate(tc_wv_model.vocab.keys()):
    word_index[key] = counter

with open('tc_word_index.json', 'w') as fp:
    json.dump(word_index, fp)

# build xxx index
from annoy import AnnoyIndex

tc_index = AnnoyIndex(200)

i = 0
for key in tc_wv_model.vocab.keys():
    v = tc_wv_model[key]
    tc_index.add_item(i, v)
    i += 1

tc_index.build(10)
tc_index.save('tc_index_build10.index')

# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# for item in tc_index.get_nns_by_item(word_index[u'自然语言处理'], 11):
# print(reverse_word_index[item])
