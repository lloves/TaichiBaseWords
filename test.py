import json
from annoy import AnnoyIndex

from gensim.models import KeyedVectors
from collections import OrderedDict

fname = 'tc_index_build10.index'

def load():
    with open('tc_word_index.json','r') as f:
        data = json.load(f)
        return data


text = ''
# word_index = OrderedDict()
# word_index = json.loads(load(), object_pairs_hook=OrderedDict);  
word_index = load()

annoy_index2 = AnnoyIndex(200)
annoy_index2.load(fname, prefault=False)

print(word_index[u'自然语言处理'])

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])   
 
for item in annoy_index2.get_nns_by_item(word_index[u'自然语言处理'], 11):
	print(reverse_word_index[item])
