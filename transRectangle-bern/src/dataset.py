import os
import pandas as pd
import numpy as np
import random
from collections import Counter

class KnowledgeGraph:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.entity_num = 0
        self.relation_num = 0
        self.concept_num = 0
        self.entity_dict = {}
        self.relation_dict = {}
        self.concept_dict = {}
        self.entities = []
        self.concepts = []
        self.triples = []
        self.triples_num = 0
        self.instance_of_num = 0
        self.instance_of = []
        self.instance_of_ok = {}
        self.subclass_of_num = 0
        self.subclass_of = []
        self.subclass_of_ok = {}
        self.train_num = 0
        self.left_entity = []
        self.right_entity = []
        self.left_num = []
        self.right_num = []

        self.concept_instance = []
        self.instance_concept = []
        self.sub_up_concept = []
        self.up_sub_concept = []
        self.instance_brother = []
        self.concept_brother = []
        self.test_triples = []
        self.valid_triples = []
        self.test_triple_num = 0
        self.valid_triple_num = 0
        '''load dicts train data'''
        self.load_dicts()
        self.load_train_data()
        '''generate candidate data'''
        self.generate_candidate_data()
        '''construct pools after loading'''
        self.triples_pool = set(self.triples)
        self.golden_triple_pool = set(self.triples) | set(self.valid_triples) | set(self.test_triples)
        self.instance_of_pool = set(self.instance_of)
        self.subclass_of_pool = set(self.subclass_of)

    def load_dicts(self):
        entity_dict_file = "instance2id.txt"
        relation_dict_file = "relation2id.txt"
        concept_dict_file = "concept2id.txt"
        print('-----Loading entity dict-----')
        entity_df = pd.read_table(os.path.join('../../data/',self.data_dir,'Train', entity_dict_file), header=None, skiprows=[0])
        # print(entity_df[0])
        # print('------')
        # print(entity_df[1])
        self.entity_dict = dict(zip(entity_df[0], entity_df[1]))
        self.entity_num = len(self.entity_dict)
        self.entities = list(self.entity_dict.values())
        print('#entity: {}'.format(self.entity_num))
        print('-----Loading relation dict-----')
        relation_df = pd.read_table(os.path.join('../../data/',self.data_dir,'Train', relation_dict_file), header=None, skiprows=[0])
        self.relation_dict = dict(zip(relation_df[0], relation_df[1]))
        self.relation_num = len(self.relation_dict)
        print('#relation: {}'.format(self.relation_num))
        print('-----Loading concept dict-----')
        concept_df = pd.read_table(os.path.join('../../data/',self.data_dir,'Train', concept_dict_file), header=None, skiprows=[0])
        self.concept_dict = dict(zip(concept_df[0], concept_df[1]))
        self.concept_num = len(self.concept_dict)
        self.concepts = list(self.concept_dict.values())
        print('#concept: {}'.format(self.concept_num))

        self.concept_instance = [[] for _ in range(self.concept_num)]
        self.instance_concept = [[] for _ in range(self.entity_num)]
        self.sub_up_concept = [[] for _ in range(self.concept_num)]
        self.up_sub_concept = [[] for _ in range(self.concept_num)]
        self.instance_brother = [[] for _ in range(self.entity_num)]
        self.concept_brother = [[] for _ in range(self.concept_num)]

    def load_train_data(self):
        instance_of_file = "instanceOf2id.txt"
        subclass_of_file = "subClassOf2id.txt"
        triple_file = "triple2id.txt"

        print('-----Loading instance_of triples-----')
        instance_of_df = pd.read_table(os.path.join('../../data/',self.data_dir,'Train', instance_of_file), header=None, sep=' ')
        # print(instance_of_df[0])
        # print(instance_of_df[1])
        self.instance_of = list(zip(instance_of_df[0], instance_of_df[1]))
        self.instance_of_num = len(self.instance_of)
        print('#instance of :{}'.format(self.instance_of_num))
        self.instance_of_ok = dict(zip(self.instance_of, [1 for i in range(len(self.instance_of))]))
        for instance_of_item in self.instance_of:
            self.instance_concept[instance_of_item[0]].append(instance_of_item[1])
            self.concept_instance[instance_of_item[1]].append(instance_of_item[0])

        print('-----Loading training triples-----')
        triple_df = pd.read_table(os.path.join('../../data/',self.data_dir,'Train', triple_file), header=None, sep=' ', skiprows=[0])
        self.triples = list(zip(triple_df[0], triple_df[1], triple_df[2]))
        self.triples_num = len(self.triples)
        print('#triples:{}'.format(self.triples_num))

        print('-----Loading subclass_of triples-----')
        subclass_of_df = pd.read_table(os.path.join('../../data/',self.data_dir,'Train', subclass_of_file), header=None, sep=' ')
        self.subclass_of = list(zip(subclass_of_df[0], subclass_of_df[1]))
        self.subclass_of_num = len(self.subclass_of)
        print('#subclass of:{}'.format(self.subclass_of_num))
        self.subclass_of_ok = dict(zip(self.subclass_of, [1 for i in range(len(self.subclass_of))]))
        for subclass_of_item in self.subclass_of:
            self.sub_up_concept[subclass_of_item[0]].append(subclass_of_item[1])
            self.up_sub_concept[subclass_of_item[1]].append(subclass_of_item[0])
        
        self.train_num = self.triples_num + self.instance_of_num + self.subclass_of_num
        print('#train_num:{}'.format(self.train_num))

        print('-----Loading test triples data-----')
        test_df = pd.read_csv('../../data/' + self.data_dir + '/Test/triple2id_positive.txt', header=None, sep=' ', skiprows=[0])
        self.test_triples = list(zip(test_df[0], test_df[1], test_df[2]))
        self.test_triple_num = len(self.test_triples)

        print('-----Loading valid triples data-----')
        valid_df = pd.read_csv('../../data/' + self.data_dir + '/Valid/triple2id_positive.txt', header=None, sep=' ', skiprows=[0])
        self.valid_triples = list(zip(valid_df[0], valid_df[1], valid_df[2]))
        self.valid_triple_num = len(self.valid_triples)

        self.left_entity = [Counter() for i in range(self.relation_num)]
        self.right_entity = [Counter() for i in range(self.relation_num)]

        # bern set
        for h, t, r in self.triples:
            self.left_entity[r][h] += 1
            self.right_entity[r][t] += 1

        self.left_num = [float(sum(c.values())) / float(len(c)) for c in self.left_entity]
        self.right_num = [float(sum(c.values())) / float(len(c)) for c in self.right_entity]


    def generate_candidate_data(self):
        print('-----generate instance candidate data-----')
        for i in range(len(self.instance_concept)):
            for j in range(len(self.instance_concept[i])):
                for k in range(len(self.concept_instance[self.instance_concept[i][j]])):
                    if self.concept_instance[self.instance_concept[i][j]][k] != i:
                        self.instance_brother[i].append(self.concept_instance[self.instance_concept[i][j]][k])
        print('-----generate concept candidate data-----')
        for i in range(len(self.sub_up_concept)):
            for j in range(len(self.sub_up_concept[i])):
                for k in range(len(self.up_sub_concept[self.sub_up_concept[i][j]])):
                    if self.up_sub_concept[self.sub_up_concept[i][j]][k] != i:
                        self.concept_brother[i].append(self.up_sub_concept[self.sub_up_concept[i][j]][k])
    

    def next_raw_batch(self, batch_number):
        rand_idx_triples = np.random.permutation(self.triples_num)
        rand_idx_instance = np.random.permutation(self.instance_of_num)
        rand_idx_subclass = np.random.permutation(self.subclass_of_num)
        
        start_triples = 0 
        start_instance = 0 
        start_subclass = 0 
        start = 0
        cycle_num =0
        while start < self.train_num:
            
            if cycle_num < 99:
                triples_end = start_triples + self.triples_num//batch_number
                instance_end = start_instance + self.instance_of_num//batch_number
                subclass_end = start_subclass + self.subclass_of_num//batch_number
            else:
                triples_end = self.triples_num
                instance_end = self.instance_of_num
                subclass_end = self.subclass_of_num
            
            raw_triples_list = [self.triples[i] for i in rand_idx_triples[start_triples:triples_end]]
            raw_instance_list = [self.instance_of[i] for i in rand_idx_instance[start_instance:instance_end]]
            raw_subclass_list = [self.subclass_of[i] for i in rand_idx_subclass[start_subclass:subclass_end]]
            start_triples = triples_end
            start_instance = instance_end
            start_subclass = subclass_end

            raw_list = [raw_triples_list, raw_instance_list, raw_subclass_list]
            print('start:{}'.format(start), end='\r')
            yield raw_list
            cycle_num += 1
            start = start_triples + start_instance + start_subclass

    def generate_training_batch(self, in_queue, out_queue):
        while True:
            raw_batch, epoch = in_queue.get()
            if epoch < 1000:
                cut = 10 - epoch * 8 // 1000
            else:
                cut = 2
            if raw_batch is None:
                return
            else:
                triple_batch_pos = raw_batch[0]
                instance_batch_pos = raw_batch[1]
                subclass_batch_pos = raw_batch[2]

                triple_batch_neg = []
                instance_batch_neg = []
                subclass_batch_neg = []
                corrupt_head_prob = np.random.binomial(1, 0.5)
                for head, tail, relation in triple_batch_pos:
                    head_neg = head
                    tail_neg = tail
                    while True:
                        # bern
                        pr = float(self.right_num[relation]) / (self.right_num[relation] + self.left_num[relation])
                        corrupt_head_prob = random.uniform(0, 1) > pr
                        if corrupt_head_prob:
                            if len(self.instance_brother[head]) != 0:
                                if random.randint(0,9) < cut:
                                    head_neg = random.choice(self.entities)
                                else:
                                    head_neg = random.choice(self.instance_brother[head])
                            else:
                                head_neg = random.choice(self.entities)
                        else:
                            if len(self.instance_brother[tail]) != 0:
                                if random.randint(0,9) < cut:
                                    tail_neg = random.choice(self.entities)
                                else:
                                    tail_neg = random.choice(self.instance_brother[tail])
                            else:
                                tail_neg = random.choice(self.entities)
                        if (head_neg, tail_neg, relation) not in self.triples_pool:
                            break
                    triple_batch_neg.append((head_neg, tail_neg, relation))

                corrupt_head_prob = np.random.binomial(1, 0.5)
                for head, tail in instance_batch_pos:
                    head_neg = head
                    tail_neg = tail
                    while True:
                        if corrupt_head_prob:
                            if len(self.instance_brother[head]) != 0:
                                if random.randint(0,9) < cut:
                                    head_neg = random.choice(self.entities)
                                else:
                                    head_neg = random.choice(self.instance_brother[head])
                            else:
                                head_neg = random.choice(self.entities)
                        else:
                            if len(self.concept_brother[tail]) != 0:
                                if random.randint(0,9) < cut:
                                    tail_neg = random.choice(self.concepts)
                                else:
                                    tail_neg = random.choice(self.concept_brother[tail])
                            else:
                                tail_neg = random.choice(self.concepts)
                        if (head_neg, tail_neg) not in self.instance_of_pool:
                            break
                    instance_batch_neg.append((head_neg, tail_neg))
                corrupt_head_prob = np.random.binomial(1, 0.5)
                for head, tail in subclass_batch_pos:
                    head_neg = head
                    tail_neg = tail
                    while True:
                        if corrupt_head_prob:
                            if len(self.concept_brother[head]) != 0:
                                if random.randint(0,9) < cut:
                                    head_neg = random.choice(self.concepts)
                                else:
                                    head_neg = random.choice(self.concept_brother[head])
                            else:
                                head_neg = random.choice(self.concepts)
                        else:
                            if len(self.concept_brother[tail]) != 0:
                                if random.randint(0,9) < cut:
                                    tail_neg = random.choice(self.concepts)
                                else:
                                    tail_neg = random.choice(self.concept_brother[tail])
                            else:
                                tail_neg = random.choice(self.concepts)
                        if (head_neg, tail_neg) not in self.subclass_of_pool:
                            break
                    subclass_batch_neg.append((head_neg, tail_neg))

                batch = ((triple_batch_pos, triple_batch_neg), (instance_batch_pos, instance_batch_neg),
                         (subclass_batch_pos, subclass_batch_neg))
                out_queue.put(batch)
    