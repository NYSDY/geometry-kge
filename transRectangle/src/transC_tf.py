import math
import timeit
import time
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from dataset import KnowledgeGraph
import os

class TransC:
    def __init__(self, kg: KnowledgeGraph,
                 embedding_dim, margin_value, margin_instance_value, margin_subclass_value, 
                 score_func, batch_number, learning_rate, n_generator, n_rank_calculator):
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.margin_value = margin_value
        self.margin_instance_value = margin_instance_value
        self.margin_subclass_value = margin_subclass_value
        self.score_func = score_func
        self.batch_number = batch_number
        self.learning_rate = learning_rate
        self.n_generator = n_generator
        self.n_rank_calculator = n_rank_calculator
        '''ops for training'''
        self.instance_of_pos = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='instance_pos')
        self.instance_of_neg = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='instance_neg')
        self.subclass_of_pos = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='subclass_pos')
        self.subclass_of_neg = tf.placeholder(dtype=tf.int32, shape=[None, 2], name='subclass_neg')
        self.triple_pos = tf.placeholder(dtype=tf.int32, shape=[None, 3], name='triple_pos')
        self.triple_neg = tf.placeholder(dtype=tf.int32, shape=[None, 3], name='triple_neg')
        self.margin = tf.placeholder(dtype=tf.float32, shape=[None], name='margin')
        self.margin_instance = tf.placeholder(dtype=tf.float32, shape=[None], name='margin_instance')
        self.margin_subclass = tf.placeholder(dtype=tf.float32, shape=[None], name='margin_subclass')
        self.margin_positive = tf.placeholder(dtype=tf.float32, shape=[None], name='margin_positive')
        self.margin_instance_positive = tf.placeholder(dtype=tf.float32, shape=[None], name='margin_instance_positive')
        self.margin_subclass_positive = tf.placeholder(dtype=tf.float32, shape=[None], name='margin_subclass_positive')
        self.train_triple_op = None
        self.train_instance_op = None
        self.train_subclass_op = None
        self.train_op = None
        self.train_op_embedding = None
        self.train_op_concept_radius = None
        self.loss_subclass = None
        self.loss_instance = None
        self.loss_triple = None
        self.loss = None
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.merge = None
        '''ops for evaluation'''
        self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[3])
        self.idx_head_prediction = None
        self.idx_tail_prediction = None
        self.r_trainable = True
        self.concept_trainable = True
        '''embeddings'''
        bound = 6 / math.sqrt(self.embedding_dim)
        with tf.variable_scope('embedding'):
            self.entity_embedding = tf.get_variable(name='entity',
                                                    shape=[kg.entity_num, self.embedding_dim],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound))
            tf.summary.histogram(name=self.entity_embedding.op.name, values=self.entity_embedding)
            self.relation_embedding = tf.get_variable(name='relation',
                                                      shape=[kg.relation_num, self.embedding_dim],
                                                      initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                                maxval=bound))
            tf.summary.histogram(name=self.relation_embedding.op.name, values=self.relation_embedding)
        with tf.variable_scope('concept'):
            self.concept_embedding = tf.get_variable(name='concept',
                                                     shape=[kg.concept_num, self.embedding_dim],
                                                     initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                               maxval=bound))
            tf.summary.histogram(name=self.concept_embedding.op.name, values=self.concept_embedding)
        with tf.variable_scope('radius'):
            self.concept_r_embedding = tf.get_variable(name='concept_r', trainable=self.r_trainable,
                                                        shape=[kg.concept_num, self.embedding_dim],
                                                        initializer=tf.random_uniform_initializer(minval=0.5,
                                                                                                    maxval=0.5))
            tf.summary.histogram(name=self.concept_r_embedding.op.name, values=self.concept_r_embedding)
        self.build_graph()
        self.build_eval_graph()
        # Create a saver.
        self.saver = tf.train.Saver()

    def build_graph(self):
        with tf.name_scope('normalization'):
            self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, dim=1)
            self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, dim=1)
            self.concept_embedding = tf.nn.l2_normalize(self.concept_embedding, dim=1)
        with tf.name_scope('training'):
            distance_pos, distance_neg = self.infer(self.triple_pos, self.triple_neg)
            tf.summary.histogram(name='triple_score_pos', values=distance_pos)
            tf.summary.histogram(name='triple_score_neg', values=distance_neg)
            self.loss_triple = self.calculate_loss(distance_pos, distance_neg, self.margin, self.margin_positive)
            score_pos1, score_neg1 = self.infer_instance_of(self.instance_of_pos, self.instance_of_neg)
            tf.summary.histogram(name='instance_score_pos', values=score_pos1)
            tf.summary.histogram(name='instance_score_neg', values=score_neg1)

            self.loss_instance = self.caculate_instance_of_loss(score_pos1, score_neg1, self.margin_instance,
                                                                self.margin_instance_positive)
            score_pos2, score_neg2 = self.infer_subclass_of(self.subclass_of_pos, self.subclass_of_neg)
            self.loss_subclass = self.caculate_subclass_of_loss(score_pos2, score_neg2, self.margin_subclass,
                                                                self.margin_subclass_positive)
            tf.summary.histogram(name='subclass_score_pos', values=score_pos2)
            tf.summary.histogram(name='subclass_score_neg', values=score_neg2)

            self.loss = self.loss_triple + self.loss_instance  + self.loss_subclass
            
            tf.summary.scalar(name=self.loss_triple.op.name, tensor=self.loss_triple)
            tf.summary.scalar(name=self.loss_instance.op.name, tensor=self.loss_instance)
            tf.summary.scalar(name=self.loss_subclass.op.name, tensor=self.loss_subclass)
            tf.summary.scalar(name=self.loss.op.name, tensor=self.loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            # self.train_triple_op = optimizer.minimize(self.loss_triple, global_step=self.global_step)
            # self.train_instance_op = optimizer.minimize(self.loss_instance, global_step=self.global_step)
            # self.train_subclass_op = optimizer.minimize(self.loss_subclass, global_step=self.global_step)
            trainable_var_radius = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='radius')
            trainable_var_concept = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='concept')
            trainable_var_embedding = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='embedding')

            self.train_op_concept_radius = optimizer.minimize(self.loss_instance + self.loss_subclass, global_step=self.global_step, var_list=trainable_var_concept + trainable_var_radius)
            self.train_op_embedding = optimizer.minimize(self.loss_triple, global_step=self.global_step, var_list=trainable_var_embedding)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.merge = tf.summary.merge_all()

    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.idx_head_prediction, self.idx_tail_prediction = self.evaluate(self.eval_triple)

    def infer(self, triple_pos, triple_neg):
        with tf.name_scope('lookup'):
            head_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 0])
            tail_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 1])
            relation_pos = tf.nn.embedding_lookup(self.relation_embedding, triple_pos[:, 2])
            head_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 0])
            tail_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 1])
            relation_neg = tf.nn.embedding_lookup(self.relation_embedding, triple_neg[:, 2])
        with tf.name_scope('link'):
            distance_pos = head_pos + relation_pos - tail_pos
            distance_neg = head_neg + relation_neg - tail_neg
            # print(tf.shape(distance_pos))
            # print('=======================================')
        return distance_pos, distance_neg

    def infer_instance_of(self, instance_of_pos, instance_of_neg):
        with tf.name_scope('lookup_instance_of'):
            head_pos = tf.nn.embedding_lookup(self.entity_embedding, instance_of_pos[:, 0])
            tail_pos = tf.nn.embedding_lookup(self.concept_embedding, instance_of_pos[:, 1])
            r_pos = tf.nn.embedding_lookup(self.concept_r_embedding, instance_of_pos[:, 1])
            head_neg = tf.nn.embedding_lookup(self.entity_embedding, instance_of_neg[:, 0])
            tail_neg = tf.nn.embedding_lookup(self.concept_embedding, instance_of_neg[:, 1])
            r_neg = tf.nn.embedding_lookup(self.concept_r_embedding, instance_of_neg[:, 1])
        with tf.name_scope('caculate_instance_of_score'):
            distance_pos = tf.abs(head_pos - tail_pos) - r_pos
            distance_neg = tf.abs(head_neg - tail_neg) - r_neg
            score_pos = tf.reduce_sum(tf.where(tf.greater(distance_pos, 0), distance_pos, tf.subtract(distance_pos, distance_pos)), axis=1)
            score_neg = tf.reduce_sum(tf.where(tf.less(distance_neg, 0), distance_neg, tf.subtract(distance_neg, distance_neg)), axis=1)

            pos_dimesion_num = tf.count_nonzero(tf.where(tf.less(distance_pos, 0), distance_pos, tf.subtract(distance_pos, distance_pos)))
            neg_dimesion_num = tf.count_nonzero(tf.where(tf.less(distance_neg, 0), distance_neg, tf.subtract(distance_neg, distance_neg)))
            tf.summary.scalar(name='instanceOf_positive_dimesion_number', tensor=pos_dimesion_num)
            tf.summary.scalar(name='instanceOf_negative_dimension_number', tensor=neg_dimesion_num)
        
        return score_pos, score_neg
    
    def infer_subclass_of(self, subclass_of_pos, subclass_of_neg):
        with tf.name_scope('lookup_subclass_of'):
            head_pos = tf.nn.embedding_lookup(self.concept_embedding, subclass_of_pos[:, 0])
            tail_pos = tf.nn.embedding_lookup(self.concept_embedding, subclass_of_pos[:, 1])
            rh_pos = tf.nn.embedding_lookup(self.concept_r_embedding, subclass_of_pos[:, 0])
            rt_pos = tf.nn.embedding_lookup(self.concept_r_embedding, subclass_of_pos[:, 1])
            head_neg = tf.nn.embedding_lookup(self.concept_embedding, subclass_of_neg[:, 0])
            tail_neg = tf.nn.embedding_lookup(self.concept_embedding, subclass_of_neg[:, 1])
            rh_neg = tf.nn.embedding_lookup(self.concept_r_embedding, subclass_of_neg[:, 0])
            rt_neg = tf.nn.embedding_lookup(self.concept_r_embedding, subclass_of_neg[:, 1])
        with tf.name_scope('caculate_subclass_of_score'):
            distance_pos = tf.abs(head_pos - tail_pos)  - rt_pos + rh_pos
            distance_neg = tf.abs(head_neg - tail_neg)  - rt_neg + rh_neg
            
            score_pos = tf.reduce_sum(tf.where(tf.greater(distance_pos, 0), distance_pos, tf.subtract(distance_pos, distance_pos)), axis=1)
            score_neg = tf.reduce_sum(tf.where(tf.less(distance_neg, 0), distance_neg, tf.subtract(distance_neg, distance_neg)), axis=1)
            pos_dimesion_num = tf.count_nonzero(tf.where(tf.less(distance_pos, 0), distance_pos, tf.subtract(distance_pos, distance_pos)))
            neg_dimesion_num = tf.count_nonzero(tf.where(tf.less(distance_neg, 0), distance_neg, tf.subtract(distance_neg, distance_neg)))
            tf.summary.scalar(name='subClassOf_positive_dimesion_number', tensor=pos_dimesion_num)
            tf.summary.scalar(name='subClassOf_negative_dimension_number', tensor=neg_dimesion_num)
        return score_pos, score_neg

    def caculate_subclass_of_loss(self, score_pos, score_neg, margin_subclass, margin_subclass_positive):
        with tf.name_scope('subclass_of_loss'):
            loss_subclass_of = tf.reduce_sum((tf.nn.relu(margin_subclass - score_neg + score_pos)),
                                             name='max_margin_subclass_loss')
        return loss_subclass_of

    def caculate_instance_of_loss(self, score_pos, score_neg, margin_instance, margin_instance_positive):
        with tf.name_scope('instance_of_loss'):
            loss_instance_of = tf.reduce_sum((tf.nn.relu(margin_instance - score_neg + score_pos)),
                                             name='max_margin_instance_loss')
        return loss_instance_of
            
    def calculate_loss(self, distance_pos, distance_neg, margin, margin_positive):
        with tf.name_scope('loss'):
            if self.score_func == 'L1':  # L1 score
                score_pos = tf.reduce_sum(tf.abs(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.abs(distance_neg), axis=1)
            else:  # L2 score
                score_pos = tf.reduce_sum(tf.square(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.square(distance_neg), axis=1)
            loss = tf.reduce_sum(tf.nn.relu(margin + score_pos - score_neg),
                                 name='triple_max_margin_loss')
        return loss

    def launch_training(self, session, summary_writer, epoch):

        raw_batch_queue = mp.Queue()
        training_batch_queue = mp.Queue()
        process_list = []
        for _ in range(self.n_generator):
            p = mp.Process(target=self.kg.generate_training_batch, kwargs={'in_queue': raw_batch_queue,
                                                                    'out_queue': training_batch_queue})
            process_list.append(p)
            p.start()

        print('-----Start training-----')
        start = timeit.default_timer()
        for raw_batch in self.kg.next_raw_batch(self.batch_number):
            raw_batch_queue.put((raw_batch, epoch))

        for _ in range(self.n_generator):
            raw_batch_queue.put((None,epoch))
        print('-----Constructing training batches-----')
        epoch_loss = 0
        n_used = 0
        for i in range(self.batch_number):
            triple_batch, instance_batch, subclass_batch = training_batch_queue.get()
            # print('triple_batch_pos:{}, instance_batch_pos:{}, subclass_batch_pos:{}'.format(len(triple_batch[0]), len(instance_batch[0]),len(subclass_batch[0])))
            margin_list = [self.margin_value for _ in range(len(triple_batch[0]))]
            margin_instance_list = [self.margin_instance_value for _ in range(len(instance_batch[0]))]
            margin_subclass_list = [self.margin_subclass_value for _ in range(len(subclass_batch[0]))]
            margin_positive_list = [0.01 for _ in range(len(triple_batch[0]))]
            margin_instance_positive_list = [0.01 for _ in range(len(instance_batch[0]))]
            margin_subclass_positive_list = [0.01 for _ in range(len(subclass_batch[0]))]
            # print(np.shape(margin_list))
            epoch_tmp = epoch % 200
            if epoch_tmp < 100:
                batch_loss, _, summary = session.run(fetches=[self.loss, self.train_op_embedding, self.merge],
                                                     feed_dict={self.triple_pos: triple_batch[0],
                                                                self.triple_neg: triple_batch[1],
                                                                self.instance_of_pos: instance_batch[0],
                                                                self.instance_of_neg: instance_batch[1],
                                                                self.subclass_of_pos: subclass_batch[0],
                                                                self.subclass_of_neg: subclass_batch[1],
                                                                self.margin: margin_list,
                                                                self.margin_instance: margin_instance_list,
                                                                self.margin_subclass: margin_subclass_list,
                                                                self.margin_positive: margin_positive_list,
                                                                self.margin_subclass_positive: margin_subclass_positive_list,
                                                                self.margin_instance_positive: margin_instance_positive_list
                                                                })
            else:
                batch_loss, _, summary = session.run(fetches=[self.loss, self.train_op_concept_radius, self.merge],
                                                     feed_dict={self.triple_pos: triple_batch[0],
                                                                self.triple_neg: triple_batch[1],
                                                                self.instance_of_pos: instance_batch[0],
                                                                self.instance_of_neg: instance_batch[1],
                                                                self.subclass_of_pos: subclass_batch[0],
                                                                self.subclass_of_neg: subclass_batch[1],
                                                                self.margin: margin_list,
                                                                self.margin_instance: margin_instance_list,
                                                                self.margin_subclass: margin_subclass_list,
                                                                self.margin_positive: margin_positive_list,
                                                                self.margin_subclass_positive: margin_subclass_positive_list,
                                                                self.margin_instance_positive: margin_instance_positive_list
                                                                })
            summary_writer.add_summary(summary, global_step=self.global_step.eval(session=session))
            epoch_loss += batch_loss
            n_used += len(triple_batch[0]) + len(instance_batch[0]) + len(subclass_batch[0])
            print('[{:.3f}s] #train_num: {}/{} avg_loss: {:.6f}'.format(timeit.default_timer() - start,
                                                                        n_used,
                                                                        self.kg.train_num - self.kg.subclass_of_num,
                                                                        batch_loss / (len(triple_batch[0]) + len(instance_batch[0]))), end='\r')


        print('epoch loss: {:.3f}'.format(epoch_loss))
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print('-----Finish training-----')
        # self.check_norm(session=session)
        for i in process_list:
            i.terminate()

    def evaluate(self, eval_triple):
        with tf.name_scope('lookup'):
            head = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[0])
            tail = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[1])
            relation = tf.nn.embedding_lookup(self.relation_embedding, eval_triple[2])
        with tf.name_scope('link'):
            distance_head_prediction = self.entity_embedding + relation - tail
            distance_tail_prediction = head + relation - self.entity_embedding
        with tf.name_scope('rank'):
            if self.score_func == 'L1':  # L1 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_head_prediction), axis=1),
                                                        k=self.kg.entity_num)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.abs(distance_tail_prediction), axis=1),
                                                        k=self.kg.entity_num)
            else:  # L2 score
                _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_head_prediction), axis=1),
                                                     k=self.kg.entity_num)
                _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_tail_prediction), axis=1),
                                                     k=self.kg.entity_num)
        return idx_head_prediction, idx_tail_prediction

    def launch_evaluation(self, session):
        eval_result_queue = mp.JoinableQueue()
        rank_result_queue = mp.Queue()
        print('-----Start evaluation-----')
        start = timeit.default_timer()
        for _ in range(self.n_rank_calculator):
            mp.Process(target=self.calculate_rank, kwargs={'in_queue': eval_result_queue,
                                                           'out_queue': rank_result_queue}).start()
        n_used_eval_triple = 0
        for eval_triple in self.kg.test_triples:
            idx_head_prediction, idx_tail_prediction = session.run(fetches=[self.idx_head_prediction,
                                                                            self.idx_tail_prediction],
                                                                   feed_dict={self.eval_triple: eval_triple})
            eval_result_queue.put((eval_triple, idx_head_prediction, idx_tail_prediction))
            n_used_eval_triple += 1
        print('[{:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
                                                            n_used_eval_triple,
                                                            self.kg.test_triple_num))
        for _ in range(self.n_rank_calculator):
            eval_result_queue.put(None)
        print('-----Joining all rank calculator-----')
        eval_result_queue.join()
        print('-----All rank calculation accomplished-----')
        print('-----Obtaining evaluation results-----')
        '''Raw'''
        head_meanrank_raw = 0
        head_meanrank_reciprocal_raw = 0
        head_hits10_raw = 0
        tail_meanrank_raw = 0
        tail_meanrank_reciprocal_raw = 0

        tail_hits10_raw = 0

        '''Filter'''
        head_meanrank_filter = 0
        head_meanrank_reciprocal_filter = 0
        head_hits10_filter = 0
        head_hits5_filter = 0
        head_hits3_filter = 0
        head_hits1_filter = 0
        tail_meanrank_filter = 0
        tail_meanrank_reciprocal_filter = 0
        tail_hits10_filter = 0
        tail_hits5_filter = 0
        tail_hits3_filter = 0
        tail_hits1_filter = 0

        for _ in range(n_used_eval_triple):
            head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter = rank_result_queue.get()
            head_meanrank_raw += head_rank_raw
            head_meanrank_reciprocal_raw += 1 / (head_rank_raw + 1)
            if head_rank_raw < 10:
                head_hits10_raw += 1
            tail_meanrank_raw += tail_rank_raw
            tail_meanrank_reciprocal_raw += 1 / (tail_rank_raw + 1)
            if tail_rank_raw < 10:
                tail_hits10_raw += 1
            head_meanrank_filter += head_rank_filter
            head_meanrank_reciprocal_filter += 1 / (head_rank_filter + 1)
            if head_rank_filter < 10:
                head_hits10_filter += 1
            if head_rank_filter < 5:
                head_hits5_filter += 1
            if head_rank_filter < 3:
                head_hits3_filter += 1
            if head_rank_filter < 1:
                head_hits1_filter += 1
            tail_meanrank_filter += tail_rank_filter
            tail_meanrank_reciprocal_filter += 1 / (tail_rank_filter + 1)
            if tail_rank_filter < 10:
                tail_hits10_filter += 1
            if tail_rank_filter < 5:
                tail_hits5_filter += 1
            if tail_rank_filter < 3:
                tail_hits3_filter += 1
            if tail_rank_filter < 1:
                tail_hits1_filter += 1
        print('-----Raw-----')
        head_meanrank_raw /= n_used_eval_triple
        head_meanrank_reciprocal_raw /= n_used_eval_triple
        head_hits10_raw /= n_used_eval_triple
        tail_meanrank_raw /= n_used_eval_triple
        tail_meanrank_reciprocal_raw /= n_used_eval_triple
        tail_hits10_raw /= n_used_eval_triple
        # print('-----Head prediction-----')
        # print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_raw, head_hits10_raw))
        # print('-----Tail prediction-----')
        # print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_raw, tail_hits10_raw))
        mean_rank_raw = round((head_meanrank_raw + tail_meanrank_raw) / 2, 3)
        mean_rank_reciprocal_raw = round((head_meanrank_reciprocal_raw + tail_meanrank_reciprocal_raw) / 2, 3)
        hits10_raw = round((head_hits10_raw + tail_hits10_raw) / 2, 3)
        print('------Average------')
        print('MeanRank: {:.3f}, MeanRankReciprocal:{:.3f}, Hits@10: {:.3f}'
              .format((head_meanrank_raw + tail_meanrank_raw) / 2,
                      (head_meanrank_reciprocal_raw + tail_meanrank_reciprocal_raw) / 2,
                      (head_hits10_raw + tail_hits10_raw) / 2))
        print('-----Filter-----')
        head_meanrank_filter /= n_used_eval_triple
        head_meanrank_reciprocal_filter /= n_used_eval_triple
        head_hits10_filter /= n_used_eval_triple
        head_hits5_filter /= n_used_eval_triple
        head_hits3_filter /= n_used_eval_triple
        head_hits1_filter /= n_used_eval_triple
        tail_meanrank_filter /= n_used_eval_triple
        tail_meanrank_reciprocal_filter /= n_used_eval_triple
        tail_hits10_filter /= n_used_eval_triple
        tail_hits5_filter /= n_used_eval_triple
        tail_hits3_filter /= n_used_eval_triple
        tail_hits1_filter /= n_used_eval_triple
        # print('-----Head prediction-----')
        # print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_filter, head_hits10_filter))
        # print('-----Tail prediction-----')
        # print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_filter, tail_hits10_filter))

        mean_rank_filter = round((head_meanrank_filter + tail_meanrank_filter) / 2, 3)
        meanrank_reciprocal_filter = round((head_meanrank_reciprocal_filter + tail_meanrank_reciprocal_filter) / 2, 3)
        hits10_filter = round((head_hits10_filter + tail_hits10_filter) / 2, 3)
        hits5_filter = round((head_hits5_filter + tail_hits5_filter) / 2, 3)
        hits3_filter = round((head_hits3_filter + tail_hits3_filter) / 2, 3)
        hits1_filter = round((head_hits1_filter + tail_hits1_filter) / 2, 3)
        print('-----Average-----')
        print('MeanRank: {:.3f}, MeanRankReciprocal: {:.3f}, Hits@10: {:.3f}, Hits@5: {:.3f}, Hits@3: {:.3f}, '
              'Hits@1: {:.3f}'.format((head_meanrank_filter + tail_meanrank_filter) / 2,
                                      (head_meanrank_reciprocal_filter + tail_meanrank_reciprocal_filter) / 2,
                                      (head_hits10_filter + tail_hits10_filter) / 2,
                                      (head_hits5_filter + tail_hits5_filter) / 2,
                                      (head_hits3_filter + tail_hits3_filter) / 2,
                                      (head_hits1_filter + tail_hits1_filter) / 2)
              )
        print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
        print('-----Finish evaluation-----')
        
        

        with open('mrr.csv', 'a', encoding='utf-8') as f:
            f.write(str(mean_rank_raw) + ',' +
                    str(mean_rank_reciprocal_raw) + ',' +
                    str(hits10_raw) + ',' +
                    str(mean_rank_filter) + ',' +
                    str(meanrank_reciprocal_filter) + ',' +
                    str(hits10_filter) + ',' +
                    str(hits5_filter) + ',' +
                    str(hits3_filter) + ',' +
                    str(hits1_filter) + '\n'
                    )

    def calculate_rank(self, in_queue, out_queue):
        while True:
            idx_predictions = in_queue.get()
            if idx_predictions is None:
                in_queue.task_done()
                return
            else:
                eval_triple, idx_head_prediction, idx_tail_prediction = idx_predictions
                head, tail, relation = eval_triple
                head_rank_raw = 0
                tail_rank_raw = 0
                head_rank_filter = 0
                tail_rank_filter = 0
                for candidate in idx_head_prediction[::-1]:
                    if candidate == head:
                        break
                    else:
                        head_rank_raw += 1
                        if (candidate, tail, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            head_rank_filter += 1
                for candidate in idx_tail_prediction[::-1]:
                    if candidate == tail:
                        break
                    else:
                        tail_rank_raw += 1
                        if (head, candidate, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            tail_rank_filter += 1
                out_queue.put((head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter))
                in_queue.task_done()

    def check_norm(self, session):
        print('-----Check norm-----')
        entity_embedding = self.entity_embedding.eval(session=session)
        relation_embedding = self.relation_embedding.eval(session=session)
        concept_embedding = self.concept_embedding.eval(session=session)
        entity_norm = np.linalg.norm(entity_embedding, ord=2, axis=1)
        relation_norm = np.linalg.norm(relation_embedding, ord=2, axis=1)
        concept_norm = np.linalg.norm(concept_embedding, ord=2, axis=1)
        print('entity norm: {}'.format(entity_norm))
        print('relation norm: {}'.format(relation_norm))
        print('concept norm: {}'.format(concept_norm))

    def save_embedding(self, session):
        time1 = time.strftime('%Y%m%d',time.localtime(time.time()))
        self.saver.save(session, "../tmp/" + str(time1) + "model.ckpt", global_step=self.global_step)
        if not os.path.isdir('../vector/' + self.kg.data_dir):
            os.makedirs('../vector/' + self.kg.data_dir)
            
        f2 = open("../vector/" + self.kg.data_dir + "/relation2vec.vec", 'w', encoding='utf-8')
        f3 = open("../vector/" + self.kg.data_dir +  "/entity2vec.vec", 'w', encoding='utf-8')
        f4 = open("../vector/" + self.kg.data_dir + "/concept2vec.vec", 'w', encoding='utf-8')
        relation_array = self.relation_embedding.eval(session=session)
        entity_array = self.entity_embedding.eval(session=session)
        concept_array = self.concept_embedding.eval(session=session)
        concept_r_array = self.concept_r_embedding.eval(session=session)

        for i in range(self.kg.relation_num):
            for ii in range(self.embedding_dim):
                ii_tmp = str(round(relation_array[i][ii], 6))
                f2.write(ii_tmp + '\t')
            f2.write('\n')

        for i in range(self.kg.entity_num):
            for ii in range(self.embedding_dim):
                ii_tmp = str(round(entity_array[i][ii], 6))
                f3.write(ii_tmp + '\t')
            f3.write('\n')

        for i in range(self.kg.concept_num):
            for ii in range(self.embedding_dim):
                ii_tmp = str(round(concept_array[i][ii], 6))
                f4.write(ii_tmp + '\t')
            f4.write('\n')
            for ii in range(self.embedding_dim):
                ii_tmp = str(round(concept_r_array[i][ii], 6))
                f4.write(ii_tmp + '\t')
            f4.write('\n')
        f2.close()
        f3.close()
        f4.close()

