# encoding=UTF-8
import numpy as np
import math
import pandas as pd
import timeit


class TestIsA:
    def __init__(self, data_set):
        self.data_set = data_set
        self.dim = 0
        self.delta_sub_max = []
        self.delta_sub_min = []
        self.delta_ins_max = []
        self.delta_ins_min = []
        self.delta_ins = 0
        self.delta_sub = 0
        self.get_max_min = True
        self.delta_sub_dim = []
        self.delta_ins_dim = []
        self.ins_test_num = 0
        self.sub_test_num = 0
        self.ins_wrong = []
        self.ins_right = []
        self.sub_wrong = []
        self.sub_right = []
        self.instance_vec = []
        self.concept_vec = []
        self.concept_r = []
        self.mix = False
        self.valid = True



    def load_vector(self):
        f1 = open("../vector/" + self.data_set + "/entity2vec.vec", 'r', encoding='utf-8')
        f2 = open("../vector/" + self.data_set + "/concept2vec.vec", 'r', encoding='utf-8')
        
        self.instance_vec = list()
        while True:
            line = f1.readline()
            if not line:
                break
            line = line.strip('\n').split('\t')[:-1]
            line_list = list(map(float, line))
            self.instance_vec.append(line_list)
        self.dim = len(self.instance_vec[0])
        self.concept_vec = list()
        self.concept_r = list()
        while True:
            line_concept = f2.readline().strip('\n')
            line_r = f2.readline().strip('\n')
            if not line_r:
                break
            line_concept = line_concept.split('\t')[:-1]
            line_concept_list = list(map(float, line_concept))
            self.concept_vec.append(line_concept_list)

            line_r = line_r.split('\t')[:-1]
            line_r_list = list(map(float, line_r))
            self.concept_r.append(line_r_list)
        
        def n_dim_statistics(concept_r):
            concept_r_array = np.array(concept_r)
            max_value = np.max(concept_r_array)
            min_value = np.min(concept_r_array)
            mean_value = np.mean(concept_r_array)
            varianc_value = np.var(concept_r_array)
            return max_value, min_value, mean_value, varianc_value
        # 统计半径分布
        with open('radius_distribution.txt', 'w', encoding='utf-8') as f:
            for temp_r_list in self.concept_r:
                for i in n_dim_statistics(temp_r_list):
                    f.write(str(i) + '\t')
                f.write('\n')

        return True
    
    def prepare(self):
        print('-----prepare data -----')
        if self.valid:
            if self.mix:
                fin = open("../../data/" + self.data_set + "/M-Valid/instanceOf2id_negative.txt", 'r', encoding='utf-8')
                fin_right = open("../../data/" + self.data_set + "/M-Valid/instanceOf2id_positive.txt", 'r', encoding='utf-8')
            else:
                fin = open("../../data/" + self.data_set + "/Valid/instanceOf2id_negative.txt", 'r', encoding='utf-8')
                fin_right = open("../../data/" + self.data_set + "/Valid/instanceOf2id_positive.txt", 'r', encoding='utf-8')
        else:
            if self.mix:
                fin = open("../../data/" + self.data_set + "/M-Test/instanceOf2id_negative.txt", 'r', encoding='utf-8')
                fin_right = open("../../data/" + self.data_set + "/M-Test/instanceOf2id_positive.txt", 'r', encoding='utf-8')
            else:
                fin = open("../../data/" + self.data_set + "/Test/instanceOf2id_negative.txt", 'r', encoding='utf-8')
                fin_right = open("../../data/" + self.data_set + "/Test/instanceOf2id_positive.txt", 'r', encoding='utf-8')

        self.ins_test_num = int(fin.readline().strip('\n'))
        self.ins_test_num = int(fin_right.readline().strip('\n'))

        self.ins_wrong = []
        self.ins_right = []
        for i in range(self.ins_test_num):
            tmp = list(map(int, fin.readline().strip('\n').split(' ')))
            self.ins_wrong.append((tmp[0], tmp[1]))
            tmp = list(map(int, fin_right.readline().strip('\n').split(' ')))
            self.ins_right.append((tmp[0], tmp[1]))

        fin.close()
        fin_right.close()

        if self.valid:
            if self.mix:
                fin = open("../../data/" + self.data_set + "/M-Valid/subClassOf2id_negative.txt", 'r', encoding='utf-8')
                fin_right = open("../../data/" + self.data_set + "/M-Valid/subClassOf2id_positive.txt", 'r', encoding='utf-8')
            else:
                fin = open("../../data/" + self.data_set + "/Valid/subClassOf2id_negative.txt", 'r', encoding='utf-8')
                fin_right = open("../../data/" + self.data_set + "/Valid/subClassOf2id_positive.txt", 'r', encoding='utf-8')
        else:
            if self.mix:
                fin = open("../../data/" + self.data_set + "/M-Test/subClassOf2id_negative.txt", 'r', encoding='utf-8')
                fin_right = open("../../data/" + self.data_set + "/M-Test/subClassOf2id_positive.txt", 'r', encoding='utf-8')
            else:
                fin = open("../../data/" + self.data_set + "/Test/subClassOf2id_negative.txt", 'r', encoding='utf-8')
                fin_right = open("../../data/" + self.data_set + "/Test/subClassOf2id_positive.txt", 'r', encoding='utf-8')

        self.sub_test_num = int(fin.readline().strip('\n'))
        self.sub_test_num = int(fin_right.readline().strip('\n'))

        self.sub_wrong = []
        self.sub_right = []
        for i in range(self.sub_test_num):
            tmp = list(map(int, fin.readline().strip('\n').split(' ')))
            self.sub_wrong.append((tmp[0], tmp[1]))
            tmp = list(map(int, fin_right.readline().strip('\n').split(' ')))
            self.sub_right.append((tmp[0], tmp[1]))

        fin.close()
        fin_right.close()

    def calculate_ins_dim_threshold(self):
        '''统计每一维度的最佳阈值'''
        print('-----start valid-----')
        self.delta_ins_max = [0 for _ in range(len(self.concept_vec[0]))]
        self.delta_ins_min = [0 for _ in range(len(self.concept_vec[0]))]

        best_ans = [0 for _ in range(len(self.concept_vec[0]))]

        def sum_dim(instance, concept):
            '''统计每一维度符合要求的数量，返回TP等值用于test'''
            dis = np.subtract(self.concept_r[concept], (np.abs(np.subtract(self.instance_vec[instance], self.concept_vec[concept]))))
            dis_temp_pos = np.where(dis >= self.delta_ins_dim, 1, 0)
            dis_temp_neg = np.where(dis < self.delta_ins_dim, 1, 0)
            # 统计最大值最小值，用于阈值设置
            if self.get_max_min:
                self.delta_ins_max = np.where(dis > self.delta_ins_max, dis, self.delta_ins_max)
                self.delta_ins_min = np.where(dis < self.delta_ins_min, dis, self.delta_ins_min)
            return dis_temp_pos, dis_temp_neg

        def test_dim():
            ans_array = np.array([[0 for _ in range(len(self.concept_vec[0]))] for _ in range(4)])
            for i in range(self.ins_test_num):
                TP, FN = sum_dim(self.ins_right[i][0], self.ins_right[i][1])
                FP, TN = sum_dim(self.ins_wrong[i][0], self.ins_wrong[i][1])
                ans_array[0] += TP
                ans_array[1] += FN
                ans_array[2] += TN
                ans_array[3] += FP

            return_ans = []
            for i in range(len(self.concept_vec[0])):
                return_ans.append((ans_array[0][i] + ans_array[2][i])*100 / (ans_array[0][i] + ans_array[1][i] + ans_array[2][i] + ans_array[3][i]))
            return return_ans
        
        self.delta_ins_dim = [0 for _ in range(self.dim)]
        # 统计一遍最大值最小值
        self.get_max_min = True
        current_ans = test_dim()
        # print('不设置阈值(为0）每一维度的accuracy：{}:'.format(current_ans))
        # print('最大值：{}, 最小值：{}'.format(self.delta_ins_max, self.delta_ins_min))
        self.get_max_min = False

        best_delta_ins_dim = [0 for _ in range(self.dim)]
        # 开始valid 各维度最佳阈值
        for i in range(100):
            for j in range(self.dim):
                self.delta_ins_dim[j] = self.delta_ins_min[j] + (self.delta_ins_max[j] - self.delta_ins_min[j]) * i / 100

            current_ans = test_dim()
            
            for k in range(self.dim):
                if current_ans[k] > best_ans[k]:
                    best_ans[k] = current_ans[k]
                    best_delta_ins_dim[k] = self.delta_ins_dim[k]
        
        for i in range(self.dim):
            self.delta_ins_dim[i] = best_delta_ins_dim[i]
        
        # print('每一维度最佳阈值：{}'.format(self.delta_ins_dim))
        # print('每一维度最佳accuracy：{}'.format(best_ans))
    
    def calculate_sub_dim_threshold(self):
        '''统计每一维度的最佳阈值'''
        print('-----start valid-----')
        self.delta_sub_max = [0 for _ in range(len(self.concept_vec[0]))]
        self.delta_sub_min = [0 for _ in range(len(self.concept_vec[0]))]

        best_ans = [0 for _ in range(len(self.concept_vec[0]))]

        def sum_dim(concept1, concept2):
            '''统计每一维度符合要求的数量，返回TP等值用于test'''
            dis = np.subtract(np.subtract(self.concept_r[concept2], self.concept_r[concept1]), np.abs(np.subtract(self.concept_vec[concept1], self.concept_vec[concept2])))
            dis_temp_pos = np.where(dis >= self.delta_sub_dim, 1, 0)
            dis_temp_neg = np.where(dis < self.delta_sub_dim, 1, 0)
            # 统计最大值最小值，用于阈值设置
            if self.get_max_min:
                self.delta_sub_max = np.where(dis > self.delta_sub_max, dis, self.delta_sub_max)
                self.delta_sub_min = np.where(dis < self.delta_sub_min, dis, self.delta_sub_min)
            return dis_temp_pos, dis_temp_neg

        def test_dim():
            ans_array = np.array([[0 for _ in range(len(self.concept_vec[0]))] for _ in range(4)])
            for i in range(self.sub_test_num):
                TP, FN = sum_dim(self.sub_right[i][0], self.sub_right[i][1])
                FP, TN = sum_dim(self.sub_wrong[i][0], self.sub_wrong[i][1])
                ans_array[0] += TP
                ans_array[1] += FN
                ans_array[2] += TN
                ans_array[3] += FP

            return_ans = []
            for i in range(len(self.concept_vec[0])):
                return_ans.append((ans_array[0][i] + ans_array[2][i])*100 / (ans_array[0][i] + ans_array[1][i] + ans_array[2][i] + ans_array[3][i]))
            return return_ans
        
        self.delta_sub_dim = [0 for _ in range(self.dim)]
        # 统计一遍最大值最小值
        self.get_max_min = True
        current_ans = test_dim()
        # print('不设置阈值(为0）每一维度的accuracy：{}:'.format(current_ans))
        # print('最大值：{}, 最小值：{}'.format(self.delta_sub_max, self.delta_sub_min))
        self.get_max_min = False

        best_delta_sub_dim = [0 for _ in range(self.dim)]
        # 开始valid 各维度最佳阈值
        for i in range(100):
            for j in range(self.dim):
                self.delta_sub_dim[j] = self.delta_sub_min[j] + (self.delta_sub_max[j] - self.delta_sub_min[j]) * i / 100

            current_ans = test_dim()
            
            for k in range(self.dim):
                if current_ans[k] > best_ans[k]:
                    best_ans[k] = current_ans[k]
                    best_delta_sub_dim[k] = self.delta_sub_dim[k]
        
        for i in range(self.dim):
            self.delta_sub_dim[i] = best_delta_sub_dim[i]
        
        # print('每一维度最佳阈值：{}'.format(self.delta_sub_dim))
        # print('每一维度最佳accuracy：{}'.format(best_ans))

    def run_valid(self):
        '''测试100维度中选择通过多少比例的维度判断为正样例'''
        print('-----start dim valid-----')
        
        ins_best_answer = 0
        ins_best_delta = 0
        sub_best_answer = 0
        sub_best_delta = 0
        for i in range(100):
            f =  i
            self.delta_ins = f
            self.delta_sub = f 
            ans = self.test()
            if ans[0] > ins_best_answer:
                ins_best_answer = ans[0]
                ins_best_delta = f
            if ans[1] > sub_best_answer:
                sub_best_answer = ans[1]
                sub_best_delta = f
        print("delta_ins is " + str(ins_best_delta) + ". The best ins accuracy on valid data is " + str(ins_best_answer)
            + "%")
        print("delta_sub is " + str(sub_best_delta) + ". The best sub accuracy on valid data is " + str(sub_best_answer)
            + "%")
        self.delta_ins = ins_best_delta
        self.delta_sub = sub_best_delta
        

    def test(self):
        TP_ins, TN_ins, FP_ins, FN_ins = 0, 0, 0, 0
        TP_sub, TN_sub, FP_sub, FN_sub = 0, 0, 0, 0
        TP_ins_dict, TN_ins_dict, FP_ins_dict, FN_ins_dict = dict(), dict(), dict(), dict()
        concept_set = dict()

        def check_instance(instance, concept):
            dis = np.subtract(self.concept_r[concept], (np.abs(np.subtract(self.instance_vec[instance], self.concept_vec[concept]))))
            dis_temp = np.where(dis > 0, dis, 0)
            
            if np.sum(dis_temp > 0) > self.delta_ins:
                return True
            else:
                return False


        def check_sub_class(concept1, concept2):
            dis = np.subtract(np.subtract(self.concept_r[concept2], self.concept_r[concept1]), 
                              np.abs(np.subtract(self.concept_vec[concept1], self.concept_vec[concept2])))
            dis_temp = np.where(dis > self.delta_sub_dim, 1, 0)
            if np.sum(dis_temp) > self.delta_sub:
                return True
            else:
                return False
            
        for i in range(self.ins_test_num):
            if check_instance(self.ins_right[i][0], self.ins_right[i][1]):
                TP_ins += 1
                if self.ins_right[i][1] in TP_ins_dict:
                    TP_ins_dict[self.ins_right[i][1]] += 1
                else:
                    TP_ins_dict[self.ins_right[i][1]] = 1
            else:
                FN_ins += 1
                if self.ins_right[i][1] in FN_ins_dict:
                    FN_ins_dict[self.ins_right[i][1]] += 1
                else:
                    FN_ins_dict[self.ins_right[i][1]] = 1

            if not check_instance(self.ins_wrong[i][0], self.ins_wrong[i][1]):
                TN_ins += 1
                if self.ins_wrong[i][1] in TN_ins_dict:
                    TN_ins_dict[self.ins_wrong[i][1]] += 1
                else:
                    TN_ins_dict[self.ins_wrong[i][1]] = 1
            else:
                FP_ins += 1
                if self.ins_wrong[i][1] in FP_ins_dict:
                    FP_ins_dict[self.ins_wrong[i][1]] += 1
                else:
                    FP_ins_dict[self.ins_wrong[i][1]] = 1
            concept_s = self.ins_right[i][1]
            concept_m = self.ins_wrong[i][1]
            concept_set[concept_s] = None
            concept_set[concept_m] = None

        for i in range(self.sub_test_num):
            if check_sub_class(self.sub_right[i][0], self.sub_right[i][1]):
                TP_sub += 1
            else:
                FN_sub += 1
            if not check_sub_class(self.sub_wrong[i][0], self.sub_wrong[i][1]):
                TN_sub += 1
            else:
                FP_sub += 1

        if self.valid:
            ins_ans = (TP_ins + TN_ins) * 100 / (TP_ins + TN_ins + FN_ins + FP_ins)
            sub_ins = (TP_sub + TN_sub) * 100 / (TP_sub + TN_sub + FN_sub + FP_sub)
            tmp_tuple = (ins_ans, sub_ins)
            return tmp_tuple
        else:
            instance_out_dict = {}
            print("instanceOf triple classification:")
            print("TP:{}, TN:{}, FP:{}, FN:{}".format(TP_ins, TN_ins, FP_ins, FN_ins))
            if TP_ins == 0:
                TP_ins = 1
            accuracy_ins = (TP_ins + TN_ins) * 100 / (TP_ins + TN_ins + FN_ins + FP_ins)
            precision_ins = TP_ins * 100 / (TP_ins + FP_ins)
            recall_ins = TP_ins * 100 / (TP_ins + FN_ins)
            F1_ins = 2 * precision_ins * recall_ins / (precision_ins + recall_ins)
            print("accuracy: {:.2f}%".format(accuracy_ins))
            print("precision: {:.2f}%".format(precision_ins))
            print("recall: {:.2f}%".format(recall_ins))
            print("F1-score: {:.2f}%".format(F1_ins))
            instance_out_dict['accuracy'] = round(accuracy_ins, 2)
            instance_out_dict['precision'] = round(precision_ins,2)
            instance_out_dict['recall'] = round(recall_ins, 2)
            instance_out_dict['F1'] = round(F1_ins, 2)


            subclass_out_dict = {}
            print("subClassOf triple classification:")
            print("TP:{}, TN:{}, FP:{}, FN:{}".format(TP_sub, TN_sub, FP_sub, FN_sub))
            if TP_sub == 0:
                TP_sub = 1
            accuracy_sub = (TP_sub + TN_sub) * 100 / (TP_sub + TN_sub + FN_sub + FP_sub)
            precision_sub = TP_sub * 100 / (TP_sub + FP_sub)
            recall_sub = TP_sub * 100 / (TP_sub + FN_sub)
            F1_sub = 2 * precision_sub * recall_sub / (precision_sub + recall_sub)
            print("accuracy: {:.2f}%".format(accuracy_sub))
            print("precision: {:.2f}%".format(precision_sub))
            print("recall: {:.2f}%".format(recall_sub))
            print("F1-score: {:.2f}%".format(F1_sub))

            subclass_out_dict['accuracy'] = round(accuracy_sub, 2)
            subclass_out_dict['precision'] = round(precision_sub, 2)
            subclass_out_dict['recall'] = round(recall_sub, 2)
            subclass_out_dict['F1'] = round(F1_sub)

        
            # don't understand

            """for item in sorted(concept_set):
                index = item
                TP_ins = TP_ins_dict[item]
                TN_ins = TN_ins_dict[item]
                FN_ins = FN_ins_dict[item]
                FP_ins = FP_ins_dict[item]
                accuracy = (TP_ins + TN_ins) * 100 / (TP_ins + TN_ins + FN_ins + FP_ins)
                precision = TP_ins * 100 / (TP_ins + FP_ins)
                recall = TP_ins * 100 / (TP_ins + FN_ins)
                p = TP_ins * 100 / (TP_ins + FP_ins)
                r = TP_ins * 100 / (TP_ins + FN_ins)
                f1 = 2 * p * r / (p + r)"""
            
            tmp_tuple = (instance_out_dict, subclass_out_dict,self.delta_ins, self.delta_sub)
            return tmp_tuple

    def run(self):
        self.load_vector()
        # prepare for valid to load valid data


        self.mix = False
        self.valid = True
        self.prepare()
        self.calculate_ins_dim_threshold()
        self.calculate_sub_dim_threshold()
        self.run_valid()
        # test
        self.valid = False
        self.prepare()
        instance_out_dict, subclass_out_dict, delta_ins, delta_sub = self.test()

        # m数据集
        self.mix = True
        self.valid = True
        self.prepare()
        self.calculate_ins_dim_threshold()
        self.calculate_sub_dim_threshold()
        self.run_valid()
        # test
        self.valid = False
        self.prepare()
        m_instance_out_dict, m_subclass_out_dict, m_delta_ins, m_delta_sub = self.test()
        return (instance_out_dict, subclass_out_dict, m_instance_out_dict, m_subclass_out_dict,delta_ins, delta_sub,  m_delta_ins, m_delta_sub)

def test_isA(data_set):
    test_isa_example = TestIsA(data_set)
    result_tuple = test_isa_example.run()
    return result_tuple

if __name__ == '__main__':
    test_isA('YAGO39K')