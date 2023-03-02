from dataset import KnowledgeGraph
from transC_tf import TransC
import tensorflow as tf
import argparse
from classification_test_isA import test_isA
from classification_test_normal import test_triple
import os
import time
import timeit
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def save_output(epoch, args, instance_out_dict, subclass_out_dict, m_instance_out_dict, m_subclass_out_dict, triple_out_dict,delta_ins, delta_sub,  m_delta_ins, m_delta_sub):
    with open('args.csv', 'a', encoding='utf-8') as f:
        f.write(str(epoch) + ',' + 
                str(args.embedding_dim) + ',' + 
                str(args.batch_number) + ',' + 
                str(args.score_func) + ',' +
                str(args.margin_value) + ',' + 
                str(args.margin_instance_value) + ',' + 
                str(args.margin_subclass_value) + ',' +
                str(args.learning_rate) + ',' + 
                str(delta_ins) + ',' + 
                str(delta_sub) + ',' + 
                str(m_delta_ins) + ',' + 
                str(m_delta_sub) + ',' + 
                str(instance_out_dict['accuracy']) + ',' +
                str(instance_out_dict['precision']) + ',' + 
                str(instance_out_dict['recall']) + ',' +
                str(instance_out_dict['F1']) + ',' + 
                str(subclass_out_dict['accuracy']) + ',' +
                str(subclass_out_dict['precision']) + ',' + 
                str(subclass_out_dict['recall']) + ',' +
                str(subclass_out_dict['F1']) + ',' + 
                str(triple_out_dict['accuracy']) + ',' +
                str(triple_out_dict['precision']) + ',' + 
                str(triple_out_dict['recall']) + ',' +
                str(triple_out_dict['F1']) +  ',' +
                str(m_instance_out_dict['accuracy']) + ',' +
                str(m_instance_out_dict['precision']) + ',' + 
                str(m_instance_out_dict['recall']) + ',' +
                str(m_instance_out_dict['F1']) + ',' + 
                str(m_subclass_out_dict['accuracy']) + ',' +
                str(m_subclass_out_dict['precision']) + ',' + 
                str(m_subclass_out_dict['recall']) + ',' +
                str(m_subclass_out_dict['F1']) + ',' + 
                '\n'
                )


def main():
    parser = argparse.ArgumentParser(description='TransC')
    parser.add_argument('--data_dir', type=str, default='YAGO39K')
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--margin_value', type=float, default=1)
    parser.add_argument('--margin_instance_value', type=float, default=0.1)
    parser.add_argument('--margin_subclass_value', type=float, default=0.1)
    parser.add_argument('--score_func', type=str, default='L2')
    parser.add_argument('--batch_number', type=int, default=100)
    # parser.add_argument('--batch_number', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--n_generator', type=int, default=32)
    parser.add_argument('--n_rank_calculator', type=int, default=32)
    parser.add_argument('--ckpt_dir', type=str, default='../ckpt/')
    parser.add_argument('--summary_dir', type=str, default='../summary/')
    parser.add_argument('--max_epoch', type=int, default=5000)
    parser.add_argument('--eval_freq', type=int, default=500)
    args = parser.parse_args()
    # print(args)
    kg = KnowledgeGraph(data_dir=args.data_dir)
    kge_model = TransC(kg=kg, embedding_dim=args.embedding_dim, margin_value=args.margin_value, margin_instance_value=args.margin_instance_value,
                       margin_subclass_value=args.margin_subclass_value,
                       score_func=args.score_func, batch_number=args.batch_number, learning_rate=args.learning_rate,
                       n_generator=args.n_generator, n_rank_calculator=args.n_rank_calculator)
    gpu_config = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_config)
    with tf.Session(config=sess_config) as sess:
        print('-----Initializing tf graph-----')
        tf.global_variables_initializer().run()
        print('-----Initialization accomplished-----')
        kge_model.check_norm(session=sess)
        summary_writer = tf.summary.FileWriter(logdir=args.summary_dir, graph=sess.graph)
        for epoch in range(args.max_epoch):
            print('=' * 30 + '[EPOCH {}]'.format(epoch) + '=' * 30)
            kge_model.launch_training(session=sess, summary_writer=summary_writer, epoch=epoch)
            if (epoch + 1) % args.eval_freq == 0:
                kge_model.launch_evaluation(session=sess)
                print('-----save embedding-----')
                kge_model.save_embedding(session=sess)
                print('------test classification isA-----')
                instance_out_dict, subclass_out_dict, m_instance_out_dict, m_subclass_out_dict,delta_ins, delta_sub,  m_delta_ins, m_delta_sub = test_isA(args.data_dir)
                print('------test classification normal-----')
                triple_out_dict = test_triple(args.data_dir)
                save_output(epoch + 1, args, instance_out_dict, subclass_out_dict, m_instance_out_dict, m_subclass_out_dict, triple_out_dict,delta_ins, delta_sub,  m_delta_ins, m_delta_sub)

    print(args)


if __name__ == '__main__':
    code_start_time = time.asctime(time.localtime(time.time()))
    main()
    code_end_time = time.asctime(time.localtime(time.time()))
    print('start time:{}'.format(code_start_time))
    print('end time:{}'.format(code_end_time))

