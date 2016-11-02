import random
import string

n_data = 100
n_class = 2 #start from 1
f = open('gt.txt','r')
f2 = open('overlapping_test.txt', 'w')
f3 = open('input.txt','r')

train_list = []
for line in f3:
    train_list.append(line.strip().split()[1])

f = open('gt.txt','r')
grammar_list = []
class_list = []
for line in f:
    grammar_list.append(line.strip().split()[1])
    class_list.append(line.strip().split()[0])

f = open('input.txt','r')
for action1_label_ind, line in enumerate(grammar_list):
    seq = list(line)
    for j in xrange(n_data):
        data_line = ''
        label_line = ""
        flag = False
        action2Ind = 0

        #find second lable
        label_flag = False
        while not label_flag:
            action2_label_ind = random.randint(0,n_class-1)
            if action2_label_ind != action1_label_ind:
                label_flag = True
        action2 = grammar_list[action2_label_ind]

        while not flag:
            repeated_num = random.randint(0,3)
            data_line = ''.join(random.sample(string.uppercase, repeated_num))
            label_line = ''.join('_'*repeated_num)
            for each in seq:
                repeated_num = random.randint(1, 5)
                data_line += each*repeated_num
                label_line += class_list[action1_label_ind]*repeated_num

                repeated_num = random.randint(0, 3)
                data_line += ''.join(random.sample(string.uppercase, repeated_num))
                label_line += '_'*repeated_num

                length = random.randint(0, 2)
                action2_seq = action2[action2Ind:action2Ind+length]
                action2Ind += length
                for item in action2_seq:
                    repeated_num = random.randint(1, 5)
                    data_line += item*repeated_num
                    label_line += class_list[action2_label_ind]*repeated_num

                    repeated_num = random.randint(0, 3)
                    data_line += ''.join(random.sample(string.uppercase, repeated_num))
                    label_line += '_'*repeated_num
            if data_line not in train_list:
               flag = True
        f2.write(label_line + ' ' + class_list[action1_label_ind] + ' ' + class_list[action2_label_ind] + ' ' + data_line + '\n')
