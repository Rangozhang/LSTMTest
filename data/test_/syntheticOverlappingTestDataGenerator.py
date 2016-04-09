import random
import string

n_data = 100
n_class = 10 #start from 1
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
 for ind, line in enumerate(grammar_list):
     seq = list(line)
     for j in xrange(n_data):
         data_line = ''
         flag = False
         action2Ind = 0
         label_flag = False
         while not label_flag:
             action2_label = random.randint(1,10)
             if action2_label != int(class_list[ind]):
                 label_flag = True
         action2 = grammar_list[action2_label-1]
         while not flag:
             data_line = ''.join(random.sample(string.uppercase, random.randint(0,3)))
             for each in seq:
                 repeated_num = random.randint(1, 5)
                 for w in xrange(repeated_num):
                     data_line += each
                 data_line += ''.join(random.sample(string.uppercase, random.randint(0,3)))
                 length = random.randint(0, 3)
                 action2_line = ''
                 action2_seq = action2[action2Ind:action2Ind+length]
                 action2Ind += length
                 for item in action2_seq:
                     repeated_num = random.randint(1, 5)
                     for w in xrange(repeated_num):
                         action2_line += item
                     action2_line += ''.join(random.sample(string.uppercase, random.randint(0, 3)))
                 data_line += action2_line
             if data_line not in train_list:
                flag = True
         f2.write(class_list[ind] + ' ' + class_list[action2_label-1] + ' ' + data_line + '\n')