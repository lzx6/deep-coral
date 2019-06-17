# -*- coding: utf-8 -*-
"""
@ project: Deep_Coral
@ author: lzx
@ file: load_pkl.py
@ time: 2019/6/16 17:00
"""
import pickle
import matplotlib.pyplot as plt
'''coral'''
f = open('result_norm/test_s_sta.pkl','rb')
src_data = pickle.load(f)
f1 = open('result_norm/test_t_sta.pkl','rb')
tgt_data = pickle.load(f1)
f_training = open('result_norm/training_state.pkl','rb')
training_sta = pickle.load(f_training)
'''no coral'''
F = open('result_norm_no/test_s_sta.pkl','rb')
src_data_no = pickle.load(F)
F1 = open('result_norm_no/test_t_sta.pkl','rb')
tgt_data_no = pickle.load(F1)
print(training_sta[0])
epochs = []
accuracy_src = []
accuracy_tgt = []
accuracy_src_no = []
accuracy_tgt_no = []
loss_class = []
loss_coral = []
for i in range(len(src_data)):
    epochs.append(src_data[i]['epoch'])
    accuracy_src.append(src_data[i]['accuracy'])
    accuracy_tgt.append(tgt_data[i]['accuracy'])
    accuracy_src_no.append(src_data_no[i]['accuracy'])
    accuracy_tgt_no.append(tgt_data_no[i]['accuracy'])
    for j in range(len(training_sta[i])):
        loss_coral.append(training_sta[i][j]['coral_loss'])
        loss_class.append(training_sta[i][j]['classification_loss'])
# print(loss_class)
#
#
print(epochs,accuracy_src,accuracy_tgt)
# plt.subplot(211)
# plt.plot(epochs,accuracy_tgt,label = 'target acc',marker='*')
# plt.plot(epochs,accuracy_src,label = 'source acc',marker='^')
# plt.plot(epochs,accuracy_tgt_no,label = 'target acc no',marker='.')
# plt.plot(epochs,accuracy_src_no,label = 'source acc no',marker='+')
# plt.legend()
# plt.title('coral and no coral acc')
# plt.grid(True)
# plt.subplot(212)
plt.plot([x for x in range(len(loss_class))],loss_class,label = 'loss class',marker='*')
plt.plot([x for x in range(len(loss_coral))],loss_coral,label = 'loss coral',marker='^')
plt.legend()
plt.title('loss coral')
plt.grid(True)
plt.show()
