# coding: UTF-8
import pickle
vocab = {}
vocab_d={10:'我',13:'我们',121:'你', 762:'你们', 60:'他', 303:'她', 252:'它', 66:'他们', 2256:'她们', 645:'它们', 50002:'我的', 50003:'我们的', 50004:'你的', 50005:'你们的', 50006:'他的', 50007:'她的', 50008:'它的',50009:'他们的', 50010:'她们的', 50011:'它们的', 50012:'我自己', 50013:'我们自己', 50014:'你自己', 50015:'你们自己', 50016:'他自己', 50017:'她自己', 50018:'他们自己', 50019:'她们自己', 50020:'它们自己', 50021:'它自己', 50022:'我自己的'}
with open('./vocab_list_path','rb') as f:
	vocab_dict = pickle.load(f)
keys=vocab_dict.keys()

num = 0	
for k in keys:
	if num <= 50023:
		vocab[vocab_dict[k]]=k
		num += 1
#input(vocab[303])
pro_ = []
pro_sents = []
pro_pre_file = open('pro_pre_file.txt', 'w')
with open('./pro_pre.txt', 'rb') as f:
	pro_batches = pickle.load(f)
	#input(pro_pre)
	for i in range(len(pro_batches)):
		for s in pro_batches[i]:
			pro_sents.append(s)
			#input(pro_sents)
	#input(len(pro_sents))
	for sent in pro_sents:
		pro_line = ""
		for ids in sent:
			#input(ids)
			#input(vocab_d.keys())
			if int(ids) in vocab.keys():
				#input(ids)
				pro_line = pro_line + vocab[int(ids)] + ' '
				#input(pro_line) 
			else:
				pro_line = pro_line + '@@' + vocab[int(ids/60000)] + ' ' 
		
		pro_.append(pro_line)
                #pro_pre_file.write(' '.join(pro_))
		pro_line = []
	print(pro_)
	input(len(pro_))
	for line in pro_:
        	pro_pre_file.write(line+"\n")
	#input(len(pro_))


	

