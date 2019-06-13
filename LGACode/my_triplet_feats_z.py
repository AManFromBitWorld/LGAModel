from keras.layers import Embedding, LSTM, Bidirectional, Dropout
import tensorflow as tf 
import argparse
import numpy as np 
import sys
from sklearn.metrics import roc_auc_score
import scipy.sparse as sp 
import pickle
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import operator
from tensorflow.python.ops import rnn, rnn_cell

def BiRNN(x, rnn_size, embed_size, n_steps,  reuse = False):

	with tf.variable_scope("lstm"):
		if reuse:
			tf.get_variable_scope().reuse_variables()
		x = tf.transpose(x, [1, 0, 2])
		x = tf.reshape(x, [-1, embed_size])
		x = tf.split(0, n_steps, x)

		lstm_fw_cell = rnn_cell.BasicLSTMCell(rnn_size, forget_bias=1.0)
		lstm_bw_cell = rnn_cell.BasicLSTMCell(rnn_size, forget_bias=1.0)

		outputs,_,_ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

		res = outputs[-1]
		return res



class Model():
	def __init__(self, args, infer = False):

		if infer:
			args.batch_size = args.n_item
		
		self.user = tf.placeholder(tf.int32, shape=[args.batch_size,1])
		self.pitem = tf.placeholder(tf.int32, shape=[args.batch_size,1])
		self.nitem = tf.placeholder(tf.int32, shape=[args.batch_size,1])
		if args.vision_flag:
			self.pvision = tf.placeholder(tf.float32, shape=[args.batch_size, args.vision_feat_size])
			self.nvision = tf.placeholder(tf.float32, shape=[args.batch_size, args.vision_feat_size])
		if args.text_flag:
			self.ptext = tf.placeholder(tf.int32, shape=[args.batch_size, args.seq_maxlen])
			self.ntext = tf.placeholder(tf.int32, shape=[args.batch_size, args.seq_maxlen])

			text_embed = Embedding(input_dim = args.vocab_size, output_dim = args.text_embed_size, input_length = args.seq_maxlen)

			ptext_embed = text_embed(self.ptext)
			ntext_embed = text_embed(self.ntext)

			ptext_rnn = BiRNN(ptext_embed, args.rnn_size, embed_size = args.text_embed_size, n_steps = args.seq_maxlen)
			ntext_rnn = BiRNN(ntext_embed, args.rnn_size, embed_size = args.text_embed_size, n_steps = args.seq_maxlen, reuse = True)


			# rnn_layer = Bidirectional(LSTM(output_dim = args.rnn_size,
			# 								input_dim = args.text_embed_size,
			# 								input_length = args.seq_maxlen,
			# 								dropout_W = 0.3,
			# 								dropout_U = 0.3))

			# ptext_rnn = rnn_layer(ptext_embed)
			# ntext_rnn = rnn_layer(ntext_embed)




		u_embed = Embedding(input_dim = args.n_user, output_dim = args.embedding_size, input_length = 1)
		i_embed = Embedding(input_dim = args.n_item, output_dim = args.embedding_size, input_length = 1)

		user_latent = tf.reshape(u_embed(self.user), [args.batch_size, args.embedding_size])
		pitem_latent = tf.reshape(i_embed(self.pitem), [args.batch_size, args.embedding_size])
		nitem_latent = tf.reshape(i_embed(self.nitem), [args.batch_size, args.embedding_size])



		# self.pscore = tf.reduce_sum(tf.mul(user_latent, pitem_latent),-1)
		# self.nscore = tf.reduce_sum(tf.mul(user_latent, nitem_latent),-1)
		if args.vision_flag==False and args.text_flag==False:
			self.pscore = self.scoring(user_latent, pitem_latent)
			self.nscore = self.scoring(user_latent, nitem_latent, reuse = True)

		elif args.vision_flag == False:
			self.pscore = self.scoring(user_latent, pitem_latent, text = ptext_rnn)
			self.nscore = self.scoring(user_latent, nitem_latent, text = ntext_rnn, reuse = True)
		elif args.text_flag == False:
			self.pscore = self.scoring(user_latent, pitem_latent, vision = self.pvision)
			self.nscore = self.scoring(user_latent, nitem_latent, vision = self.nvision, reuse = True)
		else:
			self.pscore = self.scoring(user_latent, pitem_latent, vision = self.pvision, text = ptext_rnn)
			self.nscore = self.scoring(user_latent, nitem_latent, vision = self.nvision, text = ntext_rnn, reuse = True)




		self.loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(self.pscore - self.nscore)))
		self.train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)

		# print self.pscore.get_shape()
		for v in tf.trainable_variables():
			print v, v.name



	def scoring(self, user, item, vision = None, text = None, reuse = False):
		if reuse:
			tf.get_variable_scope().reuse_variables()

		vec = tf.concat(1,[user, item, tf.mul(user,item)])
		if vision != None:
			# vec = tf.concat(1, [vec, vision])
			vision_text_vec = vision 
		if text != None:
			vision_text_vec = tf.concat(1,[vision_text_vec, text])
			# vec = tf.concat(1, [vec, text])
		# vec = tf.concat(1, [user, item, tf.mul(user,item),feature])

		if vision != None or text != None:

			hidden_size = [128,64,5]
			vision_text_size = vision_text_vec.get_shape().as_list()[-1]
			for i, size in enumerate(hidden_size):
				with tf.variable_scope('hidden_vision_text'+str(i)):
					if i==0:
						shape = [vision_text_size,size]
					else:
						shape = [hidden_size[i-1], size]
					w = tf.get_variable('w'+str(i), shape, initializer = tf.random_normal_initializer(stddev=0.02))
					b = tf.get_variable('b'+str(i), [size], initializer = tf.constant_initializer(0.0))
					# print i, prev_layer.get_shape(), w.get_shape()
					vision_text_vec = tf.matmul(vision_text_vec, w)+b
					# vision_text_vec = tf.nn.sigmoid(vision_text_vec)

			vec = tf.concat(1, [vec, vision_text_vec])


		vec_size = vec.get_shape().as_list()[-1]


		hidden_size = [128,32,1]
		hidden_layer = vec
		for i,size in enumerate(hidden_size):
			with tf.variable_scope('hidden'+str(i)):
				if i==0:
					shape = [vec_size,size]
				else:
					shape = [hidden_size[i-1],size]
				w = tf.get_variable('w'+str(i), shape, initializer = tf.random_normal_initializer(stddev=0.02))
				b = tf.get_variable('b'+str(i), [size], initializer = tf.constant_initializer(0.0))
				# print i, prev_layer.get_shape(), w.get_shape()
				hidden_layer = tf.matmul(hidden_layer, w)+b
		return hidden_layer



		# vec_size = vec.get_shape().as_list()[-1]

		# with tf.variable_scope("hiden1"):
		# 	w1 = tf.get_variable('w1',[vec_size,64], initializer = tf.random_normal_initializer(stddev = 0.02))
		# 	b1 = tf.get_variable('b1', [64], initializer = tf.constant_initializer(0.0))
		# 	hidden = tf.matmul(vec,w1)+b1

		# with tf.variable_scope('score'):
		# 	w2 = tf.get_variable('w2',[64,1], initializer = tf.random_normal_initializer(stddev=0.02))
		# 	b2 = tf.get_variable('b2', [1], initializer = tf.constant_initializer(0.0))
		# 	output = tf.matmul(hidden, w2)+b2
		# return output







filepath = '/home/jingwei/Desktop/dl/'
class DataLoader():
	def __init__(self,batch_size):
		self.batch_size = batch_size

		train = np.genfromtxt(filepath+'user-video-pair-train2.txt',dtype='|S', delimiter = ',')
		test = np.genfromtxt(filepath+'user-video-pair-test2.txt',dtype='|S', delimiter = ',')
		data = np.concatenate([train,test], axis=0).astype(np.int32)

		self.n_user = len(np.unique(data[:,0]))
		self.n_item = len(np.unique(data[:,1]))



		data = data[:len(train)]


		self.matrix = sp.lil_matrix((self.n_user, self.n_item))

		self.user_train = {}

		for record in data:
			u,v = record[0]-1, record[1]-1
			self.matrix[u, v] = 1
			self.user_train.setdefault(u,[]).append(v)

		self.matcoo = self.matrix.tocoo()
		self.train_size = len(self.matcoo.row)

		#######################read vision feature####################
		self.vision_mat = np.load('vision-feats.npy')
		self.vision_feat_size = self.vision_mat.shape[1]


		#######################read text feature#####################


		self.process_text()
		# fr = open(filepath+'u.item')
		# data = fr.readlines()
		# fr.close()
		# vision_mat = []
		# for line in data:
		# 	line = line.strip()
		# 	listfromline = line.split('|')
		# 	vision_mat.append(map(int,listfromline[-19:]))
		# self.vision_mat = np.array(vision_mat)
		# self.vision_feat_size = self.vision_mat.shape[1]


		
		# self.matrix = np.zeros((self.n_user, self.n_item))
		# for record in data:
		# 	self.matrix[record[0]-1,record[1]-1] = record[2]
		# print self.n_user, self.n_item

	def process_text(self):
		videoDict = pickle.load(open('videoDict.pkl','rb'))
		sorted_videoId = sorted(videoDict.items(), key = operator.itemgetter(1))

		# tweets = np.genfromtxt('tweets.txt', delimiter='#~~#', dtype = '|S')
		tweets = [ line.strip().split(" ## ") for line in open('exp_sameTweetsWithoutID(Video_Tweets).txt').readlines()]
		tweetsDict = dict((url,tweet) for url,tweet in tweets)


		filtered_tweets = []
		for url in zip(*sorted_videoId)[0]:
			if url not in tweetsDict:
				print " [!]Error, url not in the file"
				filtered_tweets.append("")
			else:
				filtered_tweets.append(tweetsDict[url])

		words = []
		for sent in filtered_tweets:
			words.extend(sent.lower().split())
		word_counter = Counter(words)

		self.vocab = [word for word in word_counter if word_counter[word]>10]
		self.vocab_size = len(self.vocab)+1
		self.word2idx = dict((w,i+1) for i,w in enumerate(self.vocab))

		self.sequences = [[self.word2idx[word] for word in sent.lower().split() if word in self.word2idx] for sent in filtered_tweets]
		self.seq_maxlen = max(map(len,self.sequences))
		self.sequences = pad_sequences(self.sequences, self.seq_maxlen)
		#self.sequences = pad_sequences(self.sequences, padding = 'post', maxlen = 4)

		print 'pad_sequence', self.sequences.shape, self.vocab_size


	def sample_negative(self,u):
		res = []
		for i, user in enumerate(u):
			if user not in self.user_train:
				res.append(np.random.randint(0,self.n_item))
			else:
				filtered_index = list(set(range(self.n_item)).difference(set(self.user_train[user])))
				res.append(np.random.choice(filtered_index))
		return np.array(res)

	def next_batch(self):
		begin = self.pointer*self.batch_size
		end = (self.pointer+1)*self.batch_size
		self.pointer+=1
		# nitem = np.random.randint(0,self.n_item,(self.batch_size))
		nitem = self.sample_negative(self.matcoo.row[begin:end])

		return self.matcoo.row[begin:end].reshape(-1,1), \
				self.matcoo.col[begin:end].reshape(-1,1), self.vision_mat[self.matcoo.col[begin:end]], self.sequences[self.matcoo.col[begin:end]], \
				nitem.reshape(-1,1), self.vision_mat[nitem], self.sequences[nitem]
				# np.random.randint(0,self.n_item, (self.batch_size, 1))
		# user = []
		# pitem = []
		# nitem = []
		# for i in range(self.n_user):
		# 	user.append(i)
		# 	a,b = np.random.choice(range(self.n_item), 2, replace = False)
		# 	while self.matrix[i,a]==self.matrix[i,b]:
		# 		a,b = np.random.choice(range(self.n_item),2,replace = False)
		# 	if self.matrix[i,a]>self.matrix[i,b]:
		# 		pitem.append(a)
		# 		nitem.append(b)
		# 	else:
		# 		pitem.append(b)
		# 		nitem.append(a)
		# return np.array(user).reshape(-1,1), np.array(pitem).reshape(-1,1), np.array(nitem).reshape(-1,1)

	def val_data(self):
		# test = np.genfromtxt(filepath+'ua.test').astype(np.int32)
		# matrix = sp.lil_matrix((self.n_user, self.n_item))
		# for record in test:
		# 	matrix[record[0]-1, record[1]-1] = int(record[2]>=4)
		test = np.genfromtxt(filepath+'user-video-pair-test2.txt',delimiter = ',').astype(np.int32)
		matrix = np.zeros((self.n_user, self.n_item))
		for record in test:
			matrix[record[0]-1,record[1]-1] = 1

		return matrix
		# ratings = []
		# for i in range(self.n_user):
		# 	users.extend([i]*self.n_item)
		# 	items.extend(range(self.n_item))
		# 	ratings.extend(list(matrix[i]))
		# return np.array(users).reshape(-1,1), np.array(items).reshape(-1,1), ratings
		# return test[:,0].reshape(-1,1)-1, test[:,1].reshape(-1,1)-1, test[:,2].reshape(-1,1)
	def reset_pointer(self):
		self.pointer = 0


# def validation(model, data_loader, sess, batch_size):
# 	val_user, val_item, val_score = data_loader.val_data()
# 	total_batch = int(len(val_user)/batch_size)
# 	val_score = val_score[:total_batch*batch_size]
# 	y_preds = []
# 	for i in range(total_batch):
# 		begin = i*batch_size
# 		end = (i+1)*batch_size
# 		temp_val_user = val_user[begin:end]
# 		temp_val_item = val_item[begin:end]

# 		temp_val_pfeats = data_loader.vision_mat[temp_val_item]

# 		temp_val_pfeats = temp_val_pfeats.reshape(batch_size, data_loader.vision_feat_size)

# 		temp_val_score = val_score[begin:end]
# 		pred_score = sess.run(model.pscore, feed_dict = {model.user: temp_val_user,
# 														model.pitem:temp_val_item,
# 														model.pvision: temp_val_pfeats})
# 		# temp_val_score =  map(int,temp_val_score>=3)
# 		# print pred_score.dtype, temp_val_score.dtype
# 		# for i in range(len(temp_val_score)):
# 		# 	print pred_score[i], temp_val_score[i]
# 		# pred_score
# 		y_preds.extend(list(pred_score.reshape(batch_size)))
		# res.append(roc_auc_score(temp_val_score, pred_score))
	# pred_score = sess.run(model.pscore, feed_dict = {model.user:val_user, model.pitem:val_item})
	# val_auc_score = roc_auc_score(val_score, pred_score)
	# print ("\tvalidation roc_auc_score:{}".format(roc_auc_score(val_score, y_preds)))

def test(args, k=5):
	data_loader = DataLoader(args.batch_size)
	args.n_user = data_loader.n_user
	args.n_item = data_loader.n_item

	print args.n_user, args.n_item

	args.vision_feat_size = data_loader.vision_feat_size
	args.seq_maxlen = data_loader.seq_maxlen
	args.vocab_size = data_loader.vocab_size
	# args.batch_size = data_loader.n_user
	model = Model(args, infer = True)

	saver = tf.train.Saver(tf.all_variables())

	auc = []
	precision = []

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		ckpt = tf.train.get_checkpoint_state("./checkpoint/")
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print (" [!] Load parameters success!!!")
		else:
			print (" [*] Load parameters failed...")
		mat = data_loader.val_data()

		for uid, row in enumerate(mat):
			users = np.array([uid]*len(row)).reshape(-1,1)
			items = np.array(range(len(row))).reshape(-1,1)
			ratings = row

			feed = {model.user: users, model.pitem: items}
			if args.vision_flag:
				feed[model.pvision] = data_loader.vision_mat
			if args.text_flag:
				feed[model.ptext] = data_loader.sequences

			prediction = sess.run(model.pscore, feed_dict = feed)
			prediction = prediction.reshape(-1)

			if uid not in data_loader.user_train:
				continue

			#### filter for test
			#filtered_index = list(set(range(len(row))).difference(set(data_loader.user_train[uid])))

			#if len(np.unique(row[filtered_index]))==1:
			#	continue

			#auc.append(roc_auc_score(row[filtered_index], prediction[filtered_index]))

			#print prediction.shape, uid, len(mat), len(row[filtered_index])
			#### filter for test
			
			#### trick
			if len(np.unique(row))==1:
				continue
			auc.append(roc_auc_score(row, prediction))
			print prediction.shape, uid, len(mat), len(row)
			#### trick
			
			top_k = set(np.argsort(-prediction)[:k])
			true_pid = set(np.where(row == 1)[0])
			precision.append(len(top_k&true_pid)/float(k))

		print ("roc_auc_score:{}. precision@{}:{}".format(np.mean(auc), k, np.mean(precision)))





def train(args):
	data_loader = DataLoader(args.batch_size)
	args.n_user = data_loader.n_user
	args.n_item = data_loader.n_item
	args.vision_feat_size = data_loader.vision_feat_size
	args.seq_maxlen = data_loader.seq_maxlen
	args.vocab_size = data_loader.vocab_size
	# args.batch_size = data_loader.n_user
	model = Model(args)

	saver = tf.train.Saver(tf.all_variables())
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		ckpt = tf.train.get_checkpoint_state("./checkpoint/")
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print (" [!] Load parameters success!!!")
		for e in range(args.nb_epochs):
			data_loader.reset_pointer()
			# validation(model, data_loader, sess, args.batch_size)
			total_batch = int(data_loader.train_size/args.batch_size)
			for b in range(total_batch):
				user,pitem, pvision, ptext, nitem, nvision, ntext = data_loader.next_batch()


				feed = {model.user:user, model.pitem:pitem, model.nitem: nitem}
				if args.vision_flag:
					feed[model.pvision] = pvision
					feed[model.nvision] = nvision
				if args.text_flag:
					feed[model.ptext] = ptext
					feed[model.ntext] = ntext

				loss,_ = sess.run([model.loss, model.train_op], feed_dict = feed)


				sys.stdout.write("\r {}/{}, {}/{}. loss:{}".format(e,args.nb_epochs,b,total_batch,loss))
				sys.stdout.flush()

				if (e*args.nb_batches+b)%1000 == 0 or (e == args.nb_epochs-1 and b == args.nb_batches-1):
					saver.save(sess, './checkpoint/'+"model.ckpt", global_step = e*args.nb_batches+b)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--n_user", type=int, default = 943, help="number of users")
	parser.add_argument("--n_item", type=int, default = 1682, help = 'number of items')
	parser.add_argument("--batch_size", type=int, default = 64, help = 'batch size for training')
	parser.add_argument("--nb_epochs", type = int, default = 100, help = 'number of epochs')
	parser.add_argument("--nb_batches", type=int, default = 100, help='number of batches')
	parser.add_argument('--embedding_size', type=int, default=64, help = 'embedding size of latent factors')
	parser.add_argument("--vision_feat_size", type=int, default = 19, help = 'size of movie vision features')
	parser.add_argument("--vision_flag", type=bool, default = True, help = 'enable vision features')
	parser.add_argument("--text_flag", type = bool, default = True, help = 'enable text features')
	parser.add_argument("--seq_maxlen", type = int, default = 100, help = 'maximum length of sentence')
	parser.add_argument("--vocab_size", type = int, default = 100, help = 'vocabulary size of tweets')
	parser.add_argument("--text_embed_size", type = int, default = 128, help = 'vocabulary size of tweets')
	parser.add_argument("--rnn_size", type = int, default = 128, help = 'size of rnn output layers')
	parser.add_argument("--train_test", type = str, default = 'train', help = 'size of rnn output layers')
	args = parser.parse_args()
	if args.train_test == 'train':
		train(args)
	else:
		test(args)


	# model = Model(args)





if __name__ == "__main__":
	main()