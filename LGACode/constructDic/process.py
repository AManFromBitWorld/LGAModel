import copy
import numpy as np 
import pickle
from collections import Counter
filename = 'UserVideo_list.txt'

def getData(filename):
	data = np.genfromtxt(filename, dtype='|S', delimiter = ',')
	print data.shape
	user = np.unique(data[:,0])
	video = np.unique(data[:,1])

	mat = [(u,v) for u,v in data]
	mat = list(set(mat))

	processData(mat)



def processData(mat):

	userThreshold = 5
	videoThreshold = 5

	user, video = zip(*mat)

	user_count = Counter(user)
	video_count = Counter(video)

	print len(user_count)

	# new_user = [u for u in user_count if user_count[u]>userThreshold]
	new_user = dict((u,True) for u in user_count if user_count[u]>userThreshold)
	print len(new_user)

	# new_video = [v for v in video_count if video_count[v]>videoThreshold]
	new_video = dict((v, True) for v in video_count if video_count[v]>videoThreshold)


	new_mat = [(u,v) for u,v in mat if u in new_user and v in new_video]

	saveData(new_mat)



	# np.savetxt('userVideoThreshold.txt', new_mat, fmt='%d',delimiter=',')
	# print len(new_mat)

	# while len(user)!=len(new_user) or len(video)!=len(new_video):
	# 	user = copy.copy(new_user)
	# 	video = copy.copy(new_video)
	# 	new_mat = [(u,v) for u,v in mat if u in new_user and v in new_video]
	# 	user_count = Counter(zip(*new_mat)[0])
	# 	video_count = Counter(zip(*new_mat)[1])

	# 	new_user = dict((u,True) for u in user_count if user_count[u]>userThreshold)
	# 	new_video = dict((v, True) for v in video_count if video_count[v]>videoThreshold)
	# 	print len(new_user), len(new_video)
	# print len(new_mat)

def saveData(mat):
	userDict = dict((u,i+1) for i,u in enumerate(set(zip(*mat)[0])))
	videoDict = dict((v,i+1) for i,v in enumerate(set(zip(*mat)[1])))
	# print userDict.keys()
	# print videoDict.keys()
	# pickle.dump(userDict, open('userDict.pkl', 'wb'))
	# pickle.dump(videoDict, open("videoDict.pkl", 'wb'))

	new_mat = [[userDict[u],videoDict[v]] for u,v in mat]
	np.savetxt('user-video-pair.txt', new_mat, fmt='%d', delimiter = ',')



if __name__ == "__main__":
	getData(filename)
