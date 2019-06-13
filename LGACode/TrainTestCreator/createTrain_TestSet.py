import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np 
filename = 'test.txt'
# filename2 = '1.txt'

X = np.genfromtxt(filename, delimiter = ',').astype(np.int32)
y = range(len(X))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.57, random_state=42)
np.savetxt('train_xxx.txt', X_train, delimiter = ',', fmt='%d')
np.savetxt('test_xxx.txt', X_test, delimiter = ',',fmt='%d')
print len(X_train)
print len(X_test)

# f = open(filename2,'w')
# f.write(X_text)


