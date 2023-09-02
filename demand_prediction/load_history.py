import pickle
from matplotlib import pyplot
import numpy as np
path = "modn=200n=200.pickle"
with open(path, 'rb') as f:
    history = pickle.load(f)
    

path2 = "modn=400n=400.pickle"
with open(path2, 'rb') as f:
    history2 = pickle.load(f)

path3 = "modn=600n=600.pickle"
with open(path3, 'rb') as f:
    history3 = pickle.load(f)

path4 = "modn=100n=100.pickle"
with open(path4, 'rb') as f:
    history4 = pickle.load(f)

path5 = "modn=800n=800.pickle"
with open(path5, 'rb') as f:
    history5 = pickle.load(f)


print(path,":",np.array(history['val_loss']).min())
print(path2,":",np.array(history2['val_loss']).min())
print(path3,":",np.array(history3['val_loss']).min())
print(path4,":",np.array(history4['val_loss']).min())
print(path5,":",np.array(history5['val_loss']).min())
#pyplot.plot(history['loss'], label='train')
pyplot.plot(history4['val_loss'], label='n = 100,100')
pyplot.plot(history['val_loss'], label='n = 200,200')
pyplot.plot(history2['val_loss'], label='n = 400,400')
pyplot.plot(history3['val_loss'], label='n = 600,600')
pyplot.plot(history5['val_loss'], label='n = 800,800')


pyplot.legend(loc='upper left')
pyplot.ylim(4,8)
pyplot.xlabel('epoch')
pyplot.ylabel('loss')



pyplot.savefig("load_history2.png")
pyplot.show()
pyplot.figure(0)
