from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential, Graph
from keras.objectives import mean_squared_error 
from sklearn.preprocessing import StandardScaler 
from scipy.misc import imresize
import numpy as np
import scipy.io
import theano
import sys

def MLP(d,m,q):
    model = Sequential()
    model.add(Dense(m, input_dim=d, activation="relu"))
    model.add(Dense(m, activation="relu"))
    model.add(Dense(q))
    model.compile('rmsprop','mean_squared_error')
    return model

def weighted_loss(base_loss,l):
    def loss_function(y_true, y_pred):
        return l*base_loss(y_true,y_pred)
    return loss_function

def distillation(d,m,q,t,l):
    graph = Graph()
    graph.add_input(name='x', input_shape=(d,))
    graph.add_node(Dense(m), name='w1', input='x')
    graph.add_node(Activation('relu'), name='z1', input='w1')
    graph.add_node(Dense(m), name='w2', input='z1')
    graph.add_node(Activation('relu'), name='z2', input='w2')
    graph.add_node(Dense(q), name='w3', input='z2')
    graph.add_output(name='hard', input='w3')
    graph.add_output(name='soft', input='w3')

    loss_hard = weighted_loss(mean_squared_error,1.-l)
    loss_soft = weighted_loss(mean_squared_error,l)

    graph.compile('rmsprop', {'hard':loss_hard, 'soft':loss_soft})
    return graph

def load_data(fname, tag,arm,n=-1):
  mat  = scipy.io.loadmat(fname)
  x    = mat[tag][:,0:21]
  star = np.setdiff1d(range(21,28),[20+arm])
  xs   = mat[tag][:,star]
  y    = mat[tag][:,20+arm]

  if n > 0:
    i  = np.random.permutation(x.shape[0])[0:n]
    x  = x[i]
    xs = xs[i]
    y  = y[i]

  x  = x.astype(np.float32)
  xs = xs.astype(np.float32)
  y  = y.astype(np.float32)
  return x, xs, y

np.random.seed(0)
N = int(sys.argv[1])
M = 20

outfile = open('result_sarcos_' + str(N), 'w')

for A in [1,2,3,4,5,6,7]:
  ax_tr, axs_tr, ay_tr = load_data('/home/dlopez/data/sarcos_inv.mat', 'sarcos_inv', A)
  x_te, xs_te, y_te = load_data('/home/dlopez/data/sarcos_inv_test.mat', 'sarcos_inv_test', A)
  
  s_x   = StandardScaler().fit(ax_tr)
  s_xs  = StandardScaler().fit(axs_tr)
  s_y   = StandardScaler().fit(ay_tr)
  
  ax_tr  = s_x.transform(ax_tr)
  axs_tr = s_xs.transform(axs_tr)
  ay_tr  = s_y.transform(ay_tr)
  
  x_te  = s_x.transform(x_te)
  xs_te = s_xs.transform(xs_te)
  y_te  = s_y.transform(y_te)

  for rep in xrange(10):
    i     = np.random.permutation(ax_tr.shape[0])[0:N]
    x_tr  = ax_tr[i]
    xs_tr = axs_tr[i]
    y_tr  = ay_tr[i]

    # big mlp
    mlp_big = MLP(xs_tr.shape[1],M,1)
    mlp_big.fit(xs_tr, y_tr, nb_epoch=50, verbose=0)
    err_big = np.mean(np.power(mlp_big.predict_classes(xs_te,verbose=0)-y_te,2))
    
    # student mlp
    for t in [1,2,5,10,20,50]:
      for L in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        soften = theano.function([mlp_big.layers[0].input], mlp_big.layers[2].get_output(train=False))
        ys_tr  = soften(xs_tr)/t
        
        mlp_student = distillation(x_tr.shape[1],M,1,t,L)
        mlp_student.fit({'x':x_tr, 'hard':y_tr, 'soft':ys_tr}, nb_epoch=50, verbose=0)
        err_student = np.mean(np.power(mlp_student.predict({'x':x_te})['hard']-y_te,2))
        
        line = [N, A, round(err_big,3), t, L, round(err_student,3)]
        outfile.write(str(line)+'\n')
  
outfile.close()
