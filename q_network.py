# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:58:39 2015

@author: ka
"""
import lasagne
import theano
import theano.tensor as T
import numpy as np
from updates import deepmind_rmsprop

class DeepQLearner:
    """
    Deep Q-learning network using Lasagne.
    """

    def __init__(self, input_width, input_height, num_actions,
                 num_frames, discount, learning_rate, rho,
                 rms_epsilon, momentum, clip_delta, freeze_interval,
                 batch_size, update_rule,
                 batch_accumulator, state_count, input_scale=255.0):
                     
        self.state_count=state_count
        self.input_width = input_width
        self.input_height = input_height
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.discount = discount
        self.rho = rho
        self.lr = learning_rate
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        self.clip_delta = clip_delta
        self.freeze_interval = freeze_interval

        self.update_counter = 0
        
        self.l_out = self.build_nature_network_dnn(input_width, input_height,
                                        num_actions, num_frames, batch_size)
        
        if self.freeze_interval > 0:
            self.next_l_out = self.build_nature_network_dnn(input_width,
                                                 input_height, num_actions,
                                                 num_frames, batch_size)
            self.reset_q_hat()

        states = T.matrix('states')
        next_states = T.matrix('next_states')
        rewards = T.col('rewards')
        actions = T.icol('actions')
        terminals = T.icol('terminals')

#buferis inputu viso batch
        self.states_shared = theano.shared(
            np.zeros((batch_size, state_count),
                     dtype=theano.config.floatX))

#buferis i koki state patenka visiem
        self.next_states_shared = theano.shared(
            np.zeros((batch_size, state_count),
                     dtype=theano.config.floatX))

#po 1 reward kiekvienam episode, tai kaip del atskiru veiksmu?
        self.rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

#po 1 priimta action kiekvienam episode
        self.actions_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

#?? turbut 0 ir 1, ar paskutine verte ar ne
        self.terminals_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

#paima qvals ir nexxt qvals ir grazina skirtumus batchui, viskas tik pirmam kartui

        q_vals = lasagne.layers.get_output(self.l_out, states / input_scale)
        if self.freeze_interval > 0:
            next_q_vals = lasagne.layers.get_output(self.next_l_out,
                                                    next_states / input_scale)
        else:
            next_q_vals = lasagne.layers.get_output(self.l_out,
                                                    next_states / input_scale)
            next_q_vals = theano.gradient.disconnected_grad(next_q_vals)

        target = (rewards +
                  (T.ones_like(terminals) - terminals) *
                  self.discount * T.max(next_q_vals, axis=1, keepdims=True))
        diff = target - q_vals[T.arange(batch_size),
                               actions.reshape((-1,))].reshape((-1, 1))

#neaisku
        if self.clip_delta > 0:
            diff = diff.clip(-self.clip_delta, self.clip_delta)

        if batch_accumulator == 'sum':
            loss = T.sum(diff ** 2)
        elif batch_accumulator == 'mean':
            loss = T.mean(diff ** 2)
        else:
            raise ValueError("Bad accumulator: {}".format(batch_accumulator))


#
        params = lasagne.layers.helper.get_all_params(self.l_out)
        givens = {
            states: self.states_shared,
            next_states: self.next_states_shared,
            rewards: self.rewards_shared,
            actions: self.actions_shared,
            terminals: self.terminals_shared
        }
        if update_rule == 'deepmind_rmsprop':
            updates = deepmind_rmsprop(loss, params, self.lr, self.rho,
                                       self.rms_epsilon)
        elif update_rule == 'rmsprop':
            updates = lasagne.updates.rmsprop(loss, params, self.lr, self.rho,
                                              self.rms_epsilon)

        elif update_rule == 'adam':
            updates = lasagne.updates.adam(loss, params, self.lr, self.rho, self.rho,                                              self.rms_epsilon)
                                              
        elif update_rule == 'adagrad':
            updates = lasagne.updates.adagrad(loss, params, self.lr,
                                              self.rms_epsilon)
        elif update_rule == 'sgd':
            updates = lasagne.updates.sgd(loss, params, self.lr)
            
        elif update_rule == 'momentum':
            updates = lasagne.updates.momentum(loss, params, self.lr, self.momentum)

        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))

        if self.momentum > 0:
            updates = lasagne.updates.apply_momentum(updates, None,
                                                     self.momentum)

        self._train = theano.function([], [loss, q_vals], updates=updates,
                                      givens=givens)
        self._q_vals = theano.function([], q_vals,
                                       givens={states: self.states_shared})
                                        
    def build_nature_network_dnn(self, input_width, input_height, output_dim,
                                 num_frames, batch_size):
        """
        Build build_nature_network_dnn
        """
#        b=sqrt(6/(Neuronspries+Neuronspo))

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, self.state_count)
        )

        l_hidden0 = lasagne.layers.DenseLayer(
            l_in,
            num_units=500,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Uniform(range=(np.sqrt(6/(l_in.shape[1]+100.0))), std=None, mean=0.0)
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_hidden0,
            num_units=100,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Uniform(range=np.sqrt(6/(500.0+300.0)), std=None, mean=0.0)

        )
        
        l_hidden2 = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=300,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Uniform(range=np.sqrt(6/(100.0+100.0)), std=None, mean=0.0)
        )
#        l_hidden3 = lasagne.layers.DenseLayer(
#            l_hidden2,
#            num_units=100,
#            nonlinearity=lasagne.nonlinearities.rectify,
#            W=lasagne.init.HeUniform(),
#            b=lasagne.init.Uniform(range=np.sqrt(6/(300.0+300.0)), std=None, mean=0.0)
#        )
#        
#        l_hidden31 = lasagne.layers.DenseLayer(
#            l_hidden3,
#            num_units=300,
#            nonlinearity=lasagne.nonlinearities.rectify,
#            W=lasagne.init.HeUniform(),
#            b=lasagne.init.Uniform(range=np.sqrt(6/(100.0+100.0)), std=None, mean=0.0)
#        )
        
        l_hidden4 = lasagne.layers.DenseLayer(
            l_hidden2,
            num_units=100,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Uniform(range=np.sqrt(6/(300.0+output_dim)), std=None, mean=0.0)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden4,
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        return l_out
        
    def build_nature_network_dnn_test(self, input_width, input_height, output_dim,
                                 num_frames, batch_size):
        """
        Build build_nature_network_dnn
        """
#        from lasagne.layers import dnn

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, self.state_count)
        )

        l_hidden0 = lasagne.layers.DenseLayer(
            l_in,
            num_units=50,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.01)
        )



        l_out = lasagne.layers.DenseLayer(
            l_hidden0,
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        return l_out
        
#    def build_nature_network_dnn(self, input_width, input_height, output_dim,
#                                 num_frames, batch_size):
#        """
#        Build build_nature_network_dnn
#        """
#        from lasagne.layers import dnn
#
#        l_in = lasagne.layers.InputLayer(
#            shape=(batch_size, num_frames, input_width, input_height)
#        )
#
#        l_conv1 = dnn.Conv2DDNNLayer(
#            l_in,
#            num_filters=32,
#            filter_size=(8, 8),
#            stride=(4, 4),
#            nonlinearity=lasagne.nonlinearities.rectify,
#            W=lasagne.init.HeUniform(),
#            b=lasagne.init.Constant(.1)
#        )
#
#        l_conv2 = dnn.Conv2DDNNLayer(
#            l_conv1,
#            num_filters=64,
#            filter_size=(4, 4),
#            stride=(2, 2),
#            nonlinearity=lasagne.nonlinearities.rectify,
#            W=lasagne.init.HeUniform(),
#            b=lasagne.init.Constant(.1)
#        )
#
#        l_conv3 = dnn.Conv2DDNNLayer(
#            l_conv2,
#            num_filters=64,
#            filter_size=(3, 3),
#            stride=(1, 1),
#            nonlinearity=lasagne.nonlinearities.rectify,
#            W=lasagne.init.HeUniform(),
#            b=lasagne.init.Constant(.1)
#        )
#
#        l_hidden1 = lasagne.layers.DenseLayer(
#            l_conv3,
#            num_units=512,
#            nonlinearity=lasagne.nonlinearities.rectify,
#            W=lasagne.init.HeUniform(),
#            b=lasagne.init.Constant(.1)
#        )
#
#        l_out = lasagne.layers.DenseLayer(
#            l_hidden1,
#            num_units=output_dim,
#            nonlinearity=None,
#            W=lasagne.init.HeUniform(),
#            b=lasagne.init.Constant(.1)
#        )
#
#        return l_out
        
    def reset_q_hat(self):
        all_params = lasagne.layers.helper.get_all_param_values(self.l_out)
        lasagne.layers.helper.set_all_param_values(self.next_l_out, all_params)
        
    def train(self, states, actions, rewards, next_states, terminals):
        """
        Train one batch.

        Arguments:

        states - b x f x h x w numpy array, where b is batch size,
                 f is num frames, h is height and w is width.
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        next_states - b x f x h x w numpy array
        terminals - b x 1 numpy boolean array (currently ignored)

        Returns: average loss
        """        
        
        self.states_shared.set_value(states)
        self.next_states_shared.set_value(next_states)
        self.actions_shared.set_value(actions)
        self.rewards_shared.set_value(rewards)
        self.terminals_shared.set_value(terminals)
  #jeigu neisupadatina ir negereja resetina parametrus networko i pradinius
        if (self.freeze_interval > 0 and
            self.update_counter % self.freeze_interval == 0):
            self.reset_q_hat()
        loss, _ = self._train()
        self.update_counter += 1
        return np.sqrt(loss)
        
    def choose_action(self, state, epsilon):
        
        if epsilon < 0.97 and epsilon > 0.5:
            q_vals = self.q_vals(state)
#            
#        if np.random.rand() < epsilon:
#            
#            #more of 3 - kas antras 3 - biski greiciau apsimokina
#            if np.random.randint(0, 2)>0:
#                return 3
#            else:
#                return np.random.randint(0, self.num_actions-1)
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.num_actions)       

        q_vals = self.q_vals(state)
        return np.argmax(q_vals)
        
    def q_vals(self, state):
        states = np.zeros((self.batch_size, self.state_count), dtype=theano.config.floatX)
        states[0, :] = state
        self.states_shared.set_value(states)
        return self._q_vals()[0]