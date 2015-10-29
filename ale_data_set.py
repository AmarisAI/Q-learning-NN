# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 20:37:15 2015

@author: ka
"""
import theano
import numpy as np
#import shift

floatX = theano.config.floatX

class DataSet(object):
    """ Class represents a data set that stores a fixed-length history.
    """

    def __init__(self, inputCount, max_steps=1500, phi_length=1,
                 capacity=None):
        """  Construct a DataSet.
        Arguments:
            width,height - image size
            max_steps - the length of history to store.
            phi_length - number of images to concatenate into a state.
            capacity - amount of memory to allocate (just for debugging.)
        """
        self.inputCount=inputCount
        self.count = 0
        self.max_steps = max_steps
        self.phi_length = phi_length
        if capacity == None:
            self.capacity = max_steps + 1# + int(np.ceil(max_steps * .1))
        else:
            self.capacity = capacity
        self.states = np.zeros((self.capacity, self.inputCount), dtype='float32')
        self.actions = np.zeros(self.capacity, dtype='int32')
        self.rewards = np.zeros(self.capacity, dtype='float32')
        self.terminal = np.zeros(self.capacity, dtype='bool')
        self.prices = np.zeros((self.capacity, 4), dtype='float32')
        self.his_count = 0

    def add_sample(self, state, action, reward, terminal):
        self.states[self.count, ...] = state
        self.actions[self.count] = action
        self.rewards[self.count] = reward
        self.terminal[self.count] = terminal
        self.count += 1
        

        # Shift the final max_steps back to the beginning.
        if self.count == self.capacity:
            roll_amount = self.capacity - self.max_steps
            roll_amount=int(roll_amount)
            self.states = self.shift2d_float32(self.states, roll_amount)           
            self.actions = np.roll(self.actions, -roll_amount)
            self.rewards = np.roll(self.rewards, -roll_amount)
            self.terminal = np.roll(self.terminal, -roll_amount)
            self.count = self.max_steps
            
    def add_sample_history(self, state, action, reward, terminal, price):
        self.states[self.his_count, ...] = state
        self.actions[self.his_count] = action
        self.rewards[self.his_count] = reward
        self.terminal[self.his_count] = terminal
        self.prices[self.his_count,...] = price
        self.his_count += 1
        
    def history_reset(self):
        self.states = np.zeros((self.capacity, self.inputCount), dtype='float32')
        self.actions = np.zeros(self.capacity, dtype='int32')
        self.rewards = np.zeros(self.capacity, dtype='float32')
        self.terminal = np.zeros(self.capacity, dtype='bool')
        self.prices = np.zeros((self.capacity, 4), dtype='float32')
        self.his_count = 0
        
            
    def shift2d_float32(self, data, shift_amt):
        for i in xrange(shift_amt, data.shape[0]):
            data[i-shift_amt,...] = data[i,...]
        return (data)
        
    def _min_index(self):
        return max(0, self.count - self.max_steps)

    def _max_index(self):
        return self.count - (self.phi_length + 1)

    def __len__(self):
        """ Return the total number of avaible data items. """
        return max(0, (self._max_index() - self._min_index()) + 1)
        
    def random_batch(self, batch_size):

        count = 0
        states, actions, rewards, terminals, next_states = \
            self._empty_batch(batch_size)

#
#        states[count, ...] = [-1,-1,1,-1,-1]
#        actions[count, 0] = 3
#        rewards[count, 0] = -1
#        terminals[count, 0] = True
#        next_states[count, ...] = [-1,-1,1,-1,-1]
#        count += 1
#        
#        states[count, ...] = [1,1,1,-1,-1]
#        actions[count, 0] = 3
#        rewards[count, 0] = -1
#        terminals[count, 0] = True
#        next_states[count, ...] = [1,1,1,-1,-1]
#        count += 1
#
#
#        states[count, ...] = [-1,-1,-1,1,-1]
#        actions[count, 0] = 1
#        rewards[count, 0] = -1
#        terminals[count, 0] = True
#        next_states[count, ...] = [-1,-1,1,-1,-1]
#        count += 1
#        
#        states[count, ...] = [1,1,-1,1,-1]
#        actions[count, 0] = 2
#        rewards[count, 0] = -1
#        terminals[count, 0] = True
#        next_states[count, ...] = [1,1,1,-1,-1]
#        count += 1
#        
#        states[count, ...] = [-1,-1,-1,-1,1]
#        actions[count, 0] = 2
#        rewards[count, 0] = -1
#        terminals[count, 0] = True
#        next_states[count, ...] = [-1,-1,1,-1,-1]
#        count += 1
#        
#        states[count, ...] = [1,1,-1,-1,1]
#        actions[count, 0] = 2
#        rewards[count, 0] = -1
#        terminals[count, 0] = True
#        next_states[count, ...] = [1,1,-1,-1,1]
#        count += 1
#        


#GAL net nereikia riboti pavyzdziu GAL
#        while count < (batch_size/1):
#                    index = np.random.randint(self._min_index(), self._max_index()+1)
#                    end_index = index + self.phi_length - 1          
#            
#            #if np.alltrue(np.logical_not(self.terminal[index:index+1])): 
#                
#                #sisteminiai kai - is viso 5 + nepamirst next_states ta pati                
#                #pagalvoti apie max_q is visiskai kito actiono
#                #1- state[x,x,1,x,x] action [3]
#                #2- state[x,x,0,1,x] action [1,2] 
#                #3- state[x,x,0,x,1] action [1,2]   
##                if ((self.states[index,2] > 0 and self.actions[index]==3) \
##                or (self.states[index,2] > 0 and self.actions[index]==0) \
##                or (self.states[index,3] > 0 and (self.actions[index]==1 or self.actions[index]==2)) \
##                or (self.states[index,4] > 0 and (self.actions[index]==1 or self.actions[index]==2))):
#           # if self.rewards[index] == -1 or self.rewards[index] == 0:
#                    states[count, ...] = self.states[end_index,...]
#                    actions[count, 0] = self.actions[end_index]
#                    rewards[count, 0] = self.rewards[end_index]
#                    terminals[count, 0] = self.terminal[end_index]
#                    next_states[count, ...] = self.states[end_index+1,...]
#                    count += 1

        whileIndex=0
        while count < batch_size/2:
            index = np.random.randint(self._min_index(), self._max_index()+1)
            end_index = index + self.phi_length - 1          
            whileIndex+=1
            
            #if np.alltrue(np.logical_not(self.terminal[index:index+1])): 
#                if not ((self.states[index,2] > 0 and self.actions[index]==3) \
#                or (self.states[index,2] > 0 and self.actions[index]==0) \
#                or (self.states[index,3] > 0 and (self.actions[index]==1 or self.actions[index]==2)) \
#                or (self.states[index,4] > 0 and (self.actions[index]==1 or self.actions[index]==2))):
            if whileIndex>400:
                break            
            if (self.rewards[index]>0) :
                    states[count, ...] = self.states[end_index,...]
                    actions[count, 0] = self.actions[end_index]
                    rewards[count, 0] = self.rewards[end_index]
                    terminals[count, 0] = self.terminal[end_index]
                    next_states[count, ...] = self.states[end_index+1,...]
                    count += 1  
#Idesiu kelis sisteminius ir kitus visus kitus
#padaryti 0 -> 3 ir 1or2 -> 1 or 2, deti minimaliai: manual visalaik
        while count < batch_size*2/3:
            index = np.random.randint(self._min_index(), self._max_index()+1)
            end_index = index + self.phi_length - 1          
            
            #if np.alltrue(np.logical_not(self.terminal[index:index+1])): 
#                if not ((self.states[index,2] > 0 and self.actions[index]==3) \
#                or (self.states[index,2] > 0 and self.actions[index]==0) \
#                or (self.states[index,3] > 0 and (self.actions[index]==1 or self.actions[index]==2)) \
#                or (self.states[index,4] > 0 and (self.actions[index]==1 or self.actions[index]==2))):
            if (self.states[index,1] >0 or self.states[index,2]>0) and (self.actions[index] ==0 or self.actions[index] ==3) :
                    states[count, ...] = self.states[end_index,...]
                    actions[count, 0] = self.actions[end_index]
                    rewards[count, 0] = self.rewards[end_index]
                    terminals[count, 0] = self.terminal[end_index]
                    next_states[count, ...] = self.states[end_index+1,...]
                    count += 1        
        
        while count < batch_size*3/4:
            index = np.random.randint(self._min_index(), self._max_index()+1)
            end_index = index + self.phi_length - 1          
            
            #if np.alltrue(np.logical_not(self.terminal[index:index+1])): 
#                if not ((self.states[index,2] > 0 and self.actions[index]==3) \
#                or (self.states[index,2] > 0 and self.actions[index]==0) \
#                or (self.states[index,3] > 0 and (self.actions[index]==1 or self.actions[index]==2)) \
#                or (self.states[index,4] > 0 and (self.actions[index]==1 or self.actions[index]==2))):
            if not (((self.states[index,1] >0 or self.states[index,2]>0) and (self.actions[index] ==1 or self.actions[index] ==2)) or ((self.states[index,0] >0 ) and (self.actions[index] ==3))):
                    states[count, ...] = self.states[end_index,...]
                    actions[count, 0] = self.actions[end_index]
                    rewards[count, 0] = self.rewards[end_index]
                    terminals[count, 0] = self.terminal[end_index]
                    next_states[count, ...] = self.states[end_index+1,...]
                    count += 1
                    
        while count < batch_size:
            index = np.random.randint(self._min_index(), self._max_index()+1)
            end_index = index + self.phi_length - 1          
            
            #if np.alltrue(np.logical_not(self.terminal[index:index+1])): 
#                if not ((self.states[index,2] > 0 and self.actions[index]==3) \
#                or (self.states[index,2] > 0 and self.actions[index]==0) \
#                or (self.states[index,3] > 0 and (self.actions[index]==1 or self.actions[index]==2)) \
#                or (self.states[index,4] > 0 and (self.actions[index]==1 or self.actions[index]==2))):
            #if not (((self.states[index,3] >0 or self.states[index,4]>0) and (self.actions[index] ==1 or self.actions[index] ==2)) or ((self.states[index,2] >0 ) and (self.actions[index] ==3))):
            states[count, ...] = self.states[end_index,...]
            actions[count, 0] = self.actions[end_index]
            rewards[count, 0] = self.rewards[end_index]
            terminals[count, 0] = self.terminal[end_index]
            next_states[count, ...] = self.states[end_index+1,...]
            count += 1

        return states, actions, rewards, next_states, terminals
        
    def _empty_batch(self, batch_size):
        # Set aside memory for the batch
        states = np.empty((batch_size, self.inputCount),
                          dtype=floatX)
        actions = np.empty((batch_size, 1), dtype='int32')
        rewards = np.empty((batch_size, 1), dtype=floatX)
        terminals = np.empty((batch_size, 1), dtype=bool)

        next_states = np.empty((batch_size, self.inputCount), dtype=floatX)
        return states, actions, rewards, terminals, next_states