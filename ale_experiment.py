# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 23:07:47 2015

@author: ka
"""
import logging
import numpy as np
import ale_data_set
import matplotlib.pyplot as plt
import time
from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc

class ALEExperiment(object):
    def __init__ (self, ale, agent, resized_width, resized_height,
                 num_epochs, epoch_length, test_length, death_ends_episode):
        
        self.ale = ale
        self.agent = agent
        self.num_epochs = num_epochs
        self.epoch_length = epoch_length
        self.test_length = test_length
        self.death_ends_episode = death_ends_episode
        self.min_action_set = ale.getActionCount()
        self.resized_width = resized_width
        self.resized_height = resized_height
        #TODO: cia buvo 3d for rgb, neaisku ar veiks neconvo toks. 2. Dimensijos inputu
        self.screen_buffer= np.empty((1,ale.getInputCount()),float)
        self.counter=0

        self.buffer_length = 2
        self.buffer_count = 0

        self.terminal_lol = False # Most recent episode ended on a loss of life

        self.history = ale_data_set.DataSet(self.agent.network.state_count,
                                            max_steps=self.ale.timeseries.getTestLen()-200,
                                            phi_length=1)#self.phi_length
        plt.ion()

        self.fig=plt.figure()
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax1)
        
        


    def run(self):
        """
        Run the desired number of training epochs, a testing epoch
        is conducted after each training epoch.
        """
        for epoch in range(1, self.num_epochs + 1):
             self.run_epoch(epoch, self.epoch_length)
            # issaugo networka cpickle
             self.agent.finish_epoch(epoch)

             if self.test_length > 0:
                 self.agent.start_testing()
                 self.run_epoch(epoch, self.test_length, True)
                 self.agent.finish_testing(epoch)
                 
                 if epoch==1 or epoch % 50 ==0:
                     start_time = time.time()
                     self.plotHistory()
                     total_time = time.time() - start_time
                     print("total time:")
                     print(total_time) #6.5s
                     print((self.history.actions[0:500]))

                 
    def run_epoch(self, epoch, num_steps, testing=False):
        """ Run one 'epoch' of training or testing, where an epoch is defined
        by the number of steps executed.  Prints a progress report after
        every trial

        Arguments:
        epoch - the current epoch number
        num_steps - steps per epoch
        testing - True if this Epoch is used for testing and not training

        """
        self.terminal_lol = False # Make sure each epoch starts with a reset.
        steps_left = num_steps
        
        
        if testing:
            steps_left=self.ale.timeseries.getTestLen()-200 #
            self.ale.reset_game_testing_first()
            self.counter=0
            self.history.history_reset()
            
        while steps_left > 0:
            prefix = "testing" if testing else "training"
            logging.info(prefix + " epoch: " + str(epoch) + " steps_left: " +
                         str(steps_left))
                
            _, num_steps = self.run_episode(steps_left, testing)

            steps_left -= num_steps
    
    def run_episode(self, max_steps, testing):
        """Run a single training episode.

        The boolean terminal value returned indicates whether the
        episode ended because the game ended or the agent died (True)
        or because the maximum number of steps was reached (False).
        Currently this value will be ignored.

        Return: (terminal, num_steps)

        """
        if testing:
                self.ale.reset_game_testing()
        else:
                self.ale.reset_game()
        
        # Make sure the screen buffer is filled at the beginning of
        # each episode...
        if testing:
            self._act(0,True,True)
        else: 
            self._act(0,False,True)

        #null action +setup parameters
        action = self.agent.start_episode(self.screen_buffer,testing)
        num_steps = 1
        while True:
            #jei netestina, tai blogai ismoko je grazinu 0 reward pirmam
        
            reward = self._act(action, testing)
            
            self.terminal_lol = (self.death_ends_episode and not testing)
            terminal = self.ale.game_over() or self.terminal_lol
            num_steps += 1            
            
            if terminal or num_steps >= max_steps:
                self.agent.end_episode(reward, terminal)
                break

            action = self.agent.step(reward, self.screen_buffer)

        return terminal, num_steps
        
    def _act(self, action, testing=False, beginning=False):
        """Perform the indicated action for a single frame, return the
        resulting reward and store the resulting screen image in the
        buffer

        """
        if testing:
            prices=self.ale.getCurrentPrice()
            states=self.ale.getCurrentState(testing)
        reward = self.ale.act(action, testing)
        if beginning==True:
            reward=0
            self.ale.gameOver = 0
        if testing:
            self.history.add_sample_history(states, action, reward, self.ale.gameOver, prices)
        #index = self.buffer_count % self.buffer_length
 #print(reward)

        self.screen_buffer=self.ale.getCurrentState(testing)
        self.buffer_count += 1
        self.counter += 1 #siaip nereikalingas sitas turbut
        return reward
        
    def plotHistory(self):

        #kaip jforex plotas pirkimas - linija - uzdarymas. Spalva sekmingas ar ne

                
        
        x = np.linspace(0,len(self.history.rewards)-1,len(self.history.rewards) )
        y= self.history.prices[:,0]
        y2= np.cumsum(self.history.rewards[:])

        self.ax1.clear()
        self.ax2.clear()
        
        print((self.history.actions[0:500]))
        
        xx=[]
        yy=[]
        xx2=[]
        yy2=[]
        openi=0
        close=0
        x0=[]
        y0=[]
        x1=[]
        y1=[]        
        
#        if self.agent.epsilon<0.95:
#            a=1
        
        
        for i in range(0,len(self.history.actions)):
            if self.history.actions[i]==1: 
                xx.append(self.history.prices[i,0])
                x0.append(i)
                openi=1
            if openi==1:
                if self.history.terminal[i]==True:
                    yy.append(self.history.prices[i,0])
                    y0.append(i)
                    openi=0
            if self.history.actions[i]==2: 
                xx2.append(self.history.prices[i,0])
                x1.append(i)
                close=1
            if close==1:
                if self.history.terminal[i]==True:
                    yy2.append(self.history.prices[i,0])
                    y1.append(i)
                    close=0
        if len(x0)!=len(y0):
            del x0[-1]
        if len(x1)!=len(y1):
            del x1[-1]
            
        if len(xx)==len(yy) and len(xx2)==len(yy2):
            self.ax1.plot([x0,y0],[xx,yy], 'xr-')
            self.ax1.plot([x1,y1],[xx2,yy2], 'xb-')
        
        #line1, = self.ax1.plot(x, y, 'k-') # Returns a tuple of line objects, thus the comma
        #line1.set_ydata(y)
        
        candlestick_ohlc(self.ax1, np.array([x,y,y,y,y]).transpose(), width=0.6)
       # self.ax1.autoscale_view()        
        self.ax1.set_xlim([0, len(x)-1])
        self.ax1.set_ylim([min(self.history.prices[:-1,3]), max(self.history.prices[:-1,2])])        
        
        line2, = self.ax2.plot(x, y2, 'k-') # Returns a tuple of line objects, thus the comma
        line2.set_ydata(y2)

        self.fig.canvas.draw()
        #time.sleep(1e-6)


























