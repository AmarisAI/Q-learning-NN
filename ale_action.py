# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 22:54:14 2015

@author: ka
"""
import random
import numpy as np

class ale(object):
    
    def __init__(self,timeseries, num_epoch_steps):
        self.timeseries=timeseries
        self.action_set={0,1,2,3}
        self.beginningPosition=0
        self.currentPosTime=0
        self.num_epoch_steps=num_epoch_steps
        self.currentPosState=0
        self.gameOver=1
        self.states={0,1,2} #1-nothing, 2-bought, 3-sold, 4-orderBalance
        self.priceOrder=0
        self.timeOrder=0
        
    def getActionCount(self):
        #0 NULL
        #1 BUY
        #2 SELL
        #3 CLOSE
        return len(self.action_set)
        
    def reset_game(self):
        self.beginningPosition=random.randint(0, self.timeseries.getTrainLen()-self.num_epoch_steps*2-100)      
        self.currentPosTime=self.beginningPosition
        self.currentPosState=0
        self.gameOver=0
        
    def reset_game_testing_first(self):
        self.beginningPosition=0
        self.currentPosTime=self.beginningPosition
        self.currentPosState=0
        self.gameOver=0
        
    def reset_game_testing(self):
        self.currentPosState=0
        self.gameOver=0
        
    def act(self, action, testing=False, spread=0.0002, sl=0.0007, tp=0.0009):       
        
        #nera high ir low
        sl=-sl
        sltp=True
    #tp ir sl
        if sltp:
            if not testing:
                if (self.currentPosState==1):
                    #tp buy
                    if ((self.timeseries.train[self.currentPosTime]-self.priceOrder)-spread)>tp:
                        action=3
                    #sl buy
                    if ((self.timeseries.train[self.currentPosTime]-self.priceOrder)-spread)<sl:
                        action=3
                if (self.currentPosState==2):
                    #tp buy
                    if ((-self.timeseries.train[self.currentPosTime]+self.priceOrder)-spread)>tp:
                        action=3
                    #sl buy
                    if ((-self.timeseries.train[self.currentPosTime]+self.priceOrder)-spread)<sl:
                        action=3
            else:
                if (self.currentPosState==1):
                    #tp buy
                    if ((self.timeseries.test[self.currentPosTime]-self.priceOrder)-spread)>tp:
                        action=3
                    #sl buy
                    if ((self.timeseries.test[self.currentPosTime]-self.priceOrder)-spread)<sl:
                        action=3
                if (self.currentPosState==2):
                    #tp buy
                    if ((-self.timeseries.test[self.currentPosTime]+self.priceOrder)-spread)>tp:
                        action=3
                    #sl buy
                    if ((-self.timeseries.test[self.currentPosTime]+self.priceOrder)-spread)<sl:
                        action=3
    
    
        if  (self.currentPosState==0) and  (action==0):
            self.currentPosTime=self.currentPosTime+1
            #self.gameOver=1
            return -1
        #pabandau 0->3 kaip 0->0
        elif (self.currentPosState==0) and (action == 3):
            self.currentPosTime=self.currentPosTime+1
            #self.gameOver=1
            return -1
        elif (self.currentPosState!=0) and (action == 0):
            self.currentPosTime=self.currentPosTime+1
            return 0
        elif (self.currentPosState==0) and (action == 1):
            self.currentPosState=1
            if testing:
                self.priceOrder=self.timeseries.test[self.currentPosTime]
            else:
                self.priceOrder=self.timeseries.train[self.currentPosTime]
            self.timeOrder=self.currentPosTime
            self.currentPosTime=self.currentPosTime+1
            return 0
        elif (self.currentPosState==0) and (action == 2):
            self.currentPosState=2
            if testing:
                self.priceOrder=self.timeseries.test[self.currentPosTime]
            else:
                self.priceOrder=self.timeseries.train[self.currentPosTime]
            self.timeOrder=self.currentPosTime
            self.currentPosTime=self.currentPosTime+1
            return 0
        elif (self.currentPosState==1) and (action == 3):
            
            
#            if testing:
#                print("BUY:")
#                print(self.timeseries.train[self.currentPosTime]-self.priceOrder)
#                print(self.timeseries.feature1[self.currentPosTime])
#                print("steps "+ str(self.currentPosTime-self.timeOrder))
#                print("CurrentState: "+ str(self.getCurrentState()))
#                print("")

            self.currentPosState=0
            self.gameOver=1
            #return np.clip((self.timeseries.train[self.currentPosTime]-self.priceOrder),-1,1)#-0.5/(self.currentPosTime-self.timeOrder)#10000*(self.timeseries.train[self.currentPosTime]-self.priceOrder)
            #return np.clip(100*(self.timeseries.train[self.currentPosTime]-self.priceOrder),-1,1)#-0.5/(self.currentPosTime-self.timeOrder)#10000*(self.timeseries.train[self.currentPosTime]-self.priceOrder)     
               
            self.currentPosTime+=1          
            if testing:
                return 100*np.clip((self.timeseries.test[self.currentPosTime-1]-self.priceOrder)-spread,-1,1)
            else:
                return 100*np.clip((self.timeseries.train[self.currentPosTime-1]-self.priceOrder)-spread,-1,1) 
#            
#           
#            
#            if np.clip((self.timeseries.train[self.currentPosTime]-self.priceOrder),-1,1)>0:
#                return np.clip(a,-1,1)
#            else:
#                return np.clip(-a-0.1,-1,-0.1)
        elif (self.currentPosState==2) and (action == 3):
            
#            if testing:
#                print("SELL:")
#                print(-self.timeseries.train[self.currentPosTime]+self.priceOrder)
#                print(self.timeseries.feature1[self.currentPosTime])
#                print("steps "+ str(self.currentPosTime-self.timeOrder))
#                print("CurrentState: "+ str(self.getCurrentState()))
#                print("")
            
            self.currentPosState=0
            self.gameOver=1
            #print (np.clip(1000*(-self.timeseries.train[self.currentPosTime]+self.priceOrder),-1,1)-0.5/(self.currentPosTime-self.timeOrder))#10000*(-self.timeseries.train[self.currentPosTime]+self.priceOrder)
            #return np.clip((-self.timeseries.train[self.currentPosTime]+self.priceOrder),-1,1)#-0.5/(self.currentPosTime-self.timeOrder)#10000*(-self.timeseries.train[self.currentPosTime]+self.priceOrder)
#            a=0.1*(self.currentPosTime-self.timeOrder)-0.1 
#            self.currentPosTime=self.currentPosTime+1            
#            if np.clip((-self.timeseries.train[self.currentPosTime]+self.priceOrder),-1,1) > 0:
#                return np.clip(a,-1,1)
#            else: 
#                return np.clip(-a-0.1,-1,-0.1)
            self.currentPosTime+=1          
            if testing:
                return 100*np.clip((-self.timeseries.test[self.currentPosTime-1]+self.priceOrder)-spread,-1,1)
            else:
                return 100*np.clip((-self.timeseries.train[self.currentPosTime-1]+self.priceOrder)-spread,-1,1) 

           # return np.clip(100*(-self.timeseries.train[self.currentPosTime]+self.priceOrder),-1,1)#-0.5/(self.currentPosTime-self.timeOrder)#10000*(-self.timeseries.train[self.currentPosTime]+self.priceOrder)
        else:
            self.currentPosState=0            
            self.currentPosTime=self.currentPosTime+1
            self.gameOver=1
            return -1
    
    def getCurrentState(self,testing=False):
        return np.concatenate((self.stateToCat(testing), self.getCurrentFeatures(self.currentPosTime,testing) ), axis=0)
    
    def game_over(self):
        return self.gameOver
        
    def getInputCount(self):
        return self.timeseries.countFeatures()+len(self.states)
        
    def stateToCat(self,testing=False):
        buff=np.zeros(len(self.states))
        buff[...]=-1
        buff[self.currentPosState]=1
        
        
        #cia current balance
#        if self.currentPosState==1:
#            if not testing:
#                buff[-1]=-self.priceOrder+self.timeseries.train[self.currentPosTime]
#            else:
#                buff[-1]=-self.priceOrder+self.timeseries.test[self.currentPosTime]
#        elif self.currentPosState==2:
#            if not testing:
#                buff[-1]=self.priceOrder-self.timeseries.train[self.currentPosTime]
#            else:
#                buff[-1]=self.priceOrder-self.timeseries.test[self.currentPosTime]
#        else:
#            buff[-1]=0
        return buff
        
    def getCurrentPrice(self):
        #grazina OHLC, tik testing
        return [self.timeseries.test[self.currentPosTime],self.timeseries.test[self.currentPosTime],self.timeseries.test[self.currentPosTime],self.timeseries.test[self.currentPosTime]]

    def getCurrentFeatures(self, currentPosTime, testing):
        if testing:
            return self.timeseries.testFeatures[currentPosTime]
        else:
            return self.timeseries.trainFeatures[currentPosTime]