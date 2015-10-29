# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 23:31:23 2015

@author: ka
"""
import numpy as np
import pandas as pd

class timeseries(object):
    
    def __init__(self, train, test):
        #train ir test siuo atveju yra failu adresai
        #TODO: inputs laikini - isbandyti 10s trading
        
        self.train=(np.sin(np.array(range(1,500)) * np.pi / 40. ))/80#pd.read_csv(train)['Close'][140000:143000].values
        self.test=(np.sin(np.array(range(1,500)) * np.pi / 40. ))/80#pd.read_csv(train)['Close'][140000:143000].values
        
        self.trainFeatures=self.generateFeatures(self.train)  
        self.testFeatures=self.generateFeatures(self.test) 
        
        self.featureCount=len(self.testFeatures[0])
        
    def countFeatures(self):
        return self.featureCount
        
    def getInputCount(self):
        return self.inputCount

    def getTrainLen(self):
        return len(self.train) 

    def getTestLen(self):
        return len(self.test) 
    
    def generateFeatures(self, data):
        feature1=self.subtract_diff(data, 1)
        feature1=2.0*(((feature1-np.mean(feature1))/np.std(feature1))>0)-1
#        feature1=self.subtract_diff(data, 1)
#        feature1=(feature1-np.mean(feature1))/np.std(feature1)
        feature2=self.subtract_diff(data, 3)
        feature2=(feature2-np.mean(feature2))/np.std(feature2)
        feature5=self.subtract_diff(data, 5)
        feature5=(feature5-np.mean(feature5))/np.std(feature5)
        feature8=self.subtract_diff(data, 8)
        feature8=(feature8-np.mean(feature8))/np.std(feature8)
        feature12=self.subtract_diff(data, 12)
        feature12=(feature12-np.mean(feature12))/np.std(feature12)
        feature17=self.subtract_diff(data, 17)
        feature17=(feature17-np.mean(feature17))/np.std(feature17)

        return np.column_stack((feature1,feature2,feature5,feature8,feature12,feature17))
        
    def moving_average(self,a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
#        for r in range (0,n-1):
#            print(r)
        return ret[n - 1:] / n
        
    def subtract_diff(self,data, n):
        ret=data[0:-n]
        for r in range (0,n):
            ret=np.insert(ret,0,0)
        ret=np.subtract(data,ret)
        return ret

        