# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 22:36:08 2015

@author: ka
"""
import q_network
import cPickle
import ale_action
import import_timeseries
import ale_experiment
import ale_agent

def launch(defaults):
    
    timeseries=import_timeseries.timeseries(defaults.trainfile, defaults.testfile)
    ale=ale_action.ale(timeseries, defaults.steps_per_epoch)
    
#re-run last script ctrl+F6    
#test
#    ale.reset_game()
#    print(ale.getCurrentState())
#    print(ale.act(2))  
#    print(ale.priceOrder)  
#    #print("balance: "+ str(ale.stateToCat()[-1]))
#    print(ale.getCurrentState())
#    print(ale.act(0))  
#    #print("balance: "+ str(ale.stateToCat()[-1]))
#    print(ale.getCurrentState())
#    print(ale.act(0))  
#    #print("balance: "+ str(ale.stateToCat()[-1]))
#    print(ale.getCurrentState())
#    print(ale.act(0))  
#    
#    print(ale.getCurrentState())
#    print(ale.act(0))  
#    #print("balance: "+ str(ale.stateToCat()[-1]))
#    print(ale.getCurrentState())
#    print(ale.act(0))  
#    #print("balance: "+ str(ale.stateToCat()[-1]))
#    print(ale.getCurrentState())
#    print(ale.act(0))  
#   #print("balance: "+ str(ale.stateToCat()[-1]))
#    print(ale.timeseries.train[ale.currentPosTime])
#    print(ale.getCurrentState())
#    print(ale.act(3)) 
#    
#    assert (0==1)
#    
    nn_file=None
    if nn_file is None:
       network = q_network.DeepQLearner(defaults.WIDTH,
                                         defaults.HEIGHT,
                                         ale.getActionCount(),
                                         defaults.phi_length, #num_frames - tipo istorija time
                                         defaults.discount,
                                         defaults.learning_rate,
                                         defaults.rms_decay,
                                         defaults.rms_epsilon,
                                         defaults.momentum,
                                         defaults.CLIP_DELTA, 
                                         defaults.FREEZE_INTERVAL,
                                         defaults.batch_size,
                                         defaults.update_rule,
                                         defaults.batch_accumulator,
                                         ale.getInputCount())
    else:
       handle = open(nn_file, 'r')
       network = cPickle.load(handle)
       
    agent = ale_agent.NeuralAgent(network,
                                  defaults.epsilon_start,
                                  defaults.epsilon_min,
                                  defaults.epsilon_decay,
                                  defaults.replay_memory_size,
                                  defaults.experiment_prefix,
                                  defaults.replay_start_size,
                                  defaults.update_frequency)
                                  
    experiment = ale_experiment.ALEExperiment(ale, agent,
                                              defaults.WIDTH,
                                              defaults.HEIGHT,
                                              defaults.epochs,
                                              defaults.steps_per_epoch,
                                              defaults.steps_per_test,
                                              defaults.death_ends_episode)


    experiment.run()
    