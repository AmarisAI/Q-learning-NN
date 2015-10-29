# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 22:33:58 2015

@author: ka
"""

import launcher

class Defaults:
    WIDTH=32
    HEIGHT=32
    num_actions=4
#Qlearn
    discount=0.95
#train params
    learning_rate=0.0001#0.002
    rms_decay=0.99#0.99 #rho
    rms_epsilon=1e-6#1e-6
    momentum=0.1
#other
    batch_accumulator="mean"
    update_rule="rmsprop"#"sgd"#"deepmind_rmsprop"#"rmsprop" momentum adagrad adam
    batch_size=60
    phi_length=1 #num_frames - tipo istorija time
#neaiskus
    CLIP_DELTA=0
    FREEZE_INTERVAL=-1
    
##Neural agent
    epsilon_start=1.0
    epsilon_min=0.1
    epsilon_decay=1e5
    replay_memory_size=1e3 #sitas nedidelis turi buti 1e3
    experiment_prefix="exper_"
    replay_start_size=80 #kiek sukaupia randomu
    update_frequency=1#pvz niekad 30stepu nepadaro be mirimo

#epochos ir pan

    steps_per_epoch=50 #max minuciu nemirsta; Savaitgaliu nepamirst
    epochs=200000 #200
    steps_per_test=300
    death_ends_episode=False
    
    trainfile="train_EURUSD_UTC_1 Min_Bid_2014.07.01_2014.11.29.csv"
    testfile="test_EURUSD_UTC_1 Min_Bid_2014.11.30_2015.05.07.csv"
    
#nauji
    test_no_reset=1

#conda update matplotlib    
launcher.launch(Defaults)
