# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 22:33:58 2015

@author: ka
"""

#prefix = "testing" if testing else "training"

#TODO:? dabar as pats idedu state current balance, bet gal reiktu padaryti vidine laikina atminti kad jis pats galvotu
#TODO:? speti ne vien q_val, bet ir busimas states, kad butu galima planuoti. Turi tureti visas galimybes kurti ir suvokti ka kuria kompo programas
#TODO:? tam reikia ir temporal memory
#TODO:? CNN ir 3time scales 

#TODO:? RNN+memory, atsakyti klausimams apie teksta, sitas stateofart: http://arxiv.org/pdf/1502.05698.pdf,  http://arxiv.org/abs/1410.5401, http://arxiv.org/pdf/1506.03340v1.pdf https://www.google.com/search?client=ubuntu&channel=fs&q=memory+networks+for+question+answering&ie=utf-8&oe=utf-8
# kaip ir sake, sunku kad jis zinotu ne vien kas pasakyta, bet common knowledge

#+++  ideti state current state - kiek yra skirtumas nuo pirkimo
#+++  test_set prasideda nuo nulio ir tesiaisi visas, game_over ivyksta tik kai 
#+++  test feature atskirai
#+++  num_steps >= len(test_data)
#+++  visas vieno testo vertes(history) saugo isorine funcija
#+++  ir sita loga galima uzplotint po kiekvieno test
#+++  ideti tp ir sl
#+++  padaryti kaip jforex
#+++  padaryti, kad kai epsilion < 0.9, tu 0 -> 3 ir 1or2 -> 1 or 2, deti minimaliai: manual visalaik
#+++  Normalise real-valued data. Subtract the mean and divide by standard deviation.
#kazkodel labai greitis priklauso nuo kiek data points is pradziu uzgeneruou
#jis nesupranta tp ir sl, nes buna tas pats state ir uz action 0, duoda 1 arba 0. Gal del to ir duoda DeepMind po 3 screenus, kad kryti nustatytu
#tiesiog daro 2->3 arba 1->3, nes nezino kada nupirko, ir su sltp


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