# IMPORTANT NOTE: 
# when you hand in, the two functions in this file must be self-contained, i.e., when we call the functions
# SetLearningParams() and CreateModel() from our own code (without importing anything else you may have put in this file)
# and use them to train a model, this must reproduce your submitted model 

# WARNING: the model architecture and learning parameters used below are just an example of how to do things
#          they are not necessarily good settings


import tensorflow as tf

def SetLearningParams(): 
    # WARNING: DO NOT CHANGE THE STRUCTURE OF DICT RETURNED BY THIS FUNCTION!!
    
    STUDENT_LASTNAME = "Meert"     #   example:  "Dambre" 
    STUDENT_FIRSTNAME = "Wout"      #   example:  "Joni"
    STUDENT_ID = "01906865"        #   example:  "000386365"

    # make sure student data above is filled in 
    assert STUDENT_ID is not None and STUDENT_LASTNAME is not None and STUDENT_FIRSTNAME is not None, 'Please fill in your First and Last Name and Student Id'
    
    learningparams = {}
    learningparams['student_name'] = f'{STUDENT_LASTNAME} {STUDENT_FIRSTNAME}'
    learningparams['student_seed'] = int(str(STUDENT_ID))

    learningparams['batch_size'] = 32

    # If you feel you need to use a larger batch size than your system memory allows, you can use gradient accumulation,
    # For, e.g. gradient accumumation accross two steps (=batches), 
    # you effectively double the batch size that determines each weight update step
    learningparams['gradient_accumulation'] = None
    
    # The number of epochs set below defines the maximal number of epochs in your training run, 
    # This must be long enough for learning to have converged
    # when combined with early stopping, the effective number of epochs can be smaller
    
    learningparams['epochs'] = 20
    
    # first define fixed (initial) learning rate or learning rate schedule
    lr_or_schedule = 1.0e-3
    learningparams['lrschedule'] = tf.keras.optimizers.schedules.ExponentialDecay(lr_or_schedule, decay_steps=100000, decay_rate=0.96, staircase=True)
    
    # we advise you to use Adam, but you are free to use alternatives
    learningparams['optimizer'] = tf.keras.optimizers.Adam(lr_or_schedule,                                                     gradient_accumulation_steps=learningparams['gradient_accumulation'],)

    # Define callbacks
    # for example, this is where you introduce early stopping when doing long training runs 
    # this is also where you would introduce a generic W&B logger (if W&B is used in your main training code)
    # examples for how to use both are given below, but always check whether other function arguments may be useful for you
    learningparams['callbacks'] = [#WandbCallback(monitor = 'train_accuracy', save_model= False, save_graph = False),
                                   tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=5,                               restore_best_weights=True,verbose = 1,),
                                    ]
    
    return learningparams

def CreateModel(modelname = 'mymodel', num_classes = 41):

    inputs = tf.keras.Input(shape=(64, 64, 3))
    scaled = tf.keras.layers.Rescaling(1./255)(inputs)

    # make sure you check the many other parameters you can set when defining a convolutional layer (as discussed in class)
    # Block 1
    conv_0_1 = tf.keras.layers.Conv2D(32, 3, padding = 'same', activation='relu')(scaled)
    conv_0_2 = tf.keras.layers.Conv2D(32, 3, padding = 'same', activation='relu')(conv_0_1)
    red_0 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv_0_2)
    bnorm_0 = tf.keras.layers.BatchNormalization()(red_0)
  
    # Block 2
    #drop_1 = tf.keras.layers.Dropout(0.0)(red_0)
    conv_1_1 = tf.keras.layers.Conv2D(64, 3, padding = 'same', activation='relu')(bnorm_0)
    conv_1_2 = tf.keras.layers.Conv2D(64, 3, padding = 'same', activation='relu')(conv_1_1)
    red_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv_1_2)
    bnorm_1 = tf.keras.layers.BatchNormalization()(red_1)
    
    # Block 3 with dropout
    conv_2_1 = tf.keras.layers.Conv2D(128, 3, padding = 'same', activation='relu')(bnorm_1)
    conv_2_2 = tf.keras.layers.Conv2D(128, 3, padding = 'same', activation='relu')(conv_2_1)
    red_2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv_2_2)
    bnorm_2 = tf.keras.layers.BatchNormalization()(red_2)
    drop_2 = tf.keras.layers.Dropout(0.3)(bnorm_2)

    # Block 4 with dropout
    conv_3_1 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(drop_2)
    conv_3_2 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(conv_3_1)
    red_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_3_2)
    bnorm_3 = tf.keras.layers.BatchNormalization()(red_3)
    drop_3 = tf.keras.layers.Dropout(0.4)(bnorm_3)  

    global_pool = tf.keras.layers.GlobalAveragePooling2D()(drop_3)
    
    # Flatten and Dense Layers
    dense_in = tf.keras.layers.Flatten()(global_pool)
    drop_dense1 = tf.keras.layers.Dropout(0.5)(dense_in)
    
    logits = tf.keras.layers.Dense(num_classes)(drop_dense1)
    
    conv_model = tf.keras.Model(inputs=inputs, outputs=logits, name=modelname)
    
    return conv_model
