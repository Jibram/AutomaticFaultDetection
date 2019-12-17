
import numpy as np
from numpy import random as rng
import matplotlib.pyplot as plt
import math 
from scipy.spatial import distance
import matplotlib.pyplot as plt

from preprocessing.calculateError import *
from sgcrfpy.sgcrf import SparseGaussianCRF

# Constants
# Values in temp discharge and airflow can be different, but these values 
# are based on expected values suggested by the Home Ventilation Insitute.
MAX_EXPECTED_TEMP = 78
MIN_EXPECTED_TEMP = 66
MAX_EXPECTED_FLOW = 400
MIN_EXPECTED_FLOW = 200 
SAMPLES_PER_DAY = 1440      # 1440 minutes in a day, 1 minute increments.
WEEKS = 2                   # Half month worth of "data"
TOTAL_SAMPLES = WEEKS * 7 * SAMPLES_PER_DAY
EPOCHS = 5

# ======================= Generate data ===================================
# Data is modeled as follows
#           Day of week     Minute of day       Temp Discharge  Air Flow
#                           [Increment of 1]    [Expected]      [Expected]
# Values:   [1-7]           0-1439              66-78           200-400
# This code below can be replaced with just a parsed N-length input of time
# -series data that follows the same format. 
# =========================================================================
rng.seed(1)
ts_day = np.array([((day // SAMPLES_PER_DAY) % 7 + 1) for day in range(TOTAL_SAMPLES+1)])
ts_min = np.array([sample % SAMPLES_PER_DAY for sample in range(TOTAL_SAMPLES+1)])
ts_data_T = np.floor(rng.rand(TOTAL_SAMPLES+1) * (MAX_EXPECTED_TEMP - MIN_EXPECTED_TEMP) + MIN_EXPECTED_TEMP)
ts_data_F = np.floor(rng.rand(TOTAL_SAMPLES+1) * (MAX_EXPECTED_FLOW - MIN_EXPECTED_FLOW) + MIN_EXPECTED_FLOW)
ts = np.column_stack((ts_day, ts_min, ts_data_T, ts_data_F))

# 75-25 split for the long time series, assuming our system has 1 month of data
# to train/test our model
testCount = TOTAL_SAMPLES*3//4
X_TRAIN = ts[:testCount-1,:]
Y_TRAIN = ts[1:testCount,:]
X_TEST = ts[testCount:ts.shape[0]-1,:]
Y_TEST = ts[testCount+1:ts.shape[0],]

# We will eventually train a seperaate SGCRF model with a shorter time sequence
# and treat it as ground truth. We will guess out 12 iterations and make a
# distribution with the 12 guesses.
Y_LENGTH = 30

longModel = SparseGaussianCRF()

print("\nLong term model training on historical data...") 
for i in range(EPOCHS):

    longModel.fit(X_TRAIN, Y_TRAIN)
    loss = np.round(longModel.lnll[-1], 2)

    # This will give us our longModel's predictions in the training data.
    pred_train = longModel.predict(X_TRAIN)
    prediction = longModel.predict(X_TEST)

    # Error functions given to us by the Bejarano paper 
    rmse_train = RMSE(Y_TRAIN, pred_train)
    rmse_days = RMSE(Y_TEST, prediction)
    maerr = MAE(Y_TEST,prediction)
    avgRMSE_train = np.round(np.mean(rmse_train),4)
    avgRMSE_test = np.round(np.mean(rmse_days),4)

    relerror0_train, relerror1_train, avg0_train, avg1_train = RelE(pred_train, Y_TRAIN, 4)
    relerror0_test, relerror1_test, avg0_test, avg1_test = RelE(prediction, Y_TEST, 4)
    
    print("Loss:", loss, "AVG train RMSE:", avgRMSE_train, "AVG test RMSE:", avgRMSE_test, "RelErrs (train-test)",avg0_train,avg0_test)
    
# Constants for new data
dayOfWeek = 1                       # Sunday
minuteOfDay = 720                   # Noon  #
lengthOfShortData = 20              # 20 Samples takes 20 minutes to gather

# =============== GENERATE NEW DATA ==========================
tt_day = np.array([1 for i in range(lengthOfShortData)])
tt_min = np.array([minuteOfDay + sample for sample in range(lengthOfShortData)])
tt_data_T = np.floor(rng.rand(lengthOfShortData) * (MAX_EXPECTED_TEMP - MIN_EXPECTED_TEMP) + MIN_EXPECTED_TEMP)
tt_data_F = np.floor(rng.rand(lengthOfShortData) * (MAX_EXPECTED_FLOW - MIN_EXPECTED_FLOW) + MIN_EXPECTED_FLOW)
tt = np.column_stack((tt_day, tt_min, tt_data_T, tt_data_F))

# 75-25 split of the data
compareCount = lengthOfShortData*3//4
cX_TRAIN = tt[:compareCount-1,:]
cY_TRAIN = tt[1:compareCount,:]
cX_TEST = tt[compareCount:tt.shape[0]-1,:]
cY_TEST = tt[compareCount+1:tt.shape[0],]

shortModel = SparseGaussianCRF(learning_rate=0.3)

print("\nShort term model training on 'new' data...") 
for i in range(EPOCHS):

    # VERY LIKELY WANT TO USE AN DYNAMICALLY STOPPING MECHANISM FOR THIS LOOP, 
    # WE WANT TO EXIT ONCE OUR MODEL IS NO LONGER GETTING MORE ACCURATE BY A SIGNIFICANT MARGIN.
    shortModel.fit(cX_TRAIN,cY_TRAIN)
    loss = np.round(shortModel.lnll[-1], 2)

    # This will give us our shortModel's predictions in the training data.
    pred_train = shortModel.predict(cX_TRAIN)
    prediction = shortModel.predict(cX_TEST)

    # Error functions given to us by the Bejarano paper 
    rmse_train = RMSE(cY_TRAIN, pred_train)
    rmse_days = RMSE(cY_TEST, prediction)
    maerr = MAE(cY_TEST,prediction)
    avgRMSE_train = np.round(np.mean(rmse_train),4)
    avgRMSE_test = np.round(np.mean(rmse_days),4)

    relerror0_train, relerror1_train, avg0_train, avg1_train = RelE(pred_train, cY_TRAIN, 4)
    relerror0_test, relerror1_test, avg0_test, avg1_test = RelE(prediction, cY_TEST, 4)
    
    print("Loss:", loss, "AVG train RMSE:", avgRMSE_train, "AVG test RMSE:", avgRMSE_test, "RelErrs (train-test)",avg0_train,avg0_test)

LMPrediction = longModel.predict(cX_TEST).astype(int)
SMPrediction = shortModel.predict(cX_TEST).astype(int)

# Originally, I wanted to find the distance between a JOINT probability distribution function as Temperature and Flow are very likely joint values.
# However, given the constraint of the library, I will do a sum of shannon distances between the temperature predictions and flow predictions.
# LM = Long Model, SM = Short Model, T = Temperature, F = Flow
LMT = []
LMF = []
for prediction in LMPrediction:
    LMT.append(prediction[2])
    LMF.append(prediction[3])

SMT = []
SMF = []
for prediction in SMPrediction:
    SMT.append(prediction[2])
    SMF.append(prediction[3])

binsT = np.linspace(MIN_EXPECTED_TEMP, MAX_EXPECTED_TEMP, MAX_EXPECTED_TEMP-MIN_EXPECTED_TEMP + 1)
binsF = np.linspace(MIN_EXPECTED_FLOW,MAX_EXPECTED_FLOW, 11)

histLMT, _ = np.histogram(LMT,binsT) 
histLMF, _ = np.histogram(LMF,binsF)
histSMT, _ = np.histogram(SMT,binsT)
histSMF, _ = np.histogram(SMF,binsF)

Tjsd = distance.jensenshannon(histLMT/len(LMT), histSMT/len(SMT))
Fjsd = distance.jensenshannon(histLMF/len(LMF), histSMF/len(SMF))

# If the distance is any greater than 0.1, the long term model is incorrect
# Assumption: The Short Term Model should be the accurate one as it has the "real data".
print("Total distance between two models:", (Tjsd + Fjsd)/2)