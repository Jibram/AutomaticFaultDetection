

#--------------------------------------------
# helper functions
#--------------------------------------------

# combining the data into placeholders
def getPlaceholders(org_data,comp_data):
    y1 = [] # correct values
    y2 = [] # computed values
    for i in range(len(org_data[0])):
        temp1 = []
        temp2 = []
        for j in range(len(org_data)):
            temp1.append(org_data[j][i])
            temp2.append(comp_data[j][i])
        y1.append(temp1)
        y2.append(temp2)
    return y1,y2

#--------------------------------------------
# error calculation
#--------------------------------------------
# Relative Error
def relativeError(pred, real):
    sumTotal = 0
    sumTotal_1 = 0 # adding 1 to the real value
    totalSamples = len(real)
    for j in range(totalSamples):
        if real[j]>0:
            sumTotal +=abs(pred[j]-real[j])/abs(real[j])
        sumTotal_1 += abs(pred[j]-real[j])/(abs(real[j]) + 1)
    return float(sumTotal)/float(totalSamples), float(sumTotal_1)/float(totalSamples)


def RelE(predicted, real, days):
    sumValue0 = 0
    sumValue1 = 0
    listErrDay0 = []
    listErrDay1 = []
    real, predicted = getPlaceholders(real, predicted)
    for i in range(days):
        #print predicted.shape, real.shape
        relE0, relE1 = relativeError(predicted[i], real[i])
        listErrDay0.append(relE0)
        listErrDay1.append(relE1)
        sumValue0 += relE0
        sumValue1 += relE1
    avgERR0 = float(sumValue0)/float(days)
    avgERR1 = float(sumValue1)/float(days)
    return listErrDay0, listErrDay1, avgERR0, avgERR1

# MSE : Mean Squared Error
def MSE(org_data,comp_data):
    try:
        if len(org_data) != len(comp_data):
            raise ValueError("length of original Y and computed Y does not match")
        y_org_sample, y_calc_sample = getPlaceholders(org_data,comp_data)
        mse = []
        for n in range(len(y_org_sample)):#time
            y_org = y_org_sample[n]
            y_calc = y_calc_sample[n]
            sum_value = 0
            for i in range(len(y_org)):
               diff = float(float(y_org[i])-float(y_calc[i]))
               sqrd_diff = diff ** 2
               sum_value += sqrd_diff
            mse.append(float(sum_value/len(y_org)))
        return mse
    except ValueError as err:
        print("Error: ",err)

# RMSE : Root Mean Squared Error
def RMSE(org_data,comp_data):
    mse = MSE(org_data,comp_data)
    rmse = []
    for data in mse:
        rmse.append(float(data ** 0.5))
    return rmse

# MAE : mean Absolute Error
def MAE(org_data,comp_data):
    try:
        if len(org_data) != len(comp_data):
            raise ValueError("length of original Y and computed Y does not match")
        y_org_sample, y_calc_sample = getPlaceholders(org_data,comp_data)
        mae = []
        for n in range(len(y_org_sample)):
            y_org = y_org_sample[n]
            y_calc = y_calc_sample[n]
            sum_value = 0
            for i in range(len(y_org)):
                diff = abs(float(y_org[i])-float(y_calc[i]))
                sum_value += diff
            mae.append(float(sum_value/len(y_org)))
        return mae
    except ValueError as err:
        print("Error: ",err)
    

#----------------------------------------------
# testing functions
#----------------------------------------------

#def main():
#    x1 = [[1,2,3],[4,5,6]]
#    x2 = [[1.3,6.2,3.3],[4.1,5,5.8]]
#    mse = MSE(x1,x2)
#    rmse = RMSE(x1,x2)
#    mae = MAE(x1,x2)
#    print "mse: ",mse
#    print "rmse: ",rmse
#    print "mae: ",mae

#main()
