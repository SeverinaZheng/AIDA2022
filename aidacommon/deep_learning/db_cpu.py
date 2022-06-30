from aida.aida import *;
from aidacommon.dbAdapter import DataConversion;
import time;
import sys;
import pandas as pd;
import numpy as np;
import torch;
import torch.nn as nn;
import collections;
host = 'localhost'; dbname = 'bixi'; user = 'bixi'; passwd = 'bixi'; jobName = 'torchLinear'; port = 55660;
dw = AIDA.connect(host,dbname,user,passwd,jobName,port);


# create dummy data for training
#x_values = [i for i in range(2000000)]
#x_train = np.array(x_values, dtype=np.float32)
#x_train = x_train.reshape(-1, 1)

#y_values = [2*i + 1 for i in x_values]
#y_train = np.array(y_values, dtype=np.float32)
#y_train = y_train.reshape(-1, 1)
start = time.time()
freqStations = dw.tripdata2017.filter(Q('stscode', 'endscode', CMP.NE)).aggregate(
    ('stscode', 'endscode', {COUNT('*'): 'numtrips'}), ('stscode', 'endscode')).filter(Q('numtrips', C(50), CMP.GTE));

gtripData = dw.gmdata2017.join(dw.tripdata2017, ('stscode', 'endscode'), ('stscode', 'endscode'), COL.ALL,COL.ALL).join(freqStations, ('stscode', 'endscode'), ('stscode', 'endscode'),('id', 'duration', 'gdistm', 'gduration'));

guniqueTripDist = gtripData[:, ['gdistm']].distinct().order('gdistm');

gtestTripDist = guniqueTripDist[::3];
gtrainTripDist = guniqueTripDist.filter(Q('gdistm', gtestTripDist, CMP.NOTIN));

gtrainData = gtripData.project(('gdistm', 'duration')).filter(Q('gdistm', gtrainTripDist, CMP.IN));

gmaxdist = guniqueTripDist.max('gdistm');
gmaxduration = gtripData.max('duration');
gtrainData = gtrainData.project((1.0 * F('gdistm') / gmaxdist, 1.0 * F('duration') / gmaxduration));

gtrainDataSet = dw._ones((gtrainData.numRows, 1), ("x0",)).hstack(gtrainData[:, ['gdistm']]);
#print(DataConversion.extract_X(gtrainDataSet)[:,1].reshape(-1,1))
gtrainDataSetDuration = gtrainData[:, ['duration']];
#print(np.shape(DataConversion.extract_X(gtrainDataSetDuration)))

end = time.time()
rt = end - start;
print("query time: " + str(rt))

def get_model():
    #class linearRegression(torch.nn.Module):
    #    def __init__(self, inputSize, outputSize):
    #        super(linearRegression, self).__init__()
    #        self.linear = torch.nn.Linear(inputSize, outputSize)

    #    def forward(self, x):
    #        out = self.linear(x)
    #        return out
    model = nn.Sequential(collections.OrderedDict([
        ("layer", nn.Linear(1, 1)),
    ]));
    return model

dw.x_train = gtrainDataSet
dw.y_train = gtrainDataSetDuration
dw.epoch_done = 0
dw.criterion = nn.MSELoss()
model = get_model();
dw.model = model

def regression(dw):
    model = dw.model
    x_train = DataConversion.extract_X(dw.x_train)[:,1].reshape(-1,1).astype(np.float32)
    y_train =DataConversion.extract_X(dw.y_train).astype(np.float32) 
    criterion = dw.criterion
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0000000000001)
    inputs = Variable(torch.from_numpy(x_train))
    print(np.shape(inputs))
    labels = Variable(torch.from_numpy(y_train))
    for epoch in range(100):

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

    # get output from the model, given the inputs
        outputs = model(inputs)

    # get loss for the predicted output
        loss = criterion(outputs, labels)
    # get gradients w.r.t to parameters
        loss.backward()

    # update parameters
        optimizer.step()

    new_var = Variable(torch.Tensor([[4.0]]))
    pred_y = model(new_var)
    return pred_y

return_mesg = dw._X(regression)
print("server time: " + str(return_mesg))
