import pandas as pd
import featureengineering as fe
import datagathering as dg

aapl = dg.gethist("AAPL", "20y")

data = fe.addengineeredfeatures(aapl)

corr_matrix = data.corr()

print(corr_matrix)