from data.read_data import Data
from build_features import  build_wcl_features

d = Data()
data = d.read_wcl_data()

features = build_wcl_features(data)


