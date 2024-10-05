import bagpy
from bagpy import bagreader
import pandas as pd

bag = bagreader('Our Data (bag)/bc2_hw_t8.bag')
js_est = bag.message_by_topic('/svan/js_est')
js_est = pd.read_csv(js_est)

print("Number of columns:", js_est.shape[1])