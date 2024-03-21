from preprocessing import missing_rate, detect_missed, type_summary, detect_type
import pandas as pd

data = pd.read_csv('hotel_canceling.csv')
rate = missing_rate(data)
features = detect_missed(rate)
features_rate = [(data[i].isnull().sum()/data.shape[0]).round(4) for i in features]
print(f'Feature with missing value: {features} and their corresponding missing rates are {features_rate}')

all_type = type_summary(data)
feature_type = detect_type(all_type, data)
for t in all_type.keys():
    print(f"{t} has these features: {feature_type[t]}")

print('Below are the content of features with data type string')
for i in feature_type[str]:
    print(f"Feature {i} has contnet: {data[i].unique()}")

print('Below are the count of labels of features with data type string')
for i in feature_type[str]:
    print(f"Feature {i} has {len(data[i].unique())} unique labels.")