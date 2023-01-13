# small python file for quick testing - git ignored
import pandas as pd

input1 = pd.read_csv('output_weekday.csv')
input2 = pd.read_csv('output_weekday_original.csv')

print(input1.compare(input2))
