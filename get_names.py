import tushare as ts
import os
import pandas as pd

ts.set_token("3404e77dbe323ba4582d677ace412c0bc257f72b39f956b7bf8f975f")
pro = ts.pro_api()
df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,list_date')
output_path = '/nvme5/xtang/gp-workspace/gp-data/tushare_stock_basic.csv'
df.to_csv(output_path, index=False)
print("Saved basic info to:", output_path)
