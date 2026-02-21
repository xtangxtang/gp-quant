from bs4 import BeautifulSoup
import requests
import csv
import json
import pandas as pd
import time
from datetime import datetime
import os
import pandas as pd
import threading
from selenium import webdriver
import argparse
from fake_useragent import UserAgent
import shutil

def get_daily(symbols, working_path, __time=""):
    colnames=["成交时间", "成交价", "涨跌幅", "价格变动", "成交量(手)", "成交额(元)", "性质"] 
    os.chdir(working_path)
    print("working_path " + working_path)
    os.chdir(working_path + "/gp_daily")

    if __time == "":
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        today_time = datetime.today().strftime('%Y-%m-%d')  
    else:
        today_time = __time

    print(f"today: {today_time}")
    throttle = 2
    for sysmbol in symbols:
        print("sysmbol: " + sysmbol)
        time.sleep(throttle)
        csv_dir = f"{working_path}/gp_daily/{sysmbol}"
        if os.path.exists(csv_dir) == False:
            os.mkdir(csv_dir)
        os.chdir(csv_dir)
        csv_file = f"{csv_dir}/{today_time}.csv"
        if os.path.exists(csv_file):
            tmp_df = pd.read_csv(csv_file, delimiter=",")
            if not tmp_df.empty :
                last_row = tmp_df.iloc[-1]
                print(f"{sysmbol} {today_time}.csv" + last_row["成交时间"] )
                if last_row["成交时间"] < "15:00:00":
                    # exit(2)
                    print(csv_file + last_row["成交时间"] + " is not 15:00:00, remove file" )
                    os.remove(csv_file)                
                else:
                    continue
            else:
                os.remove(csv_file)
            # try:
            #     total_detail_df = pd.read_csv(f"{today_time}.csv", delimiter=",")
            # except:
            #     print(f"got exception while reading {sysmbol} csv file")
            #     os.chdir(working_path)
            #     continue   
        total_detail_df = pd.DataFrame(columns=colnames)
        total_detail_df.to_csv(csv_file)                 
        total_detail_df = total_detail_df.set_index("成交时间")
        retry = 0        
        page_range = range(100, 0, -1)        
        for i in page_range:
            url = f"http://vip.stock.finance.sina.com.cn/quotes_service/view/vMS_tradedetail.php?symbol={sysmbol}&date={today_time}&page={i}"
            while True:
                try: 
                    ua=UserAgent()
                    # print('User-Agent :' + ua.random)
                    hdr = {'User-Agent': ua.random,
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                        'Accept-Encoding': 'none',
                        'Accept-Language': 'en-US,en;q=0.8',
                        'Connection': 'keep-alive'}                    
                    html=requests.get(url, timeout=20, headers=hdr).content 
                except Exception as e:
                    print(f"got exception while reading {sysmbol} url" + str(e))
                    if retry < 3:
                        retry = retry + 1
                        time.sleep(5)
                        continue
                    else:
                        break 
                break               
            try:                        
                df_tmp = pd.read_html(html)
                # print(df_tmp)
                if (df_tmp[3]['成交时间'].eq('该股票没有交易数据')).any():
                    print("retry index " + str(i))
                    continue
                print(df_tmp[3])              

                # open(file_name, 'w').close()
                detail_df_tmp = df_tmp[3].set_index("成交时间")     
                total_detail_df = pd.concat([detail_df_tmp,total_detail_df]).drop_duplicates()
            except Exception as e:
                print(f"got exception while looping {sysmbol}: " + str(e))
                print(df_tmp)
                # exit(0)
                time.sleep(15)
                continue
        total_detail_df = total_detail_df.iloc[::-1]
        total_detail_df.to_csv(csv_file)
        print(f"finish {sysmbol}")        

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        # print("Current Time =", current_time)
        os.chdir(working_path)
        # time.sleep(10)
    os.chdir(working_path)

def create_folder(symbols):
    for sysmbol in symbols:
        if not os.path.exists(f"gp_daily/{sysmbol}"):
            os.makedirs(f"gp_daily/{sysmbol}")

def rename_file():
    dir_list = os.listdir("gp_daily")
    cwd = os.getcwd()
    os.chdir("gp_daily")
    for symb_dir in dir_list:
        os.chdir(symb_dir)
        if os.path.exists("2023-6-2.csv"):
            os.rename("2023-6-2.csv", "2023-06-02.csv")
        os.chdir("../")
    os.chdir(cwd)    

def remove_files_not_self_list(self_gplist):
    dir_list = os.listdir("gp_daily")
    cwd = os.getcwd()
    os.chdir("gp_daily")
    for file in dir_list:
        if file in self_gplist:            
            continue
        else:
            if os.path.exists(file):
                shutil.rmtree(file)
                print("remove :" + file)    

    os.chdir(cwd)

def divide_chunks(l, n):     
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]                    



if __name__ == "__main__":

    chunks_num = 5
    working_path = os.getcwd()

    json_file = os.path.join(working_path, "self_gplist.json")
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            self_gplist = json.load(f)
        print(f"Loaded {len(self_gplist)} symbols from {json_file}")
    else:
        self_gplist = ["sz002409", "sz301323", "sh688114", "sh688508"]
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self_gplist, f, indent=4)
        print(f"Created default {json_file}")
    
    # chunks_num =5
    # remove_files_not_self_list(self_gplist)

    today_time = ""
    
    # Fix potential division by zero if self_gplist is smaller than chunks_num
    chunk_size = max(1, int(len(self_gplist) / chunks_num))
    self_gplist_cks = list(divide_chunks(self_gplist, chunk_size))
    
    # Adjust chunks_num to actual number of chunks created
    actual_chunks_num = len(self_gplist_cks)
    
    self_gplist_threads = []
    for i in range(actual_chunks_num):    
        t = threading.Thread(target=get_daily, args=(self_gplist_cks[i], working_path, today_time))
        t.name = f"sh_zhuban_{i}"
        print(t.getName())
        self_gplist_threads.append(t)

    for i in range(actual_chunks_num):
        print("i " + str(i))
        self_gplist_threads[i].start()
        os.chdir(working_path)
        
    for i in range(actual_chunks_num):
        self_gplist_threads[i].join()
        os.chdir(working_path)  

    get_daily(self_gplist, working_path, "")