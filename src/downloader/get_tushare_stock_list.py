import argparse
import json
import os
import tushare as ts

def main():
    p = argparse.ArgumentParser(description="Download tradable stock list from Tushare")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory")
    p.add_argument("--token", type=str, required=True, help="Tushare API token")
    
    args = p.parse_args()
    
    ts.set_token(args.token)
    pro = ts.pro_api()
    
    print("Fetching tradable stock list from Tushare...")
    # list_status='L' means listed (tradable)
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,list_date')
    
    if df is None or df.empty:
        print("Error: Failed to fetch stock list or list is empty.")
        return

    symbols = []
    for ts_code in df['ts_code']:
        # Convert 000001.SZ to sz000001
        parts = ts_code.split('.')
        if len(parts) == 2:
            code, ext = parts
            symbols.append(f"{ext.lower()}{code}")
        else:
            symbols.append(ts_code)
            
    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.getcwd(), out_dir)
    os.makedirs(out_dir, exist_ok=True)
        
    out_file = os.path.join(out_dir, "tushare_gplist.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(symbols, f, indent=2)
        
    print(f"Successfully saved {len(symbols)} tradable symbols to {out_file}")

if __name__ == "__main__":
    main()
