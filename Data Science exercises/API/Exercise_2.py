import pandas as pd
import requests

currency_val ="USD" # "GBP"
from_date = "2020-12-01"
to_date = "2020-12-31"
url = f"https://public.opendatasoft.com/api/records/1.0/search/?dataset=euro-exchange-rates&sort=date&facet=currency&rows=30&facet=date&q=date:[{from_date}+TO+{to_date}]&refine.currency={currency_val}"


response_j = requests.get(url)
response_d = response_j.json()
print(response_d)


rates=[]
dates=[]
currency=[]
res_dict=response_d
for rec in res_dict["records"]:
    print(rec["fields"])
    rates.append((rec["fields"]["rate"]))
    dates.append((rec["fields"]["date"]))
    currency.append((rec["fields"]["currency"]))


df =pd.DataFrame({"date":dates,"rate":rates,"currency":currency})
print(df)

"""
בתרגיל זה התבקש לכתוב קריאת API שמקבלת את שערי ההמרה בין דולר לארו בתאריך 
1 ב-2020 עד 31 בדצמבר 2020
ולאחר מכן להציג אותם כDataFrame
"""