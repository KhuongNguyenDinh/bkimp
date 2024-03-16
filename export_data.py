import requests
import csv
import datetime
from prometheus_parser import parse

prometheus_url = ""  
ACCESS_TOKEN = ""

headers = {'Authorization': 'Bearer'}

response = requests.get(prometheus_url, headers=headers)

if response.status_code == 200:
    parsed_metrics = parse(response.text)

    current_datetime = datetime.datetime.now()
    formatted_date = current_datetime.strftime("%Y%m%d")
    formatted_month = current_datetime.strftime("%Y%m")

    prometheus_node = ""

    start_date = current_datetime - datetime.timedelta(days=7)  
    end_date = current_datetime  

    filtered_metrics = [metric for metric in parsed_metrics if start_date.date() <= metric[1].date() <= end_date.date()
                        and (metric[0] == "cpu_usage" or metric[0] == "date_time")]

    filename = f"data{formatted_date}{formatted_month}_{prometheus_node}_days_range.csv"

    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for metric in filtered_metrics:
            csv_writer.writerow(metric)

    print(f"Metrics exported to '{filename}'")
    return 1
else:
    print(f"Failed to fetch metrics. Status Code: {response.status_code}")
    return 0

