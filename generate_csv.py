import csv
import random
from datetime import datetime, timedelta
import sys

def generate_and_save_csv(n, csv_path):
    """
    Generate n records and write to csv_path.
    Fields: timestamp, heart_beat, temperature, speed
    """
    start = datetime.now()
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp","heart_beat","temperature","speed"])
        writer.writeheader()
        for i in range(n):
            rec = {
                "timestamp": (start + timedelta(seconds=60*i)).isoformat(),
                "heart_beat": random.randint(60,100),
                "temperature": round(random.uniform(36.0,37.5),1),
                "speed": round(random.uniform(0,10),2)
            }
            writer.writerow(rec)
    print(f"Wrote {n} records to {csv_path}")

if __name__ == "__main__":
    # Usage: python generate_csv.py <num_records> <output.csv>
    if len(sys.argv) != 3:
        print("Usage: python generate_csv.py <num_records> <output.csv>")
        sys.exit(1)

    n = int(sys.argv[1])
    path = sys.argv[2]
    generate_and_save_csv(n, path)