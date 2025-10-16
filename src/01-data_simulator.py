import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import timedelta
import os

N_ROWS = 18000
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
DATA_PATH = 'data/simulated_shipments.csv'

os.makedirs('data', exist_ok=True) 

def simulate_supply_chain_data(n_rows):
    data = []
    
    origins = ['WH_A_North', 'WH_B_South', 'WH_C_Port']
    destinations = ['City_1_East', 'City_2_West', 'Port_Z_Gulf']
    products = ['Electronics', 'Perishables', 'Machinery']
    modes = ['Truck', 'Rail', 'Ocean']
    
    for i in range(n_rows):
        order_id = f'ORD-{i:05d}'
        scheduled_dispatch = fake.date_time_between(start_date='-3y', end_date='now', tzinfo=None)
        
        # Determine Mode and Base Transit Time
        shipping_mode = random.choice(modes)
        if shipping_mode == 'Ocean':
            scheduled_transit_hours = random.randint(300, 1000)
        elif shipping_mode == 'Rail':
            scheduled_transit_hours = random.randint(100, 400)
        else:
            scheduled_transit_hours = random.randint(24, 200)

        # Stage 1: Warehouse/Dispatch Factors (Multi-Stage Prediction Input)
        warehouse_congestion = random.uniform(0, 10)
        strike_flag = random.choices([0, 1], weights=[0.92, 0.08], k=1)[0]
        
        dispatch_base_delay = random.normalvariate(mu=30, sigma=15) if random.random() < 0.2 else 0
        dispatch_strike_delay = strike_flag * random.normalvariate(mu=480, sigma=240)
        total_dispatch_delay_min = max(0, dispatch_base_delay + dispatch_strike_delay)
        dispatch_delay_status = 1 if total_dispatch_delay_min > 90 else 0 
        
        actual_departure = scheduled_dispatch + timedelta(minutes=total_dispatch_delay_min)
        scheduled_arrival = scheduled_dispatch + timedelta(hours=scheduled_transit_hours)

        # Stage 2: Transit/Arrival Factors (External Data Integration)
        weather_severity = random.randint(1, 5) 
        customs_delay_flag = 1 if (random.random() < 0.15) and ('Port' in destinations[random.randint(0,2)]) else 0
        
        transit_weather_delay = weather_severity * 60 * random.random() * (1 + random.random())
        transit_customs_delay = customs_delay_flag * random.normalvariate(mu=900, sigma=400)
        
        total_arrival_delay_minutes = total_dispatch_delay_min + transit_weather_delay + transit_customs_delay
        
        actual_arrival = scheduled_arrival + timedelta(minutes=total_arrival_delay_minutes)
        arrival_delay_status = 1 if (actual_arrival - scheduled_arrival).total_seconds() / 60 > 180 else 0

        # Cost Quantification (Prescriptive AI Input)
        cost_rate = 2.0 if random.choice(['Machinery', 'Perishables']) else 1.0
        flat_penalty = 500 * arrival_delay_status 
        cost_impact_usd = (total_arrival_delay_minutes * cost_rate) + flat_penalty
        
        data.append({
            'OrderID': order_id,
            'Origin_WH': random.choice(origins),
            'Destination_City': random.choice(destinations),
            'Product_Category': random.choice(products),
            'Shipping_Mode': shipping_mode,
            'ScheduledDeparture': scheduled_dispatch,
            'ScheduledArrival': scheduled_arrival,
            'ActualArrival': actual_arrival,
            'Warehouse_Congestion': warehouse_congestion,
            'Strike_Flag': strike_flag,
            'Weather_Severity': weather_severity,
            'Customs_Delay_Flag': customs_delay_flag,
            'Dispatch_Delay_Status': dispatch_delay_status,
            'Arrival_Delay_Status': arrival_delay_status,
            'Total_Delay_Minutes': total_arrival_delay_minutes,
            'Cost_Impact_USD': cost_impact_usd
        })

    df = pd.DataFrame(data)
    df['Scheduled_Transit_Days'] = (df['ScheduledArrival'] - df['ScheduledDeparture']).dt.total_seconds() / (24 * 3600)
    df['Departure_DayOfWeek'] = pd.to_datetime(df['ScheduledDeparture']).dt.dayofweek
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ Data simulated and saved to {DATA_PATH}. Shape: {df.shape}")
    return df

if __name__ == '__main__':
    simulate_supply_chain_data(N_ROWS)