import pandas as pd
import numpy as np
import os

# --- Configuration ---
INPUT_PATH = 'data/prediction_output.csv'
OPTIMIZATION_OUTPUT_PATH = 'data/final_prescriptive_recommendations.csv'

# --- Prescriptive AI Parameters (Based on Business Rules) ---
COST_EXPEDITE = 800  # Cost in USD to pay for priority shipping
COST_REROUTE = 300   # Cost in USD to find an alternative route
REDUCTION_EXPEDITE = 0.85 # Expediting reduces 85% of the expected delay cost
REDUCTION_REROUTE = 0.50  # Rerouting reduces 50% of the expected delay cost
PROBABILITY_THRESHOLD = 0.50 

def run_optimization(df):
    
    # 1. Calculate Expected Financial Loss (Risk Score)
    df['Expected_Loss_USD'] = df['Predicted_Delay_Probability'] * df['Cost_Impact_USD']
    
    # 2. Initialize Recommendation
    df['Intervention_Action'] = 'No Action'
    df['Net_Benefit_USD'] = 0.0

    # 3. Apply Prescriptive Logic
    high_risk_mask = df['Predicted_Delay_Probability'] >= PROBABILITY_THRESHOLD

    # Calculate net benefit for each possible intervention
    benefit_expedite = (df['Expected_Loss_USD'] * REDUCTION_EXPEDITE) - COST_EXPEDITE
    benefit_reroute = (df['Expected_Loss_USD'] * REDUCTION_REROUTE) - COST_REROUTE
    
    # Expedite is the best action
    expedite_mask = high_risk_mask & (benefit_expedite > 0) & (benefit_expedite >= benefit_reroute)
    
    # Reroute is the best action
    reroute_mask = high_risk_mask & (benefit_reroute > 0) & (~expedite_mask)

    # Apply actions
    df.loc[expedite_mask, 'Intervention_Action'] = 'EXPEDITE (Highest Priority)'
    df.loc[expedite_mask, 'Net_Benefit_USD'] = benefit_expedite
    
    df.loc[reroute_mask, 'Intervention_Action'] = 'REROUTE (Moderate Priority)'
    df.loc[reroute_mask, 'Net_Benefit_USD'] = benefit_reroute
    
    # 4. Filter and Sort for Dashboard
    final_recs = df[df['Intervention_Action'] != 'No Action'].sort_values(
        ['Net_Benefit_USD', 'Expected_Loss_USD'], 
        ascending=[False, False]
    )
    
    return final_recs[['OrderID', 'Predicted_Delay_Probability', 'Expected_Loss_USD', 
                       'Intervention_Action', 'Net_Benefit_USD', 'Top_Delay_Drivers']]

# Run the optimization and save
if __name__ == '__main__':
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Run 02_model_trainer_predictor.py first. Missing {INPUT_PATH}")
    else:
        df_input = pd.read_csv(INPUT_PATH)
        recommendations = run_optimization(df_input)
        recommendations.to_csv(OPTIMIZATION_OUTPUT_PATH, index=False)
        
        print("\n--- Prescriptive AI Results ---")
        print(f"Total orders recommended for intervention: {len(recommendations)}")
        print(f"Total Estimated Net Benefit from Interventions: ${recommendations['Net_Benefit_USD'].sum():,.2f}")
        print(f"✅ Final recommendations saved to {OPTIMIZATION_OUTPUT_PATH}")