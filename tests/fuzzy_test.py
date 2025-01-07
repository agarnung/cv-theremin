# Non-pytest test to test skfuzzy library

import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt 

# Function to create fuzzy sets (low, medium, high) for a variable based on its min and max values
def calculate_fuzzy_sets(variable, min_val, max_val, use_gaussian):
    if not use_gaussian:
        range_span = max_val - min_val
        low_max = min_val + range_span * 0.25     # end of 'low'
        medium_min = min_val + range_span * 0.125 # start of 'medium'
        medium_max = min_val + range_span * 0.625 # end of 'medium'
        high_min = min_val + range_span * 0.5     # start of 'high'

        # Fuzzy membership functions for low, medium, and high
        variable['low'] = fuzz.trimf(variable.universe, [min_val, min_val, low_max])
        variable['medium'] = fuzz.trimf(variable.universe, [medium_min, (min_val + max_val) / 2, medium_max])
        variable['high'] = fuzz.trimf(variable.universe, [high_min, max_val, max_val])
    else:
        range_span = max_val - min_val
        center_low = min_val + range_span * 0.125
        center_medium = (min_val + max_val) / 2
        center_high = max_val - range_span * 0.125
        sigma = range_span * 0.25  

        variable['low'] = fuzz.gaussmf(variable.universe, center_low, sigma)
        variable['medium'] = fuzz.gaussmf(variable.universe, center_medium, sigma)
        variable['high'] = fuzz.gaussmf(variable.universe, center_high, sigma)

def main():
    # Define ranges for each parameter
    min_openness, max_openness = 20, 100
    min_proximity, max_proximity = 1, 25
    min_distance, max_distance = 30, 250
    min_frequency, max_frequency = 200, 600

    # Create value ranges for the fuzzy variables
    openness_range = (min_openness, max_openness)
    proximity_range = (min_proximity, max_proximity)
    distance_range = (min_distance, max_distance)
    frequency_range = (min_frequency, max_frequency)

    print({openness_range})

    # Define input and output variables
    openness = ctrl.Antecedent(np.arange(openness_range[0], openness_range[1] + 1, 1), 'openness')
    proximity = ctrl.Antecedent(np.arange(proximity_range[0], proximity_range[1] + 1, 1), 'proximity')
    distance = ctrl.Antecedent(np.arange(distance_range[0], distance_range[1] + 1, 1), 'distance')
    frequency = ctrl.Consequent(np.arange(frequency_range[0], frequency_range[1] + 1, 1), 'frequency')

    # Create fuzzy sets for each variable
    calculate_fuzzy_sets(openness, *openness_range, True)
    calculate_fuzzy_sets(proximity, *proximity_range, True)
    calculate_fuzzy_sets(distance, *distance_range, True)
    calculate_fuzzy_sets(frequency, *frequency_range, True)

    # Define fuzzy rules
    rule1 = ctrl.Rule(proximity['low'] & distance['low'] & openness['low'], frequency['low'])
    rule2 = ctrl.Rule(proximity['high'] & distance['high'] & openness['high'], frequency['high'])
    rule3 = ctrl.Rule(proximity['medium'] & distance['medium'] & openness['medium'], frequency['medium'])

    # Create the fuzzy control system
    frequency_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    frequency_simulator = ctrl.ControlSystemSimulation(frequency_ctrl)

    # Test input values
    frequency_simulator.input['openness'] = 100
    frequency_simulator.input['proximity'] = 25
    frequency_simulator.input['distance'] = 250

    # Ensure the inputs are correct
    print("Inputs: ", frequency_simulator._get_inputs())

    # Perform fuzzy inference (calculate the output)
    frequency_simulator.compute()

    new_frequency = frequency_simulator.output['frequency']
    print(f"Calculated frequency: {new_frequency}")

    frequency.view()

    plt.show()

if __name__ == "__main__":
    main()
