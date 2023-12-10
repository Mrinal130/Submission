import numpy as np
import pandas as pd


dataset1_path = "dataset-1.csv"
dataset2_path = "dataset-2.csv"
dataset3_path = "dataset-3.csv"

dataset1_path = "dataset-1.csv"
dataset2_path = "dataset-2.csv"
dataset3_path = "dataset-3.csv"

#Python Task 1

def generate_car_matrix(df: pd.DataFrame) -> pd.DataFrame:
 matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)
    matrix.values[[range(matrix.shape[0])]*2] = 0  # Set diagonal values to 0
    return matrix
car_matrix_result = generate_car_matrix(dataset1)
print("Car Matrix Result:")
print(car_matrix_result)    

    
 def get_type_count(df: pd.DataFrame) -> dict:
 df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')], labels=['low', 'medium', 'high'])
    type_counts = df['car_type'].value_counts().to_dict()
    return type_counts
car_matrix_result = generate_car_matrix(dataset1)
print("Car Matrix Result:")
print(car_matrix_result)  


def get_bus_indexes(df: pd.DataFrame) -> list:
bus_indexes = df[df['bus'] > 2 * df['bus'].mean()].index.tolist()
    return bus_indexes
bus_indexes_result = get_bus_indexes(dataset1)
print("\nBus Indexes Result:")
print(bus_indexes_result) 


def filter_routes(df: pd.DataFrame) -> list:
   avg_truck_by_route = df.groupby('route')['truck'].mean()
    selected_routes = avg_truck_by_route[avg_truck_by_route > 7].index.tolist()
    return sorted(selected_routes)
routes_result = filter_routes(dataset1)
print("\nFiltered Routes Result:")
print(routes_result)


def multiply_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    matrix[matrix > 20] *= 0.75
    matrix[matrix <= 20] *= 1.25
    return matrix.round(1)
modified_matrix_result = multiply_matrix(car_matrix_result)
print("\nModified Matrix Result:")
print(modified_matrix_result)  

def time_check(df: pd.DataFrame) -> pd.Series:
completeness_check = (df.groupby(['id', 'id_2'])['startDay'].nunique() == 7) & \
                         (df.groupby(['id', 'id_2'])['endDay'].nunique() == 7) & \
                         (df.groupby(['id', 'id_2'])['startTime'].min() == pd.to_datetime('00:00:00')) & \
                         (df.groupby(['id', 'id_2'])['endTime'].max() == pd.to_datetime('23:59:59'))
    return completeness_check
time_check_result = time_check(dataset2)
print("\nTime Check Result:")
print(time_check_result)

#python Task 2


def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:

  # Assuming the dataframe has columns 'id_start', 'id_end', and 'distance'
    distance_matrix = df.pivot(index='id_start', columns='id_end', values='distance').fillna(0)
    
    # Calculate cumulative distances along known routes
    for col in distance_matrix.columns:
        for idx in distance_matrix.index:
            if distance_matrix.at[idx, col] == 0 and idx != col:
                possible_routes = distance_matrix.loc[idx, distance_matrix.loc[idx] != 0].index
                for route in possible_routes:
                    if distance_matrix.at[route, col] != 0:
                        distance_matrix.at[idx, col] = distance_matrix.at[idx, route] + distance_matrix.at[route, col]
    
    return distance_matrix

   distance_matrix_result = calculate_distance_matrix(dataset3)
print("Distance Matrix Result:")
print(distance_matrix_result) 


def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
unrolled_df = df.unstack().reset_index()
    unrolled_df.columns = ['id_start', 'id_end', 'distance']
    

    unrolled_df = unrolled_df[unrolled_df['id_start'] != unrolled_df['id_end']]
    
    return unrolled_df

    unrolled_distance_result = unroll_distance_matrix(distance_matrix_result)
print("\nUnrolled Distance Matrix Result:")
print(unrolled_distance_result)


def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> pd.DataFrame:
    avg_distance_reference = df[df['id_start'] == reference_id]['distance'].mean()
    threshold = 0.1 * avg_distance_reference
    
    
    result_df = df.groupby('id_start')['distance'].mean().reset_index()
    result_df = result_df[(result_df['distance'] >= avg_distance_reference - threshold) &
                          (result_df['distance'] <= avg_distance_reference + threshold)]
    
    return result_df
    reference_id = 1001400  # Replace with your desired reference ID
threshold_result = find_ids_within_ten_percentage_threshold(unrolled_distance_result, reference_id)
print("\nIDs within 10% of the Average Distance Result:")
print(threshold_result)


def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}
    
    # Calculate toll rates for each vehicle type
    for vehicle_type in rate_coefficients.keys():
        df[vehicle_type] = df['distance'] * rate_coefficients[vehicle_type]
    
    return df
    toll_rate_result = calculate_toll_rate(unrolled_distance_result)
print("\nToll Rate Result:")
print(toll_rate_result)



def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:

   discount_factors = {
        'weekday_early': 0.8,
        'weekday_midday': 1.2,
        'weekday_late': 0.8,
        'weekend': 0.7
    }
    
    # Convert 'startDay' and 'endDay' to proper case
    df['startDay'] = df['startDay'].str.capitalize()
    df['endDay'] = df['endDay'].str.capitalize()
    
    # Convert 'startTime' and 'endTime' to datetime.time()
    df['startTime'] = pd.to_datetime(df['startTime']).dt.time
    df['endTime'] = pd.to_datetime(df['endTime']).dt.time
    
    # Apply discount factors based on time intervals and weekdays/weekends
    df['discount_factor'] = 1.0
    df.loc[(df['startDay'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])) &
           (df['startTime'] >= pd.to_datetime('00:00:00').time()) &
           (df['startTime'] < pd.to_datetime('10:00:00').time()), 'discount_factor'] = discount_factors['weekday_early']
    
    df.loc[(df['startDay'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])) &
           (df['startTime'] >= pd.to_datetime('10:00:00').time()) &
           (df['startTime'] < pd.to_datetime('18:00:00').time()), 'discount_factor'] = discount_factors['weekday_midday']
    
    df.loc[(df['startDay'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])) &
           (df['startTime'] >= pd.to_datetime('18:00:00').time()) &
           (df['startTime'] <= pd.to_datetime('23:59:59').time()), 'discount_factor'] = discount_factors['weekday_late']
    
    df.loc[df['startDay'].isin(['Saturday', 'Sunday']), 'discount_factor'] = discount_factors['weekend']
    
    # Apply the discount factor to each vehicle type
    for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
        df[vehicle_type] *= df['discount_factor']
    
    return df

      time_based_toll_result = calculate_time_based_toll_rates(toll_rate_result)


print("\nTime-Based Toll Rates Result:")
print(time_based_toll_result[['id_start', 'id_end', 'discount_factor', 'able2Hov2', 'able2Hov3', 'able3Hov2', 'able3Hov3',
                               'able5Hov2', 'able5Hov3', 'able4Hov2', 'able4Hov3']])
    
    

    
