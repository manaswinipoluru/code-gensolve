import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def load_paths_from_csv(csv_file):
   
    data = pd.read_csv(csv_file, header=None)
    paths = []
    
    for _, group in data.groupby(0):
        path = group.iloc[:, -2:].to_numpy()
        paths.append(path)
    
    return paths

def complete_missing_parts(paths):
    
    completed_paths = []
    
    for path in paths:
        if len(path) > 2:
            
            hull = ConvexHull(path)
            completed_path = path[hull.vertices]
        else:
           
            x_new = np.linspace(path[:, 0].min(), path[:, 0].max(), num=100)
            y_new = np.interp(x_new, path[:, 0], path[:, 1])
            completed_path = np.column_stack((x_new, y_new))
        
        completed_paths.append(completed_path)
    
    return completed_paths

def export_to_csv(paths, output_file):
    
    records = []
    for i, path in enumerate(paths):
        for j, point in enumerate(path):
            records.append([i, j, point[0], point[1]])
    
    df = pd.DataFrame(records, columns=['Shape_ID', 'Point_ID', 'X', 'Y'])
    df.to_csv(output_file, index=False, header=False)

def plot_paths_side_by_side(original_paths, completed_paths):
   
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot original paths
    for path in original_paths:
        axes[0].plot(path[:, 0], path[:, 1], linewidth=2)
    axes[0].set_title('Original Data')
    axes[0].set_aspect('equal')

    # Plot completed paths
    for path in completed_paths:
        axes[1].plot(path[:, 0], path[:, 1], linewidth=2)
    axes[1].set_title('Completed Data')
    axes[1].set_aspect('equal')

    plt.show()

def process_csv(input_csv, output_csv):
    
    original_paths = load_paths_from_csv(input_csv)
    completed_paths = complete_missing_parts(original_paths)
    export_to_csv(completed_paths, output_csv)
    plot_paths_side_by_side(original_paths, completed_paths)

# Example usage
input_csv = 'occlusion2.csv'  # Replace with your actual input file path
output_csv = 'occlusion2_sol.csv'  # Replace with your actual output file path
process_csv(input_csv, output_csv)
