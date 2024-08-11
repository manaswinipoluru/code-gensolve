import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def load_csv_data(file_path):
    return pd.read_csv(file_path, header=None).to_numpy()


def fit_segment_to_line(XY_coords, segment_id=None):
    x_coords = XY_coords[:, 0].reshape(-1, 1)
    y_coords = XY_coords[:, 1]
    regression_model = LinearRegression()
    regression_model.fit(x_coords, y_coords)
    predicted_y = regression_model.predict(x_coords)
    
    return np.column_stack((x_coords, predicted_y))


def combine_segments(data, segment_ids):
    combined_points = []
    for seg_id in segment_ids:
        segment_data = data[data[:, 0] == seg_id]
        combined_points.append(segment_data[:, 2:4]) 
    
    combined_points = np.vstack(combined_points)
    combined_points = combined_points[np.argsort(combined_points[:, 0])]  
   
    combined_line = fit_segment_to_line(combined_points, segment_id=segment_ids[0])
    new_segment_id = segment_ids[0]
    
    merged_segment = np.column_stack((
        np.full(combined_line.shape[0], new_segment_id),
        np.full(combined_line.shape[0], new_segment_id),
        combined_line
    ))

    return merged_segment


def process_and_visualize(input_csv, output_csv, merge_groups):
    
    data = load_csv_data(input_csv)

    
    updated_data = np.copy(data)

    
    for group in merge_groups:
        merged_segment = combine_segments(updated_data, group)
        
       
        for seg_id in group:
            updated_data = updated_data[updated_data[:, 0] != seg_id]
        
       
        updated_data = np.vstack([updated_data, merged_segment])

    
    segment_ids = updated_data[:, 0]
    processed_segments = []
    for unique_id in np.unique(segment_ids):
        segment_mask = segment_ids == unique_id
        x_coords = updated_data[segment_mask, 2]
        y_coords = updated_data[segment_mask, 3]
        XY_coords = np.column_stack((x_coords, y_coords))
        straightened_line = fit_segment_to_line(XY_coords, segment_id=unique_id)
        segment_id_column = np.full(straightened_line.shape[0], unique_id)
        processed_segments.append(np.column_stack((segment_id_column, segment_id_column, straightened_line)))

    
    final_data = np.vstack(processed_segments)
    pd.DataFrame(final_data, columns=['SegmentID', 'SegmentID', 'X', 'Y']).to_csv(output_csv, index=False)

    return data, final_data


def plot_segments_with_labels(data, title, ax):
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    unique_segment_ids = data['SegmentID'].unique()
    for i, segment_id in enumerate(unique_segment_ids):
        color = colors[i % len(colors)]
        segment_data = data[data['SegmentID'] == segment_id]
        ax.plot(segment_data['X'], segment_data['Y'], color=color, linewidth=2, label=f'Segment {segment_id}')
        for _, row in segment_data.iterrows():
            ax.text(row['X'], row['Y'], f'{int(segment_id)}', fontsize=8, color=color)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()


def plot_data(data, title, ax):
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    unique_ids = np.unique(data[:, 0])
    for i, uid in enumerate(unique_ids):
        color = colors[i % len(colors)]
        segment_mask = data[:, 0] == uid
        segment_data = data[segment_mask]
        ax.plot(segment_data[:, 2], segment_data[:, 3], color=color, linewidth=2)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')


input_csv_path = 'frag2.csv'
output_csv = 'frag2_sol.csv'


merge_groups = [
    [1.0, 21.0],
    [13.0, 6.0, 5.0, 24.0],
    [14.0, 7.0, 17.0],
    [16.0, 9.0, 4.0, 23.0],
    [12.0, 0.0],
    [20.0, 19.0],
    [25.0, 12.0],
    [1.0, 15.0],
    [22.0, 8.0, 10.0]
]


original_data, processed_data = process_and_visualize(input_csv_path, output_csv, merge_groups)


fig, axs = plt.subplots(1, 2, figsize=(16, 8))


plot_data(original_data, 'Input Data', axs[0])


plot_data(processed_data, 'Processed Data', axs[1])

plt.show()
