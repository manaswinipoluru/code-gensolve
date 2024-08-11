import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter

def read_csv(csv_path):
    
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []

    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []

        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)

        path_XYs.append(XYs)

    return path_XYs

def complete_and_smooth_curve(XY, num_points=200, smoothing=0.01, window_length=7, polyorder=2):
    
    try:
      
        tck, u = splprep([XY[:, 0], XY[:, 1]], s=smoothing, per=True)
        new_points = splev(np.linspace(0, 1, num_points), tck)
        new_points = np.column_stack(new_points)
        
       
        smoothed_x = savgol_filter(new_points[:, 0], window_length, polyorder)
        smoothed_y = savgol_filter(new_points[:, 1], window_length, polyorder)
        
        return np.column_stack((smoothed_x, smoothed_y))
    except Exception as e:
        print(f"Error completing curve: {e}")
        return XY 

def plot_curves_side_by_side(original_XYs, completed_XYs):
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    for XY in original_XYs:
        for path in XY:
            path = np.array(path)  
            axs[0].plot(path[:, 0], path[:, 1], 'ro-', label='Original Curve')

    for XY in completed_XYs:
        for path in XY:
            path = np.array(path) 
            axs[1].plot(path[:, 0], path[:, 1], 'b--', label='Completed Curve')

    axs[0].set_title('Original Curves')
    axs[0].legend()
    axs[0].set_aspect('equal', adjustable='box')

    axs[1].set_title('Completed Curves')
    axs[1].legend()
    axs[1].set_aspect('equal', adjustable='box')

    plt.show()

def save_csv(completed_XYs, output_path):
    
    output_data = []
    shape_index = 0

    for shape in completed_XYs:
        for i, XY in enumerate(shape):
            for point in XY:
                row = [shape_index, i, point[0], point[1]]
                output_data.append(row)
        shape_index += 1

    np.savetxt(output_path, output_data, delimiter=',', fmt='%d,%d,%f,%f')

def main(input_csv, output_csv):
   
    original_shapes = read_csv(input_csv)

    
    completed_shapes = []
    for shape in original_shapes:
        completed_shape = [complete_and_smooth_curve(np.array(XY)) for XY in shape]
        completed_shapes.append(completed_shape)

   
    plot_curves_side_by_side(original_shapes, completed_shapes)

  
    save_csv(completed_shapes, output_csv)

if __name__ == "__main__":
    input_csv_path = "occlusion1.csv" 
    output_csv_path = "occlusion1_sol.csv" 
    
    main(input_csv_path, output_csv_path)
