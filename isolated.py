import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv


color_palette = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']

def load_csv_data(csv_path):
    data = np.genfromtxt(csv_path, delimiter=',')
    grouped_paths = []
    for id in np.unique(data[:, 0]):
        filtered_data = data[data[:, 0] == id][:, 1:]
        sub_paths = []
        for sub_id in np.unique(filtered_data[:, 0]):
            sub_path = filtered_data[filtered_data[:, 0] == sub_id][:, 1:]
            sub_paths.append(sub_path)
        grouped_paths.append(sub_paths)
    return grouped_paths


def classify_shape(contour):
    contour_approx = cv.approxPolyDP(contour, 0.02 * cv.arcLength(contour, True), True)
    shape_area = cv.contourArea(contour)
    shape_perimeter = cv.arcLength(contour, True)
    shape_circularity = 4 * np.pi * (shape_area / (shape_perimeter * shape_perimeter))

    if shape_circularity > 0.7 and len(contour_approx) > 5:
        return "Circle"
    elif len(contour_approx) == 4:
        (x, y, width, height) = cv.boundingRect(contour_approx)
        aspect_ratio = width / float(height)
        if 0.9 <= aspect_ratio <= 1.1:
            return "Square"
        else:
            return "Rectangle"
    elif len(contour_approx) == 10:
        return "Star"
    else:
        return "Polygon"


def generate_star_coords(center, outer_radius, inner_radius, points=5):
    star_coords = []
    angle_step = 2 * np.pi / (points * 2)
    for i in range(points * 2):
        angle = i * angle_step
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        star_coords.append([x, y])
    star_coords.append(star_coords[0])  
    return np.array(star_coords, dtype=np.int32)


def generate_precise_shape_coords(points, shape_type):
    if shape_type == "Circle":
        center = np.mean(points[:, 0, :], axis=0)
        radius = np.mean(np.linalg.norm(points[:, 0, :] - center, axis=1))
        angles = np.linspace(0, 2 * np.pi, 100)  
        circle_coords = np.array([
            [center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)] for angle in angles
        ])
        return circle_coords.astype(np.int32)
    elif shape_type in ["Square", "Rectangle"]:
        (x, y, width, height) = cv.boundingRect(points)
        rect_coords = np.array([
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height],
            [x, y]  
        ])
        return rect_coords
    elif shape_type == "Star":
        center = np.mean(points[:, 0, :], axis=0)
        outer_radius = np.max(np.linalg.norm(points[:, 0, :] - center, axis=1))
        inner_radius = outer_radius / 2.5  
        return generate_star_coords(center, outer_radius, inner_radius)
    else:  
        return np.vstack((points[:, 0, :], points[:, 0, :][0]))  


def process_shapes(grouped_paths):
    classified_shapes = []
    for path in grouped_paths:
        for points in path:
            blank_img = np.zeros((500, 500), dtype=np.uint8)
            points = points.astype(np.int32)
            cv.polylines(blank_img, [points], isClosed=True, color=255, thickness=2)
            contours, _ = cv.findContours(blank_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                shape_type = classify_shape(cnt)
                precise_coords = generate_precise_shape_coords(cnt, shape_type)
                classified_shapes.append((precise_coords, shape_type))
    return classified_shapes


def visualize_shapes(original_paths, detected_shapes):
    fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, figsize=(16, 8))
    
  
    for i, paths in enumerate(original_paths):
        color = color_palette[i % len(color_palette)]
        for path in paths:
            ax1.plot(path[:, 0], path[:, 1], color=color, linewidth=2)
    ax1.set_title('Original Paths')
    ax1.set_aspect('equal')
    
   
    for i, (coords, shape_type) in enumerate(detected_shapes):
        color = color_palette[1] if shape_type == "Circle" else color_palette[(i + 1) % len(color_palette)]
        if shape_type == "Circle":
            center = np.mean(coords, axis=0)
            radius = np.mean(np.linalg.norm(coords - center, axis=1))
            circle = patches.Circle(center, radius, edgecolor=color, fill=False, linewidth=2)
            ax2.add_patch(circle)
        else:
            coords = np.array(coords)
            ax2.plot(coords[:, 0], coords[:, 1], color=color, linewidth=2)
    ax2.set_title('Detected Shapes')
    ax2.set_aspect('equal')
    
    plt.show() 


def export_shapes_to_csv(shapes, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Shape', 'X', 'Y', 'Radius'])
        for i, (coords, shape_type) in enumerate(shapes):
            if shape_type == "Circle":
                center = np.mean(coords, axis=0)
                radius = np.mean(np.linalg.norm(coords - center, axis=1))
                writer.writerow([i + 1, shape_type, center[0], center[1], radius])
            else:
                for coord in coords:
                    writer.writerow([i + 1, shape_type, coord[0], coord[1], ''])


def visualize_corrected_csv(csv_path):
    shapes = []
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            shape_id, shape_type = int(row[0]), row[1]
            x, y = float(row[2]), float(row[3])
            if shape_type == "Circle":
                radius = float(row[4])
                if len(shapes) < shape_id:
                    shapes.append(([], shape_type))
                shapes[shape_id - 1][0].append((x, y, radius))
            else:
                if len(shapes) < shape_id:
                    shapes.append(([], shape_type))
                shapes[shape_id - 1][0].append((x, y))

    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for coords, shape_type in shapes:
        color = color_palette[1] if shape_type == "Circle" else color_palette[(shapes.index((coords, shape_type)) + 1) % len(color_palette)]
        if shape_type == "Circle":
            center, radius = coords[0][0:2], coords[0][2]
            circle = patches.Circle(center, radius, edgecolor=color, fill=False, linewidth=2)
            ax.add_patch(circle)
        else:
            coords = np.array(coords)
            ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=2)
    ax.set_aspect('equal')
    plt.show()  


def main(input_csv, output_csv):
    original_paths = load_csv_data(input_csv)
    detected_shapes = process_shapes(original_paths)

    for i, (coords, shape_type) in enumerate(detected_shapes):
        print(f"Shape {i + 1}: {shape_type}")

    visualize_shapes(original_paths, detected_shapes)  
    export_shapes_to_csv(detected_shapes, output_csv)
    


input_csv = 'isolated.csv'
output_csv = 'isolated_sol.csv'

main(input_csv, output_csv)
