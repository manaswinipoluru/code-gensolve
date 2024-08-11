import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import csv


color_palette = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']


def load_csv(csv_file_path):
    raw_data = np.genfromtxt(csv_file_path, delimiter=',')
    organized_data = []
    for path_id in np.unique(raw_data[:, 0]):
        path_data = raw_data[raw_data[:, 0] == path_id][:, 1:]
        segments = []
        for segment_id in np.unique(path_data[:, 0]):
            segment_points = path_data[path_data[:, 0] == segment_id][:, 1:]
            segments.append(segment_points)
        organized_data.append(segments)
    return organized_data


def classify_shape(contour):
    tolerance = 0.01 * cv.arcLength(contour, True)
    approx_curve = cv.approxPolyDP(contour, tolerance, True)
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    circularity = 4 * np.pi * (area / (perimeter ** 2))

    if len(approx_curve) == 4:
        return "Rectangle"
    elif circularity > 0.7:
        return "Ellipse"
    else:
        return "Freeform"


def create_shape_points(contour, shape_type):
    if shape_type == "Ellipse":
        center = np.mean(contour[:, 0, :], axis=0)
        distances = np.linalg.norm(contour[:, 0, :] - center, axis=1)
        radius = np.mean(distances)
        angles = np.linspace(0, 2 * np.pi, 100)
        ellipse_points = np.array([
            [center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)]
            for angle in angles
        ])
        return ellipse_points.astype(np.int32)
    elif shape_type == "Rectangle":
        x, y, width, height = cv.boundingRect(contour)
        rect_points = np.array([
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height],
            [x, y]  
        ])
        return rect_points
    else:
        return contour[:, 0, :]


def analyze_shapes(organized_data):
    identified_shapes = []
    contours_list = []
    for path in organized_data:
        for segment in path:
            canvas = np.zeros((500, 500), dtype=np.uint8)
            segment = segment.astype(np.int32)
            cv.polylines(canvas, [segment], isClosed=False, color=255, thickness=2)
            contours, _ = cv.findContours(canvas, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours_list.extend(contours)

    for contour in contours_list:
        shape_type = classify_shape(contour)
        precise_points = create_shape_points(contour, shape_type) if shape_type != "Freeform" else contour[:, 0, :]
        identified_shapes.append((precise_points, shape_type))
    
    return identified_shapes


def export_shapes_to_csv(shapes, output_csv_path):
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'ShapeType', 'X', 'Y', 'Radius'])
        for index, (points, shape_type) in enumerate(shapes):
            if shape_type == "Ellipse":
                center = np.mean(points, axis=0)
                radius = np.mean(np.linalg.norm(points - center, axis=1))
                for point in points:
                    writer.writerow([index + 1, shape_type, center[0], center[1], radius])
            else:
                for point in points:
                    writer.writerow([index + 1, shape_type, point[0], point[1], ''])


def visualize_shapes(input_data, output_shapes):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

   
    for i, segments in enumerate(input_data):
        color = color_palette[i % len(color_palette)]
        for segment in segments:
            axes[0].plot(segment[:, 0], segment[:, 1], color=color, linewidth=2, linestyle='-')
    axes[0].set_title('Input Shapes')
    axes[0].set_aspect('equal')

   
    for i, (points, shape_type) in enumerate(output_shapes):
        color = color_palette[1] if shape_type == "Ellipse" else color_palette[(i + 1) % len(color_palette)]
        if shape_type == "Ellipse":
            center = np.mean(points, axis=0)
            radius = np.mean(np.linalg.norm(points - center, axis=1))
            ellipse_patch = patches.Circle(center, radius, edgecolor=color, fill=False, linewidth=2)
            axes[1].add_patch(ellipse_patch)
        elif shape_type == "Rectangle":
            rect_patch = patches.Polygon(points, closed=True, edgecolor=color, fill=False, linewidth=2)
            axes[1].add_patch(rect_patch)
        else:
            axes[1].plot(points[:, 0], points[:, 1], linestyle='-', color=color, linewidth=2)
    axes[1].set_title('Detected Shapes')
    axes[1].set_aspect('equal')

    plt.show()


def main(input_csv_path, output_csv_path):
    organized_data = load_csv(input_csv_path)
    identified_shapes = analyze_shapes(organized_data)
    export_shapes_to_csv(identified_shapes, output_csv_path)
    visualize_shapes(organized_data, identified_shapes)

    
    output_data = pd.read_csv(output_csv_path)
    print(output_data)

input_csv_path = 'frag0.csv'
output_csv_path = 'frag01_sol.csv'

main(input_csv_path, output_csv_path)
