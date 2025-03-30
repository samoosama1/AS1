import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
from pathlib import Path

def hough_transform(edge_image, rho_resolution=0.5, theta_resolution=0.5, threshold_factor=0.3):
    height, width = edge_image.shape 
    num_theta_bins = int(180 / theta_resolution)
    theta_values = np.arange(0, 180, theta_resolution)
    theta_radians = theta_values * (np.pi / 180)
    diagonal = np.sqrt(height**2 + width**2)
    max_rho = diagonal
    num_rho_bins = int(2 * diagonal / rho_resolution)
    hough_space = np.zeros((num_rho_bins, num_theta_bins))
    cos_theta = np.cos(theta_radians)
    sin_theta = np.sin(theta_radians)
    y, x = np.nonzero(edge_image)
    num_edge_points = len(x)
    x_adjusted = x - (width / 2)
    y_adjusted = y - (height / 2)
    rho = x_adjusted[:, np.newaxis] * cos_theta[np.newaxis, :] + y_adjusted[:, np.newaxis] * sin_theta[np.newaxis, :]
    rho_bins = ((rho + max_rho) / rho_resolution).astype(int)
    valid_mask = (rho_bins >= 0) & (rho_bins < num_rho_bins)
    theta_indices = np.arange(num_theta_bins)[np.newaxis, :]
    theta_indices = np.repeat(theta_indices, num_edge_points, axis=0)
    if num_edge_points > 0:
        np.add.at(hough_space, (rho_bins[valid_mask], theta_indices[valid_mask]), 1)
    max_votes = hough_space.max() if hough_space.max() > 0 else 1
    threshold = threshold_factor * max_votes  # Changed to use threshold_factor parameter
    rho_bins, theta_bins = np.where(hough_space >= threshold)
    votes = hough_space[rho_bins, theta_bins]
    rho = (rho_bins * rho_resolution) - max_rho
    theta = (theta_bins * theta_resolution) * (np.pi / 180)
    peaks = np.column_stack((rho, theta, votes))
    if len(peaks) > 0:
        peaks = peaks[peaks[:, 2].argsort()[::-1]]
    return peaks

def select_document_lines(lines, img_shape, max_lines=4):
    """
    Select the most significant lines that likely represent document boundaries.
    
    Parameters:
    -----------
    lines : numpy.ndarray
        Array of (rho, theta, votes) representing detected lines
    img_shape : tuple
        (height, width) of the image
    max_lines : int
        Number of lines to return (typically 4 for a document)
        
    Returns:
    --------
    numpy.ndarray
        Array of selected lines in format (rho, theta, votes)
    """
    if len(lines) == 0:
        return np.empty((0, 3))
        
    # Ensure lines has votes if not already present
    if lines.shape[1] == 2:
        # If votes are missing, add dummy votes
        lines = np.column_stack((lines, np.ones(len(lines))))
    
    # Step 1: Cluster lines with similar angle
    angle_threshold = np.pi / 36  # 5 degrees
    distance_threshold = min(img_shape) * 0.02  # 2% of image dimension
    
    # Sort by votes (importance)
    lines = lines[lines[:, 2].argsort()[::-1]]
    
    # Initialize clusters
    clusters = []
    
    for line in lines:
        rho, theta, votes = line
        
        # Normalize theta to range [-pi/2, pi/2)
        theta = theta % np.pi
        if theta >= np.pi/2:
            theta -= np.pi
            rho = -rho
            
        assigned = False
        
        # Check if line fits in any existing cluster
        for i, cluster in enumerate(clusters):
            cluster_theta = cluster[0][1]
            
            # If angle is similar
            if abs(theta - cluster_theta) < angle_threshold or abs(abs(theta - cluster_theta) - np.pi) < angle_threshold:
                # If distance is similar (for parallel lines)
                if abs(rho - cluster[0][0]) < distance_threshold:
                    clusters[i].append((rho, theta, votes))
                    assigned = True
                    break
        
        # If line doesn't fit any cluster, create a new one
        if not assigned:
            clusters.append([(rho, theta, votes)])
    
    # Step 2: Select the strongest line from each cluster
    selected_lines = []
    for cluster in clusters:
        # Sort by votes and take the one with highest votes
        cluster.sort(key=lambda x: x[2], reverse=True)
        selected_lines.append(cluster[0])
    
    # Step 3: Select the most important lines (typically 4 for a document)
    # Sort by votes and take top max_lines
    selected_lines.sort(key=lambda x: x[2], reverse=True)
    selected_lines = selected_lines[:max_lines]
    
    # Convert back to numpy array
    return np.array(selected_lines)

def calculate_intersections(lines):
    """
    Calculate all intersection points between lines.
    
    Parameters:
    -----------
    lines : numpy.ndarray
        Array of (rho, theta) representing detected lines
        
    Returns:
    --------
    list
        List of intersection points (x, y)
    """
    intersections = []
    
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            rho1, theta1 = lines[i, :2]
            rho2, theta2 = lines[j, :2]
            
            # Check if lines are not parallel
            if abs(theta1 - theta2) > 1e-8 and abs(abs(theta1 - theta2) - np.pi) > 1e-8:
                # Calculate intersection
                A = np.array([
                    [np.cos(theta1), np.sin(theta1)],
                    [np.cos(theta2), np.sin(theta2)]
                ])
                b = np.array([rho1, rho2])
                
                try:
                    x, y = np.linalg.solve(A, b)
                    intersections.append((x, y))
                except np.linalg.LinAlgError:
                    # In case the lines are nearly parallel
                    pass
    
    return intersections

def find_document_corners(intersections, img_shape):
    """
    Find the four corners of the document from all intersections.
    
    Parameters:
    -----------
    intersections : list
        List of intersection points (x, y)
    img_shape : tuple
        (height, width) of the image
        
    Returns:
    --------
    numpy.ndarray
        Array of 4 corners in clockwise order starting from top-left
    """
    if len(intersections) < 4:
        return None
        
    height, width = img_shape
    
    # Filter intersections that are outside the image with some margin
    margin = 0.1  # 10% margin
    valid_intersections = []
    
    for x, y in intersections:
        if (-margin * width <= x <= (1 + margin) * width and 
            -margin * height <= y <= (1 + margin) * height):
            valid_intersections.append((x, y))
    
    if len(valid_intersections) < 4:
        return None
    
    # Convert to numpy array
    points = np.array(valid_intersections)
    
    # Find convex hull
    hull = cv2.convexHull(points.reshape(-1, 1, 2).astype(np.float32))
    
    # Approximate the hull to get a quadrilateral
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    
    # If the approximation has 4 points, use it directly
    if len(approx) == 4:
        corners = approx.reshape(-1, 2)
    else:
        # If not, find the 4 extreme points
        # Find top-left, top-right, bottom-right, bottom-left
        s = points.sum(axis=1)
        corners = np.zeros((4, 2))
        
        # Top-left has smallest sum
        corners[0] = points[np.argmin(s)]
        
        # Bottom-right has largest sum
        corners[2] = points[np.argmax(s)]
        
        # Top-right has smallest difference
        diff = points[:, 0] - points[:, 1]
        corners[1] = points[np.argmax(diff)]
        
        # Bottom-left has largest difference
        corners[3] = points[np.argmin(diff)]
    
    # Order corners (top-left, top-right, bottom-right, bottom-left)
    return order_points(corners)

def order_points(pts):
    """
    Order points in clockwise order starting from top-left.
    
    Parameters:
    -----------
    pts : numpy.ndarray
        Array of points to be ordered
        
    Returns:
    --------
    numpy.ndarray
        Ordered points
    """
    # Sort by sum (x+y) for top-left and bottom-right
    s = pts.sum(axis=1)
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Top-left has smallest sum
    rect[0] = pts[np.argmin(s)]
    
    # Bottom-right has largest sum
    rect[2] = pts[np.argmax(s)]
    
    # Sort by difference (x-y) for top-right and bottom-left
    diff = np.diff(pts.reshape(-1, 2), axis=1)
    
    # Top-right has largest difference
    rect[1] = pts[np.argmax(diff)]
    
    # Bottom-left has smallest difference
    rect[3] = pts[np.argmin(diff)]
    
    return rect

def process_document_image(image_path, output_dir='outputs', resize_factor=0.4):
    """
    Process a document image to detect edges and rectify.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    output_dir : str
        Directory to save output images
    resize_factor : float
        Factor to resize the image for processing
        
    Returns:
    --------
    dict
        Dictionary with output paths and success status
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract image number from path
    img_name = os.path.basename(image_path)
    img_num = img_name.split('.')[0]
    
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image {image_path}")
        return {"success": False}
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_image = clahe.apply(image)
    
    # Apply Gaussian blur with smaller kernel
    blurred = cv2.GaussianBlur(contrast_image, (5, 5), 1.5)
    
    # Resize for faster processing
    resized_image = cv2.resize(blurred, (int(image.shape[1] * resize_factor), 
                                          int(image.shape[0] * resize_factor)), 
                                interpolation=cv2.INTER_LINEAR)
    
    # Edge detection with tighter thresholds
    edges = cv2.Canny(resized_image, threshold1=50, threshold2=150)
    
    # Save edges image
    edges_path = f"{output_dir}/edges_{img_num}.jpg"
    cv2.imwrite(edges_path, edges)
    
    # Hough transform with finer resolution
    lines = hough_transform(edges, rho_resolution=0.5, theta_resolution=0.5, threshold_factor=0.3)
    
    print(f"Image {img_num}: Detected {len(lines)} lines")
    
    # Select document boundary lines
    selected_lines = select_document_lines(lines, edges.shape, max_lines=8)
    
    # Calculate intersections between lines
    intersections = calculate_intersections(selected_lines)
    
    # Find document corners
    corners = find_document_corners(intersections, edges.shape) if intersections else None
    
    # Create color image for visualization
    resized_image_bgr = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
    
    # Draw selected lines
    for rho, theta, _ in selected_lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        
        # Adjust for coordinate system (add width/2, height/2)
        height, width = resized_image.shape
        x0 += width / 2
        y0 += height / 2
        
        # Calculate line endpoints
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        
        cv2.line(resized_image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Draw corners if found
    if corners is not None:
        for i, corner in enumerate(corners):
            x, y = corner
            # Adjust for coordinate system (add width/2, height/2)
            x += width / 2
            y += height / 2
            cv2.circle(resized_image_bgr, (int(x), int(y)), 5, (0, 255, 0), -1)
            # Label the corner
            labels = ["TL", "TR", "BR", "BL"]
            cv2.putText(resized_image_bgr, labels[i], (int(x), int(y)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Save visualization
    lines_path = f"{output_dir}/lines_{img_num}.jpg"
    cv2.imwrite(lines_path, resized_image_bgr)
    
    # Rectify document if corners found
    warped_path = None
    if corners is not None:
        # Scale corners back to original image size
        scale = 1.0 / resize_factor
        
        # Adjust for coordinate system (add width/2, height/2)
        original_corners = np.zeros_like(corners)
        for i, (x, y) in enumerate(corners):
            original_corners[i] = [(x + width / 2) * scale, (y + height / 2) * scale]
        
        # Define destination points for a standard aspect ratio
        # Determine if portrait or landscape
        width_distance = max(
            np.linalg.norm(original_corners[0] - original_corners[1]),
            np.linalg.norm(original_corners[2] - original_corners[3])
        )
        height_distance = max(
            np.linalg.norm(original_corners[0] - original_corners[3]), 
            np.linalg.norm(original_corners[1] - original_corners[2])
        )
        
        is_portrait = height_distance > width_distance
        
        if is_portrait:
            w, h = 800, 1100  # Approximate A4 proportions
        else:
            h, w = 800, 1100
        
        dst_points = np.array([
            [0, 0],           # Top-left
            [w - 1, 0],       # Top-right
            [w - 1, h - 1],   # Bottom-right
            [0, h - 1]        # Bottom-left
        ], dtype=np.float32)
        
        # Calculate perspective transform
        M = cv2.getPerspectiveTransform(original_corners.astype(np.float32), dst_points)
        
        # Apply perspective transform
        warped = cv2.warpPerspective(image, M, (w, h))
        
        # Save warped image
        warped_path = f"{output_dir}/warped_{img_num}.jpg"
        cv2.imwrite(warped_path, warped)
    
    return {
        "success": True,
        "edges_path": edges_path,
        "lines_path": lines_path,
        "warped_path": warped_path,
        "corners_found": corners is not None
    }

# Main execution
if __name__ == "__main__":
    # Check if inputs directory provided as argument
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "WarpDoc/distorted/curved"
    output_dir = "outputs"
    
    # Create outputs directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process images in the input directory
    image_files = list(Path(input_dir).glob("*.jpg"))
    
    results = []
    for img_path in image_files:
        print(f"Processing {img_path}")
        result = process_document_image(str(img_path), output_dir)
        results.append((img_path, result))
    
    # Print summary
    print("\nProcessing Summary:")
    print("-" * 50)
    for img_path, result in results:
        status = "Success" if result["success"] else "Failed"
        corners = "Found" if result.get("corners_found", False) else "Not found"
        print(f"{img_path}: {status}, Corners: {corners}")