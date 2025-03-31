# %% [markdown]
# Abdussamet Tekin 2220356042

# %%
# Import necessary libraries
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
from pathlib import Path


# %% [markdown]
# First we implement the Hough Transform from scratch using numpy.

# %%
def hough_transform(edge_image, rho_resolution=0.5, theta_resolution=0.5, threshold_factor=0.3):
   # Get dimensions of the input edge image
   height, width = edge_image.shape 
   
   # Calculate number of theta bins and create theta values array
   num_theta_bins = int(180 / theta_resolution)
   theta_values = np.arange(0, 180, theta_resolution)
   # Convert theta values from degrees to radians
   theta_radians = theta_values * (np.pi / 180)
   
   # Calculate the diagonal length of the image (maximum possible distance)
   diagonal = np.sqrt(height**2 + width**2)
   max_rho = diagonal
   
   # Calculate number of rho bins
   num_rho_bins = int(2 * diagonal / rho_resolution)
   
   # Initialize Hough accumulator array (votes for each rho-theta pair)
   hough_space = np.zeros((num_rho_bins, num_theta_bins))
   
   # Pre-compute cosine and sine values for efficiency
   cos_theta = np.cos(theta_radians)
   sin_theta = np.sin(theta_radians)
   
   # Find coordinates of non-zero pixels (edge points)
   y, x = np.nonzero(edge_image)
   num_edge_points = len(x)
   
   # Adjust coordinates relative to center of image
   x_adjusted = x - (width / 2)
   y_adjusted = y - (height / 2)
   
   # Calculate rho values for all edge points across all theta values
   rho = x_adjusted[:, np.newaxis] * cos_theta[np.newaxis, :] + y_adjusted[:, np.newaxis] * sin_theta[np.newaxis, :]
   
   # Convert rho values to bin indices
   rho_bins = ((rho + max_rho) / rho_resolution).astype(int)
   
   # Create mask for valid bin indices (within bounds)
   valid_mask = (rho_bins >= 0) & (rho_bins < num_rho_bins)
   
   # Create array of theta indices and repeat for each edge point
   theta_indices = np.arange(num_theta_bins)[np.newaxis, :]
   theta_indices = np.repeat(theta_indices, num_edge_points, axis=0)
   
   # Increment Hough accumulator for valid points
   if num_edge_points > 0:
       np.add.at(hough_space, (rho_bins[valid_mask], theta_indices[valid_mask]), 1)
   
   # Find maximum vote count (handle empty case)
   max_votes = hough_space.max() if hough_space.max() > 0 else 1
   
   # Calculate threshold based on maximum votes
   threshold = threshold_factor * max_votes  
   
   # Find coordinates of bins that exceed threshold
   rho_bins, theta_bins = np.where(hough_space >= threshold)
   
   # Get vote counts for the identified peaks
   votes = hough_space[rho_bins, theta_bins]
   
   # Convert bin indices back to rho and theta values
   rho = (rho_bins * rho_resolution) - max_rho
   theta = (theta_bins * theta_resolution) * (np.pi / 180)
   
   # Combine rho, theta, and vote counts
   peaks = np.column_stack((rho, theta, votes))
   
   # Sort peaks by vote count in descending order
   if len(peaks) > 0:
       peaks = peaks[peaks[:, 2].argsort()[::-1]]
       
   # Return the peaks array (rho, theta, votes)
   return peaks

# %% [markdown]
# When the Hough Transform fits lines, we use this select_document_lines function to select the lines that are most likely to be document lines.

# %%
def select_document_lines(lines, img_shape, max_lines=4, horizontal_threshold=np.pi/12, vertical_threshold=np.pi/12):
    """
    Select the most significant lines that likely represent document boundaries,
    prioritizing near-horizontal and near-vertical lines.
    
    Parameters:
    -----------
    lines : numpy.ndarray
        Array of (rho, theta, votes) representing detected lines
    img_shape : tuple
        (height, width) of the image
    max_lines : int
        Number of lines to return (typically 4 for a document)
    horizontal_threshold : float
        Angle threshold in radians to consider a line as horizontal (from 0 or pi)
    vertical_threshold : float
        Angle threshold in radians to consider a line as vertical (from pi/2)
        
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
    
    # Normalize all theta values to range [0, pi)
    normalized_lines = lines.copy()
    for i, line in enumerate(normalized_lines):
        rho, theta, votes = line
        theta = theta % np.pi
        if theta >= np.pi:
            theta -= np.pi
            rho = -rho
        normalized_lines[i] = [rho, theta, votes]
    
    # Define vertical and horizontal angle ranges
    vertical_mask = ((normalized_lines[:, 1] > (np.pi/2 - vertical_threshold)) & 
                     (normalized_lines[:, 1] < (np.pi/2 + vertical_threshold)))
    
    horizontal_mask = ((normalized_lines[:, 1] < horizontal_threshold) | 
                       (normalized_lines[:, 1] > (np.pi - horizontal_threshold)))
    
    # Split lines into vertical, horizontal, and other
    vertical_lines = normalized_lines[vertical_mask]
    horizontal_lines = normalized_lines[horizontal_mask]
    other_lines = normalized_lines[~(vertical_mask | horizontal_mask)]
    
    # Sort each group by votes (importance)
    if len(vertical_lines) > 0:
        vertical_lines = vertical_lines[vertical_lines[:, 2].argsort()[::-1]]
    if len(horizontal_lines) > 0:
        horizontal_lines = horizontal_lines[horizontal_lines[:, 2].argsort()[::-1]]
    if len(other_lines) > 0:
        other_lines = other_lines[other_lines[:, 2].argsort()[::-1]]
    
    # Cluster similar lines to avoid duplicates
    angle_threshold = np.pi / 36  # 5 degrees
    distance_threshold = min(img_shape) * 0.05  # 5% of image dimension
    
    # Function to cluster lines within a group
    def cluster_lines(group_lines):
        if len(group_lines) == 0:
            return []
            
        clusters = []
        for line in group_lines:
            rho, theta, votes = line
            assigned = False
            
            for i, cluster in enumerate(clusters):
                cluster_theta = cluster[0][1]
                
                # If angle is similar
                if abs(theta - cluster_theta) < angle_threshold:
                    # If distance is similar (for parallel lines)
                    distance_diff = abs(rho - cluster[0][0])
                    if distance_diff < distance_threshold:
                        clusters[i].append((rho, theta, votes))
                        assigned = True
                        break
            
            # If line doesn't fit any cluster, create a new one
            if not assigned:
                clusters.append([(rho, theta, votes)])
        
        # Select the strongest line from each cluster
        selected = []
        for cluster in clusters:
            # Sort by votes and take the one with highest votes
            cluster.sort(key=lambda x: x[2], reverse=True)
            selected.append(cluster[0])
            
        return selected
    
    # Cluster each group
    selected_vertical = cluster_lines(vertical_lines)
    selected_horizontal = cluster_lines(horizontal_lines)
    selected_other = cluster_lines(other_lines)
    
    # Determine how many lines to select from each group
    # Prioritize vertical and horizontal lines
    n_vertical = min(len(selected_vertical), max_lines // 2)
    n_horizontal = min(len(selected_horizontal), max_lines // 2)
    
    # If we still need more lines, fill with other lines
    remaining_slots = max_lines - (n_vertical + n_horizontal)
    n_other = min(len(selected_other), remaining_slots) if remaining_slots > 0 else 0
    
    # Take the top n from each group
    final_vertical = selected_vertical[:n_vertical]
    final_horizontal = selected_horizontal[:n_horizontal]
    final_other = selected_other[:n_other]
    
    # Combine and convert to numpy array
    all_selected = final_vertical + final_horizontal + final_other
    
    # If we don't have enough lines, relax our criteria and include more from other categories
    if len(all_selected) < max_lines:
        remaining = max_lines - len(all_selected)
        
        # First try to add more vertical lines
        if n_vertical < len(selected_vertical):
            additional_vertical = min(len(selected_vertical) - n_vertical, remaining)
            all_selected.extend(selected_vertical[n_vertical:n_vertical + additional_vertical])
            remaining -= additional_vertical
        
        # Then try horizontal
        if remaining > 0 and n_horizontal < len(selected_horizontal):
            additional_horizontal = min(len(selected_horizontal) - n_horizontal, remaining)
            all_selected.extend(selected_horizontal[n_horizontal:n_horizontal + additional_horizontal])
            remaining -= additional_horizontal
        
        # Finally other lines
        if remaining > 0 and n_other < len(selected_other):
            additional_other = min(len(selected_other) - n_other, remaining)
            all_selected.extend(selected_other[n_other:n_other + additional_other])
    
    # Sort all selected lines by votes for final ranking
        all_selected.sort(key=lambda x: x[2], reverse=True)
    
    # Convert to numpy array 
        return np.array(all_selected)
  # 5% of image dimension

# C uster each group
    selected_vertical = cluster_lines(vertical_lines)
    selected_horizontal = cluster_lines(horizontal_lines)
    selected_other = cluster_lines(other_lines)

    # Determine how many lines to select from each group
    # Prioritize vertical and horizontal lines
    n_vertical = min(len(selected_vertical), max_lines // 2)
    n_horizontal = min(len(selected_horizontal), max_lines // 2)

    # If we still need more lines, fill with other lines
    remaining_slots = max_lines - (n_vertical + n_horizontal)
    n_other = min(len(selected_other), remaining_slots) if remaining_slots > 0 else 0

    # Take the top n from each group
    final_vertical = selected_vertical[:n_vertical]
    final_horizontal = selected_horizontal[:n_horizontal]
    final_other = selected_other[:n_other]

    # Combine and convert to numpy array
    all_selected = final_vertical + final_horizontal + final_other

    # If we don't have enough lines, relax our criteria and include more from other categories
    if len(all_selected) < max_lines:
        remaining = max_lines - len(all_selected)

        # First try to add more vertical lines
        if n_vertical < len(selected_vertical):
            additional_vertical = min(len(selected_vertical) - n_vertical, remaining)
            all_selected.extend(selected_vertical[n_vertical:n_vertical + additional_vertical])
            remaining -= additional_vertical

        # Then try horizontal
        if remaining > 0 and n_horizontal < len(selected_horizontal):
            additional_horizontal = min(len(selected_horizontal) - n_horizontal, remaining)
            all_selected.extend(selected_horizontal[n_horizontal:n_horizontal + additional_horizontal])
            remaining -= additional_horizontal

        # Finally other lines
        if remaining > 0 and n_other < len(selected_other):
            additional_other = min(len(selected_other) - n_other, remaining)
            all_selected.extend(selected_other[n_other:n_other + additional_other])

    # Sort all selected lines by votes for final ranking
    all_selected.sort(key=lambda x: x[2], reverse=True)

    # Convert to numpy array 
    return np.array(all_selected)

# %% [markdown]
# Then from these selected lines, we calculate the intersection points of these lines in order to detect document corners.

# %%
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

# %% [markdown]
# Using these intersection points, we find the document corners so that we can apply perspective transform to dewarp the document.

# %%
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

# %% [markdown]
# Then we order these points in a clockwise manner so we can use them for perspective transformation. The reason the ordering helps with the transformation is that it ensures the points are in the correct order for the transformation matrix to be calculated correctly.  

# %%
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

# %% [markdown]
# This process_document_image function is the main function that will be called to process the image. If will take the image path as input and return the corners of the document. The workflow is as follows: 
# 1. Read the image and convert it to grayscale.
# 2. Apply Gaussian blur to reduce noise.
# 3. Detect edges using Canny edge detection.
# 4. Perform Hough Transform to detect lines.
# 5. Select the most significant lines that likely represent document boundaries.
# 6. Calculate intersections of the selected lines.
# 7. Find the four corners of the document from the intersections.
# 8. Return the corners of the document.
# 9. Rectify the image using the corners.
# 10. Write the rectified image to the output path.

# %%
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
    os.makedirs(f"{output_dir}/edges", exist_ok=True)
    os.makedirs(f"{output_dir}/lines", exist_ok=True)
    os.makedirs(f"{output_dir}/warped", exist_ok=True)
    
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
    blurred = cv2.GaussianBlur(image, (21, 21), 5)
    
    # Resize for faster processing
    resized_image = cv2.resize(blurred, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    
    # Edge detection with tighter thresholds
    edges = cv2.Canny(resized_image, threshold1=50, threshold2=150)
    
    # Save edges image
    edges_path = f"{output_dir}/edges/{img_num}.jpg"
    os.makedirs(os.path.dirname(edges_path), exist_ok=True)
    cv2.imwrite(edges_path, edges)
    
    # Hough transform with finer resolution
    lines = hough_transform(edges, rho_resolution=0.5, theta_resolution=0.5, threshold_factor=0.5)
    
    print(f"Image {img_num}: Detected {len(lines)} lines")
    
    # Select document boundary lines with angle-based filtering
    # Define thresholds for near-horizontal and near-vertical lines (15 degrees)
    horizontal_threshold = np.pi/12  # 15 degrees from horizontal
    vertical_threshold = np.pi/12    # 15 degrees from vertical
    
    selected_lines = select_document_lines(
        lines, 
        edges.shape, 
        max_lines=8,
        horizontal_threshold=horizontal_threshold, 
        vertical_threshold=vertical_threshold
    )
    
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
        
        # Color horizontal lines in blue, vertical in green, others in red
        color = (0, 0, 255)  # Default red for others
        norm_theta = theta % np.pi
        
        if norm_theta < horizontal_threshold or norm_theta > (np.pi - horizontal_threshold):
            color = (255, 0, 0)  # Blue for horizontal
        elif np.abs(norm_theta - np.pi/2) < vertical_threshold:
            color = (0, 255, 0)  # Green for vertical
            
        cv2.line(resized_image_bgr, (x1, y1), (x2, y2), color, 2)
    
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
    lines_path = f"{output_dir}/lines/{img_num}.jpg"
    os.makedirs(os.path.dirname(lines_path), exist_ok=True)
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
        warped_path = f"{output_dir}/warped/{img_num}.jpg"
        os.makedirs(os.path.dirname(warped_path), exist_ok=True)
        cv2.imwrite(warped_path, warped)
    
    return {
        "success": True,
        "edges_path": edges_path,
        "lines_path": lines_path,
        "warped_path": warped_path,
        "corners_found": corners is not None
    }

# %% [markdown]
# This part is for testing the function with a directory of images and saving the results in a specified output directory. It can be run as a script to process all images in the input directory. The function process_document_image is called for each image, and the results are saved in the output directory. The function also prints a summary of the processing results.

# %%
if __name__ == "__main__":
    # Check if inputs directory provided as argument
    input_dirs = ["WarpDoc/distorted/curved","WarpDoc/distorted/fold","WarpDoc/distorted/incomplete","WarpDoc/distorted/perspective", "WarpDoc/distorted/random", "WarpDoc/distorted/rotate"]
    output_dirs = ["outputs/curved", "outputs/fold", "outputs/incomplete", "outputs/perspective", "outputs/random", "outputs/random"]
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        # Create outputs directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Process images in the input directory
        image_files = list(Path(input_dir).glob("*.jpg"))

        results = []
        for img_path in image_files:
            print(f"Processing {img_path}")
            result = process_document_image(str(img_path), output_dir, resize_factor=0.5)
            results.append((img_path, result))

        # Print summary
        print("\nProcessing Summary:")
        print("-" * 50)
        for img_path, result in results:
            status = "Success" if result["success"] else "Failed"
            corners = "Found" if result.get("corners_found", False) else "Not found"
            print(f"{img_path}: {status}, Corners: {corners}")

# %%




