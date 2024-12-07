import cv2
import numpy as np

def detect_lines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    return lines

# Function to check if driver crossed the lines
def is_line_crossed(lines, frame_height):
    # Define the boundaries of the lines
    # These values should be adjusted based on your specific track dimensions
    lower_bound = 100  # Example coordinate for lower line
    upper_bound = frame_height - 100  # Example coordinate for upper line
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if (y1 < lower_bound and y2 < lower_bound) or (y1 > upper_bound and y2 > upper_bound):
                return True  # Line crossed
    return False  # No line crossed


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    line_crossed_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        lines = detect_lines(frame)
        
        if lines is not None:
            if is_line_crossed(lines, frame.shape[0]):
                line_crossed_count += 1

        # Optionally, display the frame with detected lines
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    
    if line_crossed_count > 0:
        print("Test Result: Fail")
    else:
        print("Test Result: Pass")

if __name__ == "__main__":
    video_path = 't.mp4'
    process_video(video_path)
