import cv2
import numpy as np

class TrafficSignDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.lower_red = np.array([0, 70, 50])
        self.upper_red = np.array([10, 255, 255])
        self.templates = {
            "stop": cv2.Canny(cv2.imread('stop_template.jpg', 0), 50, 150)
        }

    def color_segmentation(self, hsv_frame):
        mask_red = cv2.inRange(hsv_frame, self.lower_red, self.upper_red)
        return mask_red

    def edge_detection(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    def template_matching(self, edges):
        best_match = None
        best_val = 0
        for name, template in self.templates.items():
            result = cv2.matchTemplate(edges, template, cv2.TM_CCOEFF)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_val:
                best_val = max_val
                best_match = (name, max_loc)
        return best_match

    def shape_analysis(self, contour):
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        corners = len(approx)
        return aspect_ratio, corners

    def process_frame(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self.color_segmentation(hsv_frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                roi = frame[y:y+h, x:x+w]
                edges = self.edge_detection(roi)
                match = self.template_matching(edges)
                if match:
                    aspect_ratio, corners = self.shape_analysis(contour)
                    if self.is_valid_sign(aspect_ratio, corners):
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, match[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

    def is_valid_sign(self, aspect_ratio, corners):
        if 0.8 < aspect_ratio < 1.2 and corners in [3, 4, 8]:
            return True
        return False

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            processed_frame = self.process_frame(frame)
            cv2.imshow('Traffic Sign Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = TrafficSignDetector()
    detector.run()
