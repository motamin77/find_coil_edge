import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QLabel, QFileDialog, QGridLayout, QLineEdit
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PySide6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# Class for displaying Matplotlib plots in PySide6
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        self.setParent(parent)

# Main application class
class ImageAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Analyzer with Cross-Sectional Profiles")
        self.setGeometry(100, 100, 1300, 800)

        # Class variables for storing state
        self.cv_image = None
        self.original_pixmap = None
        self.current_roi = None
        self.detected_edge = None

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QGridLayout(self.central_widget)

        # --------- Left Section: Controls and Image Display ---------
        left_layout = QVBoxLayout()
        
        self.load_button = QPushButton("1. Load Image")
        self.load_button.clicked.connect(self.load_image)
        left_layout.addWidget(self.load_button)
        
        # Add slab status label
        self.status_label = QLabel("Status: Waiting for image to be loaded")
        self.status_label.setStyleSheet("font-size: 14px; padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc;")
        self.status_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.status_label)

        self.image_label = QLabel("Load an image to begin analysis")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(400, 400)
        self.image_label.setStyleSheet("border: 2px solid #ccc; font-size: 16px;")
        left_layout.addWidget(self.image_label)

        roi_layout = QGridLayout()
        roi_layout.addWidget(QLabel("<b>2. Select Row Range (Optional)</b>"), 0, 0, 1, 2)
        self.start_row_input = QLineEdit()
        self.start_row_input.setPlaceholderText("Start Row (e.g., 100)")
        self.end_row_input = QLineEdit()
        self.end_row_input.setPlaceholderText("End Row (e.g., 200)")
        roi_layout.addWidget(QLabel("From Row:"), 1, 0)
        roi_layout.addWidget(self.start_row_input, 1, 1)
        roi_layout.addWidget(QLabel("To Row:"), 2, 0)
        roi_layout.addWidget(self.end_row_input, 2, 1)
        self.apply_roi_button = QPushButton("3. Analyze Selected Region")
        self.apply_roi_button.clicked.connect(self.analyze_selected_roi)
        self.apply_roi_button.setEnabled(False)
        left_layout.addLayout(roi_layout)
        left_layout.addWidget(self.apply_roi_button)
        self.main_layout.addLayout(left_layout, 0, 0)

        # --------- Right Section: Cross-Sectional Profile Plots ---------
        right_layout = QGridLayout()
        self.plot_canvas_1 = MplCanvas(self, width=5, height=4, dpi=100)
        self.plot_canvas_2 = MplCanvas(self, width=5, height=4, dpi=100)
        self.plot_canvas_3 = MplCanvas(self, width=5, height=4, dpi=100)
        self.plot_canvas_4 = MplCanvas(self, width=5, height=4, dpi=100)
        
        right_layout.addWidget(QLabel("Plot 1: Brightness Profile"), 0, 0)
        right_layout.addWidget(self.plot_canvas_1, 1, 0)
        right_layout.addWidget(QLabel("Plot 2: RGB Color Profile"), 0, 1)
        right_layout.addWidget(self.plot_canvas_2, 1, 1)
        right_layout.addWidget(QLabel("Plot 3: Edge Density Profile"), 2, 0)
        right_layout.addWidget(self.plot_canvas_3, 3, 0)
        right_layout.addWidget(QLabel("Plot 4: HSV Saturation Profile"), 2, 1)
        right_layout.addWidget(self.plot_canvas_4, 3, 1)
        
        self.main_layout.addLayout(right_layout, 0, 1)

        self.cv_image = None
        self.original_pixmap = None

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.cv_image = cv2.imread(file_path)
            if self.cv_image is not None:
                self.apply_roi_button.setEnabled(True)
                # Convert image to Pixmap
                rgb_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.original_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                self.current_roi = None  # Clear previous region
                self.detected_edge = None  # Clear previous edge
                
                self.start_row_input.clear()
                self.end_row_input.clear()
                height, _, _ = self.cv_image.shape
                self.end_row_input.setPlaceholderText(f"End Row (max: {height})")
                
                # Analyze and display image
                self.analyze_image(self.cv_image)
            else:
                self.image_label.setText("Error loading image!")

    def display_image(self, image_data):
        """Convert OpenCV image to Pixmap for display"""
        rgb_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.original_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.update_display()

    def update_display(self):
        """Update image display with selected region and edge line"""
        if self.cv_image is None or self.original_pixmap is None:
            return

        display_pixmap = self.original_pixmap.copy()
        painter = QPainter(display_pixmap)

        # Calculate display scale
        scale_ratio = self.original_pixmap.height() / self.cv_image.shape[0]

        # Draw selected region in yellow
        if self.current_roi is not None:
            start_y = int(self.current_roi['start_row'] * scale_ratio)
            height = int((self.current_roi['end_row'] - self.current_roi['start_row']) * scale_ratio)
            painter.setBrush(QColor(255, 255, 0, 80))
            painter.setPen(Qt.NoPen)
            painter.drawRect(0, start_y, self.original_pixmap.width(), height)

        # Draw edge line in green
        if self.detected_edge is not None:
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            edge_x = int(self.detected_edge * scale_ratio)
            
            if self.current_roi is not None:
                # Draw line only in selected region
                start_y = int(self.current_roi['start_row'] * scale_ratio)
                end_y = int(self.current_roi['end_row'] * scale_ratio)
            else:
                # Draw line across full image height
                start_y = 0
                end_y = self.original_pixmap.height()
            
            painter.drawLine(edge_x, start_y, edge_x, end_y)

        painter.end()
        self.image_label.setPixmap(display_pixmap)

    def analyze_selected_roi(self):
        if self.cv_image is None: return
        try:
            start_row_str = self.start_row_input.text()
            end_row_str = self.end_row_input.text()
            
            if not start_row_str or not end_row_str:
                self.current_roi = None
                self.analyze_image(self.cv_image)
                return

            start_row = int(start_row_str)
            end_row = int(end_row_str)
            height, _, _ = self.cv_image.shape
            
            if not (0 <= start_row < end_row <= height):
                print(f"Invalid values! Values must be between 0 and {height}.")
                return

            # Store selected region in class variables
            self.current_roi = {
                'start_row': start_row,
                'end_row': end_row
            }
            
            # Analyze selected region
            roi_image = self.cv_image[start_row:end_row, :]
            self.analyze_image(roi_image)
            
        except ValueError:
            print("Please enter valid numeric values.")    
            
    def analyze_image(self, image_to_analyze):
        """Execute cross-sectional profile analysis scenarios"""
        if image_to_analyze.size == 0: return # Skip if ROI is empty
        
        # Remove 5 pixels from the right side of the image
        image_to_analyze = image_to_analyze[:, :-5]
        
        # Initial image analysis
        gray_image = cv2.cvtColor(image_to_analyze, cv2.COLOR_BGR2GRAY)
        mean_intensity_profile = np.mean(gray_image, axis=0)
        
        # Check for slab presence in the image
        if np.max(mean_intensity_profile) <= 100:
            status = "Status: No slab detected in the image ❌"
            self.status_label.setStyleSheet("font-size: 14px; padding: 5px; background-color: #ffebee; color: #c62828; border: 1px solid #ef9a9a;")
            self.detected_edge = None
        elif np.min(mean_intensity_profile) >= 90:
            status = "Status: Full slab detected ✓"
            self.status_label.setStyleSheet("font-size: 14px; padding: 5px; background-color: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7;")
            self.detected_edge = None
        else:
            # Detect edges only if slab is not complete
            edges = self.detect_slab_edges(image_to_analyze)
            
            if edges:
                self.detected_edge = edges[0]  # Store edge position
                status = f"Status: Slab edge detected at column {self.detected_edge} ✓"
                self.status_label.setStyleSheet("font-size: 14px; padding: 5px; background-color: #e3f2fd; color: #1565c0; border: 1px solid #90caf9;")
            else:
                status = "Status: Partial slab visible, but no edge detected ⚠"
                self.status_label.setStyleSheet("font-size: 14px; padding: 5px; background-color: #fff3e0; color: #ef6c00; border: 1px solid #ffcc80;")
                self.detected_edge = None
        
        # Update user interface
        self.status_label.setText(status)
        self.plot_brightness_profile(image_to_analyze)
        self.plot_rgb_profile(image_to_analyze)
        self.plot_edge_profile(image_to_analyze)
        self.plot_saturation_profile(image_to_analyze)
        
        # Update image display
        self.update_display()

    def plot_brightness_profile(self, image_data):
        """Scenario 1: Plot average brightness profile for each column"""
        gray_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        # Calculate mean for each column (axis=0)
        mean_intensity_profile = np.mean(gray_image, axis=0)
        
        self.plot_canvas_1.axes.clear()
        self.plot_canvas_1.axes.plot(mean_intensity_profile, color='k')
        self.plot_canvas_1.axes.set_title("Brightness Profile")
        self.plot_canvas_1.axes.set_xlabel("Pixel Column")
        self.plot_canvas_1.axes.set_ylabel("Average Intensity")
        self.plot_canvas_1.axes.set_xlim([0, len(mean_intensity_profile)])
        self.plot_canvas_1.axes.set_ylim([0, 256])
        self.plot_canvas_1.draw()

    def plot_rgb_profile(self, image_data):
        """Scenario 2: Plot average RGB color profile for each column"""
        b, g, r = cv2.split(image_data)
        mean_b_profile = np.mean(b, axis=0)
        mean_g_profile = np.mean(g, axis=0)
        mean_r_profile = np.mean(r, axis=0)

        self.plot_canvas_2.axes.clear()
        self.plot_canvas_2.axes.plot(mean_b_profile, color='blue', label='Blue')
        self.plot_canvas_2.axes.plot(mean_g_profile, color='green', label='Green')
        self.plot_canvas_2.axes.plot(mean_r_profile, color='red', label='Red')
        self.plot_canvas_2.axes.set_title("RGB Color Profile")
        self.plot_canvas_2.axes.set_xlabel("Pixel Column")
        self.plot_canvas_2.axes.set_ylabel("Average Intensity")
        self.plot_canvas_2.axes.legend()
        self.plot_canvas_2.axes.set_xlim([0, len(mean_b_profile)])
        self.plot_canvas_2.axes.set_ylim([0, 256])
        self.plot_canvas_2.draw()

    def plot_edge_profile(self, image_data):
        """Scenario 3: Plot edge density profile for each column"""
        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        mean_edge_profile = np.mean(gradient_magnitude, axis=0)

        self.plot_canvas_3.axes.clear()
        self.plot_canvas_3.axes.plot(mean_edge_profile, color='purple')
        self.plot_canvas_3.axes.set_title("Edge Density Profile")
        self.plot_canvas_3.axes.set_xlabel("Pixel Column")
        self.plot_canvas_3.axes.set_ylabel("Average Edge Strength")
        self.plot_canvas_3.axes.set_xlim([0, len(mean_edge_profile)])
        self.plot_canvas_3.draw()
        
    def plot_saturation_profile(self, image_data):
        """Scenario 4: Plot color saturation profile for each column"""
        hsv_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2HSV)
        saturation_channel = hsv_image[:, :, 1]
        
        mean_saturation_profile = np.mean(saturation_channel, axis=0)

        self.plot_canvas_4.axes.clear()
        self.plot_canvas_4.axes.plot(mean_saturation_profile, color='orange')
        self.plot_canvas_4.axes.set_title("HSV Saturation Profile")
        self.plot_canvas_4.axes.set_xlabel("Pixel Column")
        self.plot_canvas_4.axes.set_ylabel("Average Saturation")
        self.plot_canvas_4.axes.set_xlim([0, len(mean_saturation_profile)])
        self.plot_canvas_4.axes.set_ylim([0, 256])
        self.plot_canvas_4.draw()

    def detect_slab_edges(self, image):
        """Detect slab edge using maximum Average Edge Strength value"""
        # Calculate edge density profile
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_strength_profile = np.mean(edge_magnitude, axis=0)
        
        # Find column with maximum Average Edge Strength
        edge_position = np.argmax(edge_strength_profile)
        max_strength = edge_strength_profile[edge_position]
        
        # Display numerical value of maximum Edge Strength in the plot
        self.plot_canvas_3.axes.annotate(
            f'Max Strength: {max_strength:.1f}',
            xy=(edge_position, max_strength),
            xytext=(10, 10),
            textcoords='offset points',
            color='red',
            bbox=dict(facecolor='white', edgecolor='red', alpha=0.7)
        )
        
        # Draw vertical lines on the plots
        self.plot_canvas_1.axes.axvline(x=edge_position, color='r', linestyle='-', alpha=0.8)
        self.plot_canvas_3.axes.axvline(x=edge_position, color='r', linestyle='-', alpha=0.8)
        
        return [edge_position]

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageAnalyzer()
    window.show()
    sys.exit(app.exec())