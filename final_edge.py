import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QLabel, QFileDialog, QGridLayout, QLineEdit, QComboBox
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
        
        # Camera selection dropdown
        self.camera_selector = QComboBox()
        self.camera_selector.addItems(["Top Camera", "Bottom Camera"])
        left_layout.addWidget(QLabel("<b>1. Select Camera Position</b>"))
        left_layout.addWidget(self.camera_selector)

        self.load_button = QPushButton("2. Load Image")
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
        roi_layout.addWidget(QLabel("<b>3. Select Row Range (Optional)</b>"), 0, 0, 1, 2)
        self.start_row_input = QLineEdit()
        self.start_row_input.setPlaceholderText("Start Row (e.g., 100)")
        self.end_row_input = QLineEdit()
        self.end_row_input.setPlaceholderText("End Row (e.g., 200)")
        roi_layout.addWidget(QLabel("From Row:"), 1, 0)
        roi_layout.addWidget(self.start_row_input, 1, 1)
        roi_layout.addWidget(QLabel("To Row:"), 2, 0)
        roi_layout.addWidget(self.end_row_input, 2, 1)
        self.apply_roi_button = QPushButton("4. Analyze Selected Region")
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

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.cv_image = cv2.imread(file_path)
            if self.cv_image is not None:
                self.apply_roi_button.setEnabled(True)
                
                self.current_roi = None
                self.detected_edge = None
                
                height, _, _ = self.cv_image.shape
                
                # Set default ROI based on camera selection
                camera_position = self.camera_selector.currentText()
                if camera_position == "Top Camera":
                    start_row = height*2 // 3
                    end_row = height
                else:  # Bottom Camera
                    start_row = 0
                    end_row = height // 3
                
                self.start_row_input.setText(str(start_row))
                self.end_row_input.setText(str(end_row))
                
                self.display_image(self.cv_image)
                # Automatically analyze the default ROI
                self.analyze_selected_roi()
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

        scale_y_ratio = self.original_pixmap.height() / self.cv_image.shape[0]

        # Draw ROI
        if self.current_roi is not None:
            start_y = int(self.current_roi['start_row'] * scale_y_ratio)
            height = int((self.current_roi['end_row'] - self.current_roi['start_row']) * scale_y_ratio)
            painter.setBrush(QColor(255, 255, 0, 80))
            painter.setPen(Qt.NoPen)
            painter.drawRect(0, start_y, self.original_pixmap.width(), height)

        # Draw detected edge
        if self.detected_edge is not None:
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            
            analysis_width = self.cv_image.shape[1] - 5
            scale_x_ratio = self.original_pixmap.width() / analysis_width if analysis_width > 0 else 1
            edge_x = int(self.detected_edge * scale_x_ratio)
            
            start_y = 0
            end_y = self.original_pixmap.height()
            if self.current_roi is not None:
                start_y = int(self.current_roi['start_row'] * scale_y_ratio)
                end_y = int(self.current_roi['end_row'] * scale_y_ratio)
            
            painter.drawLine(edge_x, start_y, edge_x, end_y)

        painter.end()
        self.image_label.setPixmap(display_pixmap)
        
    def analyze_selected_roi(self):
        if self.cv_image is None: return
        try:
            start_row = int(self.start_row_input.text())
            end_row = int(self.end_row_input.text())
            height, _, _ = self.cv_image.shape
            
            if not (0 <= start_row < end_row <= height):
                print(f"Invalid values! Values must be between 0 and {height}.")
                return

            self.current_roi = {'start_row': start_row, 'end_row': end_row}
            roi_image = self.cv_image[start_row:end_row, :]
            self.analyze_image(roi_image)
            
        except ValueError:
            # If inputs are empty or invalid, analyze the full image
            self.current_roi = None
            self.analyze_image(self.cv_image)

    def _calculate_edge_profile(self, image):
        """Helper to calculate the edge strength profile from an image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_strength_profile = np.mean(edge_magnitude, axis=0)
        return edge_strength_profile

    def analyze_image(self, image_to_analyze):
        """Execute cross-sectional profile analysis with dynamic thresholds."""
        status = "Status: Undetermined State ⚠"

        if image_to_analyze.size == 0: return
        
        # Exclude last 5 columns to avoid artifacts
        image_to_analyze = image_to_analyze[:, :-5]
        
        # 1. Calculate Profiles
        gray_image = cv2.cvtColor(image_to_analyze, cv2.COLOR_BGR2GRAY)
        mean_intensity_profile = np.mean(gray_image, axis=0)
        edge_strength_profile = self._calculate_edge_profile(image_to_analyze)
        edge_position = np.argmax(edge_strength_profile)
        
        # 2. Calculate Dynamic Thresholds based on a fixed 10% percentage
        threshold_percentage = 10.0

        width = image_to_analyze.shape[1]
        window_size = 20
        offset = 10
        
        start_1 = max(0, edge_position - window_size - offset)
        end_1 = max(0, edge_position - offset)
        start_2 = min(width, edge_position + offset)
        end_2 = min(width, edge_position + window_size + offset)
        
        high_brightness_level, low_brightness_level = 249, 0

        if end_1 > start_1 and end_2 > start_2:
            zone1 = mean_intensity_profile[start_1:end_1]
            zone2 = mean_intensity_profile[start_2:end_2]

            if zone1.size > 0 and zone2.size > 0:
                mean1, mean2 = np.mean(zone1), np.mean(zone2)
                high_brightness_level = max(mean1, mean2)
                low_brightness_level = min(mean1, mean2)
        
        brightness_difference = high_brightness_level - low_brightness_level
        margin = (threshold_percentage / 100.0) * brightness_difference
        no_slab_threshold = low_brightness_level + margin
        full_slab_threshold = high_brightness_level - margin
            
        # 3. Determine Slab Status based on RGB and Brightness
        
        # RGB analysis
        b, g, r = cv2.split(image_to_analyze)
        mean_b_profile = np.mean(b, axis=0)
        mean_g_profile = np.mean(g, axis=0)
        mean_r_profile = np.mean(r, axis=0)
        
        blue_dominant_cols = np.sum((mean_b_profile > mean_g_profile) & (mean_b_profile > mean_r_profile))
        total_cols = image_to_analyze.shape[1]
        
        # Main logic
        if brightness_difference > 10:
            # If brightness difference is significant, an edge is present
            self.detected_edge = edge_position
            status = f"Status: Slab edge detected at column {self.detected_edge} ✓"
            self.status_label.setStyleSheet("font-size: 14px; padding: 5px; background-color: #e3f2fd; color: #1565c0; border: 1px solid #90caf9;")
        else:
            # If brightness difference is too small, it's either full or none.
            # We decide based on the average brightness.
            # if np.mean(mean_intensity_profile) > 128: # Heuristic value, can be adjusted
            if blue_dominant_cols > 0.8 * total_cols:     
                 status = "Status: Full slab detected ✓"
                 self.status_label.setStyleSheet("font-size: 14px; padding: 5px; background-color: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7;")

            elif blue_dominant_cols < 0.8 * total_cols:
                status = "Status: No slab detected  ❌"
                self.status_label.setStyleSheet("font-size: 14px; padding: 5px; background-color: #ffebee; color: #c62828; border: 1px solid #ef9a9a;")
            
            self.detected_edge = None

        

        # 4. Update UI
        self.status_label.setText(status)
        self.plot_brightness_profile(gray_image, full_slab_threshold, no_slab_threshold)
        self.plot_rgb_profile(image_to_analyze)
        self.plot_edge_profile(image_to_analyze)
        self.plot_saturation_profile(image_to_analyze)

        # Draw the detected edge line on the plots if an edge was found
        if self.detected_edge is not None:
             self.plot_canvas_1.axes.axvline(x=self.detected_edge, color='r', linestyle='-', alpha=0.8)
             self.plot_canvas_3.axes.axvline(x=self.detected_edge, color='r', linestyle='-', alpha=0.8)
             self.plot_canvas_1.draw()
             self.plot_canvas_3.draw()

        self.update_display()

    def plot_brightness_profile(self, gray_image, full_slab_threshold=None, no_slab_threshold=None):
        """Plot average brightness profile for each column"""
        mean_intensity_profile = np.mean(gray_image, axis=0)
        
        self.plot_canvas_1.axes.clear()
        self.plot_canvas_1.axes.plot(mean_intensity_profile, color='k')
        self.plot_canvas_1.axes.set_title("Brightness Profile")
        self.plot_canvas_1.axes.set_xlabel("Pixel Column")
        self.plot_canvas_1.axes.set_ylabel("Average Intensity")
        self.plot_canvas_1.axes.set_xlim([0, len(mean_intensity_profile)])
        self.plot_canvas_1.axes.set_ylim([0, 256])

        if full_slab_threshold is not None:
            self.plot_canvas_1.axes.axhline(y=full_slab_threshold, color='g', linestyle='--', label=f'Full Thr: {full_slab_threshold:.1f}')
        if no_slab_threshold is not None:
            self.plot_canvas_1.axes.axhline(y=no_slab_threshold, color='r', linestyle='--', label=f'None Thr: {no_slab_threshold:.1f}')
        
        if full_slab_threshold is not None or no_slab_threshold is not None:
            self.plot_canvas_1.axes.legend()
            
        self.plot_canvas_1.draw()

    def plot_rgb_profile(self, image_data):
        """Plot average RGB color profile for each column"""
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
        """Plot edge density profile for each column"""
        edge_strength_profile = self._calculate_edge_profile(image_data)
        
        self.plot_canvas_3.axes.clear()
        self.plot_canvas_3.axes.plot(edge_strength_profile, color='purple')
        self.plot_canvas_3.axes.set_title("Edge Density Profile")
        self.plot_canvas_3.axes.set_xlabel("Pixel Column")
        self.plot_canvas_3.axes.set_ylabel("Average Edge Strength")
        self.plot_canvas_3.axes.set_xlim([0, len(edge_strength_profile)])
        self.plot_canvas_3.draw()
        
    def plot_saturation_profile(self, image_data):
        """Plot color saturation profile for each column"""
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageAnalyzer()
    window.show()
    sys.exit(app.exec())