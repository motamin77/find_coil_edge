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
    # --- CONSTANTS ---
    # Moved all hardcoded values here to make the code more dynamic and readable
    CROP_COLUMNS = 5  # Number of columns to crop from the right to avoid edge artifacts
    THRESHOLD_PERCENTAGE = 10.0  # Percentage margin for calculating dynamic thresholds
    BRIGHTNESS_WINDOW_SIZE = 20  # Window size for sampling brightness near the edge
    BRIGHTNESS_WINDOW_OFFSET = 10  # Offset from the edge to start sampling
    MIN_BRIGHTNESS_DIFF_FOR_EDGE = 10  # Minimum brightness difference to consider an edge significant
    BLUE_DOMINANCE_THRESHOLD_PERCENT = 0.8  # 80% threshold for detecting a full slab based on blue color
    SOBEL_KERNEL_SIZE = 5 # Kernel size for the Sobel edge detection filter

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

        # Setup UI elements
        self._setup_ui_controls()
        self._setup_plots()

    def _setup_ui_controls(self):
        """Helper method to initialize UI controls."""
        left_layout = QVBoxLayout()
        
        self.camera_selector = QComboBox()
        self.camera_selector.addItems(["Top Camera", "Bottom Camera"])
        left_layout.addWidget(QLabel("<b>1. Select Camera Position</b>"))
        left_layout.addWidget(self.camera_selector)

        self.load_button = QPushButton("2. Load Image")
        self.load_button.clicked.connect(self.load_image)
        left_layout.addWidget(self.load_button)
        
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

    def _setup_plots(self):
        """Helper method to initialize plot canvases."""
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
                
                camera_position = self.camera_selector.currentText()
                start_row = height // 2 if camera_position == "Top Camera" else 0
                end_row = height if camera_position == "Top Camera" else height // 2
                
                self.start_row_input.setText(str(start_row))
                self.end_row_input.setText(str(end_row))
                
                self.display_image(self.cv_image)
                self.analyze_selected_roi()
            else:
                self.image_label.setText("Error loading image!")

    def display_image(self, image_data):
        rgb_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        q_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.original_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.update_display()

    def update_display(self):
        if self.cv_image is None or self.original_pixmap is None:
            return

        display_pixmap = self.original_pixmap.copy()
        painter = QPainter(display_pixmap)
        scale_y = self.original_pixmap.height() / self.cv_image.shape[0]

        if self.current_roi is not None:
            start_y = int(self.current_roi['start_row'] * scale_y)
            height = int((self.current_roi['end_row'] - self.current_roi['start_row']) * scale_y)
            painter.setBrush(QColor(255, 255, 0, 80))
            painter.setPen(Qt.NoPen)
            painter.drawRect(0, start_y, self.original_pixmap.width(), height)

        if self.detected_edge is not None:
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            analysis_width = self.cv_image.shape[1] - self.CROP_COLUMNS
            scale_x = self.original_pixmap.width() / analysis_width if analysis_width > 0 else 1
            edge_x = int(self.detected_edge * scale_x)
            
            start_y = int(self.current_roi['start_row'] * scale_y) if self.current_roi else 0
            end_y = int(self.current_roi['end_row'] * scale_y) if self.current_roi else self.original_pixmap.height()
            
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
            self.current_roi = None
            self.analyze_image(self.cv_image)

    def _calculate_edge_profile(self, gray_image):
        """Helper to calculate edge strength profile from a pre-converted grayscale image."""
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=self.SOBEL_KERNEL_SIZE)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=self.SOBEL_KERNEL_SIZE)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        return np.mean(edge_magnitude, axis=0)

    def analyze_image(self, image_to_analyze):
        """Execute cross-sectional profile analysis with dynamic thresholds."""
        if image_to_analyze.size == 0: return
        
        # Crop columns using the class constant
        image_to_analyze = image_to_analyze[:, :-self.CROP_COLUMNS]
        
        # 1. Calculate Profiles
        # OPTIMIZATION: Convert to grayscale once and reuse it
        gray_image = cv2.cvtColor(image_to_analyze, cv2.COLOR_BGR2GRAY)
        mean_intensity_profile = np.mean(gray_image, axis=0)
        edge_strength_profile = self._calculate_edge_profile(gray_image) # Pass gray image
        edge_position = np.argmax(edge_strength_profile)
        
        # 2. Calculate Dynamic Thresholds
        width = image_to_analyze.shape[1]
        start_1 = max(0, edge_position - self.BRIGHTNESS_WINDOW_SIZE - self.BRIGHTNESS_WINDOW_OFFSET)
        end_1 = max(0, edge_position - self.BRIGHTNESS_WINDOW_OFFSET)
        start_2 = min(width, edge_position + self.BRIGHTNESS_WINDOW_OFFSET)
        end_2 = min(width, edge_position + self.BRIGHTNESS_WINDOW_SIZE + self.BRIGHTNESS_WINDOW_OFFSET)
        
        high_brightness_level, low_brightness_level = 249, 0
        if end_1 > start_1 and end_2 > start_2:
            zone1 = mean_intensity_profile[start_1:end_1]
            zone2 = mean_intensity_profile[start_2:end_2]

            if zone1.size > 0 and zone2.size > 0:
                mean1, mean2 = np.mean(zone1), np.mean(zone2)
                high_brightness_level = max(mean1, mean2)
                low_brightness_level = min(mean1, mean2)
        
        brightness_difference = high_brightness_level - low_brightness_level
        margin = (self.THRESHOLD_PERCENTAGE / 100.0) * brightness_difference
        no_slab_threshold = low_brightness_level + margin
        full_slab_threshold = high_brightness_level - margin
            
        # 3. Determine Slab Status
        b, g, r = cv2.split(image_to_analyze)
        mean_b_profile = np.mean(b, axis=0)
        mean_g_profile = np.mean(g, axis=0)
        mean_r_profile = np.mean(r, axis=0)
        
        blue_dominant_cols = np.sum((mean_b_profile > mean_g_profile) & (mean_b_profile > mean_r_profile))
        total_cols = image_to_analyze.shape[1]
        
        status = "Status: Undetermined State ⚠"
        self.detected_edge = None
        
        # Logic using class constants
        if brightness_difference > self.MIN_BRIGHTNESS_DIFF_FOR_EDGE:
            self.detected_edge = edge_position
            status = f"Status: Slab edge detected at column {self.detected_edge} ✓"
            self.status_label.setStyleSheet("font-size: 14px; padding: 5px; background-color: #e3f2fd; color: #1565c0; border: 1px solid #90caf9;")
        else:
            if blue_dominant_cols > self.BLUE_DOMINANCE_THRESHOLD_PERCENT * total_cols:
                status = "Status: Full slab detected ✓"
                self.status_label.setStyleSheet("font-size: 14px; padding: 5px; background-color: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7;")
            else: # Assuming if not blue dominant, it's no slab.
                status = "Status: No slab detected  ❌"
                self.status_label.setStyleSheet("font-size: 14px; padding: 5px; background-color: #ffebee; color: #c62828; border: 1px solid #ef9a9a;")

        # 4. Update UI
        self.status_label.setText(status)
        self.plot_all_profiles(gray_image, image_to_analyze, edge_strength_profile, full_slab_threshold, no_slab_threshold)
        self.update_display()

    def plot_all_profiles(self, gray_image, color_image, edge_profile, full_thr, none_thr):
        """A single method to update all plots."""
        # Plot 1: Brightness
        mean_intensity = np.mean(gray_image, axis=0)
        self.plot_canvas_1.axes.clear()
        self.plot_canvas_1.axes.plot(mean_intensity, color='k')
        self.plot_canvas_1.axes.set_title("Brightness Profile")
        self.plot_canvas_1.axes.set_xlabel("Pixel Column")
        self.plot_canvas_1.axes.set_ylabel("Average Intensity")
        self.plot_canvas_1.axes.set_xlim([0, len(mean_intensity)])
        self.plot_canvas_1.axes.set_ylim([0, 256])
        self.plot_canvas_1.axes.axhline(y=full_thr, color='g', linestyle='--', label=f'Full Thr: {full_thr:.1f}')
        self.plot_canvas_1.axes.axhline(y=none_thr, color='r', linestyle='--', label=f'None Thr: {none_thr:.1f}')
        self.plot_canvas_1.axes.legend()
        
        # Plot 2: RGB
        b, g, r = cv2.split(color_image)
        self.plot_canvas_2.axes.clear()
        self.plot_canvas_2.axes.plot(np.mean(b, axis=0), color='blue', label='Blue')
        self.plot_canvas_2.axes.plot(np.mean(g, axis=0), color='green', label='Green')
        self.plot_canvas_2.axes.plot(np.mean(r, axis=0), color='red', label='Red')
        self.plot_canvas_2.axes.set_title("RGB Color Profile")
        self.plot_canvas_2.axes.legend()
        self.plot_canvas_2.axes.set_xlim([0, color_image.shape[1]])
        self.plot_canvas_2.axes.set_ylim([0, 256])

        # Plot 3: Edge
        self.plot_canvas_3.axes.clear()
        self.plot_canvas_3.axes.plot(edge_profile, color='purple')
        self.plot_canvas_3.axes.set_title("Edge Density Profile")
        self.plot_canvas_3.axes.set_xlabel("Pixel Column")
        self.plot_canvas_3.axes.set_ylabel("Average Edge Strength")
        self.plot_canvas_3.axes.set_xlim([0, len(edge_profile)])

        # Plot 4: Saturation
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        mean_saturation = np.mean(hsv_image[:, :, 1], axis=0)
        self.plot_canvas_4.axes.clear()
        self.plot_canvas_4.axes.plot(mean_saturation, color='orange')
        self.plot_canvas_4.axes.set_title("HSV Saturation Profile")
        self.plot_canvas_4.axes.set_xlabel("Pixel Column")
        self.plot_canvas_4.axes.set_ylabel("Average Saturation")
        self.plot_canvas_4.axes.set_xlim([0, len(mean_saturation)])
        self.plot_canvas_4.axes.set_ylim([0, 256])

        # Draw edge line on relevant plots
        if self.detected_edge is not None:
            for canvas in [self.plot_canvas_1, self.plot_canvas_2, self.plot_canvas_3, self.plot_canvas_4]:
                canvas.axes.axvline(x=self.detected_edge, color='r', linestyle='-', alpha=0.8)
        
        # Redraw all canvases
        for canvas in [self.plot_canvas_1, self.plot_canvas_2, self.plot_canvas_3, self.plot_canvas_4]:
            canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageAnalyzer()
    window.show()
    sys.exit(app.exec())