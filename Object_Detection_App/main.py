import os

import cv2
import numpy as np
import time
import threading
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.properties import (
    StringProperty,
    NumericProperty,
    ListProperty,
    ObjectProperty,
)
from detector import YOLODetector


class ResultsTable(GridLayout):
    """Custom widget to display detection results"""

    def __init__(self, **kwargs):
        super(ResultsTable, self).__init__(**kwargs)
        self.cols = 4
        self.spacing = [5, 5]
        self.padding = [5, 5]

        # Add header
        self.add_widget(Label(text="#", bold=True, size_hint_x=0.1))
        self.add_widget(Label(text="Object", bold=True, size_hint_x=0.3))
        self.add_widget(Label(text="Confidence", bold=True, size_hint_x=0.2))
        self.add_widget(Label(text="Bounding Box", bold=True, size_hint_x=0.4))

    def update_results(self, detections):
        """Update the results table with detection data"""
        # Clear previous results (keep header)
        while len(self.children) > 4:
            self.remove_widget(self.children[0])

        # Add detection results
        for i, det in enumerate(detections):
            # Add items in reverse order (Kivy's layout is bottom-up)
            self.add_widget(Label(text=str(det["bbox"]), size_hint_x=0.4))
            self.add_widget(Label(text=f"{det['confidence']:.2f}", size_hint_x=0.2))
            self.add_widget(Label(text=det["class"], size_hint_x=0.3))
            self.add_widget(Label(text=str(i + 1), size_hint_x=0.1))


class ImageDisplay(Image):
    """Custom Image widget with aspect ratio preservation"""

    def __init__(self, **kwargs):
        super(ImageDisplay, self).__init__(**kwargs)
        self.allow_stretch = True
        self.keep_ratio = True

    def display_opencv_image(self, opencv_image):
        """Display an OpenCV image on the Kivy Image widget"""
        if opencv_image is None:
            return False

        # Convert to texture
        buf = cv2.flip(
            opencv_image, 0
        )  # Flip vertically (Kivy's origin is bottom-left)
        buf = buf.tobytes()
        texture = Texture.create(
            size=(opencv_image.shape[1], opencv_image.shape[0]), colorfmt="rgb"
        )
        texture.blit_buffer(buf, colorfmt="rgb", bufferfmt="ubyte")

        # Display the texture
        self.texture = texture
        return True


class FileChooserPopup(Popup):
    """Popup for selecting an image file"""

    load = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(FileChooserPopup, self).__init__(**kwargs)
        self.title = "Select an Image"
        self.size_hint = (0.9, 0.9)

        layout = BoxLayout(orientation="vertical")

        # File chooser
        self.file_chooser = FileChooserListView(
            filters=["*.png", "*.jpg", "*.jpeg", "*.bmp"], path=os.path.expanduser("~")
        )
        layout.add_widget(self.file_chooser)

        # Buttons
        button_layout = BoxLayout(size_hint_y=None, height=50)

        cancel_btn = Button(text="Cancel")
        cancel_btn.bind(on_release=self.dismiss)
        button_layout.add_widget(cancel_btn)

        load_btn = Button(text="Load")
        load_btn.bind(on_release=self._load)
        button_layout.add_widget(load_btn)

        layout.add_widget(button_layout)
        self.content = layout

    def _load(self, instance):
        if self.file_chooser.selection:
            self.load(self.file_chooser.selection[0])
            self.dismiss()


class ObjectDetectionApp(App):
    """Main application class"""

    status_text = StringProperty("Ready")
    confidence_value = NumericProperty(0.5)

    def build(self):
        # Initialize detector in a separate thread
        threading.Thread(target=self.init_detector, daemon=True).start()

        # Set window title and size
        self.title = "Object Detection App"
        Window.size = (1200, 800)
        Window.minimum_width = 800
        Window.minimum_height = 600

        # Initialize variables
        self.current_image_path = None
        self.original_image = None
        self.image_data = None
        self.processed_image = None
        self.detections = None
        self.detector = None
        self.detector_ready = False

        # Main layout
        main_layout = BoxLayout(orientation="vertical", padding=10, spacing=10)

        # Controls layout
        controls_layout = BoxLayout(size_hint_y=None, height=50, spacing=10)

        # Load image button
        self.load_btn = Button(text="Load Image", size_hint_x=None, width=120)
        self.load_btn.bind(on_release=self.show_load_dialog)
        controls_layout.add_widget(self.load_btn)

        # Detect button
        self.detect_btn = Button(
            text="Detect Objects", size_hint_x=None, width=120, disabled=True
        )
        self.detect_btn.bind(on_release=self.detect_objects)
        controls_layout.add_widget(self.detect_btn)

        # Confidence slider
        controls_layout.add_widget(
            Label(text="Confidence:", size_hint_x=None, width=80)
        )
        self.conf_slider = Slider(
            min=0.1, max=1.0, value=0.5, step=0.01, size_hint_x=None, width=200
        )
        self.conf_slider.bind(value=self.update_confidence)
        controls_layout.add_widget(self.conf_slider)

        # Confidence value label
        self.conf_label = Label(text="0.50", size_hint_x=None, width=50)
        controls_layout.add_widget(self.conf_label)

        # Status label
        self.status_label = Label(text=self.status_text, halign="left")
        controls_layout.add_widget(self.status_label)

        main_layout.add_widget(controls_layout)

        # Images layout
        images_layout = BoxLayout(spacing=10)

        # Original image
        original_box = BoxLayout(orientation="vertical")
        original_box.add_widget(
            Label(
                text="Original Image",
                size_hint_y=None,
                height=30,
                halign="left",
                bold=True,
            )
        )
        self.original_image_widget = ImageDisplay()
        original_box.add_widget(self.original_image_widget)
        images_layout.add_widget(original_box)

        # Processed image
        processed_box = BoxLayout(orientation="vertical")
        processed_box.add_widget(
            Label(
                text="Processed Image",
                size_hint_y=None,
                height=30,
                halign="left",
                bold=True,
            )
        )
        self.processed_image_widget = ImageDisplay()
        processed_box.add_widget(self.processed_image_widget)
        images_layout.add_widget(processed_box)

        main_layout.add_widget(images_layout)

        # Results layout
        results_box = BoxLayout(orientation="vertical", size_hint_y=0.3)
        results_box.add_widget(
            Label(
                text="Detection Results",
                size_hint_y=None,
                height=30,
                halign="left",
                bold=True,
            )
        )

        # Scrollable results
        scroll_view = ScrollView(bar_width=10)
        self.results_table = ResultsTable(size_hint_y=None)
        # Bind height to children to make sure scrolling works
        self.results_table.bind(minimum_height=self.results_table.setter("height"))
        scroll_view.add_widget(self.results_table)
        results_box.add_widget(scroll_view)

        main_layout.add_widget(results_box)

        # Bind status text to label
        self.bind(status_text=self.status_label.setter("text"))
        self.bind(confidence_value=self.update_conf_label)

        return main_layout

    def init_detector(self):
        """Initialize the YOLO detector in a separate thread"""
        try:
            self.detector = YOLODetector()
            self.detector_ready = True
            # Update UI in the main thread
            Clock.schedule_once(
                lambda dt: setattr(self, "status_text", "Model loaded successfully"), 0
            )
        except Exception as e:
            Clock.schedule_once(
                lambda dt: setattr(
                    self, "status_text", f"Error loading model: {str(e)}"
                ),
                0,
            )

    def update_confidence(self, instance, value):
        """Update confidence value when slider changes"""
        self.confidence_value = value

    def update_conf_label(self, instance, value):
        """Update confidence label text"""
        self.conf_label.text = f"{value:.2f}"

    def show_load_dialog(self, *args):
        """Show file chooser popup"""
        popup = FileChooserPopup(load=self.load_image)
        popup.open()

    def load_image(self, path):
        """Load an image from path"""
        if not path:
            return

        self.current_image_path = path
        self.status_text = f"Loaded: {os.path.basename(path)}"

        try:
            # Load image
            self.original_image = cv2.imread(path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.image_data = self.original_image

            # Display image
            self.original_image_widget.display_opencv_image(self.original_image)

            # Enable detect button
            self.detect_btn.disabled = False

            # Clear processed image and results
            self.processed_image_widget.source = ""
            # Reset results
            self.results_table.update_results([])

        except Exception as e:
            self.status_text = f"Error: {str(e)}"

    def detect_objects(self, *args):
        """Process the loaded image for object detection"""
        if not self.detector_ready:
            self.status_text = "Model is still loading, please wait..."
            return

        if self.current_image_path is None:
            self.status_text = "Error: No image loaded"
            return

        # Get confidence threshold
        confidence = self.confidence_value

        # Update UI
        self.status_text = "Detecting objects..."
        self.detect_btn.disabled = True

        # Run detection in a separate thread
        threading.Thread(
            target=self.detection_thread, args=(confidence,), daemon=True
        ).start()

    def detection_thread(self, confidence):
        """Run detection in a background thread"""
        try:
            start_time = time.time()
            processed_img, detections = self.detector.detect(
                self.image_data, conf_threshold=confidence
            )
            processing_time = time.time() - start_time

            # Update UI in the main thread
            Clock.schedule_once(
                lambda dt: self.handle_results(
                    processed_img, detections, processing_time
                ),
                0,
            )
        except Exception as e:
            Clock.schedule_once(lambda dt: self.handle_error(str(e)), 0)

    def handle_results(self, processed_img, detections, processing_time):
        """Handle detection results from worker thread"""
        self.processed_image = processed_img
        self.detections = detections

        # Display processed image
        self.processed_image_widget.display_opencv_image(processed_img)

        # Update results table
        self.results_table.update_results(detections)

        # Update status
        self.status_text = (
            f"Detected {len(detections)} objects in {processing_time:.3f} seconds"
        )

        # Re-enable detect button
        self.detect_btn.disabled = False

    def handle_error(self, error_msg):
        """Handle errors from worker thread"""
        self.status_text = f"Error: {error_msg}"
        self.detect_btn.disabled = False


if __name__ == "__main__":
    ObjectDetectionApp().run()
