import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

CLASS_LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary tumor",
}


class PredictionThread(QThread):
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, image_path, predictor=None, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.predictor = predictor

    def run(self):
        try:
            if self.predictor is None:
                from predictor import Predictor

                self.predictor = Predictor()

            result = self.predictor.predict(self.image_path)
            self.result_ready.emit(result)

        except (OSError, ValueError, RuntimeError, ImportError) as error:
            self.error_occurred.emit(str(error))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.image_path = None
        self.original_pixmap = None
        self.predictor = None
        self.worker = None

        self.setWindowTitle("NeuroVision")
        self.resize(1100, 800)
        self.setMinimumSize(850, 720)

        root = QWidget()
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(36, 28, 36, 28)
        layout.setSpacing(18)

        brand = QLabel("NEUROVISION")
        brand.setObjectName("brand")
        layout.addWidget(brand)

        title = QLabel("Brain MRI Classification")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Upload an MRI image to explore the model prediction.")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        content = QHBoxLayout()
        content.setSpacing(24)
        layout.addLayout(content, stretch=1)

        image_card = QFrame()
        image_card.setObjectName("card")

        image_layout = QVBoxLayout(image_card)
        image_layout.setContentsMargins(20, 20, 20, 20)
        image_layout.setSpacing(16)

        self.preview = QLabel("No image selected")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 320)
        self.preview.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Expanding,
        )
        self.preview.setObjectName("preview")
        image_layout.addWidget(self.preview, stretch=1)

        self.file_label = QLabel("JPG / JPEG / PNG")
        self.file_label.setWordWrap(True)
        self.file_label.setObjectName("subtitle")
        image_layout.addWidget(self.file_label)

        self.upload_button = QPushButton("Choose MRI image")
        self.upload_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.upload_button.clicked.connect(self.select_image)
        image_layout.addWidget(self.upload_button)

        content.addWidget(image_card, stretch=1)

        result_card = QFrame()
        result_card.setObjectName("card")

        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(28, 28, 28, 28)
        result_layout.setSpacing(16)

        result_heading = QLabel("Prediction")
        result_heading.setObjectName("sectionTitle")
        result_layout.addWidget(result_heading)

        self.result_label = QLabel("Waiting for an image")
        self.result_label.setObjectName("result")
        self.result_label.setWordWrap(True)
        result_layout.addWidget(self.result_label)

        self.confidence_label = QLabel("Model confidence will appear here.")
        self.confidence_label.setWordWrap(True)
        self.confidence_label.setObjectName("subtitle")
        result_layout.addWidget(self.confidence_label)

        self.probability_bars = {}
        self.probability_labels = {}

        for class_name, display_name in CLASS_LABELS.items():
            row = QHBoxLayout()

            label = QLabel(display_name)
            value = QLabel("--")
            value.setAlignment(Qt.AlignmentFlag.AlignRight)

            row.addWidget(label)
            row.addStretch()
            row.addWidget(value)
            result_layout.addLayout(row)

            bar = QProgressBar()
            bar.setRange(0, 10000)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(8)
            result_layout.addWidget(bar)

            self.probability_labels[class_name] = value
            self.probability_bars[class_name] = bar

        result_layout.addStretch()

        self.analyze_button = QPushButton("Analyze image")
        self.analyze_button.setEnabled(False)
        self.analyze_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.analyze_button.clicked.connect(self.analyze_image)
        result_layout.addWidget(self.analyze_button)

        self.status_label = QLabel("Choose an image to begin.")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("subtitle")
        result_layout.addWidget(self.status_label)

        content.addWidget(result_card, stretch=1)

        footer = QLabel(
            "Research demonstration. Confidence is a model score, "
            "not a medical diagnosis."
        )
        footer.setWordWrap(True)
        footer.setObjectName("footer")
        layout.addWidget(footer)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #0b1120;
            }
            QWidget {
                color: #e7edf8;
                font-family: "Segoe UI";
                font-size: 14px;
            }
            QLabel#brand {
                color: #6ee7c8;
                font-size: 14px;
                font-weight: 700;
            }
            QLabel#title {
                font-size: 32px;
                font-weight: 700;
            }
            QLabel#subtitle {
                color: #a2b1c8;
            }
            QFrame#card {
                background-color: #121d30;
                border: 1px solid #28364f;
                border-radius: 20px;
            }
            QLabel#preview {
                background-color: #080e19;
                color: #7d8da8;
                border-radius: 12px;
            }
            QLabel#sectionTitle {
                font-size: 18px;
                font-weight: 600;
            }
            QLabel#result {
                color: #85ecd1;
                font-size: 28px;
                font-weight: 700;
            }
            QPushButton {
                background-color: #6ee7c8;
                color: #08271f;
                border: none;
                border-radius: 10px;
                padding: 14px 20px;
                font-weight: 700;
            }
            QPushButton:hover {
                background-color: #a0f3dd;
            }
            QPushButton:disabled {
                background-color: #25334a;
                color: #8190a7;
            }
            QProgressBar {
                border: none;
                border-radius: 4px;
                background-color: #26334a;
            }
            QProgressBar::chunk {
                border-radius: 4px;
                background-color: #6ee7c8;
            }
            QLabel#footer {
                color: #8190a7;
                font-size: 12px;
            }
        """)

    def reset_probabilities(self):
        for class_name in CLASS_LABELS:
            self.probability_bars[class_name].setValue(0)
            self.probability_labels[class_name].setText("--")

    def select_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Choose MRI image",
            "",
            "Images (*.png *.jpg *.jpeg)",
        )

        if not filename:
            return

        pixmap = QPixmap(filename)

        if pixmap.isNull():
            self.status_label.setText("Unable to read the selected image.")
            return

        self.image_path = filename
        self.original_pixmap = pixmap

        self.file_label.setText(
            f"{Path(filename).name} | {pixmap.width()} x {pixmap.height()}"
        )
        self.result_label.setText("Ready to analyze")
        self.confidence_label.setText("Click Analyze image to continue.")
        self.status_label.setText("Image selected.")
        self.analyze_button.setEnabled(True)

        self.reset_probabilities()
        self.update_preview()

    def update_preview(self):
        if self.original_pixmap is not None:
            self.preview.setPixmap(
                self.original_pixmap.scaled(
                    self.preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def analyze_image(self):
        if not self.image_path:
            return

        if self.worker is not None and self.worker.isRunning():
            return

        self.upload_button.setEnabled(False)
        self.analyze_button.setEnabled(False)
        self.analyze_button.setText("Analyzing...")

        self.result_label.setText("Analyzing...")
        self.confidence_label.setText("")
        self.status_label.setText("Loading the model and processing the image...")
        self.reset_probabilities()

        self.worker = PredictionThread(
            self.image_path,
            predictor=self.predictor,
            parent=self,
        )
        self.worker.result_ready.connect(self.show_prediction)
        self.worker.error_occurred.connect(self.show_prediction_error)
        self.worker.finished.connect(self.prediction_finished)
        self.worker.start()

    def show_prediction(self, result):
        self.result_label.setText(CLASS_LABELS[result["class_name"]])
        self.confidence_label.setText(f"Model confidence: {result['confidence']:.2%}")

        for class_name, probability in zip(CLASS_LABELS, result["probabilities"]):
            self.probability_labels[class_name].setText(f"{probability:.2%}")
            self.probability_bars[class_name].setValue(round(probability * 10000))

        self.status_label.setText("Analysis completed.")

    def show_prediction_error(self, message):
        self.result_label.setText("Analysis failed")
        self.confidence_label.setText("")
        self.status_label.setText(message)

    def prediction_finished(self):
        completed_worker = self.worker
        self.predictor = completed_worker.predictor
        self.worker = None

        self.upload_button.setEnabled(True)
        self.analyze_button.setEnabled(self.image_path is not None)
        self.analyze_button.setText("Analyze image")

        completed_worker.deleteLater()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_preview()

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.status_label.setText(
                "Please wait for analysis to finish before closing."
            )
            event.ignore()
            return

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
