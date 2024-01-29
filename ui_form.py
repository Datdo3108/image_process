from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QLineEdit, QComboBox


class UIForm(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        # Create widgets
        # Widget for images
        self.image_label = QLabel('Enter Text:', self)
        self.image_label.setGeometry(10, 10, 300, 300)                  # IMAGE
        self.image_hist_label = QLabel('a', self)
        self.image_hist_label.setGeometry(350, 10, 300, 300)            #HISTOGRAM
        self.image_average_label = QLabel('b', self)
        self.image_average_label.setGeometry(10, 450, 300, 300)         # AVERAGE
        self.image_median_label = QLabel('c', self)
        self.image_median_label.setGeometry(350, 450, 300, 300)         # MEDIAN
        self.choose_button = QPushButton("Choose Image", self)
        self.choose_button.setGeometry(10, 410, 300, 40)

        self.image_kapur_label = QLabel('d', self)
        self.image_kapur_label.setGeometry(700, 10, 300, 300)
        self.image_otsu_label = QLabel('e', self)
        self.image_otsu_label.setGeometry(700, 450, 300, 300)
        self.image_kapur_pre_label = QLabel('d', self)
        self.image_kapur_pre_label.setGeometry(1050, 10, 300, 300)
        self.image_otsu_pre_label = QLabel('e', self)
        self.image_otsu_pre_label.setGeometry(1050, 450, 300, 300)

        # Histogram
        self.histogram_button = QPushButton("Show Histogram", self)
        self.histogram_button.setGeometry(350, 410, 300, 40)
        self.histogram_eq_button = QPushButton("Histogram Equalize", self)
        self.histogram_eq_button.setGeometry(350, 350, 150, 40)
        self.histogram_psnr_label = QLabel("PSNR:", self)
        self.histogram_psnr_label.setGeometry(500, 350, 150, 40)

        # Average filter
        self.average_button = QPushButton("Show Average Filter", self)
        self.average_button.setGeometry(10, 750, 150, 20)
        self.average_psnr_label = QLabel("PSNR:", self)
        self.average_psnr_label.setGeometry(160, 750, 150, 20)
        self.mask_size_label = QLabel("Enter kernel size:", self)
        self.mask_size_label.setGeometry(10, 800, 150, 20)
        self.mask_size_line_edit = QLineEdit(self)
        self.mask_size_line_edit.setGeometry(200, 800, 150, 20)
        self.mask_shape_label = QLabel("Choose mask shape:", self)
        self.mask_shape_label.setGeometry(10, 850, 150, 20)
        self.mask_shape_combo_box = QComboBox(self)
        self.mask_shape_combo_box.setGeometry(200, 850, 150, 20)
        # self.mask_shape_combo_box.addItem("Average")
        # self.mask_shape_combo_box.addItem("Gaussian")
        # self.mask_shape_combo_box.addItem("Sobel (vertical)")
        # self.mask_shape_combo_box.addItem("Sobel (horizontal)")
        self.mask_shape_combo_box.addItem("Square")
        self.mask_shape_combo_box.addItem("Circle")

        # Median filter
        self.median_button = QPushButton("Show Median Filter", self)
        self.median_button.setGeometry(350, 750, 150, 20)
        self.median_psnr_label = QLabel("PSNR:", self)
        self.median_psnr_label.setGeometry(500, 750, 150, 20)
        self.median_mask_size_label = QLabel("Enter kernel size:", self)
        self.median_mask_size_label.setGeometry(350, 800, 150, 20)
        self.median_mask_size_line_edit = QLineEdit(self)
        self.median_mask_size_line_edit.setGeometry(500, 800, 150, 20)

        # Kapur'method
        self.kapur_button = QPushButton("Show Kapur's segmentation", self)
        self.kapur_button.setGeometry(700, 410, 200, 40)

        # Otsu's method
        self.otsu_button = QPushButton("Show Otsu's segmentation", self)
        self.otsu_button.setGeometry(700, 750, 200, 40)

        # Kapur's preprocess method
        self.kapur_pre_button = QPushButton("Show Evaluation", self)
        self.kapur_pre_button.setGeometry(1050, 410, 200, 40)
        self.kapur_pre_label = QLabel("Choose preprocess: ", self)
        self.kapur_pre_label.setGeometry(1050, 350, 100, 40)
        self.kapur_pre_combo_box = QComboBox(self)
        self.kapur_pre_combo_box.setGeometry(1150, 350, 100, 40)
        self.kapur_pre_combo_box.addItem("Histogram Equalize")
        self.kapur_pre_combo_box.addItem("Average filter")
        self.kapur_pre_combo_box.addItem("Median filter")

        self.kapur_evaluate_label = QLabel(self)
        self.kapur_evaluate_label.setGeometry(1300, 300, 200, 80)

        # Otsu's preprocess method
        # self.otsu_pre_button = QPushButton("Show Otsu's segmentation", self)
        # self.otsu_pre_button.setGeometry(1050, 750, 200, 40)
        # self.otsu_pre_label = QLabel("Choose preprocess: ", self)
        # self.otsu_pre_label.setGeometry(1050, 800, 100, 40)

        self.otsu_evaluate_label = QLabel(self)
        self.otsu_evaluate_label.setGeometry(1300, 700, 200, 80)
        

