import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, QPushButton, QLabel, QFileDialog, QGroupBox, QTextEdit
from PyQt5.QtGui import QPixmap, QIcon, QColor, QFont
from PyQt5.QtCore import Qt

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.relu4 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.relu3(x)
        x = self.layer4(x)
        x = self.relu4(x)
        x = self.output_layer(x)
        return x

class SplitViewGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.tmb_image_path = None
        self.abts_image_path = None
        self.tmb_rgb = None
        self.abts_rgb = None
        self.initUI()
        self.model = self.load_model()

    def initUI(self):
        self.setWindowTitle('Identifying Aging Years of Artemisiae argyi folium')
        self.setGeometry(300, 300, 720, 600)
        self.setWindowIcon(QIcon('Aiye3.png'))

        default_font = QFont("Arial", 8)
        default_color = QColor(128, 128, 128)
        self.author_label = QLabel("Made by Shuzhi Liu", self)
        self.author_label.setGeometry(550, 5, 130, 40)  # 设置绝对坐标位置和大小
        self.author_label.setAlignment(Qt.AlignRight)
        self.author_label.setFont(default_font)
        self.author_label.setStyleSheet("QLabel { color: %s }" % default_color.name())

        # default_font = QFont("Arial", 10)
        # default_font.setBold(True)
        # default_color = QColor(0,0,0)
        # self.function_label = QLabel("Function: Identifying Aging Years of Artemisiae argyi folium", self)
        # self.function_label.setGeometry(0, 5, 660, 40)
        # self.function_label.setAlignment(Qt.AlignLeft)
        # self.function_label.setFont(default_font)
        # self.function_label.setStyleSheet("QLabel { color: %s }" % default_color.name())

        self.logo_label = QLabel(self)
        self.logo_label.setGeometry(0, 25, 200, 100)  # Adjust the position and size as needed
        self.logo_label.setPixmap(
            QPixmap('aiye1.png').scaled(200, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.logo_label.setAlignment(Qt.AlignCenter)

        self.logo_label2 = QLabel(self)
        self.logo_label2.setGeometry(525, 25, 200, 100)  # Adjust the position and size as needed
        self.logo_label2.setPixmap(
            QPixmap('aiye2.png').scaled(200, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.logo_label2.setAlignment(Qt.AlignCenter)

        main_layout = QVBoxLayout()
        channel_box = QGroupBox("Channel Selection")
        channel_box.setFixedSize(300, 70)
        channel_layout = QHBoxLayout()
        channel_layout.addStretch(1)
        self.single_channel_rb = QRadioButton("Single Channel")
        self.dual_channel_rb = QRadioButton("Dual Channel")
        self.single_channel_rb.setChecked(True)
        channel_layout.addWidget(self.single_channel_rb)
        channel_layout.addWidget(self.dual_channel_rb)
        channel_layout.addStretch(1)
        channel_box.setLayout(channel_layout)
        main_layout.addWidget(channel_box, alignment=Qt.AlignCenter)

        image_layout = QHBoxLayout()
        left_group = QGroupBox("TMB Image")
        left_group.setFixedSize(300, 300)
        left_layout = QVBoxLayout()
        self.tmb_label = QLabel()
        self.tmb_label.setFixedSize(280, 220)
        self.tmb_label.setAlignment(Qt.AlignCenter)
        self.upload_tmb_btn = QPushButton("Upload TMB Image")
        self.upload_tmb_btn.clicked.connect(lambda: self.upload_image("TMB"))
        self.extract_tmb_rgb_btn = QPushButton("Extract TMB RGB")
        self.extract_tmb_rgb_btn.clicked.connect(lambda: self.extract_rgb("TMB"))
        left_layout.addWidget(self.tmb_label)
        left_layout.addWidget(self.upload_tmb_btn)
        left_layout.addWidget(self.extract_tmb_rgb_btn)
        left_group.setLayout(left_layout)
        image_layout.addWidget(left_group)

        right_group = QGroupBox("Abts Image")
        right_group.setFixedSize(300, 300)
        right_layout = QVBoxLayout()
        self.abts_label = QLabel()
        self.abts_label.setFixedSize(280, 220)
        self.abts_label.setAlignment(Qt.AlignCenter)
        self.upload_abts_btn = QPushButton("Upload Abts Image")
        self.upload_abts_btn.clicked.connect(lambda: self.upload_image("Abts"))
        self.extract_abts_rgb_btn = QPushButton("Extract Abts RGB")
        self.extract_abts_rgb_btn.clicked.connect(lambda: self.extract_rgb("Abts"))
        right_layout.addWidget(self.abts_label)
        right_layout.addWidget(self.upload_abts_btn)
        right_layout.addWidget(self.extract_abts_rgb_btn)
        right_group.setLayout(right_layout)
        image_layout.addWidget(right_group)

        main_layout.addLayout(image_layout)

        button_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Program")
        self.run_button.clicked.connect(self.run_program)
        button_layout.addWidget(self.run_button)

        self.clear_button = QPushButton("Clear Images")
        self.clear_button.clicked.connect(self.clear_images)
        button_layout.addWidget(self.clear_button)

        main_layout.addLayout(button_layout)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFixedHeight(100)
        main_layout.addWidget(self.output_text)


        self.setLayout(main_layout)

    def clear_images(self):
        self.tmb_image_path = None
        self.abts_image_path = None
        self.tmb_rgb = None
        self.abts_rgb = None
        self.tmb_label.clear()
        self.abts_label.clear()
        self.output_text.clear()

    def upload_image(self, image_type):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, f"Select {image_type} Image", "",
                                                  "Image Files (*.png *.jpg *.jpeg *.tif)", options=options)
        if filename:
            pixmap = QPixmap(filename)
            if pixmap.isNull():
                print("Failed to load image.")
            else:
                if image_type == "TMB":
                    self.tmb_label.setPixmap(
                        pixmap.scaled(self.tmb_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    self.tmb_image_path = filename
                elif image_type == "Abts":
                    self.abts_label.setPixmap(
                        pixmap.scaled(self.abts_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    self.abts_image_path = filename

    def extract_rgb(self, image_type):
        if image_type == "TMB":
            if self.tmb_image_path:
                self.tmb_rgb = self.get_rgb(self.tmb_image_path)
                self.output_text.append(f"TMB RGB: {self.tmb_rgb}")
            else:
                self.output_text.append("Please upload a TMB image first.")
        elif image_type == "Abts":
            if self.abts_image_path:
                self.abts_rgb = self.get_rgb(self.abts_image_path)
                self.output_text.append(f"Abts RGB: {self.abts_rgb}")
            else:
                self.output_text.append("Please upload an Abts image first.")

    def run_program(self):
        if self.single_channel_rb.isChecked():
            if self.tmb_rgb and self.abts_rgb:
                self.output_text.append("Both TMB and Abts channels have values. Please select Dual Channel mode.")
            elif self.tmb_rgb:
                result = self.predict(self.tmb_rgb, "TMB")
                # self.output_text.append(f"TMB Prediction: {result}")
            elif self.abts_rgb:
                result = self.predict(self.abts_rgb, "Abts")
                # self.output_text.append(f"Abts Prediction: {result}")
            else:
                self.output_text.append("Please extract RGB values for at least one channel.")
        elif self.dual_channel_rb.isChecked():
            if self.tmb_rgb and self.abts_rgb:
                result = self.predict(np.concatenate((self.tmb_rgb, self.abts_rgb)), "Dual")
                # self.output_text.append(f"Dual Channel Prediction: {result}")
            else:
                self.output_text.append("Please extract RGB values for both channels.")
        else:
            self.output_text.append("Please select a channel and extract RGB values.")

    def get_rgb(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=20, param2=20, minRadius=0, maxRadius=20)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            x, y, _ = circles[0][0]
            rgb = img[y, x, :]
            return rgb.tolist()
        return None

    def load_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLP(input_size=4, hidden_size=4096, num_classes=5).to(device)
        model.load_state_dict(torch.load('best_model2.pth', map_location=device))
        model.eval()
        return model

    def predict(self, rgb, channel):
        def format_output(self, tmb_output, abts_output):
            output_mapping = {0: 1, 1: 3, 2: 5, 3: 10, 4: 15}
            print(f"tmb_out:{tmb_output}")
            print(f"abts_output:{abts_output}")
            tmb_year = output_mapping.get(tmb_output, None)
            abts_year = output_mapping.get(abts_output, None)

            if tmb_output is not None and abts_output is None:
                if tmb_year is None:
                    return "Forecast results out of range"
                else:
                    return f"The aging years of Artemisiae argyi folium may be {tmb_year} years"

            if abts_output is not None and tmb_output is None:
                if abts_year is None:
                    return "Forecast results out of range"
                else:
                    return f"The aging years of Artemisiae argyi folium may be {abts_year} years"

            if tmb_year is None or abts_year is None:
                return "Forecast results out of range"

            if tmb_year == abts_year:
                return f"The aging years of Artemisiae argyi folium are {tmb_year} years"
            elif abs(tmb_output - abts_output) <= 1:
                if tmb_year > abts_year:
                    return f"The aging years of Artemisiae argyi folium are from {abts_year} to {tmb_year} years"
                else :
                    return f"The aging years of Artemisiae argyi folium are from {tmb_year} to {abts_year} years"
            else:

                return "There is a large deviation in the prediction, please try again."

        if channel == "Dual":
            if self.tmb_rgb is None or self.abts_rgb is None:
                self.output_text.append("Please extract the RGB values of the TMB and Abts channels first")
                return


            tmb_result = self.run_prediction(self.tmb_rgb, "TMB")
            abts_result = self.run_prediction(self.abts_rgb, "Abts")
            formatted_output = format_output(self, tmb_result, abts_result)
            self.output_text.clear()
            self.output_text.append(formatted_output)
        else:
            rgb = np.array(rgb)

            # 根据选择的通道设置通道值
            if channel == "TMB":
                channel_value = 1
                predicted_class = self.run_prediction(rgb, "TMB")
                formatted_output = format_output(self, predicted_class, None)
            elif channel == "Abts":
                channel_value = 2
                predicted_class = self.run_prediction(rgb, "Abts")
                formatted_output = format_output(self, None, predicted_class)
            else:
                raise ValueError("Invalid channel value")

            self.output_text.clear()
            self.output_text.append(formatted_output)

            X = np.concatenate((rgb.reshape(1, -1), np.array([[channel_value]])), axis=1)

            X = torch.tensor(X, dtype=torch.float32, device=next(self.model.parameters()).device)

            # 进行预测
            self.model.eval()
            with torch.no_grad():
                output = self.model(X)
                predicted_class = torch.max(output, 1)[1].cpu().item()

            return predicted_class

    def run_prediction(self, rgb, channel):
        rgb = np.array(rgb)

        if channel == "TMB":
            channel_value = 1
        elif channel == "Abts":
            channel_value = 2
        else:
            raise ValueError("Invalid channel value")

        X = np.concatenate((rgb.reshape(1, -1), np.array([[channel_value]])), axis=1)

        X = torch.tensor(X, dtype=torch.float32, device=next(self.model.parameters()).device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
            predicted_class = torch.max(output, 1)[1].cpu().item()

        return predicted_class

def main():
    app = QApplication(sys.argv)
    ex = SplitViewGUI()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()