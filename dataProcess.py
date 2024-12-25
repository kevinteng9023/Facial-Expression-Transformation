from mtcnn import MTCNN
import cv2
import os
import numpy as np
import zipfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
import itertools
import torch.autograd as autograd


def processor(input_dir='./input', output_face_dir='./output_face'):


    # Define input and output directories
    # input_dir = './input'
    # output_face_dir = './output_face'
    # output_dir = './output_happy'

    # Initialize MTCNN face detector
    detector = MTCNN()

    # Create output directory if it doesn't exist
    if not os.path.exists(output_face_dir):
        os.makedirs(output_face_dir)


    # Process each image in the emotion folder
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Error: Cannot load the image {filename}")
                continue

            # Convert image from BGR to RGB (MTCNN requires RGB format)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect faces in the image
            results = detector.detect_faces(img_rgb)

            if len(results) == 0:
                print(f"No faces detected in {filename}")
                continue

            # Process each detected face
            for i, result in enumerate(results):
                x, y, width, height = result['box']
                x, y = max(0, x), max(0, y)
                cropped_face = img_rgb[y:y + height, x:x + width]

                keypoints = result['keypoints']
                left_eye = np.array(keypoints['left_eye'])
                right_eye = np.array(keypoints['right_eye'])

                # Calculate angle for rotation
                delta_x = right_eye[0] - left_eye[0]
                delta_y = right_eye[1] - left_eye[1]
                angle = np.arctan2(delta_y, delta_x) * 180.0 / np.pi

                # Get the center point between the eyes
                eye_center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))

                # Get the rotation matrix and rotate the image
                rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
                rotated_face = cv2.warpAffine(img_rgb, rotation_matrix, (img_rgb.shape[1], img_rgb.shape[0]))

                # Crop the rotated face
                rotated_cropped_face = rotated_face[y:y + height, x:x + width]

                # Resize the cropped face to 256*256
                resized_face = cv2.resize(rotated_cropped_face, (256, 256))

                # Save the processed face
                # output_path = os.path.join(output_face_dir, f"{filename}.jpg")
                output_path = os.path.join(output_face_dir, f"{os.path.splitext(filename)[0]}_face_{i+1}.jpg")
                cv2.imwrite(output_path, cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR))
                print(f"Saved cropped and aligned face to {output_path}")
                # break

        




# Step 5: Test and Visualize Results with Test Set
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image_path

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def getTestLoader(input_test_dir='./output_face'):

    test_dataset = TestDataset(test_dir=input_test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return test_loader



# def delete_all_files(folder_path='./input'):
#     # 確認資料夾存在
#     if not os.path.isdir(folder_path):
#         print(f"資料夾不存在: {folder_path}")
#         return
    
#     # 遍歷資料夾中的所有項目
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         try:
#             # 檢查是否為檔案或符號連結
#             if os.path.isfile(file_path) or os.path.islink(file_path):
#                 os.remove(file_path)
#                 print(f"已刪除檔案: {file_path}")
#             # 如果是資料夾，跳過或根據需求處理
#             elif os.path.isdir(file_path):
#                 print(f"跳過資料夾: {file_path}")
#         except Exception as e:
#             print(f"無法刪除 {file_path}. 原因: {e}")

# # 使用範例
# folder = 'path/to/your/folder'
# delete_all_files(folder)