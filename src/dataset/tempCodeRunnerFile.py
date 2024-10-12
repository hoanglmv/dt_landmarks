    # Đọc ảnh từ file
                image = cv2.imread(self.image_filenames[index])

                # Kiểm tra nếu ảnh là None thì bỏ qua ảnh này và lấy ảnh tiếp theo
                if image is None:
                    index = (index + 1) % len(self.image_filenames)  # Lặp lại nếu vượt quá danh sách ảnh
                    continue

                # Chuyển đổi sang ảnh xám (grayscale)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Lấy landmarks và bounding box
                landmarks = self.landmarks[index]
                bbox = self.crops[index]

                # Tọa độ gốc của bounding box
                x1 = int(bbox['left'])
                y1 = int(bbox['top'])
                x2 = int(bbox['left']) + int(bbox['width'])
                y2 = int(bbox['top']) + int(bbox['height'])

                # Tính chiều rộng và chiều cao của bounding box
                width = x2 - x1
                height = y2 - y1

                # Tăng kích thước bounding box theo scale factor
                new_width = int(width * self.bbox_scale_factor)
                new_height = int(height * self.bbox_scale_factor)

                # Điều chỉnh lại tọa độ của bounding box
                x1 = max(0, x1 - (new_width - width) // 2)
                y1 = max(0, y1 - (new_height - height) // 2)
                x2 = min(image.shape[1], x1 + new_width)
                y2 = min(image.shape[0], y1 + new_height)

                # Cắt ảnh theo bounding box mới
                cropped_image = image[y1:y2, x1:x2]

                # Scale lại tọa độ landmarks dựa trên crop
                scaled_landmarks = []
                for landmark in landmarks:
                    x_scaled = landmark[0] - x1
                    y_scaled = landmark[1] - y1
                    scaled_landmarks.append([x_scaled, y_scaled])

                scaled_landmarks = np.array(scaled_landmarks).astype('float32')

                # Resize ảnh về kích thước target (224x224)
                resized_image = cv2.resize(cropped_image, self.target_size)

                # Tính scale factor
                scale_x = self.target_size[1] / cropped_image.shape[1]
                scale_y = self.target_size[0] / cropped_image.shape[0]

                # Scale lại landmarks theo ảnh đã resize
                final_landmarks = []
                for landmark in scaled_landmarks:
                    x_resized = landmark[0] * scale_x
                    y_resized = landmark[1] * scale_y
                    final_landmarks.append([x_resized, y_resized])
                final_landmarks = np.array(final_landmarks).astype('float32')

                # Chuyển đổi ảnh thành tensor
                if self.transform:
                    resized_image = self.transform(resized_image)
                else:
                    resized_image = torch.tensor(resized_image, dtype=torch.float32).unsqueeze(0) / 255.0  # Thêm chiều kênh cho ảnh xám và chuẩn hóa

                # Chuyển đổi landmarks thành tensor
                landmarks_tensor = torch.tensor(final_landmarks.flatten(), dtype=torch.float32)

                return resized_image, landmarks_tensor
        

            
        