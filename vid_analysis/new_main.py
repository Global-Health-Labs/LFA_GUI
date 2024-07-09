#!/usr/bin/env python3

import cv2
import numpy as np
import os
from scipy.signal import savgol_filter, find_peaks
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

class VideoAnalyzer:
    def __init__(self, video_file):
        self.video_file = video_file
        self.cap = cv2.VideoCapture(video_file)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        if self.fps <= 0:
            raise ValueError("Failed to read the video file or the video has zero fps.")
        
        self.new_dir = os.path.splitext(video_file)[0]
        os.makedirs(self.new_dir, exist_ok=True)
        
        self._extract_median_frame()
        
    def _extract_median_frame(self):
        num_frames_in_30_seconds = int(self.fps * 30)
        frame_ids = np.random.choice(num_frames_in_30_seconds, size=25, replace=False)
        frames = [self._read_frame(fid) for fid in frame_ids if self._read_frame(fid) is not None]
        self.median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)

    def _read_frame(self, fid):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = self.cap.read()
        return frame if ret else None

    def generate_difference_video(self):
        output_video_path = os.path.join(self.new_dir, 'output.avi')
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0,
                              (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), True)
        all_frames = []
        last_frame = None
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            dframe = cv2.absdiff(frame, self.median_frame)
            out.write(dframe)
            all_frames.append(dframe)
            self.last_frame = frame
        # self.last_frame = frame
        self.all_frames = all_frames
        self.cap.release()
        out.release()
        self._save_last_frame()
    
    def _save_last_frame(self):
        if self.last_frame is not None:
            last_frame_pic = os.path.join(self.new_dir, 'last_frame.jpg')
            cv2.imwrite(last_frame_pic, self.last_frame)

    def analyze_frames(self):
        x_start, y_start, x_end, y_end, num_rectangles, spacing = 128, 309, 178, 410, 8, 234
        data_points = []
        
        for frame_idx, frame in enumerate(self.all_frames):
            time_in_seconds = frame_idx / self.fps
            for i in range(num_rectangles):
                x_offset = i * spacing
                roi = frame[y_start:y_end, x_start + x_offset:x_end + x_offset]
                # Calculate the mean pixel intensity along the vertical axis for BGR channels
                line_profile_B = np.mean(roi[:, :, 0], axis=1)
                line_profile_G = np.mean(roi[:, :, 1], axis=1)
                line_profile_R = np.mean(roi[:, :, 2], axis=1)
                
                # Find peaks in the line profile
                filtered_B, peaks_X_B, peaks_Y_B = self._find_lfa_peaks(line_profile_B, top=y_start)
                filtered_G, peaks_X_G, peaks_Y_G = self._find_lfa_peaks(line_profile_G, top=y_start)
                filtered_R, peaks_X_R, peaks_Y_R = self._find_lfa_peaks(line_profile_R, top=y_start)

                # Collect all data points
                for y in range(len(filtered_B)):
                    data_points.append({
                        'Time (s)': time_in_seconds,
                        'Rectangle': i + 1,
                        'Pixel': 100 - y,  # Invert pixel values
                        'Blue': filtered_B[y],
                        'Green': filtered_G[y],
                        'Red': filtered_R[y]
                    })
        
        df = pd.DataFrame(data_points)
        csv_file_path = os.path.join(self.new_dir, 'data_points.csv')
        df.to_csv(csv_file_path, index=False)

    def _find_lfa_peaks(self, line_profile, top=0):
        filtered = savgol_filter(line_profile, 3, 2)
        lowest_length = np.clip(len(filtered) // 2, 1, 50) - 1
        lowest = np.sort(filtered)[:lowest_length]
        background = np.mean(lowest)
        peaks_x, _ = find_peaks(filtered)
        peaks_y = filtered[peaks_x]

        line_vals = [top + 25, top + 50, top + 80]
        interval_vals = [20, 20, 20]

        x_intervals = [[int(line - interval - top), int(line + interval - top)] for line, interval in zip(line_vals, interval_vals)]
        peaks_xy_max = [max([[x, y] for x, y in zip(peaks_x, peaks_y) if a <= x <= b], key=lambda xy: xy[1], default=[None, None]) for a, b in x_intervals]
        peaks_xy_max.append([None, background])

        peaks_x_by_location, peaks_y_by_location = zip(*peaks_xy_max)
        return filtered, list(peaks_x_by_location), list(peaks_y_by_location)

class ImageAnalyzer:
    def __init__(self, img_path, n_lines=3, color_channels=['green']):
        self.img_path = img_path
        self.img = Image.open(img_path)
        self.n_lines = n_lines
        self.color_channels = color_channels
        self.df = pd.DataFrame()

        # Extract directory path from image path
        self.output_dir = os.path.dirname(img_path)

    def analyze(self, rois, line_positions, intervals, sample_label):
        num_rois = len(rois)
        fig, axes = plt.subplots(nrows=2, ncols=num_rois, figsize=(4 * num_rois, 8))

        for count, (start, end) in enumerate(rois):
            self._process_roi(start, end, line_positions, intervals, sample_label, count, axes[0, count], axes[1, count])

        for count in range(num_rois):
            axes[0, count].set_title(f'Sample {count + 1}')
            axes[1, count].set_ylabel('Distance (pixels)')

        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(self.output_dir, 'output.png'))
        plt.close(fig)

    def _process_roi(self, start, end, line_positions, intervals, sample_label, count, ax_img, ax_plot):
        left, top = start
        right, bottom = end

        roi = self.img.crop((left, top, right, bottom))
        roi_gray = roi.convert('L')
        nleft, nright = self._calculate_lr_border(roi_gray)
        roi_tight_gray = roi_gray.crop((nleft, 0, nright, roi.size[1]))
        roi_tight_color = roi.crop((nleft, 0, nright, roi.size[1]))

        roi_green = roi_tight_color.split()[1]
        channel_line = 255 - np.mean(np.asarray(roi_green), axis=1)
        line_peak = self._find_lfa_peaks(channel_line, top)

        self._save_plots(roi_tight_color, line_peak, ax_img, ax_plot)
        self._save_data(line_peak, sample_label, count + 1)

    def _calculate_lr_border(self, image):
        arr = np.asarray(image)
        mean_vertical = np.mean(arr, axis=0)
        gradient = np.gradient(mean_vertical)
        halfpoint = gradient.size // 2
        left = np.argmin(gradient[:halfpoint])
        right = np.argmin(gradient[halfpoint:]) + halfpoint
        return left, right

    def _find_lfa_peaks(self, line_profile, top):
        filtered = savgol_filter(line_profile, 13, 2)
        lowest_length = np.clip(len(filtered) // 2, 1, 50) - 1
        lowest = np.sort(filtered)[:lowest_length]
        background = np.mean(lowest)

        peaks_x, _ = find_peaks(filtered, prominence=5, distance=20)
        peaks_y = filtered[peaks_x]

        return filtered, list(peaks_x), list(peaks_y)

    def _save_plots(self, roi_tight_color, line_peak, ax_img, ax_plot):
        ax_img.imshow(roi_tight_color, aspect='auto')
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        ax_plot.plot(line_peak[0], range(len(line_peak[0])), 'green')
        ax_plot.plot(line_peak[2], line_peak[1], 'o', color='green')
        ax_plot.set(xlabel='green (signal)', xlim=[-25, 255], xticks=[0, 100, 200])
        ax_plot.grid(True, which='major', color='lightgray')
        ax_plot.invert_yaxis()

    def _save_data(self, line_peak, sample_label, roi_number):
        peaks_indices = line_peak[1]
        peaks_signals = line_peak[2]

        # Create a dictionary to hold the data for this ROI
        data = {
            'selection': roi_number,
            'green FC peak index': peaks_indices[0] if len(peaks_indices) > 0 else None,
            'green IPC peak index': peaks_indices[1] if len(peaks_indices) > 1 else None,
            'green TB peak index': peaks_indices[2] if len(peaks_indices) > 2 else None,
            'green FC peak signal': peaks_signals[0] if len(peaks_signals) > 0 else None,
            'green IPC peak signal': peaks_signals[1] if len(peaks_signals) > 1 else None,
            'green TB peak signal': peaks_signals[2] if len(peaks_signals) > 2 else None,
            'green background signal': np.mean(peaks_signals) if len(peaks_signals) > 0 else None
        }

        # Convert the dictionary to a DataFrame
        new_df = pd.DataFrame(data, index=[0])

        # Concatenate the new data to the existing DataFrame
        self.df = pd.concat([self.df, new_df], ignore_index=True)

    def save_dataframe(self, output_path):
        # Transpose the DataFrame to get the desired format
        self.df = self.df.set_index('selection').transpose()
        self.df.columns = range(1, len(self.df.columns) + 1)
        self.df.to_csv(output_path, index=True)

# Example usage
if __name__ == "__main__":
    try:
        video_file = '20240531_114934.240.mp4'
        video_analyzer = VideoAnalyzer(video_file)
        video_analyzer.generate_difference_video()
        video_analyzer.analyze_frames()

        img_path = os.path.join(video_analyzer.new_dir, 'last_frame.jpg')
        if os.path.exists(img_path):
            x_start = 128
            y_start = 309
            x_end = 178
            y_end = 410
            num_rectangles = 8
            spacing = 234
            
            rois = []
            for i in range(num_rectangles):
                start = (x_start + i * spacing, y_start)
                end = (x_end + i * spacing, y_end)
                rois.append((start, end))
                
            line_positions = [334, 359, 389]  # Example line positions
            intervals = [20, 20, 20]  # Example intervals
            sample_label = "sample"

            image_analyzer = ImageAnalyzer(img_path)
            image_analyzer.analyze(rois, line_positions, intervals, sample_label)
            image_analyzer.save_dataframe(os.path.join(video_analyzer.new_dir, 'image_data.csv'))
        else:
            print(f"Error: {img_path} does not exist.")
    except Exception as e:
        print(f"Error: {e}")
