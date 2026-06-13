import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import os
import time
from collections import deque


class EyeTracker:
    def __init__(self):
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
        except Exception as e:
            print(f"Error initializing MediaPipe: {e}")
            raise

        # Facial landmark indices
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380] 
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.LEFT_BROW = [70, 63, 105]
        self.RIGHT_BROW = [336, 296, 334]

        # Screen settings
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False

        # Position smoothing with larger buffer
        self.position_buffer_x = deque([0.5] * 10, maxlen=10)
        self.position_buffer_y = deque([0.5] * 10, maxlen=10)

        # Cursor control parameters - ADJUSTED
        self.center_offset_x = 0
        self.center_offset_y = 0
        self.movement_scale = 1.5  # Reduced from 2.5 for better control
        self.horizontal_scale_factor = 1.0  # Reduced from 1.2

        # Action timing
        self.last_action_time = time.time()

        # Eyebrow raise detection
        self.brow_raises = 0
        self.last_brow_time = 0
        self.brow_cooldown = 2.0 

        # Blink detection - ADJUSTED THRESHOLDS
        self.blink_start_time = 0
        self.is_blinking = False
        self.long_blink_duration = 1.5
        self.blink_threshold = 0.15  # More sensitive (was 0.19)
        self.wink_threshold = 0.12   # More sensitive (was 0.18)
        self.open_threshold = 0.25   # More lenient (was 0.23)

        # Mouth detection
        self.MOUTH_TOP = 13  
        self.MOUTH_BOTTOM = 14 
        self.mouth_open_start = 0
        self.is_mouth_open = False
        self.mouth_open_duration = 1.2 
        self.mouth_open_threshold = 0.08  # Reduced from 0.1

        # Initialization - BETTER CALIBRATION
        self.last_position = (self.screen_width / 2, self.screen_height / 2)
        self.initialized = False
        self.init_frames = 0
        self.init_required_frames = 30  # Increased from 10 for better calibration
        self.init_positions = []

        # Zoom settings
        self.zoom_cooldown = 1.0

        # Keyboard settings
        self.keyboard_open = False
        self.keyboard_cooldown = 10.0

    def initialize_center_position(self, x, y):
        """Calibrate center position by averaging multiple frames"""
        self.init_positions.append((x, y))
        self.init_frames += 1

        if self.init_frames >= self.init_required_frames:
            xs, ys = zip(*self.init_positions)
            self.center_offset_x = np.median(xs)
            self.center_offset_y = np.median(ys)
            self.initialized = True
            print(f"✓ Calibration Complete!")
            print(f"Center offset: ({self.center_offset_x:.3f}, {self.center_offset_y:.3f})")

    def get_relative_eye_position(self, iris_points, eye_points):
        """Calculate iris position relative to eye"""
        iris_center = np.mean(iris_points, axis=0)
        eye_contour = np.array(eye_points)

        eye_left = np.min(eye_contour[:, 0])
        eye_right = np.max(eye_contour[:, 0])
        eye_top = np.min(eye_contour[:, 1])
        eye_bottom = np.max(eye_contour[:, 1])

        eye_width = max(eye_right - eye_left, 1)
        eye_height = max(eye_bottom - eye_top, 1)

        # Clamp relative position to 0-1 range
        rel_x = np.clip((iris_center[0] - eye_left) / eye_width, 0, 1)
        rel_y = np.clip((iris_center[1] - eye_top) / eye_height, 0, 1)

        return (rel_x, rel_y, iris_center)

    def calculate_cursor_position(self, left_rel, right_rel):
        """Calculate smooth cursor position"""
        avg_x = (left_rel[0] + right_rel[0]) / 2
        avg_y = (left_rel[1] + right_rel[1]) / 2

        self.position_buffer_x.append(avg_x)
        self.position_buffer_y.append(avg_y)

        # Smooth with larger buffer
        smooth_x = np.mean(list(self.position_buffer_x))
        smooth_y = np.mean(list(self.position_buffer_y))

        if not self.initialized:
            self.initialize_center_position(smooth_x, smooth_y)
            return self.last_position

        # Calculate offset from center
        offset_x = (smooth_x - self.center_offset_x) * self.horizontal_scale_factor
        offset_y = (smooth_y - self.center_offset_y)

        # Map to screen coordinates
        cursor_x = self.screen_width / 2 + offset_x * self.movement_scale * self.screen_width
        cursor_y = self.screen_height / 2 + offset_y * self.movement_scale * self.screen_height

        # Clamp to screen boundaries
        cursor_x = np.clip(cursor_x, 0, self.screen_width)
        cursor_y = np.clip(cursor_y, 0, self.screen_height)

        # Extra smoothing to reduce jitter
        cursor_x = 0.8 * cursor_x + 0.2 * self.last_position[0]
        cursor_y = 0.8 * cursor_y + 0.2 * self.last_position[1]

        self.last_position = (cursor_x, cursor_y)
        return (cursor_x, cursor_y)

    def calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio"""
        try:
            A = np.linalg.norm(eye_points[1] - eye_points[5])
            B = np.linalg.norm(eye_points[2] - eye_points[4])
            C = np.linalg.norm(eye_points[0] - eye_points[3])
            ear = (A + B) / (2.0 * C)
            return ear
        except:
            return 0.5

    def detect_blinks_and_winks(self, left_eye, right_eye):
        """Detect blinks and winks with adjusted thresholds"""
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)

        # Use instance thresholds
        blink = left_ear < self.blink_threshold and right_ear < self.blink_threshold
        left_wink = left_ear < self.wink_threshold and right_ear > self.open_threshold
        right_wink = right_ear < self.wink_threshold and left_ear > self.open_threshold

        now = time.time()
        if blink and not self.is_blinking:
            self.is_blinking = True
            self.blink_start_time = now
        elif not blink and self.is_blinking:
            self.is_blinking = False

        long_blink = self.is_blinking and (now - self.blink_start_time) > self.long_blink_duration

        return blink, left_wink, right_wink, left_ear, right_ear, long_blink

    def detect_eyebrow_raise(self, mesh, img_h):
        """Detect eyebrow raises"""
        try:
            left_brow_y = np.mean([mesh[i][1] for i in self.LEFT_BROW])
            right_brow_y = np.mean([mesh[i][1] for i in self.RIGHT_BROW])
            left_eye_y = np.mean([mesh[i][1] for i in self.LEFT_EYE])
            right_eye_y = np.mean([mesh[i][1] for i in self.RIGHT_EYE])
            
            raise_threshold = img_h * 0.05  # Reduced from 0.06
            left_dist = left_eye_y - left_brow_y
            right_dist = right_eye_y - right_brow_y

            both_raised = (left_dist > raise_threshold) and (right_dist > raise_threshold)
            return both_raised, left_dist, right_dist
        except:
            return False, 0, 0

    def detect_mouth_open(self, mesh, img_h):
        """Detect mouth opening"""
        try:
            if len(mesh) <= max(self.MOUTH_TOP, self.MOUTH_BOTTOM):
                return False, 0

            top_lip = mesh[self.MOUTH_TOP][1]
            bottom_lip = mesh[self.MOUTH_BOTTOM][1]
            mouth_gap = bottom_lip - top_lip
            face_height = img_h * 0.7 

            mouth_open_ratio = mouth_gap / face_height
            is_open = mouth_open_ratio > self.mouth_open_threshold

            now = time.time()
            if is_open and not self.is_mouth_open:
                self.is_mouth_open = True
                self.mouth_open_start = now
            elif not is_open:
                self.is_mouth_open = False

            long_open = self.is_mouth_open and (now - self.mouth_open_start) > self.mouth_open_duration

            return long_open, mouth_open_ratio
        except:
            return False, 0

    def open_virtual_keyboard(self):
        """Open virtual keyboard"""
        now = time.time()
        if self.keyboard_open and (now - self.last_action_time < self.keyboard_cooldown):
            return False 

        try:
            if os.name == 'nt':  
                os.system('start osk')
            self.keyboard_open = True
            self.last_action_time = now
            return True
        except Exception as e:
            print(f"Error opening virtual keyboard: {e}")
            return False

    def zoom_in(self):
        """Zoom in"""
        pyautogui.hotkey('ctrl', 'plus')

    def zoom_out(self):
        """Zoom out"""
        pyautogui.hotkey('ctrl', 'minus')

    def draw_iris_tracking(self, frame, left_iris, right_iris, left_eye, right_eye,
                           left_center, right_center):
        """Draw eye tracking visualization"""
        cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 0), 1)
        cv2.polylines(frame, [np.array(left_iris)], True, (255, 0, 0), 1)
        cv2.polylines(frame, [np.array(right_iris)], True, (255, 0, 0), 1)
        cv2.circle(frame, (int(left_center[0]), int(left_center[1])), 2, (0, 0, 255), -1)
        cv2.circle(frame, (int(right_center[0]), int(right_center[1])), 2, (0, 0, 255), -1)
        return frame

    def process_frame(self, frame):
        """Process frame for eye tracking"""
        img_h, img_w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame

        mesh_points = np.array([
            [int(p.x * img_w), int(p.y * img_h)] for p in results.multi_face_landmarks[0].landmark
        ])

        left_iris = [mesh_points[i] for i in self.LEFT_IRIS]
        right_iris = [mesh_points[i] for i in self.RIGHT_IRIS]
        left_eye = [mesh_points[i] for i in self.LEFT_EYE]
        right_eye = [mesh_points[i] for i in self.RIGHT_EYE]

        rel_left, rel_left_y, left_center = self.get_relative_eye_position(left_iris, left_eye)
        rel_right, rel_right_y, right_center = self.get_relative_eye_position(right_iris, right_eye)

        frame = self.draw_iris_tracking(frame, left_iris, right_iris, left_eye, right_eye,
                                        left_center, right_center)

        cursor_x, cursor_y = self.calculate_cursor_position((rel_left, rel_left_y),
                                                            (rel_right, rel_right_y))

        if self.initialized:
            try:
                pyautogui.moveTo(cursor_x, cursor_y)
            except:
                pass  # Ignore pyautogui errors

            blink, left_wink, right_wink, left_ear, right_ear, long_blink = self.detect_blinks_and_winks(left_eye, right_eye)
            mouth_open, mouth_ratio = self.detect_mouth_open(mesh_points, img_h)
            now = time.time()

            # Display metrics
            cv2.putText(frame, f"L-EAR: {left_ear:.2f}", (10, img_h - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"R-EAR: {right_ear:.2f}", (10, img_h - 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Mouth: {mouth_ratio:.3f}", (10, img_h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Blink Threshold: {self.blink_threshold}", (10, img_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Blink for click
            if blink and not self.is_blinking and now - self.last_action_time > 0.5:
                pyautogui.click()
                self.last_action_time = now
                cv2.putText(frame, "CLICK!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # Winks for zoom
            elif left_wink and now - self.last_action_time > self.zoom_cooldown:
                self.zoom_out() 
                self.last_action_time = now
                cv2.putText(frame, "Zoom Out", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            elif right_wink and now - self.last_action_time > self.zoom_cooldown:
                self.zoom_in()
                self.last_action_time = now
                cv2.putText(frame, "Zoom In", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Eyebrow detection
            eyebrows_raised, left_dist, right_dist = self.detect_eyebrow_raise(mesh_points, img_h)

            if eyebrows_raised:
                if now - self.last_brow_time > 0.5 and now - self.last_brow_time < self.brow_cooldown:
                    self.brow_raises += 1
                    self.last_brow_time = now
                elif now - self.last_brow_time > self.brow_cooldown:
                    self.brow_raises = 1
                    self.last_brow_time = now

                cv2.putText(frame, f"Eyebrows: {self.brow_raises}", (50, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                if self.brow_raises == 2 and now - self.last_action_time > 0.8:
                    pyautogui.click(button='right')  
                    cv2.putText(frame, "RIGHT CLICK!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                    self.last_action_time = now

                elif self.brow_raises >= 3 and now - self.last_action_time > 0.8:
                    self.open_virtual_keyboard()
                    cv2.putText(frame, "KEYBOARD!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                    self.last_action_time = now
                    self.brow_raises = 0

        else:
            cv2.putText(frame, f"Initializing: {self.init_frames}/{self.init_required_frames}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, "LOOK AT CENTER OF SCREEN", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame


def main():
    print("=" * 50)
    print("Eye Tracker - Eye Controlled Mouse")
    print("=" * 50)
    print("Starting...")
    
    try:
        tracker = EyeTracker()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n✓ Camera initialized")
    print("Look at the CENTER OF THE SCREEN for 3 seconds to calibrate")
    print("\nControls:")
    print("- Blink: Click")
    print("- Left Wink: Zoom Out")
    print("- Right Wink: Zoom In")
    print("- 2x Eyebrow: Right Click")
    print("- 3x Eyebrow: Open Keyboard")
    print("- Press 'q': Quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        processed = tracker.process_frame(frame)
        
        cv2.imshow("Eye Controlled Mouse", processed)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nEye Tracker stopped.")


if __name__ == "__main__":
    main()