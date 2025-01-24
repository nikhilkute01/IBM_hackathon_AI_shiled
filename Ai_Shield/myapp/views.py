import os
import cv2
import re
import mediapipe as mp
import numpy as np
import pandas as pd
import requests
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings


def Index(request):
    return render(request,'index.html')

def train_yourself(request):
    return render(request, 'Trainyourself.html')

def train_yourself_video(request):
    return render(request, 'Trainyourselfvid.html')

def elbowstrike(request):
    return render(request, 'Elbowstrike.html')

def safemap(request):
    return render(request, 'Safemap.html')

def defendchat(request):
    return render(request, 'Defendchat.html')

def process_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        try:
            video_file = request.FILES['video']
            
            fs = FileSystemStorage()
            
            if not video_file.name:
                return render(request, 'Elbowstrike.html', {'error': 'Invalid video filename'})
                
            filename = fs.get_available_name(video_file.name)
            if not filename:
                return render(request, 'Elbowstrike.html', {'error': 'Could not generate filename'})
                
            video_path = os.path.join(settings.MEDIA_ROOT, filename)
            if not video_path:
                return render(request, 'Elbowstrike.html', {'error': 'Invalid video path'})
                
            fs.save(video_path, video_file)

            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return render(request, 'Elbowstrike.html', {'error': 'Could not open video file'})
                
            frame_count = 0
            total_accuracy = 0
            Elbow_Frames = 0
            total_shoulder_angle = 0
            total_elbow_accuracy = 0
            Kick_Frame = 0
            total_leg_angle = 0
            total_kick_accuracy = 0
            
            highest_elbow_accuracy = 0
            highest_kick_accuracy = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                results = pose.process(rgb_frame)

                if results and results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
                    right_elbow = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])
                    right_wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])
                    right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])

                    elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)

                    right_knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])
                    right_ankle = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])

                    leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
                    
                    if elbow_angle is not None and shoulder_angle is not None:
                        Elbow_Frames += 1
                        if elbow_angle < 70:
                            elbow_accuracy = 100 - ((70 - elbow_angle) * 2)
                        elif elbow_angle > 110:
                            elbow_accuracy = 100 - ((elbow_angle - 110) * 2)
                        else:
                            elbow_accuracy = 100 - abs(90 - elbow_angle)
                        
                        elbow_accuracy = max(0, min(100, elbow_accuracy))
                        total_elbow_accuracy += elbow_accuracy
                        total_shoulder_angle += shoulder_angle
                        highest_elbow_accuracy = max(highest_elbow_accuracy, elbow_accuracy)
                    
                    if leg_angle is not None:
                        Kick_Frame += 1
                        if leg_angle < 45:
                            kick_accuracy = 100 - ((45 - leg_angle) * 2)
                        elif leg_angle > 135:
                            kick_accuracy = 100 - ((leg_angle - 135) * 2)
                        else:
                            kick_accuracy = 100 - abs(90 - leg_angle)
                        
                        kick_accuracy = max(0, min(100, kick_accuracy))
                        total_kick_accuracy += kick_accuracy
                        total_leg_angle += leg_angle
                        highest_kick_accuracy = max(highest_kick_accuracy, kick_accuracy)
                    
                    frame_count += 1

            cap.release()
            pose.close()

            if frame_count > 0 and Elbow_Frames > 0:
                avg_elbow_accuracy = total_elbow_accuracy / Elbow_Frames
                avg_shoulder_angle = total_shoulder_angle / Elbow_Frames
                
                if Kick_Frame > 0:
                    avg_leg_angle = total_leg_angle / Kick_Frame
                    avg_kick_accuracy = total_kick_accuracy / Kick_Frame
                else:
                    avg_leg_angle = 0
                    avg_kick_accuracy = 0
                    
                overall_accuracy = (avg_elbow_accuracy * 0.6) + (avg_kick_accuracy * 0.4)
            else:
                overall_accuracy = avg_shoulder_angle = avg_leg_angle = avg_elbow_accuracy = avg_kick_accuracy = 0

            if os.path.exists(video_path):
                os.remove(video_path)

            try:
                API_KEY = "dhG2vA-qeAGs8ZuQI3E62WXCdrmUoIp7ke0z5Fdt25GD"
                
                token_response = requests.post(
                    'https://iam.cloud.ibm.com/identity/token',
                    data={
                        "apikey": API_KEY,
                        "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'
                    },
                    headers={'Content-Type': 'application/x-www-form-urlencoded'}
                )
                
                if not token_response.ok:
                    raise Exception(f"Failed to get token: {token_response.text}")
                    
                mltoken = token_response.json()["access_token"]

                payload_scoring = {
                    "input_data": [{
                         "fields": [
                                "Elbow_Frames",
                                "Back-Shoulder Angle (degrees)",
                                "Elbow_Accuracy",
                                "Kick_Frame",
                                "Leg Angle",
                                "Kick_Accuracy"
                        ],
                        "values": [[
                            int(Elbow_Frames),
                            round(float(avg_shoulder_angle), 2),
                            round(float(avg_elbow_accuracy), 2), 
                            int(Kick_Frame),
                            round(float(avg_leg_angle), 2),
                            round(float(avg_kick_accuracy), 2)
                        ]]
                    }]
                }

                response_scoring = requests.post(
                    'https://au-syd.ml.cloud.ibm.com/ml/v4/deployments/09496508-89b6-47c6-b85a-7d24ccc082ff/predictions?version=2021-05-01',
                    json=payload_scoring,
                    headers={
                        'Authorization': f'Bearer {mltoken}',
                        'Content-Type': 'application/json'
                    }
                )
                
                if not response_scoring.ok:
                    raise Exception(f"Scoring failed: {response_scoring.text}")

                ibm_response = response_scoring.json()
                if not ibm_response or 'predictions' not in ibm_response:
                    raise Exception("Invalid response format from IBM Cloud")

            except requests.exceptions.RequestException as e:
                print(f"IBM API Network Error: {str(e)}")
                return render(request, 'Elbowstrike.html', {'error': f"Network error connecting to IBM Cloud: {str(e)}"})
            except Exception as e:
                print(f"IBM API Error: {str(e)}")
                return render(request, 'Elbowstrike.html', {'error': f"Error getting prediction from IBM Cloud: {str(e)}"})

            elbow_strike_accuracy = "Not Available"

            try:
                predictions = ibm_response.get("predictions", [])
                if predictions:
                    values = predictions[0].get("values", [])
                    if values:
                        full_text = values[0][0]
                        match = re.search(r'(\d{2}-\d{2}%)', full_text)
                        if match:
                            elbow_strike_accuracy = match.group(1)
            except (IndexError, KeyError, TypeError):
                pass

            video_url = fs.url(filename)

            return render(request, 'Elbowstrike.html', { 
                'video_url': video_url,
                'Elbow_Frames': int(Elbow_Frames),
                'total_shoulder_angle': round(float(avg_shoulder_angle), 2),
                'total_elbow_accuracy': round(float(avg_elbow_accuracy), 2),
                'Kick_Frame': int(Kick_Frame),
                'Leg_Angle': round(float(avg_leg_angle), 2),
                'Kick_Accuracy': round(float(avg_kick_accuracy), 2),
                'ibm_response': elbow_strike_accuracy
            })

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return render(request, 'Elbowstrike.html', {'error': str(e)})
    
    return render(request, 'Elbowstrike.html', {'error': 'No video file uploaded'})

def calculate_angle(a, b, c):
    try:
        if a is None or b is None or c is None:
            return None
            
        ba = a - b
        bc = c - b
        
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba == 0 or norm_bc == 0:
            return None
            
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return float(np.degrees(angle))
    except Exception as e:
        print(f"Error calculating angle: {str(e)}")
        return None

def upload_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        
        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)
        video_url = fs.url(filename)
        
        return render(request, 'Elbowstrike.html', {'video_url': video_url})
        
    return render(request, 'Elbowstrike.html')
