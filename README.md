# Control_Systems_Project

I developed the door opener that checks the facemask and the vaccination against  SARS-CoV-2 (in Kazakhstan, we have a special app to check the vaccination status) using Python (OpenCV framework). Using the Arduino Serial Communication, the Python communicates to the actuator if the facemask is  present and if the person is vaccinated. The signal from OpenCV program rotates the motor which opens sliding doors.

The vacination status is checked from the smarphone using OpenCV masks.
For presence of the facemask I used OpenCV built in Face Recognizer and trained it on 4000 Kaggle Images. 
