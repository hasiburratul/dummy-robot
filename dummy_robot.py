import asyncio
from viam.robot.client import RobotClient
from viam.services.vision import VisionClient
from viam.components.camera import Camera
from viam.components.base import Base
from viam.services.slam.client import SLAMClient
import math
import random
import pyaudio
import wave
import speech_recognition as sr
import tempfile
import spacy
from fuzzywuzzy import fuzz
import pickle

# connect to the robot
async def connect():
    """
    Connect to the robot using the provided API key and address.

    Returns:
        RobotClient: The connected robot client.
    """
    opts = RobotClient.Options.with_api_key(
        api_key="",
        api_key_id="",
    )
    return await RobotClient.at_address("", opts)

nlp = spacy.load("en_core_web_sm")
MIN_SIMILARITY_SCORE = 50
currently_following = []
currently_finding = []
detected_objects = []
found_objects_history = []
home_location_history = []

# Define the commands and their corresponding actions
predefined_commands = {
    "make a square": "make_a_square",
    "make a circle": "make_a_circle",
    "hello start": "start",
    "show me something": "start",
    "what is it": "detect",
    "what is that": "detect",
    "what is this": "detect",
    "what is the object": "detect",
    "what do you see": "detect",
    "now follow the": "follow_object",
    "follow the": "follow_object",
    "follow": "follow_object",
    "grab the": "follow_object",
    "grab": "follow_object",
    "track the": "follow_object",
    "track": "follow_object",
    "find the": "find_object",
    "find": "find_object",
    "stop": "stop",
    "sit": "stop",
    "stay": "stop",
    "hi": "start",
    "hello": "start",
    "hey": "start",
    "start": "start",
    "new base" : "new_home",
    "new home" : "new_home",
    "this is home" : "new_home",
    "change home" : "new_home",
    "update home" : "new_home",
    "go home now" : "come_home",
    "come home now" : "come_home",
    "come back home" : "come_home",
    "come back" : "come_home",
    "home now" : "come_home",
    "go back" : "move_back",
    "left" : "left",
    "go left" : "left",
    "right" : "right",
    "turn left" : "turn_left",
    "turn right" : "turn_right",
    "turn around" : "turn_around",
    "turn" : "turn_around",
    "turn around now" : "turn_around",
    "move right" : "right",
    "move left" : "left",
    "move forward" : "forward",
    "move backward" : "backward",
    "move back" : "backward",
    "forward" : "forward",
    "backward" : "backward",
    "move" : "forward",
    "go" : "forward",
    "go forward" : "forward",
    "go backward" : "backward",
}


def preprocess_text(text):
    """
    Preprocess the input text by removing extra whitespace and special characters.
    """
    text = text.strip()
    text = " ".join(text.split())
    return text


def summarize_text(input_text):
    """
    Summarize the input text to one sentence using spaCy.
    """
    doc = nlp(input_text)
    sentences = [sent.text for sent in doc.sents]
    if sentences:
        return sentences[0]
    return ""


def match_command(summary_text):
    """
    Match the summary text against predefined commands.
    Use fuzzy string matching to find the best match.
    """
    best_match = None
    best_match_score = 0

    for command, action in predefined_commands.items():
        if "dummy" in summary_text.lower(): # dummy is the trigger word
            cleaned_summary = summary_text.lower().replace("dummy", "").strip()
            similarity_score = fuzz.ratio(cleaned_summary, command)
            if similarity_score > best_match_score:
                best_match_score = similarity_score
                best_match = action
            elif (
                similarity_score == best_match_score
                and similarity_score >= 50
                and fuzz.ratio(command, best_match.lower())
                < fuzz.ratio(command, action.lower())
            ):
                best_match = action

    if best_match == "follow_object":
        words = summary_text.lower().split()
        if "the" in words:
            words.remove("the")
        if "a" in words:
            words.remove("a")
        if "an" in words:
            words.remove("an")
        try:
            for i, word in enumerate(words):
                if word == "follow" or word == "grab" or word == "track":
                    if words[i + 1] == "me" or words[i + 1] == "him" or words[i + 1] == "her":
                        return best_match + "_person"
                    if words[i + 1] in detected_objects:
                        return best_match + "_" + words[i + 1]
                    else:
                        return best_match + "_What is it?"
        except IndexError:
            # Handle the IndexError here (e.g., print an error message or return a default value)
            return "No matching command found"


    if best_match == "find_object":
        words = summary_text.lower().split()
        if "the" in words:
            words.remove("the")
        if "a" in words:
            words.remove("a")
        if "an" in words:
            words.remove("an")
        try:
            for i, word in enumerate(words):
                if word == "find" and words[i + 1]:
                    return best_match + "_" + words[i + 1]
        except IndexError:
                # Handle the IndexError here (e.g., print an error message or return a default value)
                return "No matching command found"

    if best_match_score >= MIN_SIMILARITY_SCORE:
        return best_match
    else:
        return "No matching command found"

# Audio recording 
def record_audio(duration, format=pyaudio.paInt16, channels=1, rate=44100):
    p = pyaudio.PyAudio()

    stream = p.open(
        format=format, channels=channels, rate=rate, input=True, frames_per_buffer=1024
    )

    print("Recording for {} seconds...".format(duration))
    frames = []

    for _ in range(0, int(rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    print("Recording finished")

    stream.stop_stream()
    stream.close()
    p.terminate()

    return frames, rate


def write_audio_to_file(frames, rate):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wf = wave.open(f, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))
        wf.close()
        return f.name


def recognize_speech_from_audio(recognizer, audio_file_path):
    response = {"success": True, "error": None, "transcription": None}

    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        try:
            response["transcription"] = recognizer.recognize_google(audio_data)
        except sr.RequestError:
            response["success"] = False
            response["error"] = "API unavailable"
        except sr.UnknownValueError:
            response["error"] = "Unable to recognize speech"

    return response

# Square movement
async def moveInSquare(base):
    for _ in range(4):
        # moves the rover forward 500mm at 500mm/s
        await base.move_straight(500, 500)
        print("move straight")
        # spins the rover 90 degrees at 100 degrees per second
        await base.spin(100,90)
        print("spin 90 degrees")

# Default movement
async def default_movement(base):
    await base.move_straight(10, -100)
    await base.spin(360, 100)

# Get current position from SLAM
async def update_position(slam_client):
    pose = await slam_client.get_position()
    x = pose.x
    y = pose.y
    o_z = pose.o_z
    theta = pose.theta
    return x, y, theta

# Check if the object is on the left or right or too close
def leftOrRight(d, frame_height, midpoint):
    if not d:
        print("Nothing detected")
        return -1

    center_x = (d.x_min + d.x_max) / 2

    distance_estimate = frame_height - d.y_max
    print(f"Distance estimate: {distance_estimate} px unit")

    if distance_estimate < 80:
        return "too_close"  # object is too close
    if center_x < midpoint - midpoint / 6:
        return "left"  # on the left
    if center_x > midpoint + midpoint / 6:
        return "right"  # on the right

    return "center"  # basically centered

# Search for the object
async def search_for_object(base, vel, detector, last_know_side):
    # Define the search parameters
    N = 100
    search_velocity = 100  # Adjust the search velocity as needed
    search_duration = 2   # Duration (in seconds) for each search step
    search_steps = 10     # Number of search steps (adjust as needed)
    cycles_before_move = 5  # Number of cycles before moving forward
    cycles_count = 0
    angle_increment = 360 / search_steps  # Calculate the angle increment for each search step

    if last_know_side == "right":
        angle_increment = -angle_increment
    if last_know_side == "center":
        await base.move_straight(50, vel)

    while N:
        # Move the rover in a circular pattern
        await base.spin(angle_increment, search_velocity)
        await asyncio.sleep(search_duration)

        # Check for detections in the current frame
        detections = await detector.get_detections_from_camera("cam")
        for d in detections:
            if d.class_name.lower() in currently_finding and d.confidence > 0.5:
                return True

        cycles_count += 1
        if cycles_count == cycles_before_move:
            # Move forward to increase the search area
            await base.move_straight(50, vel)
            print("Moving forward to continue searching")
            cycles_count = 0  # Reset the cycle count
        N -= 1
    return False  # Target not detected in the search


# Follow the object
async def follow_object(base, detector, camera):
    spinNum = 10         # when turning, spin the motor this much
    straightNum = 100   # when going straight, spin motor this much
    vel = 100           # go this fast when moving motor
    frame = await camera.get_image(mime_type="image/jpeg")
    found = False
    last_know_side = "center"
    too_close_counter = 0

    detections = await detector.get_detections_from_camera("cam")

    for d in detections:
        if d.class_name.lower() in currently_following and d.confidence > 0.5:
            last_know_side = leftOrRight(d, frame.size[1], frame.size[0] / 2)
            found = True
    while True:
        if not found:
            print(f"Following the {currently_following[0]}")
            if last_know_side == "right":
                print(f"Turning right to follow the {currently_following[0]}")
            else:
                print(f"Turning left to follow the {currently_following[0]}")
            target_found = await search_for_object(base, vel, detector, last_know_side)
            if target_found:
                found = True
            else :
                return "not_found"
        else:
            detections = await detector.get_detections_from_camera("cam")
            for d in detections:
                if d.class_name.lower() in currently_following and d.confidence > 0.5:
                    last_know_side = leftOrRight(d, frame.size[1], frame.size[0] / 2)
                    found = True
                    break
                if not found:
                    found = False
            if last_know_side == "left":
                await base.spin(spinNum, vel)     
                await base.move_straight(straightNum, vel)
            if last_know_side == "center":
                await base.move_straight(straightNum, vel)
            if last_know_side == "right":  
                await base.spin(-spinNum, vel)
                await base.move_straight(straightNum, vel)
            if last_know_side == "too_close":
                await asyncio.sleep(1)
                too_close_counter += 1
        
        detections = await detector.get_detections_from_camera("cam")
        for d in detections:
            if d.class_name.lower() in currently_following and d.confidence > 0.5:
                last_know_side = leftOrRight(d, frame.size[1], frame.size[0] / 2)
                found = True

        if too_close_counter > 3:
            return "too_close"

# Find the object
async def find_object(base, detector, camera):
    spinNum = 10         # when turning, spin the motor this much
    straightNum = 100   # when going straight, spin motor this much
    vel = 100           # go this fast when moving motor
    frame = await camera.get_image(mime_type="image/jpeg")
    found = False
    last_know_side = "center"

    detections = await detector.get_detections_from_camera("cam")
    for d in detections:
        if d.class_name.lower() in currently_finding and d.confidence > 0.5:
            last_know_side = leftOrRight(d, frame.size[1], frame.size[0] / 2)
            found = True
    
    while True:
        if not found:
            print(f"Searching the {currently_finding[0]}")
            if last_know_side == "right":
                print(f"Turning right to search the {currently_finding[0]}")
            else:
                print(f"Turning left to search the {currently_finding[0]}")
            target_found = await search_for_object(base, vel, detector, last_know_side)
            if target_found:
                found = True
            else :
                return "not_found"
        else:
            detections = await detector.get_detections_from_camera("cam")
            for d in detections:
                if d.class_name.lower() in currently_finding and d.confidence > 0.5:
                    last_know_side = leftOrRight(d, frame.size[1], frame.size[0] / 2)
                    found = True
                    break
                if not found:
                    found = False
            if last_know_side == "left":
                await base.spin(spinNum, vel)     
                await base.move_straight(straightNum, vel)
            if last_know_side == "center":
                await base.move_straight(straightNum, vel)
            if last_know_side == "right":  
                await base.spin(-spinNum, vel)
                await base.move_straight(straightNum, vel)
            if last_know_side == "too_close":
                return "too_close"
        
        detections = await detector.get_detections_from_camera("cam")
        for d in detections:
            if d.class_name.lower() in currently_finding and d.confidence > 0.5:
                last_know_side = leftOrRight(d, frame.size[1], frame.size[0] / 2)
                found = True

# Check if current theta is within tolerance
def is_theta_within_tolerance(current_theta, desired_theta, tolerance):
    return abs(current_theta - desired_theta) <= tolerance

# Check if the robot has reached the target
def has_reached_target(x, y, target_x, target_y, tolerance=50):
    return math.sqrt((x - target_x) ** 2 + (y - target_y) ** 2) <= tolerance

# Path planning based on SLAM
async def path_planning_xy(base, slam_client, target_x, target_y, target_theta):
    x, y, theta = await update_position(slam_client)

    if not is_theta_within_tolerance(theta, 0, 2):
        if theta < 0:
            spin_angle = theta
            await base.spin(velocity=-10, angle=int(spin_angle))
        elif theta > 0:
            spin_angle = theta
            await base.spin(velocity=-10, angle=int(spin_angle))
    
    x, y, theta = await update_position(slam_client)
    while abs(x - target_x) > 50:
        x, y, theta = await update_position(slam_client)
        if x > target_x:
            print(f"Current X is {x}")
            print(f"X is out of bound by {int(x - target_x)}")
            await base.move_straight(velocity=-100, distance=int(x - target_x))
        elif x < target_x:
            print(f"Current X is {x}")
            print(f"X is out of bound by {int(target_x) - x}")
            await base.move_straight(velocity=100, distance=int(target_x - x))

    await base.spin(velocity=10, angle=90)
    while abs(y - target_y) > 50:
        x, y, theta = await update_position(slam_client)
        if y > target_y:
            print(f"Current Y is {y}")
            print(f"Y is out of bound by {int(y - target_y)}")
            await base.move_straight(velocity=-100, distance=int(y - target_y))
        elif y < target_y:
            print(f"Current Y is {y}")
            print(f"Y is out of bound by {int(target_y) - y}")
            await base.move_straight(velocity=100, distance=int(target_y - y))

# Go back to home location
async def come_home(base, slam_client, home_location_history=home_location_history):
    print(home_location_history)
    x, y, theta = await update_position(slam_client)
    target_x, target_y, target_theta = home_location_history[0][0], home_location_history[0][1], home_location_history[0][2]

    while not has_reached_target(x, y, target_x, target_y):
        print("going back to home")
        await path_planning_xy(base, slam_client, target_x, target_y, target_theta)
        x, y, theta = await update_position(slam_client)
        if not is_theta_within_tolerance(theta, target_theta, 10):
            theta -= target_theta
            if theta < 0:
                spin_angle = theta
                await base.spin(velocity=-10, angle=int(spin_angle))
            elif theta > 0:
                spin_angle = theta
                await base.spin(velocity=-10, angle=int(spin_angle))
                
# Main function
async def main():
    recognizer = sr.Recognizer()
    robot = await connect()
    base = Base.from_robot(robot, "viam_base")
    detector = VisionClient.from_robot(robot, "myPeopleDetector")
    slam_client = SLAMClient.from_robot(robot=robot, name="slam")
    camera_name = "cam"
    camera = Camera.from_robot(robot, camera_name)
    N = 1

    try:
        with open('found_objects_history.pkl', 'rb') as file:
            found_objects_history = pickle.load(file)
    except FileNotFoundError:
        # Handle the case when the file is not found
        print("File 'found_objects_history.pkl' not found.")
        found_objects_history = []
    
    try: 
        with open('home_location_history.pkl', 'rb') as file:
            home_location_history = pickle.load(file)
            print(home_location_history)
    except FileNotFoundError:
        # Handle the case when the file is not found
        print("File 'home_location_history.pkl' not found.")
        x, y, theta = await update_position(slam_client)
        home_location_history = [[x, y, theta]]

    while True:
        frames, rate = record_audio(duration=5)  # Record audio for 5 seconds
        audio_file_path = write_audio_to_file(frames, rate)

        response = recognize_speech_from_audio(recognizer, audio_file_path)
        my_response = response["transcription"]

        if response["success"]:
            if my_response == None:
                continue
            elif my_response != None:
                input_text = preprocess_text(my_response)
                summary = summarize_text(input_text)
                matched_command = match_command(summary)
                print(f"Did you say: {summary}")
                if matched_command == "start":
                    await default_movement(base)
                elif matched_command == "make_a_square":
                    await moveInSquare(base)
                elif matched_command == "detect":
                    for i in range(N):
                        detections = await detector.get_detections_from_camera("cam")
                        for d in detections:
                            if d.confidence > 0.5:
                                print(f"I see a {d.class_name}")
                                if d.class_name.lower() not in detected_objects:
                                    detected_objects.append(d.class_name.lower())
                        if not detected_objects:
                            print("I don't see anything")

                elif matched_command.startswith("follow_object_"):
                    additional_text = matched_command[len("follow_object_") :].strip()
                    if "What is it?" in additional_text:
                        print("What is it?")
                    else:
                        print(f"Following: {additional_text}")
                        currently_following.append(additional_text)
                        status = await follow_object(base, detector, camera)
                        if status == "too_close":
                            print("Stopping. The {additional_text} stopped moving.")
                            await come_home(base, slam_client, home_location_history)
                            await default_movement(base)
                        else :
                            print("I lost it")
                            await come_home(base, slam_client, home_location_history)
                        currently_following.pop(currently_following.index(additional_text))

                elif matched_command.startswith("find_object_"):
                    additional_text = matched_command[len("find_object_") :].strip()
                    found = False
                    currently_finding.append(additional_text)
                    for record in found_objects_history:
                        if additional_text in record[0]:
                            print(f"I already know where the {additional_text} is!")
                            print(f"I will take you there")
                            print(f"Last time I saw it at {record[1]}, {record[2]}, {record[3]}")
                            x, y, theta = await update_position(slam_client)
                            target_x, target_y, target_theta = record[1], record[2], record[3]

                            while not has_reached_target(x, y, target_x, target_y):
                                print("going to target")
                                await path_planning_xy(base, slam_client, target_x, target_y, target_theta)
                                x, y, theta = await update_position(slam_client)
                                if not is_theta_within_tolerance(theta, target_theta, 10):
                                    theta -= target_theta
                                    if theta < 0:
                                        spin_angle = theta
                                        await base.spin(velocity=-10, angle=int(spin_angle))
                                    elif theta > 0:
                                        spin_angle = theta
                                        await base.spin(velocity=-10, angle=int(spin_angle))
                                        
                                frame = await camera.get_image(mime_type="image/jpeg")
                                detections = await detector.get_detections_from_camera("cam")
                                for d in detections:
                                    if d.class_name.lower() in currently_finding and d.confidence > 0.5:
                                        last_know_side = leftOrRight(d, frame.size[1], frame.size[0] / 2)
                                        if last_know_side == "too_close":
                                            found = True
                                            currently_finding.pop(currently_finding.index(additional_text))
                                            print("I found it!!!")
                                            await default_movement(base)
                                            await come_home(base, slam_client, home_location_history)
                                            break
                            if not found:
                                    print("Aha you moved it from the last time I saw it")
                                    found_objects_history.pop(found_objects_history.index(record))
                    if not found:
                        print(f"Finding: {additional_text}")
                        status = await find_object(base, detector, camera)
                        if status == "too_close":
                            print("I found it!!!")
                            x, y, theta = await update_position(slam_client)
                            print(f"Current position: {x}, {y}, {theta}")
                            currently_finding.pop(currently_finding.index(additional_text))
                            detected_objects.append(additional_text)
                            found_objects_history.append([additional_text, x, y, theta])
                            await default_movement(base)
                            await come_home(base, slam_client, home_location_history)
                        else :
                            print("I couldn't find it")
                            await come_home(base, slam_client, home_location_history)
                    
                elif matched_command == "stop":
                    with open('found_objects_history.pkl', 'wb') as file:
                        pickle.dump(found_objects_history, file)
                    with open('home_location_history.pkl', 'wb') as file:
                        pickle.dump(home_location_history, file)
                    break

                elif matched_command == "new_home":
                    x, y, theta = await update_position(slam_client)
                    home_location_history.pop()
                    home_location_history.append([x, y, theta])
                    print(f"New home location: {x}, {y}, {theta}")
                
                elif matched_command == "come_home":
                    print("Going back home")
                    await come_home(base, slam_client, home_location_history)
                    await default_movement(base)
                    print("I'm home")

                elif matched_command == "left":
                    await base.spin(angle=90, velocity=100)
                    await base.move_straight( distance=100, velocity=100)

                elif matched_command == "right":
                    await base.spin(angle=-90, velocity=100)
                    await base.move_straight(distance=100, velocity=100)
                elif matched_command == "turn_left":
                    await base.spin(angle=90, velocity=100)
                elif matched_command == "turn_right":
                    await base.spin(angle=-90, velocity=100)
                elif matched_command == "turn_around":
                    await base.spin(angle=180, velocity=100)
                elif matched_command == "forward":
                    await base.move_straight(distance=300, velocity=100)
                elif matched_command == "backward":
                    await base.move_straight(distance=300, velocity=-100)
                elif matched_command == "No matching command found":
                    print("I don't understand")

        else:
            print("I didn't catch that. Error: {}".format(response["error"]))


if __name__ == "__main__":
    asyncio.run(main())
