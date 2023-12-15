# Not So Dummy 
Introduction to Robot Intelligence Final Assignment

## Project Introduction
The project aims to develop an e-pet robot, a sophisticated and interactive robotic companion. It leverages the VIAM Rover and Viam Python SDK to integrate multiple cutting-edge technologies. The core idea is to create a system that embodies a blend of robotics, computer vision, speech recognition, natural language processing (NLP), and real-time audio processing. This multifaceted approach enables the e-pet robot to operate autonomously in dynamic environments, understanding and executing voice commands with precision.

The main challenges addressed by this project include:

1. **Voice-Controlled Navigation and Interaction**: Enabling a robot to interpret and respond to voice commands in real-time. This involves the challenge of accurately converting speech to text and to interpret and execute a wide range of voice commands, catering to different phrasings or variations in speech.

2. **Autonomous Object Detection and Following**: Developing a method for the robot to identify, track, and follow specific objects within its environment.

3. **Real-Time Audio Processing**: Implementing a system for real-time audio capture, processing, and analysis to facilitate immediate response to voice commands.

4. **Navigational Intelligence Using SLAM**: Implementing SLAM to help the robot understand and navigate its surroundings, maintaining an updated map of its environment, and using this information for path planning and obstacle avoidance.

5. **Memory and Learning**: Implementing a system for the robot to remember previously detected objects and locations (like a 'home' location), enabling it to perform tasks based on past interactions and data.

## Functionalties 

1. The robot uses the `PyAudio` and `Speech Recognition` libraries to convert the speech command into text.

2. Using the `Spacy` library and Fuzzywuzzy's Levenshtein distance algorithm, the robot identifies the matched predefined command from the speech-to-text converted commands. The robot uses `Dummy` as the trigger word, and any user input without the trigger word is discarded.

   Example: `Dummy can you follow the bottle` will be matched with `follow_object_bottle`.

3. The robot can perform basic commands, such as `turn left`, `turn right` etc.

4. The robot uses a custom-trained lightweight `YOLOv5` object detection model.

5. The robot uses `SLAMClient` to locate and store its home location. The user can ask the robot to update the home location. After any successful fetch or find the robot will always return to its home.

6. Taking inspiration from fetch with dog the follow functionality has been implemented. You can put the intended fetch object infront of the robot and ask if it can identify or not. If identified you can ask the robot to follow the object. 

    The robot can not follow an object that it has not detected yet. You can use find command first in that case then ask it to follow.

    When following if the object is too close the robot stops and waits for the object to move to continue following. If the object doesn't move the dog successfully fetched the object. So, it does a happy turn and return to it's home.


7. You can ask the robot to find any object from `YOLOv5` classes. The robot roam around and try to find the object. If found the robot will store the positional data where it found the object. 

    In future if you ask again to find the same object, the robot will directly to that exact position and to check if the object is still there or not using combination of `VisionClient` and `SLAMClient`. If the object has been removed, it will start the find operation again. 

8. The robot uses an estimated distance calculation based on the bounded box of detected object to determin if it has reached close enough to the object or not. This ensures the robot doesn't crash into the object while following and finding the object.

9. The robot keeps a history of all the positional data using `pickle` for future references.  

10. The robot can follow and find humans. 
