import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/workspace/ros2_ws/src/airhockey2025/install/AirHockeyChallenge'
