# Configuration #1. Camera is directed towards an elder
C001 = {
    "A001": "enters into the service area through the door",
    "A002": "stands still without a purpose",
    "A003": "calls the robot",
    "A004": "stares at the robot",
    "A005": "lifts his arm to shake hands",
    "A006": "covers his face and cries",
    "A007": "lifts his arm for a high-five",
    "A008": "threatens to hit the robot",
    "A009": "beckons to go away",
    "A010": "turns back and walks to the door",
}

# Configuration #2. Camera is directed towards a robot (or rather towards a person who acts as a robot)
C002 = {
    "A001": "bows to the elderly person",
    "A002": "stares at the elderly person for a command",
    "A003": "approaches the elderly person",
    "A004": "scratches its head from awkwardness",
    "A005": "shakes hands with the elderly person",
    "A006": "stretches his hands to hug the elderly person",
    "A007": "high-fives with the elderly person",
    "A008": "blocks the face with arms",
    "A009": "turns back and leaves the service area",
    "A010": "bows to the elderly person",
}

# Configuration #3. Camera is placed between elder and robot, no clear labeling provided.
C003 = {
    "A001": 1,
    "A002": 2,
    "A003": 3,
    "A004": 4,
    "A005": 5,
    "A006": 6,
    "A007": 7,
    "A008": 8,
    "A009": 9,
    "A010": 10
}

# List of all possible configurations
all_configurations = [C001, C002, C003]

# Maximum number of features node can have - used for padding
max_number_of_features = 2379
