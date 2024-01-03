#!/usr/bin/env python3

# Basic Rosbag to CSV converter. Does not handle composed msg types.
# https://github.com/ros2/rosbag2/issues/473
# https://github.com/fishros/ros2bag_convert/tree/main

import argparse
import sqlite3
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from builtin_interfaces.msg import Time, Duration
import numpy
import array
import csv
from os import walk, path, mkdir
import yaml


class BagFileParser():
    def __init__(self, bag_file):
        self.conn = sqlite3.connect(bag_file)
        self.cursor = self.conn.cursor()

        ## create a message type map
        topics_data = self.cursor.execute("SELECT id, name, type FROM topics").fetchall()
        self.topic_type = {name_of: type_of for id_of, name_of, type_of in topics_data}
        self.topic_id = {name_of: id_of for id_of, name_of, type_of in topics_data}
        self.topic_msg_message = {name_of: get_message(type_of) for id_of, name_of, type_of in topics_data}

    def __del__(self):
        self.conn.close()

    # Return [(timestamp0, message0), (timestamp1, message1), ...]
    def get_messages(self, topic_name):
        topic_id = self.topic_id[topic_name]
        # Get from the db
        rows = self.cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id)).fetchall()
        # Deserialise all and timestamp them
        # return [ (timestamp,deserialize_message(data, self.topic_msg_message[topic_name])) for timestamp,data in rows]
        return {timestamp: deserialize_message(data, self.topic_msg_message[topic_name]) for timestamp, data in rows}


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Convert a rosbag from IIWA_faros to a CSV file.')
    parser.add_argument('bag_folder', help='Path to the SQL database to convert.')
    parser.add_argument('-d', '--dest', dest='csv_folder', default='',
                        help='Path where to save the CSV files. If none is provided, CSV files will be saved in same folder than SQL databases.')
    args = parser.parse_args()

    # Create directory if needed
    if (args.csv_folder != '' and not path.isdir(args.csv_folder)):
        mkdir(args.csv_folder)

    ros_primitive_types = [bool, int, float, str, bytes]
    ros_array_types = [list, array.array, numpy.ndarray]
    ros_time_types = [Time, Duration]
    # Topic to be skipped
    skip_topics = [
        "/joint_states",
        "/tf_static",
        "/tf",
        "/robot_description",
        # "/Iiwa/driver_status",
        # "/Iiwa/fsm_event",
        # "/Iiwa/fsm_state",
        # "/rosout",
        # "/parameter_events",
        # "/drill_depth",
        # "/DSG/status",
        # "/Drill/status",
        # "/Supervisor/breach_status"
    ]
    # Topic fields to be skipped
    skip_fields = [
        "header",
        "get_fields_and_field_types"
    ]

    bagfile = None
    for root, dirs, files in walk(args.bag_folder):
        for file in files:
            if file.__contains__('.db3'):
                bagfile = path.join(root, file)
                csv_file = path.join(args.csv_folder if (args.csv_folder != '') else root,
                                     file.replace('_0.db3', '.csv'))
            if file.__contains__('.yaml'):
                metadata_file = path.join(root, file)

        if bagfile and metadata_file:
            print("Reading : " + bagfile + " ...")

            # Read metadata
            with open(metadata_file, "r") as stream:
                try:
                    metadata = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)

            # Read ROS2 Bag
            parser = BagFileParser(bagfile)

            ### Metadata extraction ###
            # Prepare topics to be written
            topics_to_write = list()
            timestamps = list()
            for topic in metadata["rosbag2_bagfile_information"]["topics_with_message_count"]:
                if topic["message_count"] != 0 and not topic["topic_metadata"]["name"] in skip_topics:
                    print("Found : " + topic["topic_metadata"]["name"])
                    topics_to_write.append({
                        "name": topic["topic_metadata"]["name"],
                        "fields": list(),
                        "messages": parser.get_messages(topic["topic_metadata"]["name"])
                    })

                    first_msg = list(topics_to_write[-1]["messages"].values())[0]
                    for att in dir(first_msg):  # Get metadata from 1st msg
                        if not att.startswith('_') and att.islower() and not att in skip_fields:
                            field_type = type(getattr(first_msg, att))
                            if field_type in ros_primitive_types + ros_time_types:
                                field_size = 1
                            elif field_type in ros_array_types:
                                field_size = len(getattr(first_msg, att))
                            else:
                                print("Warning : " + str(field_type) + " of " + str(
                                    att) + " is not supported. Skipping...")
                                continue

                            topics_to_write[-1]["fields"].append({
                                "name": att,
                                "type": field_type,
                                "size": field_size
                            })

                    timestamps.extend(list(topics_to_write[-1]["messages"]))
                else:
                    print("Skipping : " + topic["topic_metadata"]["name"])

            timestamps = list(dict.fromkeys(timestamps))  # Remove duplicate ?
            timestamps.sort()

            ### CSV Writting ###
            print("Writing : " + csv_file + " ...")

            with open(csv_file, 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')  # ,quotechar='|', quoting=csv.QUOTE_MINIMAL)

                # CSV header
                csv_header = ['timestamp']
                for topic in topics_to_write:
                    for field in topic["fields"]:
                        if (field["size"] == 1):
                            csv_header.append(topic["name"] + "." + field["name"])
                        elif (field["size"] > 1):
                            csv_header.extend(
                                [topic["name"] + "." + field["name"] + '.' + str(i) for i in range(0, field["size"])])
                        else:
                            print("Empty array [" + topic["name"] + "." + field["name"] + "], skipping...")
                spamwriter.writerow(csv_header)

                # CSV data
                for timestamp in timestamps:
                    csv_row = [timestamp]
                    for topic in topics_to_write:
                        if timestamp in topic["messages"]:
                            for field in topic["fields"]:
                                if field["type"] in ros_primitive_types:
                                    csv_row.append(getattr(topic["messages"][timestamp], field["name"]))
                                elif field["type"] in ros_array_types:
                                    csv_row.extend(getattr(topic["messages"][timestamp], field["name"]))
                                elif field["type"] in ros_time_types:
                                    time = getattr(topic["messages"][timestamp], field["name"])
                                    csv_row.append(time.sec * 1e9 + time.nanosec)  # Write in nanosec
                        else:
                            for field in topic["fields"]:
                                csv_row.extend([' '] * field["size"])

                    spamwriter.writerow(csv_row)

                    bagfile = None  # Delete path of processed bag for the next one
    print("Done !")  # !/usr/bin/env python3

# Basic Rosbag to CSV converter. Does not handle composed msg types.
# https://github.com/ros2/rosbag2/issues/473
# https://github.com/fishros/ros2bag_convert/tree/main

import argparse
import sqlite3
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from builtin_interfaces.msg import Time, Duration
import numpy
import array
import csv
from os import walk, path, mkdir
import yaml


class BagFileParser():
    def __init__(self, bag_file):
        self.conn = sqlite3.connect(bag_file)
        self.cursor = self.conn.cursor()

        ## create a message type map
        topics_data = self.cursor.execute("SELECT id, name, type FROM topics").fetchall()
        self.topic_type = {name_of: type_of for id_of, name_of, type_of in topics_data}
        self.topic_id = {name_of: id_of for id_of, name_of, type_of in topics_data}
        self.topic_msg_message = {name_of: get_message(type_of) for id_of, name_of, type_of in topics_data}

    def __del__(self):
        self.conn.close()

    # Return [(timestamp0, message0), (timestamp1, message1), ...]
    def get_messages(self, topic_name):
        topic_id = self.topic_id[topic_name]
        # Get from the db
        rows = self.cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id)).fetchall()
        # Deserialise all and timestamp them
        # return [ (timestamp,deserialize_message(data, self.topic_msg_message[topic_name])) for timestamp,data in rows]
        return {timestamp: deserialize_message(data, self.topic_msg_message[topic_name]) for timestamp, data in rows}


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Convert a rosbag from IIWA_faros to a CSV file.')
    parser.add_argument('bag_folder', help='Path to the SQL database to convert.')
    parser.add_argument('-d', '--dest', dest='csv_folder', default='',
                        help='Path where to save the CSV files. If none is provided, CSV files will be saved in same folder than SQL databases.')
    args = parser.parse_args()

    # Create directory if needed
    if (args.csv_folder != '' and not path.isdir(args.csv_folder)):
        mkdir(args.csv_folder)

    ros_primitive_types = [bool, int, float, str, bytes]
    ros_array_types = [list, array.array, numpy.ndarray]
    ros_time_types = [Time, Duration]
    # Topic to be skipped
    skip_topics = [
        "/joint_states",
        "/tf_static",
        "/tf",
        # "/robot_description",
        # "/Iiwa/driver_status",
        # "/Iiwa/fsm_event",
        # "/Iiwa/fsm_state",
        # "/rosout",
        # "/parameter_events",
        # "/drill_depth",
        # "/DSG/status",
        # "/Drill/status",
        # "/Supervisor/breach_status"
    ]
    # Topic fields to be skipped
    skip_fields = [
        "header",
        "get_fields_and_field_types"
    ]

    bagfile = None
    for root, dirs, files in walk(args.bag_folder):
        for file in files:
            if file.__contains__('.db3'):
                bagfile = path.join(root, file)
                csv_file = path.join(args.csv_folder if (args.csv_folder != '') else root,
                                     file.replace('_0.db3', '.csv'))
            if file.__contains__('.yaml'):
                metadata_file = path.join(root, file)

        if bagfile and metadata_file:
            print("Reading : " + bagfile + " ...")

            # Read metadata
            with open(metadata_file, "r") as stream:
                try:
                    metadata = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)

            # Read ROS2 Bag
            parser = BagFileParser(bagfile)

            ### Metadata extraction ###
            # Prepare topics to be written
            topics_to_write = list()
            timestamps = list()
            for topic in metadata["rosbag2_bagfile_information"]["topics_with_message_count"]:
                if topic["message_count"] != 0 and not topic["topic_metadata"]["name"] in skip_topics:
                    print("Found : " + topic["topic_metadata"]["name"])
                    topics_to_write.append({
                        "name": topic["topic_metadata"]["name"],
                        "fields": list(),
                        "messages": parser.get_messages(topic["topic_metadata"]["name"])
                    })

                    first_msg = list(topics_to_write[-1]["messages"].values())[0]
                    for att in dir(first_msg):  # Get metadata from 1st msg
                        if not att.startswith('_') and att.islower() and not att in skip_fields:
                            field_type = type(getattr(first_msg, att))
                            if field_type in ros_primitive_types + ros_time_types:
                                field_size = 1
                            elif field_type in ros_array_types:
                                field_size = len(getattr(first_msg, att))
                            else:
                                print("Warning : " + str(field_type) + " of " + str(
                                    att) + " is not supported. Skipping...")
                                continue

                            topics_to_write[-1]["fields"].append({
                                "name": att,
                                "type": field_type,
                                "size": field_size
                            })

                    timestamps.extend(list(topics_to_write[-1]["messages"]))
                else:
                    print("Skipping : " + topic["topic_metadata"]["name"])

            timestamps = list(dict.fromkeys(timestamps))  # Remove duplicate ?
            timestamps.sort()

            ### CSV Writting ###
            print("Writing : " + csv_file + " ...")

            with open(csv_file, 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')  # ,quotechar='|', quoting=csv.QUOTE_MINIMAL)

                # CSV header
                csv_header = ['timestamp']
                for topic in topics_to_write:
                    for field in topic["fields"]:
                        if (field["size"] == 1):
                            csv_header.append(topic["name"] + "." + field["name"])
                        elif (field["size"] > 1):
                            csv_header.extend(
                                [topic["name"] + "." + field["name"] + '.' + str(i) for i in range(0, field["size"])])
                        else:
                            print("Empty array [" + topic["name"] + "." + field["name"] + "], skipping...")
                spamwriter.writerow(csv_header)

                # CSV data
                for timestamp in timestamps:
                    csv_row = [timestamp]
                    for topic in topics_to_write:
                        if timestamp in topic["messages"]:
                            for field in topic["fields"]:
                                if field["type"] in ros_primitive_types:
                                    csv_row.append(getattr(topic["messages"][timestamp], field["name"]))
                                elif field["type"] in ros_array_types:
                                    csv_row.extend(getattr(topic["messages"][timestamp], field["name"]))
                                elif field["type"] in ros_time_types:
                                    time = getattr(topic["messages"][timestamp], field["name"])
                                    csv_row.append(time.sec * 1e9 + time.nanosec)  # Write in nanosec
                        else:
                            for field in topic["fields"]:
                                csv_row.extend([' '] * field["size"])

                    spamwriter.writerow(csv_row)

                    bagfile = None  # Delete path of processed bag for the next one
    print("Done !")