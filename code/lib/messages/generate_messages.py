# Generate messages for the model to run
# Programmer: Eric Araujo
# Date of Creation: 14/11/2018
# Last update: 14/11/2018

import os
import sys
import json
import pandas as pd

def sequence_messages(num_repeat=10, num_loops=5, root_folder='../../'):
    """Save a csv file at filename folder containing a sequence of messages based on:
    num_repeat: how many messages of the same type will be created in sequence.
    num_loops: how many times the body of messages is going to be repeated.

    For example:
    if num_repeat = 5, it means that each message (fake news, news, holidays pics, etc.) will have 5 sequential messages one after another. In total there will be 25 messages (5 types of message x 5 repeats).
    The num_loops is about how many times the structure above is going to be replicated. So if we have num_loops=10, the 25 messages will be copied 10 times, generating 250 messages in total.
    """
    # Open JSON file with the information of the messages
    json_string = root_folder + "data/messages/messages.json"
    with open(json_string, 'r') as f:
        messages = json.load(f)
    
    # Generate 10 messages of each
    num_repeat = 10

    messages_df = pd.DataFrame()
    messages_df = pd.concat(
                    [messages_df, pd.DataFrame([messages['fake_news']]*num_repeat)], ignore_index=True)
    messages_df = pd.concat(
                    [messages_df, pd.DataFrame([messages['news']]*num_repeat)], ignore_index=True)
    messages_df = pd.concat(
                    [messages_df, pd.DataFrame([messages['holidays_pics']]*num_repeat)], ignore_index=True)
    messages_df = pd.concat(
                    [messages_df, pd.DataFrame([messages['online_course_ad']]*num_repeat)], ignore_index=True)
    messages_df = pd.concat(
                    [messages_df, pd.DataFrame([messages['cats']]*num_repeat)], ignore_index=True)

    # Repeat the same sequence num_loops times
    messages_df = pd.concat([messages_df]*num_loops, ignore_index=True)

    # Export dataframe to .csv
    filename = root_folder + "data/messages/messages.csv"
    messages_df.to_csv(filename, index=False)

    return messages_df


