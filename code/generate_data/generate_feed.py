import pandas as pd
import numpy as np
import random
from random import randint

# this script will generate a feed of n posts of the chosen category and export it to a .csv file

# random_control_values will determine how much variance there is in the control features.
# True will randomly assign values from the set [0, 0.25, 0.5, 0.75, 1]
# False will set all control variables to 0.5

# currently, the script is set to randomize the msg_med and msg_question features
# the features can be changed in the generate post function.

#############################

category = 'cat_ent'
length = 100
filename = "feed_cat_ent.csv"
random_control_values = True

#############################


categories = ['cat_per', 'cat_ent', 'cat_new', 'cat_edu', 'cat_con']
features = ['msg_rel', 'msg_qua', 'msg_sen', 'msg_sal', 'msg_com', 'msg_med', 'msg_que']
fixed_values = [0, 0.25, 0.5, 0.75, 1]


# 1. insert the prefered category from the category list
# 2. set a feature parameter to True if you want the value to be different from the fixed value
# 3. if randomize_fixed_values = True, values from the fixed_values list will be assigned for
#    the controlled features. Otherwise a standard value of 0.5 will be assigned.
# 4. adjust the if-statements to change feature values
def generate_post(category, msg_rel=False, msg_qua=False,
                  msg_sen=False, msg_sal=False, msg_com=False,
                  msg_med=True, msg_que=True, randomize_fixed_values=True):

    # set other categories equal to zero
    category_pairs = {category: 0 for category in categories}

    # randomize the fixed values if True
    if randomize_fixed_values == True:
        feature_pairs = {feature: random.choice(fixed_values) for feature in features}

    # else set the control features equal to 0.5
    else:
        feature_pairs = {feature: 0.5 for feature in features}

    # merge dictionaries
    post = {**category_pairs, **feature_pairs}

    # set the category
    post[category] = 1

    # set the variables you want change
    if msg_med == True:
        media_values = [0, 0.5, 1]
        post['msg_med'] = media_values[randint(0, 2)]

    if msg_que == True:
        post['msg_que'] = random.randint(0, 1)

    if msg_com == True:
        pass

    if msg_sal == True:
        pass

    if msg_sen == True:
        pass

    if msg_rel == True:
        pass

    if msg_qua == True:
        pass

    return(post)


def generate_feed(category, n):
    feed = pd.DataFrame()

    for i in range(n):
        post = generate_post(category, randomize_fixed_values=random_control_values)
        feed = feed.append(post, ignore_index=True)

        # rearrange columns
        feed = feed[categories + features]

    return feed


def export_feed(feed, filename):
    feed.to_csv(filename, index=False, index_label=False)


feed = generate_feed(category, length)
export_feed(feed, filename)
