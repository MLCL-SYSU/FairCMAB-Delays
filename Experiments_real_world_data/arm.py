# -*- coding: utf-8 -*-
import numpy as np
from utilities import *


def get_arm_set(input_file, delimiter='\t'):
    arm_set= []

    with open(input_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        columns = line.strip().split(delimiter)
        if len(columns) >= 5:
            value = columns[4] 
            if value not in arm_set:
                arm_set.append(value)

    return arm_set


class Arm():
    def __init__(self,arm_set_title,click_rate,mean):  
        self.data = 'Top_10_products_successful_conversions'      
        self.arm_set_title = arm_set_title 
        self.click_rate = click_rate
        self.mean = mean

    def pull(self,arm_index):
        reward = np.random.binomial(n=1, p=self.mean[arm_index], size=1)[0]
        if reward == 0 and np.random.uniform(0,1) > self.click_rate:
            delay = 0
        else:
            info = random_sample_dataset(self.data, self.arm_set_title[arm_index], delimiter='\t')
            if float(info[2]) == 0:
                delay = 0
            else:
                time_between_page_visit = (7*86400) / (float(info[2])/self.click_rate)
                delay = float(info[1]) / time_between_page_visit
            
            if delay > 50000:
                delay = 0
        return reward, delay

    def get_mean(self):
        return self.mean

    def __str__(self):
        return f"{self.type} Arm with mean = {self.mean}"