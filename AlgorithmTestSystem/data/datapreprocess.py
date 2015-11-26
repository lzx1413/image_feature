# -*- coding: utf-8 -*-
__author__ = 'lzx14'
from random import shuffle
from math import trunc
import sys

class DataPreprocess:
    """
    This class is used to do some preprocess of the feature data, inlucding remove the title of one single data ,select
    train data,querydata,basedata randomly,
    """

    def __init__(self, filename):
        self.raw_file_name = filename
        self.raw_data_number = 0

    def remove_header_of_each_row(self):
        """
        asuming that each row is a single data and this function will help you get the digital data and
        write it to the file with the "without_head_"+filename.
        :return:none
        """
        with open(self.raw_file_name, 'r') as raw_data:
            with open('without_head_' + self.raw_file_name, 'w')as pure_data:
                for single_data in raw_data:
                    single_pure_data = single_data.split(' ', 1)[1]
                    pure_data.writelines(single_pure_data)
                    self.raw_data_number += 1

    def divide_data(self, train_num_, query_num_, filename='default'):
        if filename == 'default':
            filename = 'without_head_' + self.raw_file_name
        if self.raw_data_number == 0:
            with open(filename, 'r') as file:
                for singledata in file:
                    self.raw_data_number += 1
        if train_num_ < 1:
            train_num = trunc(self.raw_data_number * train_num_)
        else:
            train_num = train_num_
        if query_num_ < 1:
            query_num = trunc(query_num_ * self.raw_data_number)
        else:
            query_num = query_num_
        with open(filename, 'r') as datafile:
            raw_data = datafile.read()
            data_list = raw_data.split('\n')
            data_size = len(data_list)
            shuffle(data_list)
            train_data = data_list[0:train_num]
            query_data = data_list[train_num:train_num + query_num]
            base_data = data_list[train_num + query_num:data_size]
            with open('train' + filename, 'w') as train_file:
                for line in train_data:
                    train_file.write('{0}\n'.format(line))
            with open('query' + filename, 'w') as query_file:
                for line in query_data:
                    query_file.write('{0}\n'.format(line))
            with open('base' + filename, 'w') as base_file:
                for line in base_data:
                    base_file.write('{0}\n'.format(line))

    def test_args(self, *args):
        for arg in args:
            print arg

    def get_several_trainsets(self, train_numbers):
        filename = 'without_head_'+self.raw_file_name
        with open(filename, 'r') as datafile:
            rawdata = datafile.read()
            print 'raw data has been read'
            data_list = rawdata.split('\n')
            data_size = len(data_list)
            shuffle(data_list)
            print 'raw data has been shuffled'
        for train_num_str in train_numbers:
            train_num = int(train_num_str)
            if train_num < data_size:
                train_data = data_list[0:train_num]
                with open(str(train_num) + '_samples_of_' + self.raw_file_name, 'w') as datafile:
                    for line in train_data:
                        datafile.write('{0}\n'.format(line))
                print str(train_num)+'train samples have saved!'
            else:
                print 'cannot get '+str(train_num)+'train samples as the raw data is not enough '




if __name__ == '__main__':
    filename = sys.argv[1]
    set_of_number = sys.argv[2:len(sys.argv)]
    data = DataPreprocess(filename)
    data.remove_header_of_each_row()
    #data.devidedata(50000, 1000)
    data.get_several_trainsets(set_of_number)
