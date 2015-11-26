# coding=utf-8
# !/usr/bin/python
import os
import re
import sys
allFileNum = 0


def printPath(level, path):
    global allFileNum
    '''''
    打印一个目录下的所有文件夹和文件
    '''
    # 所有文件夹，第一个字段是次目录的级别
    dirlist = []
    # 所有文件
    filelist = []
    filewithd = []
    filewithoutd = []
    # 返回一个列表，其中包含在目录条目的名称(google翻译)
    files = os.listdir(path)
    # 先添加目录级别
    dirlist.append(str(level))
    for f in files:
        if os.path.isdir(path + '/' + f):
            # 排除隐藏文件夹。因为隐藏文件夹过多
            if f[0] == '.':
                pass
            else:
                # 添加非隐藏文件夹
                dirlist.append(f)
        if os.path.isfile(path + '/' + f):
            # 添加文件
            filelist.append(f)
            # 当一个标志使用，文件夹列表第一个级别不打印
    i_dl = 0
    for dl in dirlist:
        if i_dl == 0:
            i_dl += 1
        else:
            # 打印至控制台，不是第一个的目录
            print '-' * (int(dirlist[0])), dl
            # 打印目录下的所有文件夹和文件，目录级别+1
            printPath((int(dirlist[0]) + 1), path + '/' + dl)
    with open('list.txt', 'w') as filewrite:
        pattern = re.compile(r'.300d')
        for fl in filelist:
            allFileNum += 1
            if re.search(pattern, fl):
                filewithd.append(fl)
            else:
                filewithoutd.append(fl)
        for fl in filewithoutd:
            filewrite.write(fl + '\n')
        filewrite.write('\n')
        for fl in filewithd:
            filewrite.write(fl + '\n')
            # 打印文件
            # filewrite.write(fl+'\n')
            # print '-' * (int(dirlist[0])), fl
            # 随便计算一下有多少个文件


if __name__ == '__main__':
    rootpath = sys.argv[1]
    printPath(1, rootpath)
    print 'the number of the files =', allFileNum
