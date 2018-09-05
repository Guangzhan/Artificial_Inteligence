

import DecisionTree
import numpy as np


if __name__ == '__main__':
    file = open('../input/05.DecisionTree/car.data')
    lines = file.readlines()
    raw_data = np.zeros([len(lines), 7], np.int32)
    for idx in range(len(lines)):
        raw_data[idx] = np.array(lines[idx].split(','), np.int32)
    file.close()
    np.random.shuffle(raw_data)

    data = raw_data.transpose()[0:6].transpose()
    label = raw_data.transpose()[6]


    tree_no_pruning = DecisionTree.Tree(label, data, None)
    tree_pre_pruning = DecisionTree.Tree(label, data, 'Pre')
    tree_post_pruning = DecisionTree.Tree(label, data, 'Post')

    test_count = len(label) // 3
    test_data, test_label = data[0:test_count], label[0:test_count]
    times_no_pruning, times_pre_pruning, times_post_pruning = 0, 0, 0
    print('正在检验结果（共 %d 条验证数据）' % test_count)
    for idx in range(test_count):
        if tree_no_pruning.predict(test_data[idx]) == test_label[idx]:
            times_no_pruning += 1
        if tree_pre_pruning.predict(test_data[idx]) == test_label[idx]:
            times_pre_pruning += 1
        if tree_post_pruning.predict(test_data[idx]) == test_label[idx]:
            times_post_pruning += 1
    print('【未剪枝】：命中 %d 次，命中率 %.2f%%' % (times_no_pruning, times_no_pruning * 100 / test_count))
    print('【预剪枝】：命中 %d 次，命中率 %.2f%%' % (times_pre_pruning, times_pre_pruning * 100 / test_count))
    print('【后剪枝】：命中 %d 次，命中率 %.2f%%' % (times_post_pruning, times_post_pruning * 100 / test_count))