
# 具有两种剪枝功能的简单决策树
# 使用信息熵进行划分，剪枝时采用激进策略（即使剪枝后正确率相同，也会剪枝）
import numpy as np

class Tree:
    def __init__(self, label, attr, pruning=None):
        self.__root = None
        boundary = len(label) // 3
        if pruning is None:
            self.__root = self.__run_build(label[boundary:], attr[boundary:],
                                           np.array(range(len(attr.transpose()))), False)
            return
        if pruning == 'Pre':
            self.__root = self.__run_build(label[boundary:], attr[boundary:],
                                           np.array(range(len(attr.transpose()))),
                                           True, attr[0:boundary], label[0:boundary])
        elif pruning == 'Post':
            self.__root = self.__run_build(label[boundary:], attr[boundary:],
                                           np.array(range(len(attr.transpose()))), False)

            self.__post_pruning(self.__root, attr[0:boundary], label[0:boundary])
        else:
            raise RuntimeError('未能识别的参数：%s' % pruning)

    @staticmethod
    # 返回使用特定属性划分下的信息熵之和
    # label: 数据标签
    # attr: 用于进行数据划分的属性
    def __get_info_entropy(label, attr):

        # >>> a = np.array([2,4,6,8,10])
        # >>> np.where(a > 5)
        #(array([2, 3, 4]),)
        result = 0.0
        for this_attr in np.unique(attr):
            sub_label, entropy = label[np.where(attr == this_attr)[0]], 0.0
            for this_label in np.unique(sub_label):
                p = len(np.where(sub_label == this_label)[0]) / len(sub_label)
                entropy -= p * np.log2(p)
            result += len(sub_label) / len(label) * entropy
        return result

    # 递归构建一颗决策树
    # data: 维度为 N * 2 的数组，每行的第 0 个数表示数据索引，第 1 个数表示数据标签
    # attr: 维度为 N * M 的数组，每行表示一条数据的属性，列数随着决策树的构建而变化
    # attr_idx: 表示每个属性在原始属性集合中的索引，用于决策树的构建
    # pre_pruning: 表示是否进行预剪枝
    # check_attr: 在预剪枝时，用作测试数据的属性集合
    # check_label: 在预剪枝时，用作测试数据的验证标签
    def __run_build(self, label, attr, attr_idx, pre_pruning, check_attr=None, check_label=None):
        node, right_count = {}, None
        max_type = np.argmax(np.bincount(label))
        if len(np.unique(label)) == 1:
            # 如果所有样本属于同一类C，则将结点标记为C
            node['type'] = label[0]
            return node
        if attr is None or len(np.unique(attr, axis=0)) == 1:
            # 如果 attr 为空或者 attr 上所有元素取值一致，则将结点标记为样本数最多的类
            node['type'] = max_type
            return node
        attr_trans = np.transpose(attr)
        min_entropy, best_attr = np.inf, None
        # 获取各种划分模式下的信息熵之和（作用和信息增益类似）
        # 并以此为信息，找出最佳的划分属性
        if pre_pruning:
            right_count = len(np.where(check_label == max_type)[0])
        for this_attr in attr_trans:
            entropy = self.__get_info_entropy(label, this_attr)
            if entropy < min_entropy:
                min_entropy = entropy
                best_attr = this_attr
        # branch_attr_idx 表示用于划分的属性是属性集合中的第几个
        branch_attr_idx = np.where((attr_trans == best_attr).all(1))[0][0]
        if pre_pruning:
            sub_right_count = 0
            check_attr_trans = check_attr.transpose()
            # branch_attr_idx 表示本次划分依据的属性属于属性集中的哪一个
            for val in np.unique(best_attr):
                # 按照预划分的特征进行划分，并统计划分后的正确率
                # branch_data_idx 表示数据集中，被划分为 idx 的数据的索引
                branch_data_idx = np.where(best_attr == val)[0]
                # predict_label 表示一次划分以后，该分支数据的预测类别
                predict_label = np.argmax(np.bincount(label[branch_data_idx]))
                # check_data_idx 表示验证集中，属性编号为 branch_attr_idx 的属性值等于 val 的项的索引
                check_data_idx = np.where(check_attr_trans[branch_attr_idx] == val)[0]
                # check_branch_label 表示按照当前特征划分以后，被分为某一类的数据的标签
                check_branch_label = check_label[check_data_idx]
                # 随后判断这些标签是否等于前面计算得到的类别，如果相等，则分类正确
                sub_right_count += len(np.where(check_branch_label == predict_label)[0])
            if sub_right_count <= right_count:
                # 如果划分后的正确率小于等于不划分的正确率，则剪枝
                node['type'] = max_type
                return node
        values = []
        for val in np.unique(best_attr):
            values.append(val)
            branch_data_idx = np.where(best_attr == val)[0]
            if len(branch_data_idx) == 0:
                new_node = {'type': np.argmax(np.bincount(label))}
            else:
                # 按照划分构造新数据，并开始递归
                branch_label = label[branch_data_idx]
                #a = np.array(np.arange(12).reshape(3, 4))

                # a
                # Out[301]:
                # array([[0, 1, 2, 3],
                #        [4, 5, 6, 7],
                #        [8, 9, 10, 11]])
                #
                # np.delete(a, 1, 0)
                # Out[302]:
                # array([[0, 1, 2, 3],
                #        [8, 9, 10, 11]])
                branch_attr = np.delete(attr_trans, branch_attr_idx, axis=0).transpose()[branch_data_idx]
                new_node = self.__run_build(branch_label, branch_attr,
                                            np.delete(attr_idx, branch_attr_idx, axis=0),
                                            pre_pruning, check_attr, check_label)
            node[str(val)] = new_node
        node['attr'] = attr_idx[branch_attr_idx]
        node['type'] = max_type
        node['values'] = values
        return node

    # 后剪枝
    # node: 当前进行判断和剪枝操作的结点
    # check_attr: 用于验证的数据属性集
    # check_label: 用于验证的数据标签集
    def __post_pruning(self, node, check_attr, check_label):
        check_attr_trans = check_attr.transpose()
        if node.get('attr') is None:
            # attr 为 None 代表叶节点
            return len(np.where(check_label == node['type'])[0])
        sub_right_count = 0
        for val in node['values']:
            sub_node = node[str(val)]
            # 找到当前分支点中，数据属于 idx 这一分支的数据的索引
            idx = np.where(check_attr_trans[node['attr']] == val)[0]
            # 使用上述数据，从子节点开始新的递归
            sub_right_count += self.__post_pruning(sub_node, check_attr[idx], check_label[idx])
        if sub_right_count <= len(np.where(check_label == node['type'])[0]):
            for val in node['values']:
                del node[str(val)]
            del node['values']
            del node['attr']
            return len(np.where(check_label == node['type']))
        return sub_right_count

    # 根据构建的决策树预测结果
    # data: 用于预测的数据，维度为M
    # return: 预测结果
    def predict(self, data):
        node = self.__root
        while node.get('attr') is not None:
            attr = node['attr']
            node = node.get(str(data[attr]))
            if node is None:
                return None
        return node.get('type')

    # 以文本形式（类JSON）打印出决策树
    def print_tree(self):
        print(self.__root)
