def tow_sum_with_dict(nums, target):
    _dict = {}
    for i, m in enumerate(nums):
        _dict[m] = i

    for i, m in enumerate(nums):
        j = _dict.get(target - m)
        if j is not None and i != j:
            return [i, j]
print(tow_sum_with_dict([1,2,3,4,5,6],9))

