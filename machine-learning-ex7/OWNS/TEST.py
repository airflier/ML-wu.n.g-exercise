import numpy as np 



def removeElement(nums, val):
    if not nums:
        return 0
    for i in range(len(nums)):
        if nums[i] == val:
            del nums[i]
    return len(nums)

removeElement([0,4,4,0,4,4,4,0,2], 4)


#help(np.argsort)
'''
movies=[]
with open('E:\lessons\ML wu.n.g\coursera-ml-py-master\coursera-ml-py-master\machine-learning-ex8\ex8\movie_ids.txt', 'r', encoding = 'utf-8') as f:
    for line in f:
        a = line.strip()
        b= a.split(' ')[1:]
        c= ' '.join(b)
        movies.append(' '.join(line.strip().split(' ')[1:]))
'''
'''
print(movies)
'''


'''
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0
        i = 1 
        for k in range(1,len(nums)):
            if nums[k] != nums[k-1]:
                nums[i] = nums[k]
                i += 1
        del nums[i:len(nums)-1]
        return i
    
    
print(removeDuplicates([1,2,4,4,5,5,6]))
'''