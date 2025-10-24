# ...existing code...
def merge(nums1, m: int, nums2, n: int) -> None:
    """
    原地将有序的 nums2 合并进 nums1（nums1 末尾有足够空间）。
    """
    i = m - 1          # nums1 有效区最后一个索引
    j = n - 1          # nums2 最后一个索引
    k = m + n - 1      # nums1 写入位置（从末尾往前）

    while j >= 0:      # 只要 nums2 还有元素就继续
        if i >= 0 and nums1[i] > nums2[j]:
            nums1[k] = nums1[i]
            i -= 1
        else:
            nums1[k] = nums2[j]
            j -= 1
        k -= 1

# 简单测试
num1 = [0]
m = 0
num2 = [1]
n = 1
merge(num1, m, num2, n)
print(num1)  # [1]
# ...existing code...