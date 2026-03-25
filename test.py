def find(array, value) -> int:
    left = 0
    right = len(array) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if array[mid] == value:
            return mid
        elif array[mid] < value:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
value = 5
print(find(array, value))