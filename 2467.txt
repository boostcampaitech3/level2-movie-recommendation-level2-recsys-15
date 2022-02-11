import sys
input = sys.stdin.readline

n = int(input())
n_list = list(map(int, input().split()))

left = 0
right = n-1
ans_left, ans_right = 0, 0
sum_lr = 0
mini = sys.maxsize

while left < right:
    sum_lr = n_list[left] + n_list[right]
    if abs(sum_lr) < mini: # 산성,알칼리성 합의 절대값이 mini보다 작으면
        ans_left, ans_right = left, right
        mini = abs(sum_lr) # mini값 업데이트
     
    if sum_lr > 0:
        right -= 1 #알칼리성 한 칸 앞으로
    elif sum_lr < 0:
        left += 1 # 산성 한 칸 뒤로
    else:
        break
       
print(n_list[ans_left], n_list[ans_right])
    
