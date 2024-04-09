import numpy as np

def amortization_schedule(principal, annual_interest_rate, years, payment_frequency=12, defer_months=0, rate_changes={}):
    # 將年利率轉換為月利率
    monthly_interest_rate = annual_interest_rate / (payment_frequency * 100)
    
    # 計算總還款期數
    total_payments = years * payment_frequency
    
    # 計算延遲還款期數
    deferred_payments = defer_months * payment_frequency
    
    # 初始化用於存儲結果的數組
    payment_schedule = np.zeros((total_payments, 5))
    
    # 計算每期固定支付金額
    fixed_payment = principal * (monthly_interest_rate * ((1 + monthly_interest_rate) ** total_payments)) / (((1 + monthly_interest_rate) ** total_payments) - 1)
    
    # 剩餘本金從貸款本金開始
    remaining_balance = principal
    
    # 檢查是否有利率變動
    rate_years = sorted(rate_changes.keys())
    rate_change_index = 0
    
    # 逐期循環
    for i in range(total_payments):
        
        # 檢查是否需要調整利率
        if rate_change_index < len(rate_years) and i == rate_years[rate_change_index] * payment_frequency:
            monthly_interest_rate = rate_changes[rate_years[rate_change_index]] / (payment_frequency * 100)
            fixed_payment = remaining_balance * (monthly_interest_rate * ((1 + monthly_interest_rate) ** (total_payments-i))) / (((1 + monthly_interest_rate) ** (total_payments-i)) - 1)
            rate_change_index += 1
            
        # 計算當期利息
        interest_payment = remaining_balance * monthly_interest_rate
        
        # 計算本金支付
        if i < deferred_payments:
            principal_payment = 0  # 在延遲還款期間不支付本金
        else:
            principal_payment = fixed_payment - interest_payment
        
        # 更新剩餘本金
        remaining_balance -= principal_payment
        

        
        # 將當期結果存儲到數組中
        payment_schedule[i] = [i+1, interest_payment, principal_payment, fixed_payment, remaining_balance]
    
    return payment_schedule

def print_amortization_schedule(payment_schedule):
    print("{:<9} {:<11} {:<9} {:<11} {:<15}".format("期數", "利息支付", "本金支付", "固定支付金額", "剩餘本金"))
    for row in payment_schedule:
        print("{:<10d} {:<15.2f} {:<15.2f} {:<15.2f} {:<15.2f}".format(int(row[0]), row[1], row[2], row[3], row[4]))

# 輸入貸款相關資訊
principal = 5000000  # 貸款本金
annual_interest_rate = 8.0  # 年利率
years = 20  # 貸款年限 
payment_frequency = 1  # 每年還款頻率 (每月一次) 整數更改 雙周的話改成26
defer_months = 0  # 延後還款月數

#要使用時新增內容
rate_changes = {4 :7.0}  # 利率變動: 若在第五年降息至 7%  要改成{5-1: 7.0}


# 生成攤還表
schedule = amortization_schedule(principal, annual_interest_rate, years, payment_frequency, defer_months, rate_changes)

# 印出攤還表
print_amortization_schedule(schedule)
