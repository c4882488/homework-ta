"""千芬＿機器學習"""

"""
簡介
1.缺失值
1.5 欄位特徵設計
2.label 轉 代碼
3.分 訓練資料、學習資料
4.特徵縮放
"""

#套件
import pandas as pd

class Data:
    
    def data_clean(self,output_log=False):
        """資料清洗"""
        df = pd.read_csv("./salary.csv")

        print(df.isnull().sum())
        print("資料無任一缺失值")
        return df

    def data_intro(self):
        """資料特徵介紹"""
        df = self.data_clean()
        print("數值資料詳細")
        numric_detail = df.describe()
        print("類別資料詳細")
        for column in df.select_dtypes(include='object').columns:
            print(f"\n{column}:")
            print(df[column].value_counts())

    def data_encoder(self):
        """資料預處理，含分割資料、資料轉換"""
        df = self.data_clean()
        for column in df.select_dtypes(include='object').columns:
            df[column] = "1"

    def main(self):
        """執行"""
        print("1.資料清洗")
        self.data_clean()
        print("2.資料特徵介紹")
        self.data_intro()
        print("3.LabelToNumeric")
        self.data_encoder()


if __name__ == "__main__":
    Data().main()
