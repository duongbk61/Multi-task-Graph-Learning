import torch
import pandas as pd
from dataset import Ponzi, Phish

def export_data(dataset_class, path, node_type, output_csv):
    print(f"\nĐang tải tập dữ liệu {dataset_class.__name__}...")
    try:
        dataset = dataset_class(path)
        data = dataset[0]
        
        x = data[node_type].x
        y = data[node_type].y
        
        # Chỉ giữ lại các node đã được gán nhãn thực sự (0: Bình thường, 1: Gian lận)
        # Các node có nhãn = 2 là unlabeled/padded theo dataset.py
        mask = (y == 0) | (y == 1)
        x_labeled = x[mask].numpy()
        y_labeled = y[mask].numpy()
        
        # Cấu hình tên cột chính xác theo bảng mô tả của bạn (14 chiêu)
        columns = [
            # Call (Dim 1-7)
            "Call_Total_Sent", "Call_Total_Recv",
            "Call_Avg_Sent", "Call_Avg_Recv",
            "Call_Balance",
            "Call_Freq_Sent", "Call_Freq_Recv",
            # Transaction (Dim 8-14)
            "Trans_Total_Sent", "Trans_Total_Recv",
            "Trans_Avg_Sent", "Trans_Avg_Recv",
            "Trans_Balance",
            "Trans_Freq_Sent", "Trans_Freq_Recv"
        ]
        
        # Chuyển đổi sang Pandas DataFrame để dễ nhìn
        df = pd.DataFrame(x_labeled, columns=columns)
        df['Label'] = y_labeled
        
        df.to_csv(output_csv, index=False)
        print(f"[THÀNH CÔNG] Đã xuất {len(df)} tài khoản ra file: {output_csv}")
        
        # In thêm một chút mẫu dữ liệu cho thân thiện
        print(df.head())
        
    except Exception as e:
        import traceback
        print(f"Lỗi khi trích xuất dữ liệu: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    # Sinh dữ liệu CSV cho Ponzi
    export_data(Ponzi, './data/Ponzi/', 'CA', 'ponzi_expert_analysis.csv')
    
    # Sinh dữ liệu CSV cho Phish
    export_data(Phish, './data/Phish/', 'EOA', 'phish_expert_analysis.csv')
