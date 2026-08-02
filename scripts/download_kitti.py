import torchvision
import os

def download_kitti():
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Bắt đầu tải dataset KITTI vào thư mục: {data_dir}...")
    try:
        # download=True sẽ yêu cầu torchvision tự động tải về
        dataset = torchvision.datasets.Kitti(root=data_dir, download=True)
        print("Đã khởi tạo xong Dataset! Tổng số mẫu:", len(dataset))
    except Exception as e:
        print("Lỗi khi dùng torchvision tải dataset KITTI:", e)

if __name__ == "__main__":
    download_kitti()
