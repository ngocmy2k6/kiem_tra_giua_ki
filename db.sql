import pyodbc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Cấu hình chuỗi kết nối SQL Server (Đã sửa lại đúng cú pháp của pyodbc)
conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"  # Hoặc {SQL Server} tùy thuộc vào driver có sẵn trên Codespaces
    "SERVER=127.0.0.1,1433;"
    "DATABASE=master;"
    "UID=sa;"
    "PWD=Vietnam@123;"
)

try:
    conn = pyodbc.connect(conn_str)
    print("=== Kết nối SQL Server thành công! ===")
except Exception as e:
    print(f"Lỗi kết nối: {e}")
    exit()

# 2. Đọc dữ liệu từ SQL
query = "SELECT Quantity, UnitPrice, (Quantity * UnitPrice) AS Revenue FROM [Order Details]"
df = pd.read_sql(query, conn)

# --- IN DỮ LIỆU RA MÀN HÌNH THEO YÊU CẦU ---
print("\n=== 5 DÒNG DỮ LIỆU ĐẦU TIÊN ===")
print(df.head())

print("\n=== THÔNG TIN TỔNG QUAN VỀ DỮ LIỆU ===")
print(df.info())
# --------------------------------------------

X = df[['Quantity', 'UnitPrice']]
y = df['Revenue']

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Dự báo
y_pred = model.predict(X_test)

# In thử một vài kết quả dự báo
print("\n=== KẾT QUẢ DỰ BÁO (5 dòng đầu) ===")
df_result = pd.DataFrame({'Thực tế': y_test, 'Dự báo': y_pred})
print(df_result.head())

# Biểu đồ kết quả (Lưu ý về Codespaces ở bên dưới)
print("\nĐang hiển thị biểu đồ...")
plt.scatter(y_test, y_pred)
plt.xlabel("Doanh thu thực tế")
plt.ylabel("Doanh thu dự báo")
plt.title("Dự báo doanh thu bán hàng")
plt.show()