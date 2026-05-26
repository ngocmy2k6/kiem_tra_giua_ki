gitimport pymssql
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import gradio as gr

# =========================
# KẾT NỐI DATABASE
# =========================
conn = pymssql.connect(
    server='127.0.0.1',
    port='1433',
    user='sa',
    password='Vietnam@123',
    database='master'
)

# Hàm này dùng để lấy dữ liệu
def draw_prediction_plot():
  
    df = pd.read_sql("SELECT Quantity, UnitPrice, (Quantity * UnitPrice) AS Revenue FROM [Order Details]", conn)

    X = df[['Quantity', 'UnitPrice']]
    y = df['Revenue']

    # Chia tập dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện mô hình hồi quy tuyến tính
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Dự báo
    y_pred = model.predict(X_test)

    # Tạo đối tượng hình ảnh để truyền lên giao diện Gradio
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, y_pred, color='blue', alpha=0.6)
    ax.set_xlabel("Doanh thu thực tế")
    ax.set_ylabel("Doanh thu dự báo")
    ax.set_title("Dự báo doanh thu bán hàng")
    
    return fig

# =========================
# GIAO DIỆN GRADIO 
# =========================
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 📊 Ứng dụng Dự báo Doanh thu Bán hàng (Gradio)")
    gr.Markdown("Biểu đồ dưới đây được vẽ dựa trên dữ liệu thật từ SQL Server thông qua mô hình Linear Regression.")
    
    # Tạo một nút bấm trên giao diện
    btn = gr.Button("Tải dữ liệu và Dự báo", variant="primary")
    
    # Component này dùng để hiển thị biểu đồ Matplotlib lên Web
    plot_output = gr.Plot(label="Biểu đồ dự báo")
    
    # Khi bấm nút, hàm draw_prediction_plot sẽ chạy và trả kết quả vào plot_output
    btn.click(fn=draw_prediction_plot, inputs=None, outputs=plot_output)

# =========================
# RUN APP
# =========================
# share=True giúp bạn tạo ra một link public có thể gửi cho bạn bè xem được luôn
app.launch()