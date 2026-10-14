import random
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential

# -------------------- Cấu hình --------------------
SEED = 42
WINDOW_SIZE = 60
EPOCHS = 10
BATCH_SIZE = 32

np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

st.set_page_config(
    page_title="Dự báo giá cổ phiếu bằng LSTM",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Dự báo giá cổ phiếu bằng LSTM")
st.write(
    "Nhập mã cổ phiếu Việt Nam để lấy dữ liệu lịch sử, huấn luyện mô hình "
    "LSTM và xem dự báo cho 10 phiên giao dịch tiếp theo."
)
st.caption(
    "Lưu ý: Đây là kết quả mô hình tham khảo, không phải lời khuyên đầu tư. "
    "Dự báo giá tương lai có độ bất định cao."
)


def normalize_ticker(raw_ticker: str) -> str:
    """Cho phép nhập VNM hoặc VNM.VN."""
    ticker = raw_ticker.strip().upper().replace(" ", "")
    if not ticker:
        return ""
    if "." not in ticker:
        ticker += ".VN"
    return ticker


@st.cache_data(ttl=1800, show_spinner=False)
def download_history(ticker: str) -> pd.DataFrame:
    end_date = date.today() + timedelta(days=1)
    df = yf.download(
        ticker,
        start="2018-01-01",
        end=end_date.isoformat(),
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        return pd.DataFrame()

    # yfinance đôi khi trả về MultiIndex columns, kể cả khi chỉ có một mã.
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            close = df.xs("Close", axis=1, level=0)
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        else:
            return pd.DataFrame()
    else:
        if "Close" not in df.columns:
            return pd.DataFrame()
        close = df["Close"]

    result = pd.DataFrame({"Close": pd.to_numeric(close, errors="coerce")})
    result = result.dropna()
    result.index = pd.to_datetime(result.index)
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result


def create_sequences(values: np.ndarray, window_size: int = WINDOW_SIZE):
    X, y = [], []
    for i in range(window_size, len(values)):
        X.append(values[i - window_size:i])
        y.append(values[i])
    return np.asarray(X), np.asarray(y)


def train_and_predict(df: pd.DataFrame):
    data = df[["Close"]].to_numpy(dtype=np.float32)

    if len(data) < 150:
        raise ValueError(
            "Dữ liệu lịch sử quá ít để huấn luyện ổn định. "
            "Hãy kiểm tra mã cổ phiếu hoặc chọn mã có lịch sử dài hơn."
        )

    train_size = int(0.8 * len(data))
    train_raw = data[:train_size]
    test_raw = data[train_size:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)

    X_train, y_train = create_sequences(train_scaled)
    if len(X_train) < 20:
        raise ValueError("Không đủ dữ liệu tạo chuỗi huấn luyện.")

    # Giữ phần cuối tập huấn luyện làm tập validation theo thứ tự thời gian.
    val_size = max(1, int(0.1 * len(X_train)))
    if len(X_train) - val_size < 1:
        raise ValueError("Không đủ dữ liệu để chia tập huấn luyện và kiểm định.")

    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train_final, y_train_final = X_train[:-val_size], y_train[:-val_size]

    model = Sequential([
        Input(shape=(WINDOW_SIZE, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train_final,
        y_train_final,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0,
    )

    # Test sequences dùng 60 phiên cuối train làm ngữ cảnh,
    # sau đó dự báo các phiên trong tập test.
    test_combined = np.vstack([train_scaled[-WINDOW_SIZE:], test_scaled])
    X_test, y_test = create_sequences(test_combined)
    if len(X_test) == 0:
        raise ValueError("Không đủ dữ liệu cho tập kiểm tra.")

    test_pred_scaled = model.predict(X_test, verbose=0)
    test_pred = scaler.inverse_transform(test_pred_scaled).ravel()
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_test_real, test_pred)))
    mae = float(mean_absolute_error(y_test_real, test_pred))
    r2 = float(r2_score(y_test_real, test_pred)) if len(y_test_real) > 1 else float("nan")

    # Dự báo cuốn chiếu 10 phiên; mỗi dự báo được đưa vào cửa sổ kế tiếp.
    last_window = scaler.transform(data[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE, 1)
    future_scaled = []
    current_window = last_window.copy()

    for _ in range(10):
        next_pred = model.predict(current_window, verbose=0)
        future_scaled.append(float(next_pred[0, 0]))
        current_window = np.concatenate(
            [current_window[:, 1:, :], next_pred.reshape(1, 1, 1)],
            axis=1,
        )

    future_prices = scaler.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)
    ).ravel()

    # Tạo 10 ngày làm việc dự kiến; đây là ngày thường, chưa loại trừ ngày lễ.
    future_dates = pd.bdate_range(
        start=pd.Timestamp(df.index[-1]).normalize() + pd.Timedelta(days=1),
        periods=10,
    )

    return {
        "model": model,
        "history": history,
        "scaler": scaler,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "test_dates": df.index[train_size:],
        "y_test_real": y_test_real,
        "test_pred": test_pred,
        "future_dates": future_dates,
        "future_prices": future_prices,
        "last_close": float(data[-1, 0]),
        "last_date": pd.Timestamp(df.index[-1]),
    }


with st.form("prediction_form"):
    ticker_input = st.text_input(
        "Mã cổ phiếu",
        value="VNM",
        placeholder="Ví dụ: VNM, FPT, HPG",
        help="Với cổ phiếu Việt Nam, nhập mã như VNM. Bạn cũng có thể nhập VNM.VN.",
    )
    submitted = st.form_submit_button("🔮 Dự đoán", type="primary", use_container_width=True)

if submitted:
    ticker = normalize_ticker(ticker_input)

    if not ticker:
        st.error("Vui lòng nhập mã cổ phiếu.")
    else:
        try:
            with st.spinner(f"Đang tải dữ liệu {ticker}..."):
                df = download_history(ticker)

            if df.empty:
                st.error(
                    f"Không tìm thấy dữ liệu cho mã **{ticker}**. "
                    "Hãy kiểm tra mã cổ phiếu hoặc thử lại sau."
                )
            else:
                with st.spinner("Đang huấn luyện LSTM và tạo dự báo..."):
                    result = train_and_predict(df)

                last_close = result["last_close"]
                last_forecast = float(result["future_prices"][-1])
                change = last_forecast - last_close
                change_pct = (change / last_close * 100) if last_close else 0

                st.success(f"Đã hoàn thành dự báo cho {ticker}.")
                st.caption(
                    f"Dữ liệu đóng cửa gần nhất: "
                    f"{result['last_date'].strftime('%d/%m/%Y')}"
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Giá đóng cửa gần nhất", f"{last_close:,.0f} VND")
                c2.metric(
                    "Dự báo phiên thứ 10",
                    f"{last_forecast:,.0f} VND",
                    f"{change_pct:+.2f}% so với giá gần nhất",
                )
                c3.metric("RMSE trên tập kiểm tra", f"{result['rmse']:,.2f} VND")

                st.subheader("1. Kết quả dự báo bằng chữ")
                direction = (
                    "cao hơn" if last_forecast > last_close
                    else "thấp hơn" if last_forecast < last_close
                    else "bằng"
                )
                st.write(
                    f"Theo mô hình LSTM, giá đóng cửa gần nhất của **{ticker}** là "
                    f"**{last_close:,.0f} VND**. Sau 10 phiên giao dịch dự kiến, "
                    f"mô hình cho giá khoảng **{last_forecast:,.0f} VND/cổ phiếu**, "
                    f"{direction} giá gần nhất **{abs(change):,.0f} VND "
                    f"({abs(change_pct):.2f}%)**."
                )
                st.warning(
                    "Đây là dự báo cuốn chiếu dựa trên giá đóng cửa lịch sử. "
                    "Mô hình chỉ dùng giá đóng cửa, chưa xét tin tức, kết quả kinh doanh, "
                    "thanh khoản, yếu tố thị trường hoặc các sự kiện bất thường."
                )

                st.subheader("2. Bảng dự báo 10 phiên giao dịch")
                forecast_table = pd.DataFrame({
                    "Phiên": np.arange(1, 11),
                    "Ngày dự kiến": result["future_dates"].strftime("%d/%m/%Y"),
                    "Giá dự báo (VND)": np.round(result["future_prices"], 0).astype(int),
                })
                st.dataframe(forecast_table, use_container_width=True, hide_index=True)

                st.subheader("3. Biểu đồ dự báo 10 phiên tới")
                fig, ax = plt.subplots(figsize=(12, 5))
                historical_tail = df["Close"].tail(90)
                ax.plot(
                    historical_tail.index,
                    historical_tail.values,
                    label="Giá đóng cửa lịch sử",
                    linewidth=2,
                )
                forecast_x = pd.DatetimeIndex(
                    [result["last_date"]] + list(result["future_dates"])
                )
                forecast_y = np.concatenate(
                    [[last_close], result["future_prices"]]
                )
                ax.plot(
                    forecast_x,
                    forecast_y,
                    marker="o",
                    linestyle="--",
                    label="Dự báo LSTM",
                )
                ax.set_title(f"{ticker}: giá lịch sử và dự báo 10 phiên")
                ax.set_xlabel("Ngày")
                ax.set_ylabel("Giá (VND)")
                ax.legend()
                ax.grid(True, alpha=0.25)
                fig.autofmt_xdate()
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("4. Kiểm tra mô hình trên dữ liệu chưa dùng để huấn luyện")
                m1, m2, m3 = st.columns(3)
                m1.metric("MAE", f"{result['mae']:,.2f} VND")
                m2.metric("RMSE", f"{result['rmse']:,.2f} VND")
                m3.metric("R²", f"{result['r2']:.4f}" if np.isfinite(result["r2"]) else "Không xác định")

                st.caption(
                    "MAE/RMSE càng thấp thường càng tốt khi so sánh trên cùng mã và cùng giai đoạn. "
                    "R² có thể âm; các chỉ số kiểm tra trong quá khứ không bảo đảm độ chính xác "
                    "của dự báo tương lai."
                )

                with st.expander("Xem biểu đồ Loss khi huấn luyện"):
                    fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
                    ax_loss.plot(result["history"].history["loss"], label="Train Loss")
                    ax_loss.plot(result["history"].history["val_loss"], label="Validation Loss")
                    ax_loss.set_title("Loss: tập huấn luyện và tập kiểm định")
                    ax_loss.set_xlabel("Epoch")
                    ax_loss.set_ylabel("Mean Squared Error")
                    ax_loss.legend()
                    ax_loss.grid(True, alpha=0.25)
                    st.pyplot(fig_loss)
                    plt.close(fig_loss)

                csv = forecast_table.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "⬇️ Tải bảng dự báo CSV",
                    data=csv,
                    file_name=f"du_bao_{ticker.replace('.', '_')}.csv",
                    mime="text/csv",
                )

        except Exception as exc:
            st.error(f"Không thể hoàn thành dự báo: {exc}")
            st.info(
                "Hãy kiểm tra kết nối Internet, mã cổ phiếu và phiên bản thư viện. "
                "Lần huấn luyện đầu có thể mất vài phút."
            )

st.divider()
st.caption(
    "Nguồn dữ liệu: Yahoo Finance thông qua yfinance. "
    "Ngày dự báo là ngày làm việc dự kiến, chưa loại trừ ngày nghỉ lễ của thị trường."
)
