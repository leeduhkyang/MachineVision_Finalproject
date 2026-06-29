# MachineVision_Finalproject[Final_project.py](https://github.com/user-attachments/files/29466260/Final_project.py)


"""
Members:
Nguyễn Thành Nam 		MSSV: 21146028
Li Duy Khang	 	    MSSV: 21146397

Pipelines of project
Final_Project.py — DETECT SÂN (A) + HOMOGRAPHY (C) + VĐV (B) + HEATMAP (D)
========================================================================================
Final project - Machine Vision
Cửa sổ: 1=Camera+VĐV, 2=Mask VĐV (debug), 3=Top-down+chấm điểm, 4=Heatmap (real-time).
Q=thoát (hiện heatmap thành phẩm), SPACE=tạm dừng.
"""

import cv2 as cv
import numpy as np
import sys, os

# ════════════════════════════════════════════════════════
# PHẦN 1: THAM SỐ TOÀN CỤC (Gộp từ CourtA và Pipeline)
# ════════════════════════════════════════════════════════

# Comment/Uncomment 2 dòng dươi để chọn video tạo heatmap 
VIDEO = "1_Lindan.mp4"


# --- Tham số cho Detect Sân ---
OW, OH   = 600, 1300          # kích thước ảnh warp trung gian
EI       = 30                 # bỏ qua biên warp (chống artifact mép)
W2, H2   = 610, 1340          # khung sân chuẩn (tỉ lệ ~ 6.1m : 13.4m)
N_SAMPLE = 40                 # số frame lấy mẫu khắp video
_CLAHE = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# --- Tham số cho Detect VĐV & Heatmap ---
HALF_Y = H2 // 2                       # đường lưới (670) — crop nửa dưới (sân gần)
COURT_W_CM, COURT_H_CM = 610.0, 1340.0

GREEN_LO = np.array([35, 40, 40])
GREEN_HI = np.array([85, 255, 255])

HEATMAP_EVERY = 10                     # cập nhật cửa sổ heatmap mỗi N frame (nhẹ máy)
HEAT_BLUR     = 51                     # độ lan tỏa heatmap (lẻ)
HEAT_ALPHA    = 0.9                    # độ đậm heatmap khi chồng lên sân
SAVE_HEATMAP  = "heatmap_final.png"    # None nếu không muốn lưu


# ════════════════════════════════════════════════════════
# PHẦN 2: CÁC HÀM XỬ LÝ SÂN (COURT DETECTION)
# ════════════════════════════════════════════════════════

def adaptive_court_mask(img):
    """Tự học màu sân từ vùng giữa-dưới khung rồi ngưỡng theo độ giống màu đó."""
    H, W = img.shape[:2]
    hsv = cv.cvtColor(cv.GaussianBlur(img, (5, 5), 0), cv.COLOR_BGR2HSV)
    roi = hsv[int(0.45*H):int(0.80*H), int(0.30*W):int(0.70*W)]
    hm = np.median(roi[:, :, 0]); sm = np.median(roi[:, :, 1]); vm = np.median(roi[:, :, 2])
    lo = np.array([max(hm-18, 0),   max(sm-70, 25), max(vm-90, 25)])
    hi = np.array([min(hm+18, 180), min(sm+70, 255), min(vm+90, 255)])
    mask = cv.inRange(hsv, lo, hi)
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  k, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k, iterations=2)
    return mask

def corners_from_contour(c):
    """4 góc thật của contour = 4 điểm cực theo đường chéo (không cắt góc)."""
    pc = c.reshape(-1, 2).astype(np.float32)
    s = pc.sum(1); d = pc[:, 0] - pc[:, 1]
    return np.array([pc[np.argmin(s)],   # TL
                     pc[np.argmax(d)],   # TR
                     pc[np.argmax(s)],   # BR
                     pc[np.argmin(d)]],  # BL
                    np.float32)

def _find_outer(arr, lo, hi, take, tf=0.35, sm=7):
    """Đường biên NGOÀI CÙNG trong [lo,hi]: đỉnh cục bộ vượt ngưỡng vùng."""
    a = np.convolve(arr, np.ones(sm)/sm, mode='same')
    reg = a[lo:hi]
    if reg.max() <= 0:
        return None
    thr = reg.max() * tf
    cand = [i+lo for i in range(1, len(reg)-1)
            if reg[i] > thr and reg[i] >= reg[i-1] and reg[i] >= reg[i+1]]
    return None if not cand else (cand[0] if take == 'min' else cand[-1])

def _court_quad(img):
    """Khối sân → 4 góc thô. Trả None nếu không tìm được khối đủ lớn."""
    mask = adaptive_court_mask(img)
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv.contourArea)
    if cv.contourArea(c) < 0.05 * img.shape[0] * img.shape[1]:
        return None
    return corners_from_contour(c)

def refine_via_warp(img, quad):
    """Tinh chỉnh 4 góc từ mép thảm về đúng đường biên line. Trả None nếu thất bại."""
    src = quad.astype(np.float32)
    M = cv.getPerspectiveTransform(src, np.array([[0, 0], [OW, 0], [OW, OH], [0, OH]], np.float32))
    Minv = np.linalg.inv(M)
    warp = cv.warpPerspective(img, M, (OW, OH))

    gray = _CLAHE.apply(cv.cvtColor(warp, cv.COLOR_BGR2GRAY))
    white = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv.THRESH_BINARY, 51, -12)
    horiz = cv.morphologyEx(white, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (51, 1)))
    vert  = cv.morphologyEx(white, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 51)))
    col = vert.sum(0) / 255.0
    row = horiz.sum(1) / 255.0

    L = _find_outer(col, EI, int(0.45*OW), 'min'); R = _find_outer(col, int(0.55*OW), OW-EI, 'max')
    T = _find_outer(row, EI, int(0.45*OH), 'min'); B = _find_outer(row, int(0.55*OH), OH-EI, 'max')
    if None in (L, R, T, B):
        return None
    wc = np.array([[L, T], [R, T], [R, B], [L, B]], np.float32).reshape(-1, 1, 2)
    return cv.perspectiveTransform(wc, Minv).reshape(4, 2)

def _is_valid(p, H, W):
    """Kiểm tra hình học để loại frame rác."""
    TL, TR, BR, BL = p
    if not (TL[1] < BL[1] and TR[1] < BR[1]):       # trên cao hơn dưới
        return False
    if not (TL[0] < TR[0] and BL[0] < BR[0]):       # trái đúng bên trái
        return False
    if cv.contourArea(p.astype(np.float32)) < 0.04 * H * W:
        return False
    if not cv.isContourConvex(p.astype(np.int32)):
        return False
    return True

def detect_court(video_path, max_valid=18, frame_step=8, max_read=400, verbose=True):
    """Phát hiện 4 góc sân (median nhiều frame)."""
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        if verbose: print(f"[ERR] Khong mo duoc {video_path}")
        return None

    results = []
    idx = 0
    read = 0
    while read < max_read and len(results) < max_valid:
        if not cap.grab():            
            break
        read += 1
        if idx % frame_step != 0:     
            idx += 1
            continue
        idx += 1
        ok, frame = cap.retrieve()    
        if not ok:
            continue
        H, W = frame.shape[:2]
        quad = _court_quad(frame)
        if quad is None:
            continue
        ref = refine_via_warp(frame, quad)
        if ref is not None and _is_valid(ref, H, W):
            results.append(ref)
    cap.release()

    if len(results) < 3:
        if verbose: print(f"[ERR] Chi {len(results)} frame hop le (doc {read} frame) → khong tin cay.")
        return None
    pts = np.median(np.stack(results, axis=0), axis=0).astype(np.float32)
    if verbose:
        print(f"[OK] {len(results)} frame hop le (doc {read} frame). "
              f"4 goc = {[tuple(p.astype(int)) for p in pts]}")
    return pts

def court_homography(pts_src, w=W2, h=H2):
    """Ma trận homography từ 4 góc sân ảnh → khung sân chuẩn w×h (top-down)."""
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    return cv.getPerspectiveTransform(pts_src.astype(np.float32), dst)


# ════════════════════════════════════════════════════════
# PHẦN 3: CÁC HÀM TIỆN ÍCH (VĐV & HEATMAP)
# ════════════════════════════════════════════════════════

def show(name, img, w):
    h = int(img.shape[0] * w / img.shape[1])
    cv.imshow(name, cv.resize(img, (w, h)))

def get_foot(frame, court_mask):
    H, W = frame.shape[:2]
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    green = cv.inRange(hsv, GREEN_LO, GREEN_HI)
    on_court = cv.bitwise_and(cv.bitwise_not(green), court_mask)
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8))
    clean = cv.morphologyEx(on_court, cv.MORPH_OPEN, k)

    cnts, _ = cv.findContours(clean, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    for c in cnts:
        a = cv.contourArea(c)
        if a < 400:
            continue
        x, y, w, h = cv.boundingRect(c)
        ar = h / float(w + 1e-5)
        if ar < 0.6 or ar > 4.5 or w > W * 0.25:
            continue
        m = cv.moments(c)
        if m["m00"] == 0:
            continue
        cy = m["m01"] / m["m00"]
        if cy > H / 2 and a > best_area:
            best_area = a
            best = (x, y, w, h)
    if best is None:
        return None, None, clean
    x, y, w, h = best
    return (x + w // 2, y + h), best, clean

def draw_court(w, h):
    GREEN = (60, 120, 40); WHITE = (255, 255, 255); t = 2
    court = np.full((h, w, 3), GREEN, np.uint8)
    m = 100
    x_out_l, x_out_r = 0, w - 1
    x_in_l, x_in_r = int(0.46*m), w - int(0.46*m)
    x_center = w // 2
    y_top, y_bot = 0, h - 1
    y_net = h // 2
    y_short_top = y_net - int(1.98*m); y_short_bot = y_net + int(1.98*m)
    y_long_top = int(0.76*m); y_long_bot = h - int(0.76*m)
    cv.rectangle(court, (x_out_l, y_top), (x_out_r, y_bot), WHITE, t)
    cv.line(court, (x_in_l, y_top), (x_in_l, y_bot), WHITE, t)
    cv.line(court, (x_in_r, y_top), (x_in_r, y_bot), WHITE, t)
    cv.line(court, (x_out_l, y_net), (x_out_r, y_net), WHITE, t)
    cv.line(court, (x_out_l, y_short_top), (x_out_r, y_short_top), WHITE, t)
    cv.line(court, (x_out_l, y_short_bot), (x_out_r, y_short_bot), WHITE, t)
    cv.line(court, (x_out_l, y_long_top), (x_out_r, y_long_top), WHITE, t)
    cv.line(court, (x_out_l, y_long_bot), (x_out_r, y_long_bot), WHITE, t)
    cv.line(court, (x_center, y_short_top), (x_center, y_long_top), WHITE, t)
    cv.line(court, (x_center, y_short_bot), (x_center, y_long_bot), WHITE, t)
    return court

def overlay_heatmap_on_court(heat_accum, w, h, blur=HEAT_BLUR, alpha=HEAT_ALPHA):
    blurred = cv.GaussianBlur(heat_accum, (blur, blur), 0)
    if blurred.max() > 0:
        norm = np.power(blurred / blurred.max(), 0.5)          # sqrt-gamma kéo vùng nhạt lên
        heat_color = cv.applyColorMap((norm*255).astype(np.uint8), cv.COLORMAP_JET)
        heat_mask = (blurred / blurred.max() * alpha)[:, :, None]
    else:
        heat_color = np.zeros((h, w, 3), np.uint8)
        heat_mask = np.zeros((h, w, 1), np.float32)
    court = draw_court(w, h).astype(np.float32)
    out = court * (1 - heat_mask) + heat_color.astype(np.float32) * heat_mask
    return out.astype(np.uint8)


# ════════════════════════════════════════════════════════
# PHẦN 4: MAIN PIPELINE THỰC THI CHÍNH
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ===== 1. DETECT SÂN (1 LẦN) =====
    print("[...] Dang detect san...")
    pts = detect_court(VIDEO)
    if pts is None:
        print("[ERR] Khong detect duoc san. Dung.")
        sys.exit()
        
    M = court_homography(pts)
    pts_i = pts.astype(np.int32)

    cap = cv.VideoCapture(VIDEO)
    ok0, f0 = cap.read()
    if not ok0:
        print("[ERR] Khong the doc video roi. Dung.")
        sys.exit()
        
    H, W = f0.shape[:2]
    court_mask = np.zeros((H, W), np.uint8)
    cv.fillPoly(court_mask, [pts_i], 255)

    heat_accum = np.zeros((H2, W2), np.float32)    # ma trận tích lũy heatmap

    # ===== 2. PHÁT VIDEO =====
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    print("[OK] Phat video. Q=thoat (hien heatmap), SPACE=tam dung.")
    paused = False
    frame_idx = 0
    
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            foot, bbox, dbg = get_foot(frame, court_mask)

            cam = frame.copy()
            cv.polylines(cam, [pts_i.reshape(-1, 1, 2)], True, (0, 0, 255), 2)
            for p in pts_i:
                cv.circle(cam, tuple(p), 6, (0, 255, 255), -1)

            top = cv.warpPerspective(frame, M, (W2, H2))

            if foot is not None:
                x, y, w, h = bbox
                cv.rectangle(cam, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv.circle(cam, foot, 9, (0, 255, 255), -1)
                pt = cv.perspectiveTransform(
                    np.array([[[float(foot[0]), float(foot[1])]]], np.float32), M)[0][0]
                ix, iy = int(round(pt[0])), int(round(pt[1]))
                x_cm = pt[0] * (COURT_W_CM / W2); y_cm = pt[1] * (COURT_H_CM / H2)
                if 0 <= ix < W2 and 0 <= iy < H2:
                    heat_accum[iy, ix] += 1.0                 # TÍCH LŨY HEATMAP
                    cv.circle(top, (ix, iy), 12, (0, 0, 255), -1)
                    cv.circle(top, (ix, iy), 14, (255, 255, 255), 2)
                cv.putText(cam, f"X={x_cm:.0f}cm Y={y_cm:.0f}cm", (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            show("1. Camera + VDV", cam, 800)
            show("2. Mask VDV (debug)", dbg, 360)
            show("3. Top-down (nua duoi)", top[HALF_Y:H2, 0:W2], 320)

            if frame_idx % HEATMAP_EVERY == 0:
                heat_img = overlay_heatmap_on_court(heat_accum, W2, H2)
                show("4. Heatmap (real-time)", heat_img[HALF_Y:H2, 0:W2], 320)

        key = cv.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            paused = not paused

    cap.release()

    # ===== 3. HEATMAP THÀNH PHẨM =====
    final = overlay_heatmap_on_court(heat_accum, W2, H2)[HALF_Y:H2, 0:W2]   # nửa dưới
    print(f"[OK] Tong diem tich luy: {heat_accum.sum():.0f} | o cao nhat: {heat_accum.max():.0f}")
    if SAVE_HEATMAP:
        cv.imwrite(SAVE_HEATMAP, final)
        print(f"[OK] Da luu: {SAVE_HEATMAP}")
    show("HEATMAP thanh pham (nua duoi)", final, 360)
    print("Nhan phim bat ky de thoat...")
    cv.waitKey(0)
    cv.destroyAllWindows()
    print("[OK] Ket thuc.")
