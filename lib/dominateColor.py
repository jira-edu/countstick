from scipy import stats
import cv2

def hue_to_color_name(h):
    if (h >= 0 and h <= 10) or (h >= 170 and h <= 179):
        return "แดง"
    elif 11 <= h <= 20:
        return "ส้ม"
    elif 21 <= h <= 35:
        return "เหลือง"
    elif 36 <= h <= 85:
        return "เขียว"
    elif 86 <= h <= 100:
        return "ฟ้า"
    elif 101 <= h <= 130:
        return "น้ำเงิน"
    elif 131 <= h <= 160:
        return "ม่วง"
    elif 161 <= h <= 169:
        return "ชมพู"
    else:
        return "ไม่ทราบสี"

def getColor(img):
    # แปลงภาพจาก BGR เป็น HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ตัดขาว/เหลืองจากไฟ
    mask_color = hsv[:,:,1] > 50   
    mask_no_glare = hsv[:,:,2] < 230
    mask = mask_color & mask_no_glare
    filtered_hsv = hsv[mask]

    # เอาเฉพาะ H
    hues = filtered_hsv[:, 0]   
    mode_result = stats.mode(hues)
    mode_value = mode_result.mode.item()
    mode_count = mode_result.count.item()

    return hue_to_color_name(mode_value)