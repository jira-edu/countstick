import os

# ตั้งค่าโฟลเดอร์ที่ต้องการเปลี่ยนชื่อไฟล์
folder_path = "/path/to/your/folder"

# อ่านไฟล์ทั้งหมดในโฟลเดอร์ (ไม่รวมโฟลเดอร์ย่อย)
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# เรียงลำดับชื่อไฟล์ (เพื่อให้ผลลัพธ์คาดเดาได้)
files.sort()

# หาจำนวนหลัก (digits) ที่ต้องใช้ เช่น 3 หลัก -> 001
total = len(files)
digits = len(str(total))

for index, filename in enumerate(files, start=1):
    # แปลง index เช่น 1 → 001
    new_name_number = str(index).zfill(digits)

    # แยกนามสกุลไฟล์
    _, ext = os.path.splitext(filename)

    # สร้างชื่อใหม่
    new_name = f"{new_name_number}{ext}"

    # path เก่าและใหม่
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)

    # เปลี่ยนชื่อไฟล์
    os.rename(old_path, new_path)

print("Done! Renamed", total, "files.")
