import PIL
from PIL import Image

imgresize = Image.open('noi.png') #ไฟล์ภาพต้นฉบับ
imgresize = imgresize.resize((250, 250)) #ปรับขนาดไฟล์ภาพตามต้องการ
imgresize.save('qr.png')