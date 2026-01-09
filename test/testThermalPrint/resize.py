import PIL
from PIL import Image

imgresize = Image.open('logo-countstick.png') #ไฟล์ภาพต้นฉบับ
imgresize = imgresize.resize((200, 200)) #ปรับขนาดไฟล์ภาพตามต้องการ
imgresize.save('logo.png')