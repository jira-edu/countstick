from escpos.printer import Usb
import usb.core
import usb.util
from promptpay import qrcode
from PIL import Image

def getQr(amount):
    img = qrcode.generate_payload(promptPayNumber, amount)
    qrcode.to_file(img, "QRcode.png")
    imgresize = Image.open('QRcode.png')
    imgresize = imgresize.resize((200, 200)) #ปรับขนาดไฟล์ภาพตามต้องการ
    imgresize.save('QRcode.png')

def printOut(green=0,red=0,purple=0,blue=0):    

    device = usb.core.find(idVendor=0x0483, idProduct=0x070b)
    if device is None:
        messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถเชื่อมต่อเครื่องพิมพ์ใบเสร็จ USB ได้")
        return
    
    now = datetime.now()
    timestamp = now.strftime("%d/%m/20%y - %H:%M:%S")
    
    p = Usb(0x0483, 0x070b)

    p.set(align='center')
    p.image("../images/logo.png", impl="bitImageColumn")
    p.set(bold=True)
    p.text("COUNTSTICK MALA\n")
    p.textln("1/2 Moo 1, Nong Pling, Mueang,")
    p.textln("Kamphaeng Phet 62000")
    p.text("Tel. 055841823\n")
    p.textln(timestamp)

    p.set(align='left')
    p.textln("--------------------------------")
    p.text("Order\t\tQty\tPrice\n")
    p.textln("--------------------------------")
    p.set(bold=False)
  
    p.textln("Mala Soup\t"+"1"+"\t"+str(soupPrice))
    if(green>0):
        greenTotal = green*greenPrice
        p.textln("Green stick\t"+str(green)+"\t"+str(greenTotal))
    if(red>0):
        redTotal = red*redPrice
        p.textln("Red stick\t"+str(red)+"\t"+str(redTotal))
    if(purple>0):
        purpleTotal = purple*purplePrice
        p.textln("Purple stick\t"+str(purple)+"\t"+str(purpleTotal))
    if(blue>0):
        blueTotal = blue*bluePrice
        p.textln("Blue stick\t"+str(blue)+"\t"+str(blueTotal))
    total = greenTotal+redTotal+purpleTotal+blueTotal

    p.set(bold=True)
    p.textln("--------------------------------")
    p.text("TOTAL\t\t\t"+str(total)+"\n")

    getQr(total)

    p.set(align='center')
    p.textln("--------------------------------")
    p.image("QRcode.png", impl="bitImageColumn")    
    p.textln("THANK YOU!")
    p.cut()
    p.close()

if __name__ == "__main__":
    print("Main")
    from tkinter import messagebox
    from datetime import datetime
    greenPrice = 5
    redPrice = 10
    purplePrice = 20
    bluePrice = 30
    soupPrice = 39
    promptPayNumber = '0926582873'
    printOut(green=1,red=1,purple=1,blue=1)
    

