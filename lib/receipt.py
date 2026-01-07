from escpos.printer import Usb
import usb.core
import usb.util
from promptpay import qrcode
from PIL import Image
from tkinter import messagebox
from datetime import datetime

promptPayNumber = "0926582873"

def getQr(amount):
    img = qrcode.generate_payload(promptPayNumber, amount)
    qrcode.to_file(img, "QRcode.png")
    imgresize = Image.open('QRcode.png')
    imgresize = imgresize.resize((200, 200)) #ปรับขนาดไฟล์ภาพตามต้องการ
    imgresize.save('QRcode.png')

def printOut(soupPrice,greenPrice,redPrice,purplePrice,bluePrice,green=0,red=0,purple=0,blue=0):

    device = usb.core.find(idVendor=0x0483, idProduct=0x070b)
    if device is None:
        messagebox.showerror("ข้อผิดพลาด", f"ไม่สามารถเชื่อมต่อเครื่องพิมพ์ใบเสร็จ USB ได้")
        return
    
    now = datetime.now()
    timestamp = now.strftime("%d/%m/20%y - %H:%M:%S")
    
    p = Usb(0x0483, 0x070b)

    p.set(align='center')
    p.image("images/logo.png", impl="bitImageColumn")
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
  
    p.textln(f"Soup {soupPrice} Baht\t1\t{soupPrice}")
    # p.textln("Mala Soup\t"+"1"+"\t"+str(soupPrice))
    greenTotal = green*greenPrice
    if(green>0):
        p.textln(f"Green {greenPrice} Baht\t{green}\t{greenTotal}")
    redTotal = red*redPrice
    if(red>0):
        p.textln(f"Red {redPrice} Baht\t{red}\t{redTotal}")
    purpleTotal = purple*purplePrice
    if(purple>0):
        p.textln(f"Purple {purplePrice} Baht\t{purple}\t{purpleTotal}")
    blueTotal = blue*bluePrice
    if(blue>0):
        p.textln(f"Blue {bluePrice} Baht\t{blue}\t{blueTotal}")
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
    

