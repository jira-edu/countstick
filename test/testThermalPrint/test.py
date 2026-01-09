from escpos.printer import Usb
print("Raedy")
# 0483:070b
p = Usb(0x0483, 0x070b)

p.set(align='center')
p.set(bold=True)
p.text("Countstick Mala\n")

p.set(align='left')
p.set(bold=False)
p.text("blue x 10")
p.text("\t\t")
p.text("200 ฿\n")

p.set(align='left')
p.text("12345678901234567890123456789012\n")

p.set(align='center')
p.image("qr.png", impl="bitImageColumn")
p.text("***Thank you***\n")

p.cut()
print("Finished print!")