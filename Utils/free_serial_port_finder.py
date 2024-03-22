import serial.tools.list_ports
def findSerialPort():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        try:
            ser = serial.Serial(p.device, 115200)
            ser.close()
            return p.device
        except serial.SerialException:
            pass
    return None
