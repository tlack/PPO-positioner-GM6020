ITERS = 500

from math import floor
import os
import can

def send(bus, data):
    bus.stop_all_periodic_tasks(remove_tasks=True)
    msg = can.Message(arbitration_id=0x1ff, dlc=8, data=data, extended_id=False, check=True)
    # bus.send(msg)
    bus.send_periodic(msg, 0.1, 0.5) 
    print("Message sent on {}".format(bus.channel_info))

def speed(bus, speed):
    if speed < 0:
        if speed < -1: speed = -1
        n = 255 - (127 * (abs(speed)/1.0))
    else:
        if speed > 1: speed = 1
        n = 127 * speed
    n = floor(n)
    print(speed, n)
    send(bus, [0,0, 0,0, 0,0, n, 0])

def recv(bus):
    msg = bus.recv()
    # print(msg)

os.system('sudo ifconfig can0 down')
os.system('sudo ip link set can0 type can bitrate 1000000')
os.system('sudo ifconfig can0 up')
bus = can.interface.Bus(channel = 'can0', bustype = 'socketcan_ctypes')

while True:
    req = input('enter speed -1 .. 1: ')
    print(req)
    if req == '': break
    req = float(req)
    i = 0
    speed(bus, req)
    while i < ITERS:
        recv(bus)
        i += 1

os.system('sudo ifconfig can0 down')

