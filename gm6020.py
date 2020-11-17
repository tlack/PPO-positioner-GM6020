# Very crude software to control a DJI RoboMaster GM6020 via CAN with 
# Linux CAN library.

# Warning uses iffy sudo ifconfig junk

DEBUG=0
MOTOR_NUMBER=4
SEND_FREQ=0.01
SEND_TIME=0.02
RECV_SLEEP=0.0005

from math import floor
import os
import time
import threading

import can

def dbg(txt):
    if DEBUG:
        print('>',txt)

class Motor:

    def start(self, motor=MOTOR_NUMBER):
        os.system('sudo ifconfig can0 down')
        os.system('sudo ip link set can0 type can bitrate 1000000')
        os.system('sudo ifconfig can0 up')
        self.bus = can.interface.Bus(channel = 'can0', bustype = 'socketcan_ctypes')
        self.motor = motor
        self.state = None
        x = threading.Thread(target=self.recv_loop)
        x.start()

    def recv_loop(self):
        while True:
            self.recv()
            time.sleep(RECV_SLEEP)

    def recv(self):
        msg = self.bus.recv()
        # print(msg)
        msg = msg.data
        # XXX parse
        pos = msg[0] << 8 | msg[1]
        # print(pos)
        rpm = msg[2] << 8 | msg[3]
        amps = msg[4] << 8 | msg[5]
        temp = msg[6]

        self.state = {"pos": pos, "rpm":rpm, "amps":amps, "temp":temp}
        return self.state
        # print(msg)

    def send(self, data, duration=SEND_TIME, freq=SEND_FREQ):
        self.bus.stop_all_periodic_tasks(remove_tasks=True)
        msg = can.Message(arbitration_id=0x1ff, dlc=8, data=data, extended_id=False, check=True)
        # self.bus.send(msg)
        self.bus.send_periodic(msg, freq, duration) 

    def speed(self, speed):
        if speed < 0:
            if speed < -1: speed = -1
            n = 255 - (127 * (abs(speed)/1.0))
        else:
            if speed > 1: speed = 1
            n = 127 * speed
        n = floor(n)
        dbg(f'speed {speed} -> {n}')
        data = [0] * 8
        data[(self.motor-1)*2] = n
        self.send(data)
        return (speed, n)



