import numpy as np
import pynput


class Keyboard_listener():
    """Background thread for getting (x,y) velocity reference from the keyboard arrows.
    The arrows provide the velocity direction, while the norm is fixed through set_v().
    To smoothen the reference, the returned vel is averaged over a sliding window of size vhist_size.
    Listens to ESC to stop simulation scripts (since plt messes up with CTRL+C).
    Listens to 's' key to populate a toggle flag.
    """
    def __init__(self, vhist_size=3, v_default=1):
        self.vx = 0.
        self.vy = 0.
        self.vnorm = v_default
        self.vhist = np.zeros((2, vhist_size))
        self.vel_thread = pynput.keyboard.Listener(on_press = self.on_press, on_release = self.on_release)
        self.vel_thread.start()
        self.stop = False
        self.flag = False;


    def __del__(self):
        self.vel_thread.join()


    def set_v(self, v):
        self.vnorm = v


    def on_press(self,key):
        try:
            if key == pynput.keyboard.Key.up:
                self.vx = 1
            elif key == pynput.keyboard.Key.left:
                self.vy = 1
            elif key == pynput.keyboard.Key.down:
                self.vx = -1
            elif key == pynput.keyboard.Key.right:
                self.vy = -1
            elif key == pynput.keyboard.Key.esc:
                self.stop = True
            elif key == pynput.keyboard.Key.tab:
                self.flag = not self.flag
                print('flag:', self.flag)
        except:
            pass


    def on_release(self,key):
        if key == pynput.keyboard.Key.up or key == pynput.keyboard.Key.down:
            self.vx = 0
        elif key == pynput.keyboard.Key.left or key == pynput.keyboard.Key.right:
            self.vy = 0


    def get_v(self):
        v = np.array([self.vx, self.vy])
        norm = np.linalg.norm(v)
        v = np.zeros(2) if norm == 0 else v/norm
        self.vhist[:,:-1] = self.vhist[:,1:]
        self.vhist[:,-1] = v*self.vnorm
        return np.mean(self.vhist, axis=1)


    def get_stop(self):
        return self.stop


    def get_flag(self):
        return 1. if self.flag else 0.
