
##############################
# Book order simulator
##############################

import wx
import time
import numpy as np

class EventTimer:
    def __init__(self, freq, num_events=1000):
        self.events = np.random.exponential(scale=1./freq, size=num_events)
        self.last_t = 0
        self.last_event = 0

    def update(self, t):
        dt = self.events[self.last_event]
        if dt <= t - self.last_t:
            self.last_t = t
            self.last_event += 1
            return True
        return False

class Queue:
    def __init__(self, display, freq_can, freq_lim):
        self.size = 0
        self.display = display
        self.can_events = EventTimer(freq_can)
        self.lim_events = EventTimer(freq_lim)

    def update(self, t):
        if self.can_events.update(t):
            self.decrement()
        if self.lim_events.update(t):
            self.increment()            
        
    def increment(self):
        self.set_size(self.size + 1)

    def decrement(self):
        self.set_size(self.size - 1)

    def set_size(self, size):
        self.size = size
        self.display.SetLabel( str(size) )
        
class Book:
    def __init__(self, displays):
        freq = 5
        self.queues = [ Queue(disp, freq, freq) for disp in displays ]
        self.start = time.time()
        
    def update(self, event):
        t = time.time() - self.start
        map(lambda q: q.update(t), self.queues)

        
class Window(wx.Frame):
  
    def __init__(self, parent, title):
        super(Window, self).__init__(parent, title=title, 
                                      size=(800, 600))
            
        self.InitUI()
        self.InitBook()
        self.Centre()
        self.Show()     
        
    def InitUI(self):
    
        panel = wx.Panel(self)
        font = wx.Font(24, wx.SWISS, wx.NORMAL, wx.BOLD, True, 'Verdana')
        
        vbox = wx.BoxSizer(wx.VERTICAL)
        n = 5
        
        self.bid_queues = [ wx.StaticText(panel, wx.ID_ANY, label='0') for i in range(n)]
        map(lambda q: q.SetFont(font), self.bid_queues)
        map(lambda q: vbox.Add(q, 1), self.bid_queues)

        panel.SetSizer(vbox)

    def InitBook(self):
        self.book = Book(self.bid_queues)
        self.timer = wx.Timer(self)
        self.timer.Start(100)
        self.Bind(wx.EVT_TIMER, self.book.update, self.timer)
    
if __name__ == '__main__':
  
    app = wx.App()
    Window(None, title='Carnet d\'ordre')
    app.MainLoop()

