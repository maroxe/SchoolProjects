##############################
# Book order simulator
##############################

import sys
import wx
import time
import numpy as np

WIN_SIZE = (100, 800)

class EventTimer:
    def __init__(self, freq):
        self.last_t = 0
        self.dt = np.random.exponential(scale=1./freq)
        self.freq = freq
        
    def update(self, t):
        if self.dt <= t - self.last_t:
            self.last_t = t
            self.dt = np.random.exponential(scale=1./self.freq)
            return True
        return False

class Queue:
    def __init__(self, display):
        self.size = 0
        self.display = display


    def increment(self):
        self.set_size(self.size + 1)

    def decrement(self):
        self.set_size(self.size - 1)

    def set_size(self, size):
        if size < 0: return
        self.size = size
        self.display.SetLabel( str(size) )

class Price(Queue):
    def set_size(self, size):
        Queue.set_size(self, size)
        self.display.SetLabel( str(size) + '$')


        
class Book:
    def __init__(self, bid_disp, ask_disp, price_disp, steps=1000):
        
        self.ask_queues = [ Queue(disp) for disp in ask_disp ]
        self.bid_queues = [ Queue(disp) for disp in bid_disp ]

        self.best_ask = self.ask_queues[0]
        self.best_bid = self.bid_queues[-1]

        self.price = Price(price_disp)
        
        self.start = time.time()

        freq = 5 # 5 events / s
        self.ask_can_events = EventTimer(freq*2)
        self.ask_lim_events = EventTimer(freq)
        self.bid_can_events = EventTimer(freq*2)
        self.bid_lim_events = EventTimer(freq)

        self.moves = []
        
    def update(self, event):
        t = time.time() - self.start

        if self.ask_can_events.update(t):
            self.best_ask.decrement()
        if self.ask_lim_events.update(t):
            self.best_ask.increment()

        if self.bid_can_events.update(t):
            self.best_bid.decrement()
        if self.bid_lim_events.update(t):
            self.best_bid.increment()            

        if self.best_ask.size <= 0:
            self.price.increment()
            self.reinit()
            self.moves.append(1)
            print np.mean(self.moves)
            

        if self.best_bid.size <= 0:
            self.price.decrement()
            self.reinit()
            self.moves.append(-1)
            print np.mean(self.moves)
        sys.stdout.flush()
        
    def reinit(self):
        x, y = np.random.poisson(lam=5., size=2)
        self.best_ask.set_size(x)
        self.best_bid.set_size(y)
        
        
class Window(wx.Frame):
  
    def __init__(self, parent, title):
        super(Window, self).__init__(parent, title=title, 
                                      size=WIN_SIZE)
            
        self.InitUI()
        self.InitBook()
        self.Centre()
        self.Show()     
        
    def InitUI(self):
    
        panel = wx.Panel(self)
        font = wx.Font(24, wx.SWISS, wx.NORMAL, wx.BOLD, True, 'Verdana')
        
        n = 5
        vbox = wx.BoxSizer(wx.VERTICAL)


        self.bid_queues = [ wx.StaticText(panel, wx.ID_ANY, label='0') for i in range(n)]
        map(lambda q: (q.SetBackgroundColour('red'), q.SetFont(font)), self.bid_queues)
        map(lambda q: vbox.Add(q, 1), self.bid_queues)

        vbox.Add((1, 50))

        self.price = wx.StaticText(panel, wx.ID_ANY, label = '10$')
        self.price.SetFont(font)
        vbox.Add(self.price)
        
        vbox.Add((1, 50))
        
        self.ask_queues = [ wx.StaticText(panel, wx.ID_ANY, label='0') for i in range(n)]
        map(lambda q: (q.SetBackgroundColour('green'), q.SetFont(font)), self.ask_queues)
        map(lambda q: vbox.Add(q, 1), self.ask_queues)

        panel.SetSizer(vbox)

    def InitBook(self):
        self.book = Book(self.bid_queues, self.ask_queues, self.price)
        self.timer = wx.Timer(self)
        self.timer.Start(1)
        self.Bind(wx.EVT_TIMER, self.book.update, self.timer)
    
if __name__ == '__main__':
  
    app = wx.App()
    Window(None, title='Carnet d\'ordre')
    app.MainLoop()

