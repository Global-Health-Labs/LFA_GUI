import os
import tkinter as tk
import tkinter.ttk as ttk
from ttkwidgets import TickScale
from PIL import Image, ImageTk
import pandas as pd
from tkinter import Menu, Label, Toplevel, Entry, filedialog, Button, simpledialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter,find_peaks
# class popupWindow(object):

#     value_global=""
#     def __init__(self, master):
#         self.top=Toplevel(master)
#         self.top.columnconfigure(0, weight=1)
#         self.top.columnconfigure(1,weight=3)
#         self.value=""

#         self.l=Label(self.top,text="Sample Label")
#         self.l.grid(column=0,row=0,sticky=tk.W, padx=5, pady=5)

#         self.e=Entry(self.top)
#         self.e.grid(column=1, row=0, sticky=tk.E, padx=5, pady=5)

#         self.b=Button(self.top,text='Ok',command=self.cleanup)
#         self.b.grid(column=1, row=3, sticky=tk.E, padx=5, pady=5)

#     def cleanup(self):
#         self.value=self.e.get()
#         popupWindow.value_global=self.value
#         self.top.destroy()

class MousePositionTracker(tk.Frame):
    """ Tkinter Canvas mouse position widget. """

    def __init__(self, app, root):
        # self.canvas = app.canvas
        self.root = root
        self.app = app
        self.width = self.app.canvas.winfo_width()
        self.height = self.app.canvas.winfo_height()
        self.count = 0
        # self.n = 3
        self.reset()

        # Options for areas outside rectanglar selection.
        select_outside = dict(dash=(2, 2), fill='red', outline='', state=tk.HIDDEN, stipple='gray25')
        # Separate options for area inside rectanglar selection.
        select_inside = dict(dash=(2, 2), fill='', outline='white', state=tk.HIDDEN)
        # Initial extrema of inner and outer rectangles.
        i_min_x, i_min_y, i_max_x, i_max_y = 0, 0, 1, 1
        o_min_x, o_min_y, o_max_x, o_max_y = 0, 0, self.width, self.height
        self.rects = (
            # Area *outside* selection (inner) rectangle.
            self.app.canvas.create_rectangle(o_min_x, o_min_y,  o_max_x, i_min_y, **select_outside),
            self.app.canvas.create_rectangle(o_min_x, i_min_y,  i_min_x, i_max_y, **select_outside),
            self.app.canvas.create_rectangle(i_max_x, i_min_y,  o_max_x, i_max_y, **select_outside),
            self.app.canvas.create_rectangle(o_min_x, i_max_y,  o_max_x, o_max_y, **select_outside),
            # Inner rectangle.
            self.app.canvas.create_rectangle(i_min_x, i_min_y,  i_max_x, i_max_y, **select_inside)
        )

        self.df = pd.DataFrame()

        # Create canvas cross-hair lines.
        xhair_opts = dict(dash=(3, 2), fill='white', state=tk.HIDDEN)
        self.lines = (self.app.canvas.create_line(0, 0, 0, self.height, **xhair_opts),
                      self.app.canvas.create_line(0, 0, self.width,  0, **xhair_opts))

    def cur_selection(self):
        return (self.start, self.end)

    def begin(self, event):
        self.hide()
        self.start = (event.x, event.y)  # Remember position (no drawing).

    def update(self, event):
        self.end = (event.x, event.y)
        self._update(event)
        # Current extrema of inner and outer rectangles.
        i_min_x, i_min_y,  i_max_x, i_max_y = self._get_coords()
        o_min_x, o_min_y,  o_max_x, o_max_y = 0, 0,  self.width, self.height
        # Update coords of all rectangles based on these extrema.
        self.app.canvas.coords(self.rects[0], o_min_x, o_min_y,  o_max_x, i_min_y),
        self.app.canvas.coords(self.rects[1], o_min_x, i_min_y,  i_min_x, i_max_y),
        self.app.canvas.coords(self.rects[2], i_max_x, i_min_y,  o_max_x, i_max_y),
        self.app.canvas.coords(self.rects[3], o_min_x, i_max_y,  o_max_x, o_max_y),
        self.app.canvas.coords(self.rects[4], i_min_x, i_min_y,  i_max_x, i_max_y),

        for rect in self.rects:  # Make sure all are now visible.
            self.app.canvas.itemconfigure(rect, state=tk.NORMAL)

    def _update(self, event):
        # Update cross-hair lines.
        self.app.canvas.coords(self.lines[0], event.x, 0, event.x, self.height)
        self.app.canvas.coords(self.lines[1], 0, event.y, self.width, event.y)
        self.show()

    def _get_coords(self):
        """ Determine coords of a polygon defined by the start and
            end points one of the diagonals of a rectangular area.
        """
        return (min((self.start[0], self.end[0])), min((self.start[1], self.end[1])),
                max((self.start[0], self.end[0])), max((self.start[1], self.end[1])))

    def reset(self):
        self.start = self.end = None

    def hide(self):
        self.app.canvas.itemconfigure(self.lines[0], state=tk.HIDDEN)
        self.app.canvas.itemconfigure(self.lines[1], state=tk.HIDDEN)
        for rect in self.rects:
            self.app.canvas.itemconfigure(rect, state=tk.HIDDEN)

    def show(self):
        self.app.canvas.itemconfigure(self.lines[0], state=tk.NORMAL)
        self.app.canvas.itemconfigure(self.lines[1], state=tk.NORMAL)

    def autodraw(self):
        """Setup automatic drawing; supports command option"""
        self.reset()
        self.app.canvas.bind("<Button-1>", self.begin)
        self.app.canvas.bind("<B1-Motion>", self.update)
        self.app.canvas.bind("<ButtonRelease-1>", self.quit)

    def quit(self, event):
        self.count+=1
        self.sample_label = simpledialog.askstring("Input",
                                                   "Sample label",
                                                   parent=self.root,
                                                   initialvalue=self.count)
        if self.sample_label:
            self.process_ROI()
            self.save_dataframe()
        self.hide()  # Hide cross-hairs.
        self.reset()

    def process_ROI(self):
        channel_selections = self.app.channel_listbox.curselection()
        if channel_selections == ():
            messagebox.showerror('Error','No color channel is selected. Please select at least one color channel and try again.')

        left, top = [i/self.app.canvas.aspect for i in self.start]
        right, bottom = [i/self.app.canvas.aspect for i in self.end]
        for i in range(int(self.app.n_lines.get())):
            if self.app.line_vals[i].get() < top or self.app.line_vals[i].get() > bottom:
                messagebox.showerror('Error',f'All line locators must be located in the rectangular ROI that you dragged. Reposition locator for line named "{self.app.line_name_vals[i].get()}" and re-drag your ROI.')
                return
 
        roi = self.app.img.crop((left, top, right, bottom))
        roi_gray = roi.convert('L')
        nleft, nright = self.calculate_LR_border(roi_gray)
        roi_tight_gray = roi_gray.crop((nleft, 0, nright, roi.size[1]))
        roi_tight_color = roi.crop((nleft, 0, nright, roi.size[1]))
        channel_lines = [255-np.mean(np.asarray(roi_channel),axis=1) for roi_channel in [roi_tight_gray]+list(roi_tight_color.split())]
        line_peaks = [self.find_lfa_peaks(channel_lines[i], top) for i in channel_selections]

        file_title = self.file_label+'-'+str(self.count)+'-'+self.sample_label
        save_path = os.path.normpath(os.path.join(self.dir_name, self.file_label, file_title+'.png'))

        n_channels = len(line_peaks)
        n_peaks = len(line_peaks[0][1])-1 # don't double-count background
        color_channels = [self.app.COLOR_CHANNELS[i] for i in channel_selections]
        features = [f'{self.app.line_name_vals[i].get()} peak' for i in range(n_peaks)] + ['background']
        data_types = ['index', 'signal']

        w, h = roi_tight_color.size
        fig, axes = plt.subplots(nrows=1, ncols=n_channels+1, sharex=False, sharey=True, width_ratios=[1]*(n_channels+1), figsize=[6*w/h*(n_channels+1),6])
        fig.suptitle(file_title)
        axes[0].imshow(roi_tight_color, aspect='auto')
        axes[0].set_xticks([])
        axes[0].set_ylabel('distance (pixels)')
        axes[0].set_anchor('E')
        for i in range(int(self.app.n_lines.get())):
            vals = [self.app.line_vals[i].get(),
                    self.app.line_vals[i].get() - float(self.app.interval_vals[i].get()),
                    self.app.line_vals[i].get() + float(self.app.interval_vals[i].get())
                ]
            for i, val in enumerate(vals):
                if val > 0 or val < len(line_peaks[0][0]):
                    axes[0].axhline(y = val - top, color = 'tab:gray', linestyle = '--' if i == 0 else '-')
        for i in range(len(channel_selections)):
            axes[i+1].plot(line_peaks[i][0], range(0,len(line_peaks[i][0])), color_channels[i])
            axes[i+1].plot(line_peaks[i][2], line_peaks[i][1], 'o', color=color_channels[i])
            axes[i+1].set(xlabel=f'{color_channels[i]} (signal)', xlim=[-25,255], xticks=[0,100,200])
            axes[i+1].grid(True, which='major', color='lightgray')
        pad = 0.1 # Padding around the edge of the figure
        fig.subplots_adjust(hspace=0, wspace=0, left=pad, right=1-pad, top=1-pad, bottom=pad)
      
        try:
            fig.savefig(save_path)
        except:
            messagebox.showerror('Error',f'Cannot save plot image. Check for open image file with name {save_path}')
        plt.close(fig)

        data=[[' '.join([c,f,t]), line_peaks[i][k+1][j]] for i,c in enumerate(color_channels) for k,t in enumerate(data_types) for j,f in enumerate(features) if f+t!='backgroundindex']
        df = pd.DataFrame([self.count]+[row[1] for row in data], index=['selection']+[row[0] for row in data], columns=[self.sample_label])
        self.df=pd.concat([self.df, df], axis=1)

    def calculate_LR_border(self, image):
        arr=np.asarray(image)
        mean_vertical=np.mean(arr,axis=0)
        gradient=np.gradient(mean_vertical)
        halfpoint=int(gradient.size//2)
        left=np.argmin(gradient[:halfpoint])
        right=np.argmin(gradient[halfpoint:])+halfpoint
        return left, right

    def find_lfa_peaks(self, line_profile, top):
        N = int(self.app.n_lines.get())
        filtered = savgol_filter(line_profile, 13, 2)
        lowest_length = np.clip(len(filtered)//2, 1, 50)-1
        lowest = np.sort(filtered)[0:lowest_length]
        background = np.mean(lowest) #+ 3*np.std(lowest)
        peaks_X,_=find_peaks(filtered)
        peaks_Y=filtered[peaks_X]

        X_intervals = [[int(line.get()-int(interval.get())-top), int(line.get()+int(interval.get())-top)] for line, interval in zip(self.app.line_vals[0:N], self.app.interval_vals[0:N])]
        peaks_XY_max = [max([[X, Y] for (X,Y) in zip(peaks_X, peaks_Y) if X >= a and X <= b], key=lambda x:x[1], default=[None, None]) for a, b, in X_intervals]
        peaks_XY_max.append([None, background])

        peaks_X_by_location, peaks_Y_by_location = zip(*peaks_XY_max)
        return filtered, list(peaks_X_by_location), list(peaks_Y_by_location)
    
    def save_dataframe(self):
        try:
            self.df.to_csv(self.csv_save_path, index=True)
        except:
            return
        
    def update_data(self):
        self.count = 0
        (self.dir_name, self.file_name) = os.path.split(self.app.img_path)
        (self.file_label, self.file_ext) = os.path.splitext(self.file_name)
        folder = os.path.normpath(os.path.join(self.dir_name, self.file_label))
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.csv_save_path = os.path.normpath(os.path.join(self.dir_name, self.file_label, self.file_label+'.csv'))

    def resize(self):
        self.width = self.app.canvas.winfo_width()
        self.height = self.app.canvas.winfo_height()

class Application(tk.PanedWindow):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = self.master

        self.controls = ttk.LabelFrame(self, text='Controls', padding=10, width=150)
        self.add(self.controls, padx=10, pady=10)
        self.images = ttk.LabelFrame(self, text='Image')
        self.add(self.images, padx=10, pady=10)

        self.canvas = tk.Canvas(self.images)
        self.canvas.aspect = 1.0
        self.canvas.place(relheight=1.0, relwidth=1.0)
        self.canvas.bind("<Configure>", self.resize_image)

        self.img_path = "./front.png"
        self.img = Image.open(self.img_path)
        self.img_tk = ImageTk.PhotoImage(self.img)
        self.img_container=self.canvas.create_image(0, 0, image=self.img_tk, anchor=tk.NW)
        
        # Create mouse position tracker that uses the function.
        self.posn_tracker = MousePositionTracker(self, self.parent)
        self.posn_tracker.autodraw()

        self.NUM_LINES = 3
        self.MAX_LINES = 4
        self.COLOR_CHANNELS = ['red', 'green', 'blue', 'gray']

        self.auto_rectangles = []

        self.create_controls()
    
    def create_controls(self):
        self.n_lines = tk.StringVar()
        self.n_lines.set(self.NUM_LINES) # set default number of controls on this line
        self.n_lines.trace_add('write', self.update_line_selection)
        self.open = ttk.Button(self.controls,
                               command=self.open_file,
                               text='Open image file...')
        self.analyze = ttk.Button(self.controls,
                                  command=self.auto_analysis,
                                  text='Start auto-analysis...')
        self.channel_choices = tk.StringVar(value=self.COLOR_CHANNELS)
        self.channel_listbox = tk.Listbox(self.controls,
                                          listvariable=self.channel_choices,
                                          selectmode="multiple",
                                          exportselection=0,
                                          height=0)
        self.channel_listbox.selection_set(0, len(self.channel_choices.get()))
        self.lines_entry = ttk.Entry(self.controls,
                                     textvariable=self.n_lines
                                    )
        self.line_name_vals = [tk.StringVar(self,
                                            'line '+str(i+1))
                               for i in range(self.MAX_LINES)]
        self.line_names = [ttk.Entry(self.controls,
                                     textvariable=self.line_name_vals[i])
                           for i in range(self.MAX_LINES)]
        self.line_vals = [tk.IntVar(self, 1) for i in range(self.MAX_LINES)]
        self.line_scales = [TickScale(self.controls,
                                      command=lambda value, index=i: self.update_scales(value, index),
                                      variable=self.line_vals[i],
                                      from_=1,
                                      orient=tk.VERTICAL,
                                      to=self.img.height,
                                      showvalue=True,
                                      resolution=1,
                                      length=100,
                                      labelpos=tk.W)
                            for i in range(self.MAX_LINES)]
        self.line_spinboxes = [ttk.Spinbox(self.controls,
                                           command=lambda value=None, index=i: self.update_scales(value, index),
                                           from_=1,
                                           to=self.img.height,
                                           textvariable=self.line_vals[i],
                                           width=5)
                               for i in range(self.MAX_LINES)]
        self.interval_vals = [tk.StringVar(self, 15) for _ in range(self.MAX_LINES)]
        self.interval_spinboxes = [ttk.Spinbox(self.controls,
                                               command=lambda value=None, index=i: self.update_scales(value, index),
                                               from_=1,
                                               to=100,
                                               textvariable=self.interval_vals[i],
                                               width=5)
                                   for i in range(self.MAX_LINES)]
        self.lines = [self.canvas.create_line(0,
                                              scale.get(),
                                              self.canvas.winfo_width(),
                                              scale.get())
                      for scale in self.line_scales]
        self.rectangles = [self.canvas.create_rectangle(0,
                                                        max(0,
                                                            self.line_scales[i].get() - float(self.interval_spinboxes[i].get())),
                                                        self.canvas.winfo_width(),
                                                        min(self.canvas.winfo_height(),
                                                            self.line_scales[i].get() + float(self.interval_spinboxes[i].get())),
                                                        fill='gray',
                                                        stipple='gray25')
                           for i in range(self.MAX_LINES)]

        n = 0
        self.open.grid(column=0, row=n, columnspan=2, sticky=tk.EW)
        n += 1
        self.analyze.grid(column=0, row=n, columnspan=2, sticky=tk.EW)
        n += 1
        tk.Frame(self.controls, bd=10, height=4, background='black').grid(row=n, columnspan=2, sticky=tk.EW)
        n += 1
        tk.Label(self.controls, text='(Un)select color channels').grid(column=0, row=n, columnspan=2, sticky=tk.EW)
        n += 1
        self.channel_listbox.grid(column=0, row=n, columnspan=2, sticky=tk.EW)
        n += 1
        tk.Frame(self.controls, bd=10, height=4, background='black').grid(row=n, columnspan=2, sticky=tk.EW)
        n += 1
        tk.Label(self.controls, text='Select no. of lines (1-4)').grid(column=0, row=n, columnspan=2, sticky=tk.EW)
        n += 1
        self.lines_entry.grid(column=0, row=n, columnspan=2, sticky=tk.EW)
        n += 1
        tk.Frame(self.controls, bd=10, height=4, background='black').grid(row=n, columnspan=2, sticky=tk.EW)
        n += 1
        tk.Label(self.controls, text='Label and locate line(s)').grid(column=0, row=n, columnspan=2, sticky=tk.EW)
        n += 1
        [self.line_names[i].grid(column=0, row=n+3*i, columnspan=2, sticky=tk.EW) for i in range(self.NUM_LINES)]
        n += 1
        [self.line_spinboxes[i].grid(column=0, row=n+3*i, sticky=tk.NW) for i in range(self.NUM_LINES)]
        [self.interval_spinboxes[i].grid(column=1, row=n+3*i, sticky=tk.NE) for i in range(self.NUM_LINES)]
        [self.line_scales[i].grid(column=0, row=n+1+3*i, columnspan=2, sticky=tk.W) for i in range(self.NUM_LINES)]
            
    def update_scales(self, val, i):
        scale_val = float(self.line_scales[i].get())
        spin_val = float(self.interval_spinboxes[i].get())
        self.canvas.coords(self.lines[i], 0, int(self.canvas.aspect * scale_val), self.canvas.winfo_width(), int(self.canvas.aspect * scale_val))
        self.canvas.coords(self.rectangles[i],
                            0,
                            max(0, self.canvas.aspect * (scale_val - spin_val)),
                            self.canvas.winfo_width(),
                            min(self.canvas.winfo_height(), self.canvas.aspect * (scale_val + spin_val)))

    def resize_image(self, event):  # Must accept these arguments.
        N = int(self.n_lines.get())
        c_w, i_w, c_h, i_h = self.canvas.winfo_width(), self.img.width, self.canvas.winfo_height(), self.img.height
        self.canvas.aspect = c_w / i_w if c_w / i_w < c_h / i_h else c_h / i_h
        self.img_tk = ImageTk.PhotoImage(self.img.resize((int(i_w * self.canvas.aspect), int(i_h * self.canvas.aspect)), Image.LANCZOS))
        self.canvas.itemconfig(self.img_container, image=self.img_tk)
        self.posn_tracker.resize()
        [self.line_scales[i].configure(to=self.img.height) for i in range(N)]
        [self.line_spinboxes[i].configure(to=self.img.height) for i in range(N)]
        [self.canvas.delete(rectangle) for rectangle in self.auto_rectangles]
        [self.update_scales(None, i) for i in range(N)]

    def update_line_selection(self, callback_name, callback_index, callback_method):
        if self.n_lines.get().isdigit():
            N = int(self.n_lines.get())
            if N > 0 and N < 5:
                n = 10
                [entry.grid_forget() for entry in self.line_names]
                [scale_spinbox.grid_forget() for scale_spinbox in self.line_spinboxes]
                [scale.grid_forget() for scale in self.line_scales]
                [spinbox.grid_forget() for spinbox in self.interval_spinboxes]
                [self.line_names[i].grid(column=0, row=n+3*i, columnspan=2) for i in range(N)]
                n += 1
                [self.line_spinboxes[i].grid(column=0, row=n+3*i) for i in range(N)]
                [self.interval_spinboxes[i].grid(column=1, row=n+3*i) for i in range(N)]
                [self.line_scales[i].grid(column=0, row=n+1+3*i) for i in range(N)]
                
                self.resize_image(None)

    def open_file(self, event=None):
        N = int(self.n_lines.get())
        self.img_path = filedialog.askopenfilename(defaultextension=".txt",
                                                   filetypes=[("Image files", "*.png"), ("Image files", "*.tif"), ("All Files", "*.*")])
        print(f'{self.img_path}')
        root.title(f'{os.path.basename(self.img_path)}')
        try:
            self.img = Image.open(self.img_path)
        except:
            return
        self.resize_image(None)
        [self.update_scales(None, i) for i in range(N)]
        self.posn_tracker.update_data()
        # custom control settings for testing code on NAATOS strip images
        [self.line_name_vals[i].set(s) for i, s in enumerate(['FC', 'IPC', 'TB'])]
        [self.line_vals[i].set(n) for i, n in enumerate([425, 485, 530])]
        [self.update_scales(None, i) for i in range(N)]
        [self.channel_listbox.selection_clear(i) for i in [1, 2, 3]]

    def auto_analysis(self, event=None):
        if self.img != None:
            top = int(390*self.canvas.aspect)
            bottom = int(560*self.canvas.aspect)
            left = int(32*self.canvas.aspect)
            right = int(120*self.canvas.aspect)
            last = int(self.img.size[0]*self.canvas.aspect)
            spacing = int(101*self.canvas.aspect)
            left_list = [pos for pos in range(left, last, spacing)]
            right_list = [pos for pos in range(right, last, spacing)]
            self.auto_rectangles = [self.canvas.create_rectangle(L, top, R, bottom, outline='yellow') for L, R in zip(left_list, right_list)]
            for L, R in zip(left_list, right_list):
                if L < last:
                    self.posn_tracker.start = (L, top)
                    self.posn_tracker.end = (R, bottom)
                    rectangle = self.canvas.create_rectangle(L, top, R, bottom, outline='red')
                    self.posn_tracker.quit(None)
                    self.canvas.delete(rectangle)

if __name__ == '__main__':
    root = tk.Tk()
    root.title('Image Cropper')
    root.state('zoomed')
    root.minsize(1372, 600)

    app = Application(root, orient=tk.HORIZONTAL, sashwidth=5)
    app.place(anchor=tk.NW, relwidth=1.0, relheight=1.0)
    app.update()

    app.mainloop()
